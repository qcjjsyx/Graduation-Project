#include "dataplane/dma_engine.hpp"

#include <cstring>

#include <algorithm>
#include <limits>

namespace hw::dataplane {

namespace {

AccessKind access_kind_for(hw::command::DataFormat format) {
    return format == hw::command::DataFormat::Int32 ? AccessKind::Int32Data
                                                    : AccessKind::Fp32Data;
}

std::uint64_t multiply_checked(
    std::uint64_t lhs,
    std::uint64_t rhs,
    bool* overflow) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        *overflow = true;
        return 0;
    }
    return lhs * rhs;
}

std::uint64_t add_checked(
    std::uint64_t lhs,
    std::uint64_t rhs,
    bool* overflow) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        *overflow = true;
        return 0;
    }
    return lhs + rhs;
}

// End offset of a strided row range: offset + (rows-1)*stride + row_bytes.
// Returns true when any step of the computation overflows u64, including the
// addition of `offset` itself.
bool row_range_overflow(
    std::uint64_t offset,
    std::uint64_t stride,
    std::uint32_t rows,
    std::uint64_t row_bytes,
    std::uint64_t* end) {
    if (rows == 0) {
        *end = offset;
        return false;
    }
    bool overflow = false;
    const std::uint64_t steps = rows - 1;
    const std::uint64_t step_bytes = multiply_checked(steps, stride, &overflow);
    const std::uint64_t span = add_checked(step_bytes, row_bytes, &overflow);
    const std::uint64_t total = add_checked(offset, span, &overflow);
    if (overflow) {
        return true;
    }
    *end = total;
    return false;
}

}  // namespace

const char* dma_direction_name(DmaDirection direction) {
    switch (direction) {
    case DmaDirection::MemoryToBuffer: return "ddr_to_buffer";
    case DmaDirection::BufferToMemory: return "buffer_to_ddr";
    }
    return "unknown";
}

DmaEngine::DmaEngine(
    DmaConfig config,
    MemoryService& memory,
    BufferManager& buffers,
    TraceRecorder* trace)
    : config_(config), memory_(memory), buffers_(buffers), trace_(trace) {}

void DmaEngine::set_request(const DmaRequest& request) {
    request_ = request;
}

void DmaEngine::set_request_valid(bool valid) {
    if (valid && !request_valid_) {
        issue_recorded_ = false;
    }
    request_valid_ = valid;
    if (!valid) {
        issue_recorded_ = false;
    }
}

bool DmaEngine::request_ready() const {
    return phase_ == Phase::Idle && !response_valid_;
}

bool DmaEngine::busy() const {
    return phase_ != Phase::Idle;
}

std::uint64_t DmaEngine::beats_per_row() const {
    const std::uint64_t row_bytes = active_.cols * active_.element_bytes;
    return std::max<std::uint64_t>(
        1, (row_bytes + config_.beat_bytes - 1) / config_.beat_bytes);
}

std::uint64_t DmaEngine::total_bytes() const {
    bool overflow = false;
    const std::uint64_t elements =
        multiply_checked(active_.rows, active_.cols, &overflow);
    const std::uint64_t bytes = multiply_checked(elements, active_.element_bytes, &overflow);
    return overflow ? 0 : bytes;
}

DmaEngine::Beat DmaEngine::current_beat() const {
    const std::uint64_t row_bytes = active_.cols * active_.element_bytes;
    Beat beat;
    beat.row = row_;
    beat.index_in_row = index_in_row_;
    beat.ddr_address = active_.ddr_region.base_addr + active_.ddr_offset +
                       static_cast<std::uint64_t>(row_) * active_.ddr_row_stride +
                       static_cast<std::uint64_t>(index_in_row_) * config_.beat_bytes;
    beat.buffer_offset = active_.buffer_offset +
                         static_cast<std::uint64_t>(row_) * active_.buffer_row_stride +
                         static_cast<std::uint64_t>(index_in_row_) * config_.beat_bytes;
    const std::uint64_t consumed =
        static_cast<std::uint64_t>(index_in_row_) * config_.beat_bytes;
    beat.length = std::min<std::uint64_t>(config_.beat_bytes, row_bytes - consumed);
    return beat;
}

bool DmaEngine::prepare_transfer(std::uint64_t cycle, std::string* reason) {
    (void)cycle;
    const auto& region = active_.ddr_region;
    if (region.format != hw::command::DataFormat::Fp32 &&
        region.format != hw::command::DataFormat::Int32) {
        *reason = "DMA only supports FP32/INT32 device data regions";
        return false;
    }
    if (active_.element_bytes != region.element_bytes ||
        active_.element_bytes == 0) {
        *reason = "DMA element size does not match the region data format";
        return false;
    }
    if (config_.beat_bytes == 0 ||
        config_.beat_bytes % active_.element_bytes != 0) {
        *reason = "DMA beat size must be a non-zero multiple of the element size";
        return false;
    }
    if (active_.rows == 0 || active_.cols == 0) {
        // Zero-size transfer: legal handle and offset checks only, no data
        // movement. Addresses may not be abused to bypass region checks.
        const AccessKind kind = access_kind_for(region.format);
        if (active_.ddr_offset > region.byte_size) {
            *reason = "zero-size DMA offset is outside the DDR region";
            return false;
        }
        const auto checked = memory_.check(
            region.base_addr,
            region.byte_size,
            kind,
            active_.direction == DmaDirection::BufferToMemory);
        if (!checked.ok) {
            *reason = checked.reason;
            return false;
        }
        return true;
    }

    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(active_.cols) * active_.element_bytes;
    const std::uint64_t ddr_stride = active_.ddr_row_stride != 0
                                         ? active_.ddr_row_stride
                                         : row_bytes;
    const std::uint64_t buffer_stride = active_.buffer_row_stride != 0
                                            ? active_.buffer_row_stride
                                            : row_bytes;
    if (ddr_stride < row_bytes || buffer_stride < row_bytes) {
        *reason = "DMA row stride is smaller than one row";
        return false;
    }

    std::uint64_t ddr_end = 0;
    if (row_range_overflow(
            active_.ddr_offset, ddr_stride, active_.rows, row_bytes, &ddr_end)) {
        *reason = "DMA DDR range overflows u64";
        return false;
    }
    if (ddr_end > region.byte_size) {
        *reason = "DMA DDR range exceeds the region";
        return false;
    }

    bool overflow = false;
    const std::uint64_t total =
        multiply_checked(
            multiply_checked(active_.rows, active_.cols, &overflow),
            active_.element_bytes,
            &overflow);
    if (overflow) {
        *reason = "DMA transfer size overflows u64";
        return false;
    }
    total_transfer_bytes_ = total;

    std::uint64_t buffer_end = 0;
    if (row_range_overflow(
            active_.buffer_offset, buffer_stride, active_.rows, row_bytes,
            &buffer_end)) {
        *reason = "DMA buffer range overflows u64";
        return false;
    }
    if (buffer_end > buffers_.config().capacity_bytes) {
        *reason = "DMA buffer range exceeds the scratchpad capacity";
        return false;
    }
    buffer_span_bytes_ = buffer_end - active_.buffer_offset;

    const bool write_ddr = active_.direction == DmaDirection::BufferToMemory;
    const AccessKind kind = access_kind_for(region.format);
    // Validate every row of the whole transfer before any beat is issued.
    // ddr_end was checked against region.byte_size, and the loader guarantees
    // base_addr + region.byte_size does not overflow, so absolute addresses
    // stay inside the region and the image.
    for (std::uint32_t row = 0; row < active_.rows; ++row) {
        bool overflow = false;
        const std::uint64_t row_offset = add_checked(
            active_.ddr_offset,
            multiply_checked(row, ddr_stride, &overflow),
            &overflow);
        const std::uint64_t address =
            add_checked(region.base_addr, row_offset, &overflow);
        if (overflow) {
            *reason = "DMA DDR row " + std::to_string(row) + " overflows u64";
            return false;
        }
        const auto checked = memory_.check(
            address, row_bytes, kind, write_ddr);
        if (!checked.ok) {
            *reason = "DMA DDR row " + std::to_string(row) + ": " + checked.reason;
            return false;
        }
    }
    return true;
}

void DmaEngine::accept_request(std::uint64_t cycle) {
    // The handshake consumes the valid flag: holding valid high after
    // acceptance must not re-accept the same request.
    request_valid_ = false;
    issue_recorded_ = false;
    active_ = request_;
    // Initialize every transaction counter before validation or early returns
    // so zero-length and failed requests cannot inherit a previous transfer's
    // byte counts.
    bytes_done_ = 0;
    total_transfer_bytes_ = 0;
    buffer_span_bytes_ = 0;
    memory_pending_ = false;
    memory_response_ready_ = false;
    have_lease_ = false;
    beat_read_progress_ = 0;
    beat_write_progress_ = 0;
    beat_staged_ = false;
    memory_wait_cycles_ = 0;
    buffer_wait_cycles_ = 0;
    pipeline_wait_cycles_ = 0;
    memory_backpressure_cycles_ = 0;
    response_backpressure_cycles_ = 0;
    response_ = DmaResponse{};
    response_.request_id = active_.request_id;
    response_.has_command = active_.has_command;
    response_.command_id = active_.command_id;
    response_.node_id = active_.node_id;
    response_.issue_cycle = request_issue_cycle_;
    response_.accept_cycle = cycle;
    response_.request_wait_cycles = cycle - request_issue_cycle_;
    ++accepted_count_;

    if (active_.ddr_row_stride == 0) {
        active_.ddr_row_stride =
            static_cast<std::uint64_t>(active_.cols) * active_.element_bytes;
    }
    if (active_.buffer_row_stride == 0) {
        active_.buffer_row_stride =
            static_cast<std::uint64_t>(active_.cols) * active_.element_bytes;
    }

    std::string reason;
    if (!prepare_transfer(cycle, &reason)) {
        fail_transfer(
            hw::command::StatusCode::AddressFault, reason, cycle);
        return;
    }

    const std::uint64_t total = active_.rows == 0 || active_.cols == 0
                                    ? 0
                                    : total_transfer_bytes_;
    const std::uint64_t span = active_.rows == 0 || active_.cols == 0
                                   ? 0
                                   : buffer_span_bytes_;
    BufferResult buffer_result;
    if (active_.direction == DmaDirection::MemoryToBuffer) {
        buffer_result = buffers_.begin_load(
            active_.buffer, active_.owner, active_.buffer_offset, span, cycle);
    } else {
        buffer_result = buffers_.begin_writeback(
            active_.buffer, active_.owner, active_.buffer_offset, span, cycle);
    }
    if (!buffer_result.ok) {
        fail_transfer(
            buffer_result.status, buffer_result.reason, cycle,
            buffer_result.transient);
        return;
    }
    lease_ = buffer_result.lease;
    have_lease_ = true;

    if (total == 0) {
        if (active_.direction == DmaDirection::MemoryToBuffer) {
            buffers_.finish_load(lease_.lease_id, 0, cycle);
        } else {
            buffers_.finish_writeback(lease_.lease_id, 0, cycle);
        }
        have_lease_ = false;
        response_.first_beat_cycle = cycle;
        response_.last_beat_cycle = cycle;
        finish_transfer(cycle);
        return;
    }

    total_transfer_bytes_ = total;
    row_ = 0;
    index_in_row_ = 0;
    beats_per_row_ = beats_per_row();
    next_beat_cycle_ = cycle + config_.startup_cycles;
    first_beat_started_ = false;
    phase_ = Phase::Active;
}

void DmaEngine::start_next_beat(std::uint64_t cycle) {
    const Beat beat = current_beat();
    if (active_.direction == DmaDirection::BufferToMemory) {
        if (!first_beat_started_) {
            response_.first_beat_cycle = cycle;
            first_beat_started_ = true;
        }
        // On-chip budget gate with segment progress: the whole beat is staged
        // exactly once before one DDR write is submitted. Staging is
        // (re)initialized only when a NEW beat starts; budget waits and DDR
        // backpressure retries keep the already loaded bytes and progress.
        if (!beat_staged_) {
            beat_buffer_data_.assign(static_cast<std::size_t>(beat.length), 0);
            beat_read_progress_ = 0;
            beat_staged_ = true;
        }
        while (beat_read_progress_ < beat.length) {
            const std::uint32_t want =
                static_cast<std::uint32_t>(beat.length - beat_read_progress_);
            std::uint32_t take = want;
            if (arbiter_ != nullptr) {
                take = arbiter_->acquire(
                    OnChipDomain::Read, OnChipClientKind::Dma, 0, want);
                if (take == 0) {
                    ++buffer_wait_cycles_;
                    return;
                }
            }
            std::string reason;
            std::vector<std::uint8_t> segment;
            const auto read_status = buffers_.lease_read(
                lease_.lease_id, beat.buffer_offset + beat_read_progress_, take,
                segment, cycle, &reason);
            if (read_status == hw::command::StatusCode::BufferFull) {
                ++buffer_wait_cycles_;
                return;
            }
            if (read_status != hw::command::StatusCode::Ok) {
                fail_transfer(read_status, reason, cycle);
                return;
            }
            std::memcpy(
                beat_buffer_data_.data() + beat_read_progress_, segment.data(),
                segment.size());
            beat_read_progress_ += static_cast<std::uint32_t>(segment.size());
            scratch_read_bytes_ += segment.size();
        }
    } else if (!first_beat_started_) {
        response_.first_beat_cycle = cycle;
        first_beat_started_ = true;
    }
    MemoryRequest request;
    request.request_id = memory_.next_request_id();
    request.client = MemoryClient::Dma;
    request.has_command = active_.has_command;
    request.command_id = active_.command_id;
    request.node_id = active_.node_id;
    request.label = active_.ddr_region.name;

    if (active_.direction == DmaDirection::MemoryToBuffer) {
        request.write = false;
        request.address = beat.ddr_address;
        request.length = beat.length;
        request.kind = access_kind_for(active_.ddr_region.format);
        if (!memory_.submit(request, cycle)) {
            ++memory_backpressure_cycles_;
            return;
        }
        memory_pending_ = true;
        pending_memory_id_ = request.request_id;
        return;
    }

    // BufferToMemory: the whole beat is already staged above under the read
    // budget; do NOT re-read it here (that would double-count scratchpad
    // traffic and re-read data on DDR backpressure retries).
    request.write = true;
    request.address = beat.ddr_address;
    request.length = beat.length;
    request.kind = access_kind_for(active_.ddr_region.format);
    request.write_data = beat_buffer_data_;
    if (!memory_.submit(request, cycle)) {
        ++memory_backpressure_cycles_;
        return;
    }
    memory_pending_ = true;
    pending_memory_id_ = request.request_id;
}

void DmaEngine::complete_beat(std::uint64_t cycle) {
    const Beat beat = current_beat();
    const std::uint64_t length = active_.rows == 0 || active_.cols == 0
                                     ? 0
                                     : beat.length;
    bytes_done_ += length;
    response_.last_beat_cycle = cycle;
    if (active_.direction == DmaDirection::MemoryToBuffer) {
        read_bytes_ += length;
        ddr_read_bytes_ += length;
    } else {
        write_bytes_ += length;
        ddr_write_bytes_ += length;
    }

    memory_response_ready_ = false;
    memory_pending_ = false;
    beat_read_progress_ = 0;
    beat_write_progress_ = 0;
    beat_staged_ = false;

    ++index_in_row_;
    if (index_in_row_ >= beats_per_row_) {
        index_in_row_ = 0;
        ++row_;
        if (row_ >= active_.rows) {
            finish_transfer(cycle);
            return;
        }
    }
    next_beat_cycle_ = cycle + config_.beat_interval_cycles;
}

void DmaEngine::finish_transfer(std::uint64_t cycle) {
    hw::command::StatusCode status = hw::command::StatusCode::Ok;
    std::string reason;
    if (have_lease_) {
        const std::uint64_t span = buffer_span_bytes_;
        status = active_.direction == DmaDirection::MemoryToBuffer
                     ? buffers_.finish_load(lease_.lease_id, span, cycle, &reason)
                     : buffers_.finish_writeback(
                           lease_.lease_id, span, cycle, &reason);
        have_lease_ = false;
    }
    if (status != hw::command::StatusCode::Ok) {
        fail_transfer(status, reason, cycle);
        return;
    }
    response_.status = hw::command::StatusCode::Ok;
    response_.bytes = bytes_done_;
    response_.read_bytes = active_.direction == DmaDirection::MemoryToBuffer
                               ? bytes_done_
                               : 0;
    response_.write_bytes = active_.direction == DmaDirection::BufferToMemory
                                ? bytes_done_
                                : 0;
    response_.finish_cycle = cycle;
    response_.memory_wait_cycles = memory_wait_cycles_;
    response_.buffer_wait_cycles = buffer_wait_cycles_;
    response_.pipeline_wait_cycles = pipeline_wait_cycles_;
    response_.memory_backpressure_cycles = memory_backpressure_cycles_;
    ++completed_count_;
    publish_response(cycle);
}

void DmaEngine::fail_transfer(
    hw::command::StatusCode status,
    const std::string& reason,
    std::uint64_t cycle,
    bool retryable) {
    if (have_lease_) {
        buffers_.abort_lease(lease_.lease_id, cycle);
        have_lease_ = false;
    }
    memory_pending_ = false;
    memory_response_ready_ = false;
    response_.status = status;
    response_.reason = reason;
    response_.retryable = retryable && bytes_done_ == 0;
    response_.bytes = bytes_done_;
    response_.read_bytes = 0;
    response_.write_bytes = 0;
    response_.finish_cycle = cycle;
    response_.memory_wait_cycles = memory_wait_cycles_;
    response_.buffer_wait_cycles = buffer_wait_cycles_;
    response_.pipeline_wait_cycles = pipeline_wait_cycles_;
    response_.memory_backpressure_cycles = memory_backpressure_cycles_;
    ++failed_count_;
    publish_response(cycle);
}

void DmaEngine::publish_response(std::uint64_t cycle) {
    (void)cycle;
    response_valid_ = true;
    phase_ = Phase::ResponseHold;

    pending_trace_row_ = TransactionRow{};
    pending_trace_row_.layer = "dma";
    pending_trace_row_.client = "dma";
    pending_trace_row_.request_id = response_.request_id;
    pending_trace_row_.has_command = response_.has_command;
    pending_trace_row_.command_id = response_.command_id;
    pending_trace_row_.node_id = response_.node_id;
    pending_trace_row_.region = active_.ddr_region.name;
    pending_trace_row_.address = active_.ddr_region.base_addr + active_.ddr_offset;
    pending_trace_row_.length = total_transfer_bytes_;
    pending_trace_row_.write = active_.direction == DmaDirection::BufferToMemory;
    pending_trace_row_.status = response_.status;
    pending_trace_row_.reason = response_.reason;
    pending_trace_row_.issue_cycle = response_.issue_cycle;
    pending_trace_row_.accept_cycle = response_.accept_cycle;
    pending_trace_row_.data_cycle = response_.last_beat_cycle;
    pending_trace_row_.issue_cycle = response_.issue_cycle;
    pending_trace_row_.bytes = response_.bytes;
    pending_trace_row_.request_wait_cycles = response_.request_wait_cycles;
    pending_trace_row_.memory_service_cycles = response_.memory_wait_cycles;
    pending_trace_row_.response_backpressure_cycles =
        response_.response_backpressure_cycles;
    have_pending_trace_row_ = true;
}

void DmaEngine::flush_trace_row() {
    if (trace_ != nullptr && have_pending_trace_row_) {
        pending_trace_row_.response_backpressure_cycles =
            response_.response_backpressure_cycles;
        trace_->add_transaction(pending_trace_row_);
    }
    have_pending_trace_row_ = false;
}

void DmaEngine::tick(std::uint64_t cycle) {
    // Record the issue cycle the first time the caller asserts valid, even
    // while the engine is busy; this is request wait, not transfer time.
    if (request_valid_ && !issue_recorded_) {
        request_issue_cycle_ = cycle;
        issue_recorded_ = true;
    }
    if (phase_ == Phase::ResponseHold) {
        if (response_valid_ && !response_ready_) {
            ++response_backpressure_cycles_;
            response_.response_backpressure_cycles = response_backpressure_cycles_;
        }
        if (response_ready_) {
            response_valid_ = false;
            flush_trace_row();
            phase_ = Phase::Idle;
            response_ = DmaResponse{};
            memory_wait_cycles_ = 0;
            buffer_wait_cycles_ = 0;
            pipeline_wait_cycles_ = 0;
            memory_backpressure_cycles_ = 0;
            response_backpressure_cycles_ = 0;
        }
        return;
    }
    if (phase_ == Phase::Idle) {
        if (request_valid_ && request_ready()) {
            accept_request(cycle);
        }
        return;
    }

    // Active transfer.
    if (memory_pending_) {
        ++memory_wait_cycles_;
        return;
    }
    if (memory_response_ready_) {
        if (pending_memory_response_.status != hw::command::StatusCode::Ok) {
            memory_response_ready_ = false;
            fail_transfer(
                pending_memory_response_.status,
                pending_memory_response_.reason,
                cycle);
            return;
        }
        if (active_.direction == DmaDirection::MemoryToBuffer) {
            const Beat beat = current_beat();
            const std::size_t total = pending_memory_response_.read_data.size();
            // On-chip budget gate with segment progress: a grant smaller than
            // the beat is written partially and resumed next cycle.
            std::size_t take = total - beat_write_progress_;
            if (take > 0) {
                if (arbiter_ != nullptr) {
                    const std::uint32_t granted = arbiter_->acquire(
                        OnChipDomain::Write,
                        OnChipClientKind::Dma,
                        0,
                        static_cast<std::uint32_t>(take));
                    if (granted == 0) {
                        ++buffer_wait_cycles_;
                        return;
                    }
                    take = std::min<std::size_t>(take, granted);
                }
                std::vector<std::uint8_t> segment(
                    pending_memory_response_.read_data.begin() +
                        static_cast<std::ptrdiff_t>(beat_write_progress_),
                    pending_memory_response_.read_data.begin() +
                        static_cast<std::ptrdiff_t>(beat_write_progress_ + take));
                std::string reason;
                const auto write_status = buffers_.lease_write(
                    lease_.lease_id, beat.buffer_offset + beat_write_progress_,
                    segment, cycle, &reason);
                if (write_status == hw::command::StatusCode::BufferFull) {
                    ++buffer_wait_cycles_;
                    return;
                }
                if (write_status != hw::command::StatusCode::Ok) {
                    memory_response_ready_ = false;
                    fail_transfer(write_status, reason, cycle);
                    return;
                }
                beat_write_progress_ += static_cast<std::uint32_t>(take);
                scratch_write_bytes_ += take;
            }
            if (beat_write_progress_ < total) {
                ++buffer_wait_cycles_;
                return;  // budget exhausted mid-beat
            }
            beat_write_progress_ = 0;
        }
        memory_response_ready_ = false;
        complete_beat(cycle);
        return;
    }
    if (cycle < next_beat_cycle_) {
        ++pipeline_wait_cycles_;
        return;
    }
    start_next_beat(cycle);
}

void DmaEngine::deliver_memory_response(const MemoryResponse& response) {
    if (!memory_pending_ || response.request_id != pending_memory_id_) {
        return;  // stale or not ours
    }
    memory_pending_ = false;
    pending_memory_response_ = response;
    memory_response_ready_ = true;
}

void DmaEngine::reset() {
    if (have_lease_) {
        buffers_.abort_lease(lease_.lease_id, 0);
        have_lease_ = false;
    }
    // Drop any request that was waiting before reset; it must not be accepted
    // after reset with a stale buffer handle.
    request_valid_ = false;
    memory_pending_ = false;
    memory_response_ready_ = false;
    response_valid_ = false;
    response_ready_ = false;
    have_pending_trace_row_ = false;
    issue_recorded_ = false;
    phase_ = Phase::Idle;
    response_ = DmaResponse{};
    beat_read_progress_ = 0;
    beat_write_progress_ = 0;
    beat_staged_ = false;
    memory_wait_cycles_ = 0;
    buffer_wait_cycles_ = 0;
    pipeline_wait_cycles_ = 0;
    memory_backpressure_cycles_ = 0;
    response_backpressure_cycles_ = 0;
}

}  // namespace hw::dataplane
