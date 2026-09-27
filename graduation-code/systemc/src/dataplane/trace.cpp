#include "dataplane/trace.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace hw::dataplane {

const char* status_name(hw::command::StatusCode status) {
    switch (status) {
    case hw::command::StatusCode::Ok: return "OK";
    case hw::command::StatusCode::BadCommand: return "BAD_COMMAND";
    case hw::command::StatusCode::BadDescriptor: return "BAD_DESCRIPTOR";
    case hw::command::StatusCode::AddressFault: return "ADDRESS_FAULT";
    case hw::command::StatusCode::BufferFull: return "BUFFER_FULL";
    case hw::command::StatusCode::DependencyFailed: return "DEPENDENCY_FAILED";
    case hw::command::StatusCode::PolicyNotSupported: return "POLICY_NOT_SUPPORTED";
    case hw::command::StatusCode::PivotNotFound: return "PIVOT_NOT_FOUND";
    case hw::command::StatusCode::PivotUnstable: return "PIVOT_UNSTABLE";
    case hw::command::StatusCode::NumericOverflow: return "NUMERIC_OVERFLOW";
    case hw::command::StatusCode::QuantizationSaturation: return "QUANTIZATION_SATURATION";
    case hw::command::StatusCode::PrecisionRetry: return "PRECISION_RETRY";
    case hw::command::StatusCode::StaticPivotFailure: return "STATIC_PIVOT_FAILURE";
    case hw::command::StatusCode::NonFiniteValue: return "NON_FINITE_VALUE";
    case hw::command::StatusCode::Timeout: return "TIMEOUT";
    case hw::command::StatusCode::Aborted: return "ABORTED";
    }
    return "UNKNOWN";
}

namespace {

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

void write_transactions_csv(
    const std::string& path,
    const std::vector<TransactionRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << "layer,client,request_id,has_command,command_id,node_id,region,address,"
           "length,write,status,reason,issue_cycle,accept_cycle,data_cycle,"
           "response_cycle,bytes,request_wait_cycles,memory_service_cycles,"
           "response_backpressure_cycles\n";
    for (const auto& row : rows) {
        out << csv_escape(row.layer) << ',' << csv_escape(row.client) << ','
            << row.request_id << ',' << (row.has_command ? 1 : 0) << ','
            << row.command_id << ',' << row.node_id << ','
            << csv_escape(row.region) << ',' << row.address << ',' << row.length
            << ',' << (row.write ? 1 : 0) << ','
            << status_name(row.status) << ',' << csv_escape(row.reason) << ','
            << row.issue_cycle << ',' << row.accept_cycle << ','
            << row.data_cycle << ',' << row.response_cycle << ',' << row.bytes
            << ',' << row.request_wait_cycles << ',' << row.memory_service_cycles
            << ',' << row.response_backpressure_cycles << '\n';
    }
}

void write_buffers_csv(const std::string& path, const std::vector<BufferRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << "cycle,slot,generation,owner,from_state,to_state,reason\n";
    for (const auto& row : rows) {
        out << row.cycle << ',' << row.slot << ',' << row.generation << ','
            << csv_escape(row.owner) << ',' << csv_escape(row.from_state) << ','
            << csv_escape(row.to_state) << ',' << csv_escape(row.reason) << '\n';
    }
}

}  // namespace

void TraceRecorder::write_csv(const std::string& output_dir) const {
    std::filesystem::create_directories(output_dir);
    write_transactions_csv(output_dir + "/transactions.csv", transactions_);
    write_buffers_csv(output_dir + "/buffers.csv", buffers_);
}

}  // namespace hw::dataplane
