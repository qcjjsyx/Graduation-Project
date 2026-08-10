#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_types.hpp"

namespace hw {

class NodeTaskCodec {
public:
    static constexpr std::size_t RECORD_SIZE = 128;
    using Record = std::array<std::uint8_t, RECORD_SIZE>;

    static NodeTask decode(const Record& record) {
        NodeTask task{};
        task.node_id = read_u16(record, 0);
        task.flags = read_u16(record, 2);
        task.parent_id = read_u16(record, 4);
        task.children_count = read_u16(record, 6);
        task.total_dim = read_u32(record, 8);
        task.pivot_dim = read_u32(record, 12);
        task.tile_count = read_u32(record, 16);
        task.tail_dim = read_u32(record, 20);
        task.map_table_bytes = read_u32(record, 24);
        task.reserved = read_u32(record, 28);
        task.front_q_addr = read_u64(record, 32);
        task.front_e_addr = read_u64(record, 40);
        task.update_q_addr = read_u64(record, 48);
        task.update_e_addr = read_u64(record, 56);
        task.map_table_addr = read_u64(record, 64);
        task.l_factor_addr = read_u64(record, 72);
        task.u_factor_addr = read_u64(record, 80);
        task.p_vector_addr = read_u64(record, 88);
        task.node_meta_addr = read_u64(record, 96);
        task.solve_workspace_addr = read_u64(record, 104);
        task.reserved_addr0 = read_u64(record, 112);
        task.reserved_addr1 = read_u64(record, 120);
        if (task.pivot_dim == 0 || task.total_dim == 0 ||
            task.pivot_dim > task.total_dim) {
            throw std::runtime_error("NodeTask dimensions are invalid");
        }
        if (task.reserved != 0 || task.reserved_addr0 != 0 ||
            task.reserved_addr1 != 0) {
            throw std::runtime_error("NodeTask reserved fields must be zero");
        }
        const auto expected_tiles = (task.pivot_dim + 15u) / 16u;
        const auto expected_tail = task.pivot_dim == 0 ? 0u :
            (task.pivot_dim % 16u == 0 ? 16u : task.pivot_dim % 16u);
        if (task.tile_count != expected_tiles || task.tail_dim != expected_tail) {
            throw std::runtime_error("NodeTask tile metadata is inconsistent");
        }
        return task;
    }

    static Record encode(const NodeTask& task) {
        Record record{};
        write_u16(record, 0, task.node_id);
        write_u16(record, 2, task.flags);
        write_u16(record, 4, task.parent_id);
        write_u16(record, 6, task.children_count);
        write_u32(record, 8, task.total_dim);
        write_u32(record, 12, task.pivot_dim);
        write_u32(record, 16, task.tile_count);
        write_u32(record, 20, task.tail_dim);
        write_u32(record, 24, task.map_table_bytes);
        write_u32(record, 28, task.reserved);
        write_u64(record, 32, task.front_q_addr);
        write_u64(record, 40, task.front_e_addr);
        write_u64(record, 48, task.update_q_addr);
        write_u64(record, 56, task.update_e_addr);
        write_u64(record, 64, task.map_table_addr);
        write_u64(record, 72, task.l_factor_addr);
        write_u64(record, 80, task.u_factor_addr);
        write_u64(record, 88, task.p_vector_addr);
        write_u64(record, 96, task.node_meta_addr);
        write_u64(record, 104, task.solve_workspace_addr);
        write_u64(record, 112, task.reserved_addr0);
        write_u64(record, 120, task.reserved_addr1);
        return record;
    }

    static std::vector<NodeTask> read_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open task file: " + path);
        }
        input.seekg(0, std::ios::end);
        const auto size = input.tellg();
        if (size < 0 || static_cast<std::size_t>(size) % RECORD_SIZE != 0) {
            throw std::runtime_error("tasks.bin size is not a multiple of 128 bytes");
        }
        input.seekg(0, std::ios::beg);

        std::vector<NodeTask> tasks;
        tasks.reserve(static_cast<std::size_t>(size) / RECORD_SIZE);
        Record record{};
        while (input.read(reinterpret_cast<char*>(record.data()), RECORD_SIZE)) {
            tasks.push_back(decode(record));
        }
        if (!input.eof()) {
            throw std::runtime_error("failed while reading task file: " + path);
        }
        return tasks;
    }

private:
    static std::uint16_t read_u16(const Record& data, std::size_t offset) {
        return static_cast<std::uint16_t>(data[offset]) |
               (static_cast<std::uint16_t>(data[offset + 1]) << 8);
    }

    static std::uint32_t read_u32(const Record& data, std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) {
            value |= static_cast<std::uint32_t>(data[offset + i]) << (8 * i);
        }
        return value;
    }

    static std::uint64_t read_u64(const Record& data, std::size_t offset) {
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(data[offset + i]) << (8 * i);
        }
        return value;
    }

    static void write_u16(Record& data, std::size_t offset, std::uint16_t value) {
        for (unsigned i = 0; i < 2; ++i) {
            data[offset + i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFu);
        }
    }

    static void write_u32(Record& data, std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) {
            data[offset + i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFu);
        }
    }

    static void write_u64(Record& data, std::size_t offset, std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) {
            data[offset + i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFu);
        }
    }
};

}  // namespace hw
