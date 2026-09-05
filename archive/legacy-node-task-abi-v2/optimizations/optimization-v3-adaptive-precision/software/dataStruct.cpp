#include <cstddef>
#include <cstdint>
#include <type_traits>

// C++ reference layout for artifact ABI v2. The executable codec still
// performs explicit little-endian byte reads/writes and does not reinterpret
// memory_image.bin as this native struct.
struct NodeTaskV2 {
    std::uint16_t node_id;
    std::uint16_t flags;
    std::uint16_t parent_id;
    std::uint16_t children_count;

    std::uint32_t total_dim;
    std::uint32_t pivot_dim;
    std::uint32_t tile_count;
    std::uint32_t tail_dim;
    std::uint32_t map_table_bytes;
    std::uint32_t reserved;

    std::uint64_t front_q_addr;
    std::uint64_t front_e_addr;
    std::uint64_t update_q_addr;
    std::uint64_t update_e_addr;
    std::uint64_t map_table_addr;
    std::uint64_t l_factor_addr;
    std::uint64_t u_factor_addr;
    std::uint64_t p_vector_addr;
    std::uint64_t node_meta_addr;
    std::uint64_t solve_workspace_addr;
    std::uint64_t reserved_addr0;
    std::uint64_t reserved_addr1;
};

static_assert(std::is_standard_layout_v<NodeTaskV2>);
static_assert(sizeof(NodeTaskV2) == 128);
static_assert(offsetof(NodeTaskV2, total_dim) == 8);
static_assert(offsetof(NodeTaskV2, front_q_addr) == 32);
static_assert(offsetof(NodeTaskV2, reserved_addr1) == 120);
