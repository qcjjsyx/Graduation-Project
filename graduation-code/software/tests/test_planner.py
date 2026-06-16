from src.memory.planner import plan_memory


def test_memory_planner_non_overlap():
    num_nodes = 3
    q_sizes = [128, 64, 32]
    e_sizes = [16, 16, 16]
    map_sizes = [40, 20, 10]
    update_q_sizes = [64, 32, 0]
    update_e_sizes = [2, 2, 0]
    l_factor_sizes = [48, 32, 16]
    u_factor_sizes = [48, 32, 16]
    task_desc_sizes = [80, 80, 80]
    plans, total = plan_memory(
        num_nodes,
        q_sizes,
        e_sizes,
        map_sizes,
        update_q_sizes=update_q_sizes,
        update_e_sizes=update_e_sizes,
        l_factor_sizes=l_factor_sizes,
        u_factor_sizes=u_factor_sizes,
        task_desc_sizes=task_desc_sizes,
        align=16,
    )

    ranges = []
    for node_id in range(num_nodes):
        plan = plans[node_id]
        ranges.append((plan.front_q.offset, plan.front_q.offset + plan.front_q.size))
        ranges.append((plan.front_e.offset, plan.front_e.offset + plan.front_e.size))
        ranges.append((plan.update_q.offset, plan.update_q.offset + plan.update_q.size))
        ranges.append((plan.update_e.offset, plan.update_e.offset + plan.update_e.size))
        ranges.append((plan.l_factor.offset, plan.l_factor.offset + plan.l_factor.size))
        ranges.append((plan.u_factor.offset, plan.u_factor.offset + plan.u_factor.size))
        ranges.append((plan.map_table.offset, plan.map_table.offset + plan.map_table.size))
        ranges.append((plan.task_desc.offset, plan.task_desc.offset + plan.task_desc.size))

    ranges = [r for r in ranges if r[0] != r[1]]
    ranges.sort()
    for i in range(1, len(ranges)):
        assert ranges[i - 1][1] <= ranges[i][0]
    assert total >= ranges[-1][1]
