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
    p_vector_sizes = [8, 4, 2]
    node_meta_sizes = [64, 64, 64]
    solve_workspace_sizes = [32, 16, 16]
    global_plan, plans, total = plan_memory(
        num_nodes,
        12,
        q_sizes,
        e_sizes,
        map_sizes,
        update_q_sizes=update_q_sizes,
        update_e_sizes=update_e_sizes,
        l_factor_sizes=l_factor_sizes,
        u_factor_sizes=u_factor_sizes,
        p_vector_sizes=p_vector_sizes,
        node_meta_sizes=node_meta_sizes,
        solve_workspace_sizes=solve_workspace_sizes,
        align=16,
    )

    ranges = [
        (region.offset, region.offset + region.size)
        for region in (
            global_plan.task_queue,
            global_plan.permutation,
            global_plan.rhs_q,
            global_plan.rhs_e,
            global_plan.solution_q,
            global_plan.solution_e,
        )
    ]
    for node_id in range(num_nodes):
        plan = plans[node_id]
        ranges.append((plan.front_q.offset, plan.front_q.offset + plan.front_q.size))
        ranges.append((plan.front_e.offset, plan.front_e.offset + plan.front_e.size))
        ranges.append((plan.update_q.offset, plan.update_q.offset + plan.update_q.size))
        ranges.append((plan.update_e.offset, plan.update_e.offset + plan.update_e.size))
        ranges.append((plan.l_factor.offset, plan.l_factor.offset + plan.l_factor.size))
        ranges.append((plan.u_factor.offset, plan.u_factor.offset + plan.u_factor.size))
        ranges.append((plan.map_table.offset, plan.map_table.offset + plan.map_table.size))
        ranges.append((plan.p_vector.offset, plan.p_vector.offset + plan.p_vector.size))
        ranges.append((plan.node_meta.offset, plan.node_meta.offset + plan.node_meta.size))
        ranges.append(
            (
                plan.solve_workspace.offset,
                plan.solve_workspace.offset + plan.solve_workspace.size,
            )
        )

    ranges = [r for r in ranges if r[0] != r[1]]
    ranges.sort()
    for i in range(1, len(ranges)):
        assert ranges[i - 1][1] <= ranges[i][0]
    assert total >= ranges[-1][1]
