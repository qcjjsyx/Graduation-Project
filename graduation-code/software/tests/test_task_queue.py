from software.scheduler.task_queue import sibling_friendly_order


def test_task_queue_parent_after_children():
    # Tree: 0 is parent of 1,2; 2 is parent of 3
    parent = [-1, 0, 0, 2]
    order = sibling_friendly_order(parent)
    pos = {node: i for i, node in enumerate(order)}
    for child, p in enumerate(parent):
        if p >= 0:
            assert pos[child] < pos[p]