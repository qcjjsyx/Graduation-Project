from software.scheduler.map_gen import generate_map_tables


def test_map_table_identity_overlap():
    # Node ranges: node0 [0,2], node1 [2,4], node2 [0,4]
    node_ranges = [(0, 2), (2, 4), (0, 4)]
    parent = [2, 2, -1]
    maps = generate_map_tables(node_ranges, parent)
    # Parent node 2 should have two entries from children 0 and 1
    entries = maps[2]
    assert len(entries) == 2
    entry0 = next(e for e in entries if e.child_id == 0)
    entry1 = next(e for e in entries if e.child_id == 1)
    assert entry0.row_map == [0, 1]
    assert entry0.col_map == [0, 1]
    assert entry1.row_map == [0, 1]
    assert entry1.col_map == [2, 3]