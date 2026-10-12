from src.dataStruct import MapTableEntry
from src.scheduler.map_gen import generate_map_tables


def test_map_table_identity_overlap():
    node_ranges = [(0, 1), (1, 2), (0, 2)]
    parent = [2, 2, -1]
    front_indices = [[0, 1], [1, 0], [0, 1]]
    maps = generate_map_tables(node_ranges, parent, front_indices)

    entries = maps[2]
    assert len(entries) == 2
    entry0 = next(e for e in entries if e.child_id == 0)
    entry1 = next(e for e in entries if e.child_id == 1)
    assert entry0.row_map == [0]
    assert entry0.col_map == [1]
    assert entry1.row_map == [0]
    assert entry1.col_map == [0]


def test_map_table_front_index_mapping():
    node_ranges = [(0, 1), (1, 2)]
    parent = [1, -1]
    front_indices = [[0, 1, 2], [1, 2]]
    maps = generate_map_tables(node_ranges, parent, front_indices)
    assert maps[0] == []
    assert maps[1] == [MapTableEntry(child_id=0, row_map=[0, 1], col_map=[0, 1])]

