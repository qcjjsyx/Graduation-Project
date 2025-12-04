from graphviz import Digraph

def create_hpu_atu_diagram():
    # 创建有向图
    dot = Digraph('HPU_ATU_Architecture', format='png')
    dot.attr(rankdir='TB')  # 从上到下布局
    dot.attr('node', fontname='Microsoft YaHei', fontsize='10')
    
    # 设置整体样式
    dot.attr('graph', splines='ortho')  # 使用直线连接
    
    # 主框架 - HPU部分
    with dot.subgraph(name='cluster_hpu') as hpu: # type: ignore
        hpu.attr(label='HPU (混合主元选取)', style='rounded,filled', fillcolor='#f0f5ff', fontname='Microsoft YaHei')
        
        # HPU内部节点
        hpu.node('input', '输入: 当前列', shape='plaintext')
        hpu.node('unified', '统一主元逻辑', shape='box')
        hpu.node('mode_a', '模式A (阈值法)', shape='box', style='filled', fillcolor='#e6f3ff')
        hpu.node('mode_b', '模式B (竞标赛)', shape='box', style='filled', fillcolor='#e6f3ff')
        hpu.node('output', '输出: 主元索引 ipiv', shape='plaintext')
        
        # HPU内部连接
        dot.edge('input', 'unified')
        dot.edge('unified', 'mode_a', style='dashed')
        dot.edge('unified', 'mode_b', style='dashed')
        dot.edge('mode_a', 'output', style='invis')  # 隐形连接用于布局
        dot.edge('mode_b', 'output', style='invis')  # 隐形连接用于布局
        dot.edge('unified', 'output')
    
    # ATU单元
    dot.node('atu', 'ATU\n(地址翻译单元)', shape='box', style='rounded,filled', fillcolor='#f0f8ff')
    
    # 地址映射表生成
    dot.node('addr_map', '生成地址映射表\n(addr_map)', shape='box', style='filled', fillcolor='#fff0f0')
    
    # 物理内存
    with dot.subgraph(name='cluster_memory') as memory: # type: ignore
        memory.attr(label='物理内存 (DDR/SRAM)', style='rounded,filled', fillcolor='#f8f8f8')
        memory.node('memory', '数据存储按物理地址', shape='box')
    
    # 计算核
    dot.node('compute', '计算核\n(Panel-GEPP, etc.)', shape='box', style='rounded,filled', fillcolor='#f5f0ff')
    
    # 查询过程
    dot.node('request', '请求逻辑行', shape='plaintext')
    dot.node('query', 'ATU 查询', shape='diamond', style='filled', fillcolor='#fff8dc')
    dot.node('get_addr', '获取物理地址', shape='box', style='filled', fillcolor='#f0fff0')
    dot.node('read_data', '实际数据读取', shape='plaintext')
    
    # 主要数据流连接
    dot.edge('output', 'atu')
    dot.edge('atu', 'addr_map')
    dot.edge('addr_map', 'memory')
    dot.edge('memory', 'compute', style='invis')  # 隐形连接用于布局
    dot.edge('compute', 'request')
    dot.edge('request', 'query')
    dot.edge('query', 'get_addr')
    dot.edge('get_addr', 'memory', label='查询', style='dashed')
    dot.edge('memory', 'read_data')
    dot.edge('read_data', 'compute', constraint='false')  # 不强制布局约束
    
    # 添加回环箭头
    dot.edge('compute', 'atu', label='查询请求', style='dashed', constraint='false')
    
    return dot

def create_simplified_diagram():
    """创建一个更简化的版本，更接近原图布局"""
    dot = Digraph('HPU_ATU_Simplified', format='png')
    dot.attr(rankdir='TB')
    dot.attr('node', fontname='Microsoft YaHei', fontsize='10')
    
    # 设置节点样式
    dot.attr('node', shape='box')
    
    # HPU部分
    dot.node('hpu', 'HPU (混合主元选取)\n\n[输入: 当前列]\n    ↓\n统一主元逻辑\n  ├─ 模式A (阈值法)\n  └─ 模式B (竞标赛)\n    ↓\n[输出: 主元索引 ipiv]', 
             style='rounded,filled', fillcolor='#f0f5ff')
    
    # ATU部分
    dot.node('atu', 'ATU\n(地址翻译单元)', style='rounded,filled', fillcolor='#f0f8ff')
    
    # 内存部分
    dot.node('memory', '物理内存 (DDR/SRAM)\n\n数据存储按物理地址', 
             style='rounded,filled', fillcolor='#f8f8f8')
    
    # 计算核部分
    dot.node('compute', '计算核\n(Panel-GEPP, etc.)', style='rounded,filled', fillcolor='#f5f0ff')
    
    # 查询过程节点
    dot.node('process', '请求逻辑行 → ATU 查询 → 获取物理地址\n\n↓\n\n← 实际数据读取', 
             shape='plaintext')
    
    # 连接
    dot.edge('hpu', 'atu')
    dot.edge('atu', 'memory')
    dot.edge('memory', 'compute', style='invis')  # 隐形连接用于布局
    dot.edge('compute', 'process')
    dot.edge('process', 'memory', constraint='false')
    
    return dot

def main():
    # 创建详细版本的图表
    detailed_diagram = create_hpu_atu_diagram()
    detailed_diagram.render('hpu_atu_detailed', cleanup=True)
    print("详细版本已保存为 hpu_atu_detailed.png")
    
    # 创建简化版本的图表
    simplified_diagram = create_simplified_diagram()
    simplified_diagram.render('hpu_atu_simplified', cleanup=True)
    print("简化版本已保存为 hpu_atu_simplified.png")
    
    # 在Jupyter Notebook中直接显示
    # return detailed_diagram, simplified_diagram

if __name__ == "__main__":
    main()