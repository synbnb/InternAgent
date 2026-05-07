#!/usr/bin/env python3
"""
MCTS Search Tree Visualizer
解析 mcts.log 文件并可视化 MCTS 搜索树结构
"""

import re
import sys
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple

class NodeInfo:
    """节点信息"""
    def __init__(self, node_id: str, stage: str, parent_id: Optional[str] = None):
        self.id = node_id
        self.stage = stage  # root, draft, improve, debug
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.metric: Optional[float] = None
        self.is_terminal: bool = False
        self.is_buggy: bool = False
        self.continue_improve: bool = False
        self.improve_failure: Optional[int] = None  # 记录失败次数
        self.run_status: str = "unknown"  # OK, FAIL, unknown
        
    def __str__(self):
        status_sym = "✓" if self.run_status == "OK" else "✗" if self.run_status == "FAIL" else "?"
        metric_str = f" m:{self.metric:.4f}" if self.metric else ""
        terminal_str = " [T]" if self.is_terminal else ""
        return f"{status_sym} {self.stage[:1]}{self.id[:8]}{metric_str}{terminal_str}"


def parse_log(log_file: str) -> Tuple[Dict[str, NodeInfo], str]:
    """
    解析 log 文件，提取节点树结构
    
    Returns:
        nodes: {node_id: NodeInfo}
        root_id: 根节点ID
    """
    nodes: Dict[str, NodeInfo] = {}
    root_id = None
    
    print(f"Parsing {log_file}...")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取根节点 - 支持两种格式
            # 新格式：[SELECT] Examining X, metric=Y (stage=root, ...)
            # 旧格式：[SELECT] Iter 1: examining node X (stage=root, ...)
            if 'stage=root' in line and ('[SELECT] Examining' in line or 'examining node' in line):
                match = re.search(r'Examining\s+(\w+)|examining node\s+(\w+)', line)
                if match:
                    root_id = match.group(1) or match.group(2)
                    if root_id not in nodes:
                        nodes[root_id] = NodeInfo(root_id, "root")
            
            # 提取节点创建：draft
            if '[draft] node=' in line and 'iter=1/' in line:
                match = re.search(r'node=(\w+).*branch=(\d+)', line)
                if match:
                    node_id = match.group(1)
                    if node_id not in nodes:
                        nodes[node_id] = NodeInfo(node_id, "draft", root_id)
                        if root_id and root_id in nodes:
                            nodes[root_id].children_ids.append(node_id)
            
            # 提取节点创建：improve
            if '[improve] parent=' in line and 'iter=1/' in line:
                match = re.search(r'parent=(\w+).*node=(\w+)', line)
                if match:
                    parent_id = match.group(1)
                    node_id = match.group(2)
                    if node_id not in nodes:
                        nodes[node_id] = NodeInfo(node_id, "improve", parent_id)
                        if parent_id in nodes:
                            nodes[parent_id].children_ids.append(node_id)
            
            # 提取 run 状态
            if 'node=' in line and ' run ' in line:
                match = re.search(r'node=(\w+).*run (OK|FAIL)', line)
                if match:
                    node_id = match.group(1)
                    status = match.group(2)
                    if node_id in nodes:
                        nodes[node_id].run_status = status
            
            # 提取 metric - 支持多种格式
            # 注意：使用独立的if而不是elif，确保所有格式都能被检查
            # 优先级：is the best > Comparing > Examining
            
            # 1. "Node X is the best node so far (metric: Y)" - 最高优先级
            if 'is the best node so far' in line:
                match = re.search(r'Node (\w+).*\(metric:\s*([0-9.]+)\)', line)
                if match:
                    node_id = match.group(1)
                    metric_val = float(match.group(2))
                    if node_id in nodes:
                        nodes[node_id].metric = metric_val
            
            # 2. "Comparing Node X (metric: Y) with local best Node ..." - 次优先级
            if 'Comparing Node' in line and 'metric:' in line:
                # 使用非贪婪匹配，只匹配第一个 metric
                match = re.search(r'Comparing Node (\w+).*?\(metric:\s*([0-9.]+)\)', line)
                if match:
                    node_id = match.group(1)
                    metric_val = float(match.group(2))
                    if node_id in nodes and metric_val > 0:  # 过滤0.0值
                        # 只有在还没有metric时才设置，避免覆盖更准确的"is the best"值
                        if nodes[node_id].metric is None:
                            nodes[node_id].metric = metric_val
            
            # 3. "metric=Y" 在 [SELECT] Examining 中 - 最低优先级
            if '[SELECT] Examining' in line and 'metric=' in line:
                match = re.search(r'Examining\s+(\w+).*metric=([0-9.]+)', line)
                if match:
                    node_id = match.group(1)
                    metric_val = float(match.group(2))
                    
                    if node_id in nodes :  #
                        # 只有在还没有metric时才设置，避免覆盖更准确的值
                        if nodes[node_id].metric is None:
                            nodes[node_id].metric = metric_val
            
            # 提取 terminal 状态
            if 'Terminal node' in line:
                match = re.search(r'Terminal node (\w+)', line)
                if match:
                    node_id = match.group(1)
                    if node_id in nodes:
                        nodes[node_id].is_terminal = True
            
            # 提取 improve_failure_depth
            if 'try one more time' in line:
                match = re.search(r'Node (\w+).*try one more time\((\d+)/', line)
                if match:
                    node_id = match.group(1)
                    failure_count = int(match.group(2))
                    if node_id in nodes:
                        nodes[node_id].improve_failure = failure_count
    
    print(f"Parsed {len(nodes)} nodes, root: {root_id}")
    return nodes, root_id


def print_tree_ascii(nodes: Dict[str, NodeInfo], root_id: str, max_depth: int = 10):
    """
    以 ASCII 树形式打印节点树
    """
    if root_id not in nodes:
        print("Error: Root node not found")
        return
    
    print("\n" + "="*80)
    print("MCTS Search Tree Visualization")
    print("="*80)
    print(f"Legend: ✓=OK ✗=FAIL ?=unknown, [r/d/i]=root/draft/improve, m=metric, [T]=terminal")
    print("="*80 + "\n")
    
    def dfs_print(node_id: str, prefix: str, is_last: bool, depth: int):
        if depth > max_depth:
            return
        
        node = nodes[node_id]
        
        # 打印当前节点
        current_prefix = "└── " if is_last else "├── "
        print(prefix + current_prefix + str(node))
        
        # 更新前缀
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension
        
        # 递归打印子节点
        children = node.children_ids
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            dfs_print(child_id, new_prefix, is_last_child, depth + 1)
    
    # 打印根节点
    dfs_print(root_id, "", True, 0)
    
    print("\n" + "="*80)
    print("Statistics:")
    print("="*80)
    
    # 统计信息
    stats = {
        'root': 0, 'draft': 0, 'improve': 0, 'debug': 0,
        'OK': 0, 'FAIL': 0, 'unknown': 0,
        'terminal': 0,
        'with_metric': 0
    }
    
    for node in nodes.values():
        stats[node.stage] = stats.get(node.stage, 0) + 1
        stats[node.run_status] = stats.get(node.run_status, 0) + 1
        if node.is_terminal:
            stats['terminal'] += 1
        if node.metric is not None:
            stats['with_metric'] += 1
    
    print(f"Total nodes: {len(nodes)}")
    print(f"  Root: {stats['root']}, Draft: {stats['draft']}, Improve: {stats['improve']}, Debug: {stats['debug']}")
    print(f"  Status - OK: {stats['OK']}, FAIL: {stats['FAIL']}, Unknown: {stats['unknown']}")
    print(f"  Terminal: {stats['terminal']}, With metric: {stats['with_metric']}")
    
    # 打印最佳节点
    best_nodes = [n for n in nodes.values() if n.metric is not None]
    if best_nodes:
        best = max(best_nodes, key=lambda n: n.metric)
        print(f"\nBest node: {best.id[:16]}... metric: {best.metric:.6f}")
    
    print("="*80 + "\n")


def print_tree_html(nodes: Dict[str, NodeInfo], root_id: str, output_file: str = "mcts_tree.html"):
    """
    生成 HTML 格式的可视化树（使用简单的 HTML/CSS）
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MCTS Search Tree</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tree {
            margin-top: 20px;
        }
        .node {
            margin: 2px 0;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .root { background: #e3f2fd; }
        .draft { background: #fff3e0; }
        .improve { background: #e8f5e9; }
        .debug { background: #fce4ec; }
        .ok { color: #2e7d32; font-weight: bold; }
        .fail { color: #c62828; font-weight: bold; }
        .terminal { border: 2px solid #d32f2f; }
        .indent { margin-left: 30px; }
        h1 { color: #1976d2; }
        .legend {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MCTS Search Tree Visualization</h1>
        
        <div class="legend">
            <strong>Legend:</strong><br>
            ✓=OK  ✗=FAIL  ?=unknown  |  [r/d/i]=root/draft/improve  |  m=metric  |  [T]=terminal
        </div>
        
        <div class="tree">
"""
    
    def dfs_html(node_id: str, indent_level: int):
        node = nodes[node_id]
        
        # 确定样式类
        status_class = node.run_status
        style = node.stage
        if node.is_terminal:
            style += " terminal"
        
        margin_left = indent_level * 30
        html_content = f'<div class="node {style}" style="margin-left: {margin_left}px;">'
        html_content += f'<span class="{status_class}">{str(node)}</span>'
        html_content += '</div>\n'
        
        # 递归子节点
        for child_id in node.children_ids:
            html_content += dfs_html(child_id, indent_level + 1)
        
        return html_content
    
    html += dfs_html(root_id, 0)
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML visualization saved to: {output_file}")


def print_tree_graphviz(nodes: Dict[str, NodeInfo], root_id: str, output_file: str = "mcts_tree"):
    """
    使用 Graphviz 生成带箭头的树形图
    需要安装: pip install graphviz
    系统需要安装 Graphviz: https://graphviz.org/download/
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Install with: pip install graphviz")
        print("Also ensure Graphviz executables are installed on your system.")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  macOS: brew install graphviz")
        print("  Windows: Download from https://graphviz.org/download/")
        return
    
    # 创建有向图
    g = Digraph(
        name='MCTS_Tree',
        comment='MCTS Search Tree Visualization',
        format='png',  # 可以改为 'pdf', 'svg' 等
        graph_attr={
            'rankdir': 'TB',  # Top to Bottom (从上到下)
            'splines': 'ortho',  # 正交边
            'nodesep': '0.5',
            'ranksep': '1.0',
            'bgcolor': '#f5f5f5'
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fontsize': '10',
            'fontname': 'Arial'
        },
        edge_attr={
            'arrowhead': 'vee',
            'color': '#666666',
            'penwidth': '1.5'
        }
    )
    
    # 添加节点和边的递归函数
    def add_node_recursive(node_id: str):
        if node_id not in nodes:
            return
        
        node = nodes[node_id]
        
        # 构建节点标签
        status_sym = "✓" if node.run_status == "OK" else "✗" if node.run_status == "FAIL" else "?"
        stage_name = {"root": "R", "draft": "D", "improve": "I", "debug": "DB"}.get(node.stage, node.stage[:1].upper())
        
        label_lines = [
            f"{status_sym} {stage_name} {node.id[:8]}",
        ]
        
        if node.metric is not None:
            label_lines.append(f"metric: {node.metric:.4f}")
        else:
            label_lines.append("metric: N/A")
        
        if node.is_terminal:
            label_lines.append("[TERMINAL]")
        
        label = "\n".join(label_lines)
        
        # 根据状态和阶段设置颜色
        if node.run_status == "OK":
            if node.metric is not None and node.metric > 0.9:
                fillcolor = "#90EE90"  # 浅绿色 - 高metric
            else:
                fillcolor = "#E8F5E9"  # 浅绿色 - 成功但metric一般
        elif node.run_status == "FAIL":
            fillcolor = "#FFCDD2"  # 浅红色 - 失败
        else:
            fillcolor = "#F5F5F5"  # 灰色 - 未知
        
        # 根据阶段调整颜色
        if node.stage == "root":
            fillcolor = "#BBDEFB"  # 浅蓝色 - 根节点
        elif node.stage == "draft":
            fillcolor = "#FFF3E0"  # 浅橙色 - draft节点
        
        # 添加节点
        g.node(
            node_id,
            label=label,
            fillcolor=fillcolor
        )
        
        # 递归添加子节点和边
        for child_id in node.children_ids:
            if child_id in nodes:
                add_node_recursive(child_id)
                g.edge(node_id, child_id)
    
    # 从根节点开始构建
    if root_id not in nodes:
        print(f"Error: Root node {root_id} not found in nodes")
        return
    
    add_node_recursive(root_id)
    
    # 渲染图像
    try:
        output_path = g.render(
            output_file,  # 输出文件名（不含扩展名）
            format='png',
            cleanup=True,  # 清理临时文件
            view=False  # 不自动打开
        )
        print(f"Graphviz tree visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error rendering graphviz: {e}")
        print("Make sure Graphviz executables are installed and in PATH")
        # 仍然保存源文件
        try:
            g.save(f"{output_file}.gv")
            print(f"Saved graphviz source to: {output_file}.gv")
            print("You can manually render it with: dot -Tpng {output_file}.gv -o {output_file}.png")
        except Exception as e2:
            print(f"Error saving source file: {e2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mcts.py <mcts.log> [--html|--graphviz|--all]")
        print("\nOptions:")
        print("  (no option)    - Print ASCII tree visualization (default)")
        print("  --html         - Generate HTML tree visualization")
        print("  --graphviz     - Generate Graphviz tree visualization (requires graphviz)")
        print("  --all          - Generate all formats (ASCII + HTML + Graphviz)")
        print("\nExamples:")
        print("  python visualize_mcts.py results/xxx/mcts.log")
        print("  python visualize_mcts.py results/xxx/mcts.log --html")
        print("  python visualize_mcts.py results/xxx/mcts.log --graphviz")
        print("  python visualize_mcts.py results/xxx/mcts.log --all")
        print("\nNote: For --graphviz option, you need to install:")
        print("  pip install graphviz")
        print("  And install Graphviz system package:")
        print("    Ubuntu/Debian: sudo apt-get install graphviz")
        print("    macOS: brew install graphviz")
        print("    Windows: Download from https://graphviz.org/download/")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else ""
    
    generate_html = output_format in ["--html", "--all"]
    generate_graphviz = output_format in ["--graphviz", "--all"]
    
    try:
        nodes, root_id = parse_log(log_file)
        print_tree_ascii(nodes, root_id)
        
        if generate_html:
            output_html = log_file.replace(".log", "_tree.html")
            print_tree_html(nodes, root_id, output_html)
        
        if generate_graphviz:
            output_gv = log_file.replace(".log", "_tree")
            print_tree_graphviz(nodes, root_id, output_gv)
    
    except FileNotFoundError:
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
