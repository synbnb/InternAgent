#!/usr/bin/env python3
"""
MCTS Search Tree Visualizer for Claude Code Backend
解析 Claude Code MCTS 的 mcts.log 文件并可视化 MCTS 搜索树结构
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class NodeInfo:
    """节点信息"""
    def __init__(self, node_id: str, stage: str, parent_id: Optional[str] = None):
        self.id = node_id
        self.stage = stage  # root, draft, improve
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.metric: Optional[float] = None
        self.is_terminal: bool = False
        self.continue_improve: bool = False
        self.improve_failure_depth: int = 0  # 改进失败次数
        self.run_status: str = "unknown"  # SUCCESS, FAILED, unknown

    def __str__(self):
        status_sym = "✓" if self.run_status == "SUCCESS" else "✗" if self.run_status == "FAILED" else "?"
        metric_str = f" m:{self.metric:.4f}" if self.metric else ""
        terminal_str = " [T]" if self.is_terminal else ""
        improve_fail_str = f" [F{self.improve_failure_depth}]" if self.improve_failure_depth > 0 else ""
        stage_abbrev = {"root": "R", "draft": "D", "improve": "I"}.get(self.stage, self.stage[:1])
        return f"{status_sym} {stage_abbrev}-{self.id[:8]}{metric_str}{improve_fail_str}{terminal_str}"


def parse_log_claude(log_file: str) -> Tuple[Dict[str, NodeInfo], str]:
    """
    解析 Claude Code MCTS log 文件，提取节点树结构

    Returns:
        nodes: {node_id: NodeInfo}
        root_id: 根节点ID
    """
    nodes: Dict[str, NodeInfo] = {}
    root_id = None

    print(f"Parsing Claude Code MCTS log: {log_file}...")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取根节点
            # [step] Starting from root
            # [select] Starting from node XXX, stage=root, is_terminal=False
            if '[select] Starting from node' in line and 'stage=root' in line:
                match = re.search(r'from node (\w+).*stage=root', line)
                if match:
                    node_id = match.group(1)
                    if node_id not in nodes:
                        nodes[node_id] = NodeInfo(node_id, "root")
                    root_id = node_id

            # 提取 draft 节点创建
            # [draft] node=XXX generating initial code
            if '[draft] node=' in line and 'generating initial code' in line:
                match = re.search(r'node=(\w+)', line)
                if match:
                    node_id = match.group(1)
                    if node_id not in nodes:
                        nodes[node_id] = NodeInfo(node_id, "draft", root_id)
                        if root_id and root_id in nodes:
                            # 添加为 root 的子节点（如果还不在子节点列表中）
                            if node_id not in nodes[root_id].children_ids:
                                nodes[root_id].children_ids.append(node_id)

            # 提取 improve 节点创建
            # [improve] parent=XXX node=YYY improving code
            if '[improve] parent=' in line and 'improving code' in line:
                match = re.search(r'parent=(\w+).*node=(\w+)', line)
                if match:
                    parent_id = match.group(1)
                    node_id = match.group(2)
                    if node_id not in nodes:
                        nodes[node_id] = NodeInfo(node_id, "improve", parent_id)
                        if parent_id in nodes:
                            # 添加为父节点的子节点（如果还不在子节点列表中）
                            if node_id not in nodes[parent_id].children_ids:
                                nodes[parent_id].children_ids.append(node_id)

            # 提取运行状态
            # [draft] node=XXX run SUCCESS
            # [draft] node=XXX run FAILED, attempting to fix
            if 'node=' in line and ('run SUCCESS' in line or 'run FAILED' in line):
                match = re.search(r'node=(\w+).*run (SUCCESS|FAILED)', line)
                if match:
                    node_id = match.group(1)
                    status = match.group(2)
                    if node_id in nodes:
                        nodes[node_id].run_status = status

            # 提取 metric
            # "Node XXX is the best node so far (metric: 0.923165)"
            if 'is the best node so far' in line:
                match = re.search(r'Node (\w+).*metric:\s*([0-9.]+)', line)
                if match:
                    node_id = match.group(1)
                    metric_val = float(match.group(2))
                    if node_id in nodes:
                        nodes[node_id].metric = metric_val

            # "Comparing Node XXX (metric: 0.923165) with local best Node YYY"
            elif 'Comparing Node' in line and 'metric:' in line:
                match = re.search(r'Comparing Node (\w+).*?\(metric:\s*([0-9.]+)\)', line)
                if match:
                    node_id = match.group(1)
                    metric_val = float(match.group(2))
                    if node_id in nodes and nodes[node_id].metric is None:
                        nodes[node_id].metric = metric_val

            # 提取 improvement failure depth
            # "Improvement (-0.003440) below threshold, try again (2/2)"
            if 'below threshold, try again' in line:
                match = re.search(r'try again \((\d+)/\d+\)', line)
                if match:
                    failure_count = int(match.group(1))
                    # 需要从上下文找到对应的节点
                    # 查找前面最近的 Comparing Node
                    # 这里简化处理：从前一行获取节点ID

            # 提取 terminal 状态
            # is_terminal=True
            if 'is_terminal=True' in line:
                match = re.search(r'node=(\w+)', line)
                if match:
                    node_id = match.group(1)
                    if node_id in nodes:
                        nodes[node_id].is_terminal = True

            # 提取 continue_improve 状态
            if 'continue_improve=True' in line:
                match = re.search(r'node=(\w+)', line)
                if match:
                    node_id = match.group(1)
                    if node_id in nodes:
                        nodes[node_id].continue_improve = True

    # 后处理：根据子节点确定父子关系（补充遗漏的）
    # 如果一个 improve 节点没有父节点，尝试从日志中推断

    print(f"Parsed {len(nodes)} nodes, root: {root_id}")
    return nodes, root_id


def print_tree_ascii(nodes: Dict[str, NodeInfo], root_id: str, max_depth: int = 20):
    """
    以 ASCII 树形式打印节点树
    """
    if root_id not in nodes:
        print("Error: Root node not found")
        return

    print("\n" + "="*80)
    print("Claude Code MCTS Search Tree Visualization")
    print("="*80)
    print(f"Legend: ✓=SUCCESS ✗=FAILED ?=unknown, [R/D/I]=root/draft/improve")
    print(f"        m=metric, [T]=terminal, [FN]=failed N times")
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
            if child_id in nodes:
                is_last_child = (i == len(children) - 1)
                dfs_print(child_id, new_prefix, is_last_child, depth + 1)

    # 打印根节点
    dfs_print(root_id, "", True, 0)

    print("\n" + "="*80)
    print("Statistics:")
    print("="*80)

    # 统计信息
    stats = {
        'root': 0, 'draft': 0, 'improve': 0,
        'SUCCESS': 0, 'FAILED': 0, 'unknown': 0,
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
    print(f"  Root: {stats['root']}, Draft: {stats['draft']}, Improve: {stats['improve']}")
    print(f"  Status - SUCCESS: {stats['SUCCESS']}, FAILED: {stats['FAILED']}, Unknown: {stats['unknown']}")
    print(f"  Terminal: {stats['terminal']}, With metric: {stats['with_metric']}")

    # 打印最佳节点
    best_nodes = [n for n in nodes.values() if n.metric is not None]
    if best_nodes:
        best = min(best_nodes, key=lambda n: n.metric)
        print(f"\nBest node: {best.id[:16]}... stage: {best.stage}, metric: {best.metric:.6f}")

    print("="*80 + "\n")


def print_tree_html(nodes: Dict[str, NodeInfo], root_id: str, output_file: str = "mcts_tree_claude.html"):
    """
    生成 HTML 格式的可视化树
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Claude Code MCTS Search Tree</title>
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
        .success { color: #2e7d32; font-weight: bold; }
        .failed { color: #c62828; font-weight: bold; }
        .terminal { border: 2px solid #d32f2f; }
        h1 { color: #1976d2; }
        .legend {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .best-node {
            background: #90EE90;
            font-weight: bold;
            border: 2px solid #2e7d32;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Claude Code MCTS Search Tree Visualization</h1>

        <div class="legend">
            <strong>Legend:</strong><br>
            ✓=SUCCESS  ✗=FAILED  ?=unknown  |  [R/D/I]=root/draft/improve  |  m=metric  |  [T]=terminal  |  [FN]=failed N times
        </div>

        <div class="tree">
"""

    # 找到最佳节点
    best_node_id = None
    best_nodes = [n for n in nodes.values() if n.metric is not None]
    if best_nodes:
        best = min(best_nodes, key=lambda n: n.metric)
        best_node_id = best.id

    def dfs_html(node_id: str, indent_level: int):
        if node_id not in nodes:
            return ""

        node = nodes[node_id]

        # 确定样式类
        status_class = "success" if node.run_status == "SUCCESS" else "failed" if node.run_status == "FAILED" else ""
        style_classes = [node.stage]
        if node.is_terminal:
            style_classes.append("terminal")
        if node.id == best_node_id:
            style_classes.append("best-node")

        margin_left = indent_level * 30
        html_content = f'<div class="node {" ".join(style_classes)}" style="margin-left: {margin_left}px;">'
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


def print_tree_graphviz(nodes: Dict[str, NodeInfo], root_id: str, output_file: str = "mcts_tree_claude"):
    """
    使用 Graphviz 生成树形图
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Install with: pip install graphviz")
        return

    # 找到最佳节点
    best_node_id = None
    best_nodes = [n for n in nodes.values() if n.metric is not None]
    if best_nodes:
        best = min(best_nodes, key=lambda n: n.metric)
        best_node_id = best.id

    g = Digraph(
        name='MCTS_Tree_Claude',
        comment='Claude Code MCTS Search Tree',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
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

    def add_node_recursive(node_id: str):
        if node_id not in nodes:
            return

        node = nodes[node_id]

        # 构建标签
        status_sym = "✓" if node.run_status == "SUCCESS" else "✗" if node.run_status == "FAILED" else "?"
        stage_name = {"root": "ROOT", "draft": "DRAFT", "improve": "IMPROVE"}.get(node.stage, node.stage.upper())

        label_lines = [
            f"{status_sym} {stage_name}",
            f"ID: {node.id[:8]}",
        ]

        if node.metric is not None:
            label_lines.append(f"metric: {node.metric:.4f}")

        if node.is_terminal:
            label_lines.append("[TERMINAL]")

        if node.improve_failure_depth > 0:
            label_lines.append(f"[F{node.improve_failure_depth}]")

        label = "\n".join(label_lines)

        # 设置颜色
        if node.id == best_node_id:
            fillcolor = "#90EE90"  # 最佳节点 - 亮绿色
        elif node.run_status == "SUCCESS":
            fillcolor = "#E8F5E9"  # 成功 - 浅绿色
        elif node.run_status == "FAILED":
            fillcolor = "#FFCDD2"  # 失败 - 浅红色
        else:
            fillcolor = "#F5F5F5"  # 未知 - 灰色

        if node.stage == "root":
            fillcolor = "#BBDEFB"  # 根节点 - 浅蓝色
        elif node.stage == "draft":
            fillcolor = "#FFF3E0"  # draft - 浅橙色

        g.node(node_id, label=label, fillcolor=fillcolor)

        # 递归子节点
        for child_id in node.children_ids:
            if child_id in nodes:
                add_node_recursive(child_id)
                g.edge(node_id, child_id)

    add_node_recursive(root_id)

    try:
        output_path = g.render(output_file, format='png', cleanup=True, view=False)
        print(f"Graphviz visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error rendering graphviz: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mcts_claude.py <mcts.log> [--html|--graphviz|--all]")
        print("\nOptions:")
        print("  (no option)    - Print ASCII tree visualization (default)")
        print("  --html         - Generate HTML tree visualization")
        print("  --graphviz     - Generate Graphviz tree visualization (requires graphviz)")
        print("  --all          - Generate all formats (ASCII + HTML + Graphviz)")
        print("\nExamples:")
        print("  python visualize_mcts_claude.py results/xxx/mcts.log")
        print("  python visualize_mcts_claude.py results/xxx/mcts.log --html")
        print("  python visualize_mcts_claude.py results/xxx/mcts.log --all")
        sys.exit(1)

    log_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else ""

    generate_html = output_format in ["--html", "--all"]
    generate_graphviz = output_format in ["--graphviz", "--all"]

    try:
        nodes, root_id = parse_log_claude(log_file)

        if not root_id:
            print("Error: Could not find root node in log")
            sys.exit(1)

        print_tree_ascii(nodes, root_id)

        if generate_html:
            output_html = log_file.replace(".log", "_tree_claude.html")
            print_tree_html(nodes, root_id, output_html)

        if generate_graphviz:
            output_gv = log_file.replace(".log", "_tree_claude")
            print_tree_graphviz(nodes, root_id, output_gv)

    except FileNotFoundError:
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
