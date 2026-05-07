from typing import Dict, List, Set, Any, Optional
from enum import Enum


class NodeExecutionStatus(Enum):
    """节点状态枚举"""
    PENDING = "pending"    # 未执行
    EXECUTED = "executed"  # 已执行


class DirectedGraph:
    """
    有向图数据结构
    
    支持节点和边的属性管理，以及依赖状态跟踪
    """
    
    def __init__(self):
        # 节点数据: {node_id: {attributes, status, in_edges, out_edges}}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        
        # 边数据: {edge_id: {from_node, to_node, attributes}}
        self.edges: Dict[str, Dict[str, Any]] = {}
        
        # 边ID计数器
        self._edge_counter = 0
    
    def add_node(self, node_id: str, **attributes) -> bool:
        """
        添加节点
        
        Args:
            node_id: 节点唯一标识符
            **attributes: 节点的属性
            
        Returns:
            bool: 是否添加成功（如果节点已存在则返回False）
        """
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = {
            'attributes': attributes,
            'status': NodeExecutionStatus.PENDING,
            'in_edges': set(),   # 指向该节点的边ID集合
            'out_edges': set()   # 从该节点出发的边ID集合
        }
        return True
    
    def add_edge(self, from_node: str, to_node: str, **attributes) -> Optional[str]:
        """
        添加边
        
        Args:
            from_node: 起始节点ID
            to_node: 目标节点ID
            **attributes: 边的属性
            
        Returns:
            Optional[str]: 边ID，如果添加失败则返回None
        """
        # 检查节点是否存在
        if from_node not in self.nodes or to_node not in self.nodes:
            return None
        
        # 生成边ID
        edge_id = f"edge_{self._edge_counter}"
        self._edge_counter += 1
        
        # 创建边
        self.edges[edge_id] = {
            'from_node': from_node,
            'to_node': to_node,
            'attributes': attributes
        }
        
        # 更新节点的边集合
        self.nodes[from_node]['out_edges'].add(edge_id)
        self.nodes[to_node]['in_edges'].add(edge_id)
        
        return edge_id
    
    def get_node_attributes(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点属性"""
        if node_id in self.nodes:
            return self.nodes[node_id]['attributes']
        return None
    
    def get_edge_attributes(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """获取边属性"""
        if edge_id in self.edges:
            return self.edges[edge_id]['attributes']
        return None
    
    def set_node_status(self, node_id: str, status: NodeExecutionStatus) -> bool:
        """
        设置节点状态
        
        Args:
            node_id: 节点ID
            status: 节点状态
            
        Returns:
            bool: 是否设置成功
        """
        if node_id in self.nodes:
            self.nodes[node_id]['status'] = status
            return True
        return False
    
    def get_node_status(self, node_id: str) -> Optional[NodeExecutionStatus]:
        """获取节点状态"""
        if node_id in self.nodes:
            return self.nodes[node_id]['status']
        return None
    
    def get_ready_nodes(self) -> List[str]:
        """
        获取所有没有前置依赖的节点（可执行的节点）
        
        规则：
        1. 没有入边的节点（根节点）
        2. 所有前置节点都已执行的节点
        
        Returns:
            List[str]: 可执行节点ID列表
        """
        ready_nodes = []
        
        for node_id, node_data in self.nodes.items():
            # 跳过已执行的节点
            if node_data['status'] == NodeExecutionStatus.EXECUTED:
                continue
            
            # 检查是否所有前置节点都已执行
            is_ready = True
            
            for edge_id in node_data['in_edges']:
                edge = self.edges[edge_id]
                from_node = edge['from_node']
                from_node_status = self.nodes[from_node]['status']
                
                # 如果前置节点未执行，则当前节点不可执行
                if from_node_status != NodeExecutionStatus.EXECUTED:
                    is_ready = False
                    break
            
            if is_ready:
                ready_nodes.append(node_id)
        
        return ready_nodes
    
    def get_dependent_nodes(self, node_id: str) -> List[str]:
        """
        获取指定节点的所有依赖节点（直接依赖）
        
        Args:
            node_id: 节点ID
            
        Returns:
            List[str]: 依赖节点ID列表
        """
        if node_id not in self.nodes:
            return []
        
        dependent_nodes = []
        for edge_id in self.nodes[node_id]['in_edges']:
            edge = self.edges[edge_id]
            dependent_nodes.append(edge['from_node'])
        
        return dependent_nodes
    
    def get_dependent_nodes_recursive(self, node_id: str) -> Set[str]:
        """
        获取指定节点的所有依赖节点（递归，包括间接依赖）
        
        Args:
            node_id: 节点ID
            
        Returns:
            Set[str]: 所有依赖节点ID集合
        """
        if node_id not in self.nodes:
            return set()
        
        visited = set()
        stack = [node_id]
        
        while stack:
            current_node = stack.pop()
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 添加所有前置节点到栈中
            for edge_id in self.nodes[current_node]['in_edges']:
                edge = self.edges[edge_id]
                stack.append(edge['from_node'])
        
        # 移除节点本身
        visited.discard(node_id)
        return visited
    
    def get_successor_nodes(self, node_id: str) -> List[str]:
        """
        获取指定节点的所有后继节点（直接后继）
        
        Args:
            node_id: 节点ID
            
        Returns:
            List[str]: 后继节点ID列表
        """
        if node_id not in self.nodes:
            return []
        
        successor_nodes = []
        for edge_id in self.nodes[node_id]['out_edges']:
            edge = self.edges[edge_id]
            successor_nodes.append(edge['to_node'])
        
        return successor_nodes
    
    def remove_node(self, node_id: str) -> bool:
        """
        删除节点及其相关的所有边
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否删除成功
        """
        if node_id not in self.nodes:
            return False
        
        # 删除所有相关的边
        edges_to_remove = set()
        
        # 收集所有需要删除的边
        for edge_id in self.nodes[node_id]['in_edges']:
            edges_to_remove.add(edge_id)
        
        for edge_id in self.nodes[node_id]['out_edges']:
            edges_to_remove.add(edge_id)
        
        # 删除边
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # 删除节点
        del self.nodes[node_id]
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """
        删除边
        
        Args:
            edge_id: 边ID
            
        Returns:
            bool: 是否删除成功
        """
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        from_node = edge['from_node']
        to_node = edge['to_node']
        
        # 从节点的边集合中移除
        if from_node in self.nodes:
            self.nodes[from_node]['out_edges'].discard(edge_id)
        
        if to_node in self.nodes:
            self.nodes[to_node]['in_edges'].discard(edge_id)
        
        # 删除边
        del self.edges[edge_id]
        return True
    
    def has_cycle(self) -> bool:
        """
        检查图中是否存在环
        
        Returns:
            bool: 是否存在环
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for edge_id in self.nodes[node_id]['out_edges']:
                edge = self.edges[edge_id]
                neighbor = edge['to_node']
                
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def get_topological_sort(self) -> Optional[List[str]]:
        """
        获取拓扑排序结果
        
        Returns:
            Optional[List[str]]: 拓扑排序结果，如果存在环则返回None
        """
        if self.has_cycle():
            return None
        
        # 计算入度
        in_degree = {node_id: len(self.nodes[node_id]['in_edges']) 
                    for node_id in self.nodes}
        
        # 找到所有入度为0的节点
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 减少所有后继节点的入度
            for edge_id in self.nodes[current]['out_edges']:
                edge = self.edges[edge_id]
                neighbor = edge['to_node']
                in_degree[neighbor] -= 1
                
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(self.nodes) else None
    
    def reset_all_nodes_status(self):
        """重置所有节点状态为未执行"""
        for node_data in self.nodes.values():
            node_data['status'] = NodeExecutionStatus.PENDING
    
    def build_initial_graph(self, node_list: List[Dict[str, Any]], edge_list: List[Dict[str, Any]]) -> bool:
        """
        从节点列表和边列表快速构建图
        
        Args:
            node_list: 节点列表，每个元素是字典，必须包含'id'键
            edge_list: 边列表，每个元素是字典，必须包含'id'键，以及'from_node'和'to_node'键
            
        Returns:
            bool: 是否构建成功
            
        Example:
            node_list = [
                {'id': 'A', 'name': '任务A', 'priority': 1},
                {'id': 'B', 'name': '任务B', 'priority': 2}
            ]
            edge_list = [
                {'id': 'edge1', 'from_node': 'A', 'to_node': 'B', 'weight': 1.0}
            ]
        """
        try:
            # 清空现有图
            self.nodes.clear()
            self.edges.clear()
            self._edge_counter = 0
            
            # 添加所有节点
            for node_dict in node_list:
                if 'id' not in node_dict:
                    raise ValueError(f"节点字典缺少'id'键: {node_dict}")
                
                node_id = node_dict['id']
                # 复制除了'id'之外的所有属性
                attributes = {k: v for k, v in node_dict.items() if k != 'id'}
                
                if not self.add_node(node_id, **attributes):
                    raise ValueError(f"添加节点失败，可能节点ID重复: {node_id}")
            
            # 添加所有边
            for edge_dict in edge_list:
                if 'id' not in edge_dict:
                    raise ValueError(f"边字典缺少'id'键: {edge_dict}")
                if 'from_node' not in edge_dict:
                    raise ValueError(f"边字典缺少'from_node'键: {edge_dict}")
                if 'to_node' not in edge_dict:
                    raise ValueError(f"边字典缺少'to_node'键: {edge_dict}")
                
                edge_id = edge_dict['id']
                from_node = edge_dict['from_node']
                to_node = edge_dict['to_node']
                
                # 复制除了'id', 'from_node', 'to_node'之外的所有属性
                attributes = {k: v for k, v in edge_dict.items() 
                            if k not in ['id', 'from_node', 'to_node']}
                
                # 检查节点是否存在
                if from_node not in self.nodes:
                    raise ValueError(f"边引用的起始节点不存在: {from_node}")
                if to_node not in self.nodes:
                    raise ValueError(f"边引用的目标节点不存在: {to_node}")
                
                # 添加边
                self.edges[edge_id] = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'attributes': attributes
                }
                
                # 更新节点的边集合
                self.nodes[from_node]['out_edges'].add(edge_id)
                self.nodes[to_node]['in_edges'].add(edge_id)
            
            return True
            
        except Exception as e:
            # 如果构建失败，清空图并返回False
            self.nodes.clear()
            self.edges.clear()
            self._edge_counter = 0
            print(f"构建图失败: {e}")
            return False
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        获取图的统计信息
        
        Returns:
            Dict[str, Any]: 图的统计信息
        """
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        
        pending_nodes = sum(1 for node_data in self.nodes.values() 
                          if node_data['status'] == NodeExecutionStatus.PENDING)
        executed_nodes = total_nodes - pending_nodes
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'pending_nodes': pending_nodes,
            'executed_nodes': executed_nodes,
            'has_cycle': self.has_cycle()
        }
    
    def visualize_text(self) -> None:
        """
        文本形式可视化任务依赖图
        """
        # 获取拓扑排序
        topological_order = self.get_topological_sort()
        if not topological_order:
            print("警告：图中存在环，无法进行拓扑排序")
            return

        print("\n" + "="*80)
        print("任务依赖关系图 (文本形式)")
        print("="*80)

        # 计算层级
        node_levels = {}
        for node_id in topological_order:
            dependent_nodes = self.get_dependent_nodes_recursive(node_id)
            max_level = -1
            for dep_node in dependent_nodes:
                if dep_node in node_levels:
                    max_level = max(max_level, node_levels[dep_node])
            node_levels[node_id] = max_level + 1

        # 按层级显示
        max_level = max(node_levels.values()) if node_levels else 0

        for level in range(max_level + 1):
            print(f"\n层级 {level}:")
            print("-" * 40)

            level_nodes = [node_id for node_id, lvl in node_levels.items() if lvl == level]

            for i, node_id in enumerate(level_nodes):
                node_attrs = self.get_node_attributes(node_id)
                subtask = node_attrs.get('subtask', f'Task {node_id}') if node_attrs else f'Task {node_id}'
                status = self.get_node_status(node_id)
                status_str = "✓" if status == NodeExecutionStatus.EXECUTED else "○"

                print(f"  {i+1}. [{status_str}] {node_id}: {subtask}")

        # 显示依赖关系
        print(f"\n依赖关系:")
        print("-" * 40)

        for edge_id, edge_data in self.edges.items():
            source = edge_data['from_node']
            target = edge_data['to_node']
            relationship = edge_data['attributes'].get('relationship', 'unknown')
            print(f"  {source} --[{relationship}]--> {target}")

        # 显示执行顺序
        print(f"\n推荐执行顺序:")
        print("-" * 40)
        for i, node_id in enumerate(topological_order):
            node_attrs = self.get_node_attributes(node_id)
            subtask = node_attrs.get('subtask', f'Task {node_id}') if node_attrs else f'Task {node_id}'
            print(f"  {i+1}. {node_id}: {subtask}")

        print("="*80)
    
    def visualize_graph(self, save_path: str = None) -> None:
        """
        图形形式可视化任务依赖图 - 使用拓扑排序创建层次化布局
        
        Args:
            save_path: 保存图片的路径（可选）
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("需要安装 networkx 和 matplotlib: pip install networkx matplotlib")
            return

        # 拓扑排序
        topological_order = self.get_topological_sort()
        if not topological_order:
            print("警告：图中存在环，无法进行拓扑排序")
            return

        # 计算层级
        node_levels = {}
        for node_id in topological_order:
            deps = self.get_dependent_nodes_recursive(node_id)
            node_levels[node_id] = max((node_levels[d] for d in deps if d in node_levels), default=-1) + 1

        # 分组
        level_groups = {}
        for nid, lvl in node_levels.items():
            level_groups.setdefault(lvl, []).append(nid)

        # 构建 NetworkX 图
        G = nx.DiGraph()
        
        # 从 DirectedGraph 中提取节点和边信息
        for node_id, node_data in self.nodes.items():
            subtask = node_data['attributes'].get('subtask', f'Task {node_id}')
            if len(subtask) > 30:
                subtask = subtask[:27] + "..."
            G.add_node(node_id, label=subtask, level=node_levels[node_id])
        
        # 从 DirectedGraph 中提取边信息
        for edge_id, edge_data in self.edges.items():
            source = edge_data['from_node']
            target = edge_data['to_node']
            relationship = edge_data['attributes'].get('relationship', 'unknown')
            G.add_edge(source, target, relationship=relationship)

        # 层次布局
        pos = {}
        max_level = max(node_levels.values()) if node_levels else 0
        for level in range(max_level + 1):
            if level in level_groups:
                nodes_in_level = level_groups[level]
                for i, nid in enumerate(nodes_in_level):
                    y = 1.0 - (i + 1) / (len(nodes_in_level) + 1)
                    x = level / max_level if max_level > 0 else 0.5
                    pos[nid] = (x, y)

        # 绘图 - 调整图形大小和布局
        fig, ax = plt.subplots(figsize=(16, 10))
        
        node_colors = [
            'lightblue' if self.get_node_status(n) == NodeExecutionStatus.PENDING else 'lightgreen'
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3500, alpha=0.8, ax=ax)

        edge_colors = []
        for u, v, data in G.edges(data=True):
            rel = data.get('relationship', 'unknown')
            edge_colors.append(self._get_relationship_color(rel))
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20,
                               arrowstyle='->', width=2, alpha=0.7, ax=ax)

        node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight='bold', ax=ax)

        for level in range(max_level + 1):
            ax.text(level / max_level if max_level > 0 else 0.5, 1.05,
                   f'Level {level}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        legend_elements = [
            mpatches.Patch(color='red', label='requires'),
            mpatches.Patch(color='blue', label='refines'),
            mpatches.Patch(color='green', label='verifies'),
            mpatches.Patch(color='orange', label='documents'),
            mpatches.Patch(color='purple', label='handoff'),
            mpatches.Patch(color='lightblue', label='Pending'),
            mpatches.Patch(color='lightgreen', label='Executed')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        ax.set_title('Task Dependency Graph (Topological Layout)', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 调整布局以避免警告
        plt.subplots_adjust(right=0.85)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")

        plt.show()
    
    def _setup_chinese_font(self, preferred=None):
        """
        启用可同时显示中英文的字体配置。
        1) 优先使用同时包含 Latin+CJK 的字体（Noto/Source Han/YaHei/PingFang/WenQuanYi/SimHei）
        2) 若未找到，回退到 DejaVu Sans（英文 OK，中文可能缺字）
        返回：选中的 CJK 字体名（找不到则返回 None）
        """
        import os
        import matplotlib as mpl
        from matplotlib import font_manager as fm
        from matplotlib.font_manager import FontProperties

        # 英文备用（matplotlib 自带）
        latin = 'DejaVu Sans'
        _ = fm.findfont(latin, fallback_to_default=True)

        # 同时含 Latin+CJK 的常见字体（按优先级）
        name_candidates = (preferred or []) + [
            'Noto Sans CJK SC',
            'Source Han Sans SC',
            'Microsoft YaHei',
            'PingFang SC',
            'WenQuanYi Zen Hei',
            'SimHei',
            'Arial Unicode MS',
        ]
        # 常见路径（存在则注册）
        path_candidates = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
            r'C:\Windows\Fonts\simhei.ttf',    # 黑体
            r'C:\Windows\Fonts\arialuni.ttf',  # Arial Unicode MS（若有）
        ]

        available = {f.name for f in fm.fontManager.ttflist}
        cjk = None

        # 1) 按名字匹配
        for n in name_candidates:
            if n in available:
                cjk = n
                break

        # 2) 按路径注册并获取真实名字
        if not cjk:
            for p in path_candidates:
                if os.path.exists(p):
                    fm.fontManager.addfont(p)
                    cjk = FontProperties(fname=p).get_name()
                    break

        # 全局配置（统一使用主字体，确保同一文本中中英文都能显示）
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['pdf.fonttype'] = 42  # 便于矢量输出嵌入
        mpl.rcParams['ps.fonttype']  = 42

        if cjk:
            # 设为主字体；再把 DejaVu 作为兜底（个别符号缺失时）
            mpl.rcParams['font.family']     = [cjk]
            mpl.rcParams['font.sans-serif'] = [cjk, latin]
            try:
                fm._rebuild()
            except Exception:
                pass
            print(f'使用字体：{cjk}（fallback: {latin}）')
            return cjk
        else:
            # 没有中文字体时，至少英文正常
            mpl.rcParams['font.family']     = [latin]
            mpl.rcParams['font.sans-serif'] = [latin]
            print('未找到中文字体，回退到 DejaVu Sans（中文可能缺字）。')
            return None
    
    def _get_relationship_color(self, relationship: str) -> str:
        """根据关系类型获取颜色"""
        color_map = {
            "requires": "red",
            "refines": "blue", 
            "verifies": "green",
            "documents": "orange",
            "handoff": "purple"
        }
        return color_map.get(relationship, "gray")
    
    def __str__(self) -> str:
        """图的字符串表示"""
        info = self.get_graph_info()
        return (f"DirectedGraph(nodes={info['total_nodes']}, "
                f"edges={info['total_edges']}, "
                f"pending={info['pending_nodes']}, "
                f"executed={info['executed_nodes']})")
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_graph_levels(self) -> int:
        """
        获取图的层数
        
        基于拓扑排序和依赖关系分析，计算图的最大层数。
        根节点（没有依赖的节点）为第0层，其他节点的层数为其最大依赖节点层数+1。
        
        Returns:
            int: 图的层数，如果图存在环则返回-1
        """
        # 检查是否存在环
        if self.has_cycle():
            return -1
        
        # 获取拓扑排序
        topological_order = self.get_topological_sort()
        if not topological_order:
            return -1
        
        # 计算每个节点的层数
        node_levels = {}
        
        for node_id in topological_order:
            # 获取所有依赖节点（递归）
            dependent_nodes = self.get_dependent_nodes_recursive(node_id)
            
            # 计算最大依赖层数
            max_dependent_level = -1
            for dep_node in dependent_nodes:
                if dep_node in node_levels:
                    max_dependent_level = max(max_dependent_level, node_levels[dep_node])
            
            # 当前节点层数 = 最大依赖层数 + 1
            node_levels[node_id] = max_dependent_level + 1
        
        # 返回最大层数
        return max(node_levels.values()) + 1 if node_levels else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将图转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含 nodes 和 edges 的字典，格式如下：
            {
                "nodes": [
                    {
                        "node_id": "n1",
                        "type": "search",
                        "task": "任务描述"
                        ...
                    }
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2", 
                        "relationship": "关系描述"
                    }
                ]
            }
        """
        # 构建节点列表
        nodes = []
        for node_id, node_data in self.nodes.items():
            node_attrs = node_data['attributes']
            node_dict = {
                "node_id": node_id
            }
            
            # 添加所有节点属性
            node_dict.update(node_attrs)
            status = self.get_node_status(node_id)
            node_dict['status'] = status.name if hasattr(status, 'name') else str(status)
            
            nodes.append(node_dict)
        
        # 构建边列表
        edges = []
        for edge_id, edge_data in self.edges.items():
            edge_dict = {
                "from": edge_data['from_node'],
                "to": edge_data['to_node']
            }
            
            # 添加边属性
            edge_attrs = edge_data['attributes']
            if 'relationship' in edge_attrs:
                edge_dict['relationship'] = edge_attrs['relationship']
            
            # 添加其他属性（除了特殊处理的属性外）
            for key, value in edge_attrs.items():
                if key != 'relationship':
                    edge_dict[key] = value
            
            edges.append(edge_dict)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    @classmethod
    def from_dict(cls, graph_dict: Dict[str, Any]) -> 'DirectedGraph':
        """
        从字典格式创建图
        
        Args:
            graph_dict: 包含 nodes 和 edges 的字典，格式如下：
            {
                "nodes": [
                    {
                        "node_id": "n1",
                        "type": "search",
                        "task": "任务描述",
                        "knowledge": "知识信息（可选）"
                    }
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2", 
                        "relationship": "关系描述"
                    }
                ]
            }
        
        Returns:
            DirectedGraph: 新创建的图实例
        """
        graph = cls()
        
        # 添加节点
        if 'nodes' in graph_dict:
            for node_data in graph_dict['nodes']:
                if 'node_id' not in node_data:
                    raise ValueError(f"节点缺少 node_id: {node_data}")
                
                node_id = node_data['node_id']
                # 复制除了 node_id 之外的所有属性
                attributes = {k: v for k, v in node_data.items() if k != 'node_id'}
                
                if not graph.add_node(node_id, **attributes):
                    raise ValueError(f"添加节点失败，可能节点ID重复: {node_id}")
        
        # 添加边
        if 'edges' in graph_dict:
            for edge_data in graph_dict['edges']:
                if 'from' not in edge_data:
                    raise ValueError(f"边缺少 from 字段: {edge_data}")
                if 'to' not in edge_data:
                    raise ValueError(f"边缺少 to 字段: {edge_data}")
                
                from_node = edge_data['from']
                to_node = edge_data['to']
                
                # 复制除了 from 和 to 之外的所有属性
                attributes = {k: v for k, v in edge_data.items() if k not in ['from', 'to']}
                
                edge_id = graph.add_edge(from_node, to_node, **attributes)
                if edge_id is None:
                    raise ValueError(f"添加边失败，可能节点不存在: {from_node} -> {to_node}")
        
        return graph

    def validate_and_fix_graph(self) -> None:
        """
        验证并修复图的结构问题
        """
        # 1. 检查并删除answer节点的出边
        self.remove_answer_node_outgoing_edges()
        
        # 2. 检查并删除环
        self.remove_cycles()
    
    def remove_answer_node_outgoing_edges(self) -> int:
        """
        删除所有type为answer的节点的出边
        
        Returns:
            int: 删除的边数量
        """
        answer_nodes = []
        # 找到所有type为answer的节点
        for node_id, node_data in self.nodes.items():
            if node_data.get('attributes', {}).get('type') == 'answer':
                answer_nodes.append(node_id)
        
        # 删除这些节点的所有出边
        edges_to_remove = []
        for edge_id, edge_data in self.edges.items():
            if edge_data['from_node'] in answer_nodes:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        return len(edges_to_remove)
    
    def remove_cycles(self) -> int:
        """
        检测并删除图中的环，优先删除pending节点之间的边
        
        Returns:
            int: 删除的边数量
        """
        removed_edges = 0
        
        while self.has_cycle_dfs():
            # 找到环中的pending节点之间的边并删除
            cycle_edge = self.find_cycle_edge_to_remove()
            if cycle_edge:
                self.remove_edge(cycle_edge)
                removed_edges += 1
            else:
                # 如果没有找到pending节点之间的边，可能需要删除其他边
                # 这里简单地删除第一个找到的环中的边
                any_cycle_edge = self.find_any_cycle_edge()
                if any_cycle_edge:
                    self.remove_edge(any_cycle_edge)
                    removed_edges += 1
                else:
                    break  # 无法找到环中的边，退出循环
        
        return removed_edges
    
    def has_cycle_dfs(self) -> bool:
        """
        使用DFS检测图中是否存在环
        
        Returns:
            bool: 是否存在环
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True  # 发现环
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            # 遍历所有邻接节点
            for edge_id, edge_data in self.edges.items():
                if edge_data['from_node'] == node:
                    if dfs(edge_data['to_node']):
                        return True
            
            rec_stack.remove(node)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def find_cycle_edge_to_remove(self) -> Optional[str]:
        """
        找到环中两个pending节点之间的边
        
        Returns:
            Optional[str]: 要删除的边的ID，如果没有找到则返回None
        """
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            if node in rec_stack:
                # 发现环，查找环中pending节点之间的边
                cycle_start_idx = path.index(node)
                cycle_nodes = path[cycle_start_idx:] + [node]
                
                # 检查环中相邻节点对，找到pending节点之间的边
                for i in range(len(cycle_nodes) - 1):
                    from_node = cycle_nodes[i]
                    to_node = cycle_nodes[i + 1]
                    
                    # 检查这两个节点是否都是pending状态
                    from_status = self.nodes.get(from_node, {}).get('attributes', {}).get('status')
                    to_status = self.nodes.get(to_node, {}).get('attributes', {}).get('status')
                    
                    if from_status == 'pending' and to_status == 'pending':
                        # 找到对应的边ID
                        for edge_id, edge_data in self.edges.items():
                            if edge_data['from_node'] == from_node and edge_data['to_node'] == to_node:
                                return edge_id
                
                return None  # 环中没有pending节点之间的边
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # 遍历所有邻接节点
            for edge_id, edge_data in self.edges.items():
                if edge_data['from_node'] == node:
                    result = dfs(edge_data['to_node'])
                    if result:
                        return result
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        for node_id in self.nodes:
            if node_id not in visited:
                result = dfs(node_id)
                if result:
                    return result
        
        return None
    
    def find_any_cycle_edge(self) -> Optional[str]:
        """
        找到环中的任意一条边
        
        Returns:
            Optional[str]: 要删除的边的ID，如果没有找到则返回None
        """
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            if node in rec_stack:
                # 发现环，返回环中的第一条边
                cycle_start_idx = path.index(node)
                if cycle_start_idx < len(path) - 1:
                    from_node = path[cycle_start_idx]
                    to_node = path[cycle_start_idx + 1]
                    
                    # 找到对应的边ID
                    for edge_id, edge_data in self.edges.items():
                        if edge_data['from_node'] == from_node and edge_data['to_node'] == to_node:
                            return edge_id
                
                return None
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # 遍历所有邻接节点
            for edge_id, edge_data in self.edges.items():
                if edge_data['from_node'] == node:
                    result = dfs(edge_data['to_node'])
                    if result:
                        return result
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        for node_id in self.nodes:
            if node_id not in visited:
                result = dfs(node_id)
                if result:
                    return result
        
        return None
