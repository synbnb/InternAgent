# InternAgent记忆机制完整分析

## 🎯 概述

InternAgent实现了**四层记忆系统**，用于存储实验结果、学习成功/失败模式、跟踪历史创意，并基于经验进化提示词。

---

## 📊 记忆系统架构

### 四层记忆系统

```
┌─────────────────────────────────────────────────────────────┐
│                    InternAgent记忆系统                         │
├─────────────────────────────────────────────────────────────┤
│  1️⃣ Context Memory    - 对话历史和工作记忆                     │
│  2️⃣ Task Memory       - 实验结果的短期记忆                       │
│  3️⃣ Online Memory     - 实验完成时自动保存                       │
│  4️⃣ Long Memory       - 历史创意跟踪和提示词进化                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ Context Memory（上下文记忆）

### 作用
存储对话历史和工作记忆，用于session持久化。

### 配置
```yaml
memory:
  context_memory:
    backend: "file_system"
    file_dir: "results"
```

### 实现位置
- **文件**：`internagent/mas/memory/memory_manager.py`
- **实现**：`FileSystemMemoryManager`

### 存储内容
- Session状态（INITIAL, GENERATING, COMPLETED等）
- 历史创意（ideas）
- 迭代次数
- 反馈历史
- 工具使用统计

### 存储位置
```
results/{task_name}/traj_{session_id}.json
```

### 数据结构
```json
{
  "id": "session_1780481098",
  "task": {...},
  "ideas": [
    {
      "id": "idea_xxx",
      "text": "创意内容",
      "score": 8.5,
      "rationale": "理由",
      "method": "方法描述"
    }
  ],
  "iterations_completed": 1,
  "state": "completed",
  "feedback_history": [],
  "top_ideas": ["idea_id_1", "idea_id_2"]
}
```

---

## 2️⃣ Task Memory（任务记忆）

### 作用
**短期记忆**，记录实验结果（正/负标签），用于未来指导创意生成。

### 核心功能
1. **存储实验记录**：保存每个创意的baseline和improved结果
2. **自动标签**：根据性能提升率自动标记正/负/中性
3. **相似检索**：基于语义相似性检索相关历史记录
4. **过滤失败创意**：避免生成与失败历史相似的创意

### 配置
```yaml
memory:
  task_memory:
    enabled: true
    memory_dir: "./config/mem_store"
    top_k: 5  # 检索最相关的5条记录
    alpha: 0.5  # 关键词检索和语义检索的权重
    include_details: true
    embedding_mode: "description"  # 基于描述生成嵌入
    embedding:
      model_type: "local"
      model_name: ""
```

### 实现位置
- **文件**：`internagent/mas/memory/task_memory.py`
- **核心类**：`TaskMemoryLayer`, `TaskMemRecord`

### 存储结构

#### 文件组织
```
config/mem_store/
├── {task_name}/
│   ├── memory_store.json  # 主存储文件
│   ├── embedding_index.pkl  # 嵌入索引
│   └── metadata.json  # 元数据
```

#### TaskMemRecord结构
```python
@dataclass
class TaskMemRecord:
    record_id: str          # 唯一标识符
    name: str              # 创意名称
    title: str             # 标题
    description: str       # 详细描述
    statement: str         # 陈述
    method: str            # 方法
    
    # 实验结果
    baseline_results: Dict[str, float]     # run_0结果
    improved_results: Dict[str, float]     # run_1~run_N的best/avg
    all_run_results: List[Dict[str, float]]  # 所有运行结果
    
    # 标签：1(正), 0(中性), -1(负)
    label: int = 0
    
    # 改进详情
    improvement_rates: Dict[str, float]    # 每个指标的改进率
    overall_improvement_rate: float = 0.0  # 总体改进率
    primary_metric: Optional[str] = None  # 主要指标
    
    success: bool = False
    
    # 元数据
    task: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
```

### 核心方法

#### 1. 保存实验结果
```python
def save_experiment_result(
    idea: Dict[str, Any],
    baseline_results: Dict[str, float],
    improved_results: Dict[str, float],
    label: int
) -> TaskMemRecord
```

#### 2. 检索相似记录
```python
def retrieve_similar_ideas(
    query_idea: Dict[str, Any],
    top_k: int = 5,
    filter_label: Optional[int] = None
) -> List[TaskMemRecord]
```

#### 3. 生成指导提示
```python
def generate_guidance_prompt(
    query_idea: Dict[str, Any],
    include_failed: bool = True
) -> str
```

### 工作流程

```
实验完成
    ↓
OnlineMemorySaver自动保存
    ├─→ 读取实验结果（run_0, run_1, ...）
    ├─→ 计算改进率和标签
    └─→ 保存到TaskMemory
    ↓
生成新创意时
    ├─→ GenerationAgent检索相似记录
    ├─→ 生成指导提示（包含成功/失败模式）
    └─→ 基于历史经验调整创意
```

---

## 3️⃣ Online Memory（在线记忆）

### 作用
**实时保存**实验结果，当实验完成时自动保存到Task Memory。

### 配置
```yaml
memory:
  online_memory:
    enabled: true
    aggregation: "best"  # best/avg/last
```

### 实现位置
- **文件**：`internagent/mas/memory/online_memory.py`
- **核心类**：`OnlineMemorySaver`

### 核心功能

#### 自动保存机制
```python
class OnlineMemorySaver:
    def save_idea_result(
        idea: Dict[str, Any],
        results_dir: Path,
        session_id: Optional[str] = None,
        traj_path: Optional[Path] = None
    ) -> bool
```

#### 聚合方法
- **best**：选择性能最好的run结果
- **avg**：计算所有run的平均结果
- **last**：使用最后一个run的结果

### 触发时机
```
实验运行完成
    ↓
ExperimentRunner检测完成
    ↓
OnlineMemorySaver.save_idea_result()
    ↓
保存到TaskMemory
```

### 配置示例
```yaml
memory:
  online_memory:
    enabled: true
    aggregation: "best"  # 推荐使用best
```

---

## 4️⃣ Long Memory（长期记忆）

### 作用
**长期存储**所有历史创意，构建创意关系图，支持提示词进化。

### 两个核心组件

#### 4.1 IdeaGraph（创意图）

**功能**：
- 存储所有历史创意
- 基于相似性构建创意关系图
- 支持创意聚类和探索性评分

**技术栈**：
- **ChromaDB**：向量存储和相似性搜索
- **NetworkX**：图操作和聚类
- **OpenAI Embeddings**：文本向量化

**配置**：
```yaml
memory:
  long_memory:
    enabled: true
    idea_graph:
      similarity_threshold: 0.7  # 相似性阈值
```

**存储位置**：
```
results/{task_name}/
├── chroma_db/           # ChromaDB向量存储
│   └── {task_name}/
├── {task_name}_graph.pkl  # NetworkX图数据
└── idea_graph.png       # 可视化图
```

**数据结构**：
```python
@dataclass
class IdeaGraph:
    working_dir: str              # 工作目录
    namespace: str                # 命名空间
    similarity_threshold: float   # 相似性阈值
    
    # 内部组件
    chroma_client: ChromaClient
    collection: Collection        # ChromaDB集合
    graph: nx.Graph              # NetworkX图
```

**核心方法**：
```python
def add_idea_node(self, idea: Dict[str, Any]) -> None
def get_exploration_score(self, idea: Dict[str, Any]) -> float
def cluster_ideas(self, method: str = "louvain") -> Dict[str, List[str]]
def get_cluster_summary(self) -> Dict[str, List[str]]
```

**工作流程**：
```
创意生成完成
    ↓
添加到IdeaGraph
    ├─→ 向量化创意文本
    ├─→ 存储到ChromaDB
    ├─→ 添加节点到NetworkX图
    └─→ 计算相似性并创建边
    ↓
计算探索性评分
    ├─→ 基于邻居节点密度
    └─→ 鼓励探索不相似区域
```

#### 4.2 PromptEvolver（提示词进化器）

**功能**：
- 分析正负经验库
- 识别最佳方法
- 生成新的任务方向
- 更新背景信息

**配置**：
```yaml
memory:
  long_memory:
    prompt_evolver:
      enabled: true
      evolution_interval: 1  # 每轮进化
      num_candidates: 3     # 生成3个候选提示
```

**核心方法**：
```python
def evolve_prompt(
    current_prompt: Dict[str, Any],
    memory_records: List[TaskMemRecord]
) -> Dict[str, Any]
```

**进化策略**：
1. 分析最佳方法（基于性能指标）
2. 提取成功模式
3. 生成新的背景描述
4. 更新任务方向

---

## 🔗 记忆系统协同工作

### 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                      第N轮实验开始                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  1. 加载历史记忆                            │
        │   ├─ Context Memory: 加载session状态       │
        │   ├─ Task Memory: 检索相似历史记录          │
        │   └─ IdeaGraph: 加载所有历史创意            │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  2. 生成创意（GenerationAgent）              │
        │   ├─ 使用Task Memory的指导提示             │
        │   ├─ 过滤与失败历史相似的创意                │
        │   └─ 基于IdeaGraph计算探索性评分             │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  3. 运行实验（ExperimentRunner）            │
        │   └─ 执行多个创意的实验                      │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  4. 在线保存（OnlineMemory）                │
        │   └─ 自动保存每个完成的实验到Task Memory      │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  5. 更新长期记忆（Long Memory）              │
        │   ├─ IdeaGraph: 添加新创意                  │
        │   └─ PromptEvolver: 进化提示词              │
        └───────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │  6. 持久化Context Memory                   │
        │   └─ 保存session状态到traj文件             │
        └───────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      准备下一轮实验                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 记忆系统使用示例

### 示例1：GenerationAgent使用Task Memory

```python
# generation_agent.py
context = {
    "goal": session.task.to_dict(),
    "iteration": session.iterations_completed,
    "feedback": session.feedback_history,
    "paper_lst": paper_lst,
    "task_name": task_name
}

# 如果use_memory=True，检索相似记录
if self.config.get("use_memory", False):
    similar_records = self.task_memory.retrieve_similar_ideas(
        query_idea=current_idea,
        top_k=5
    )
    
    # 生成指导提示
    guidance = self.task_memory.generate_guidance_prompt(
        query_idea=current_idea,
        include_failed=True
    )
    
    # 将指导添加到prompt
    prompt += f"\n## Historical Guidance\n{guidance}"
```

### 示例2：过滤失败创意

```python
# generation_agent.py
if self.config.get("filter_failed_ideas", False):
    # 检索失败记录
    failed_records = self.task_memory.retrieve_similar_ideas(
        query_idea=idea,
        filter_label=-1,  # 只检索负标签记录
        top_k=5
    )
    
    # 检查是否与失败记录相似
    for failed_record in failed_records:
        similarity = calculate_similarity(idea, failed_record)
        if similarity > self.config.get("failed_similarity_threshold", 0.7):
            # 重新生成创意
            idea = regenerate_idea()
            break
```

### 示例3：IdeaGraph计算探索性评分

```python
# stage.py
def _load_historical_ideas_to_graph(self):
    """加载所有历史创意到IdeaGraph"""
    for ideas_file in glob.glob("results/*_launch/session_*/ideas.json"):
        with open(ideas_file) as f:
            ideas = json.load(f)
        
        for idea in ideas:
            self.idea_graph.add_idea_node(idea)
            
    # 聚类创意
    self.idea_graph.cluster_ideas(method="louvain")
```

---

## 🔑 关键配置选项

### Task Memory配置
```yaml
memory:
  task_memory:
    enabled: true                          # 启用任务记忆
    memory_dir: "./config/mem_store"       # 存储目录
    top_k: 5                               # 检索记录数
    alpha: 0.5                             # 混合检索权重
    embedding_mode: "description"          # 嵌入模式
```

### Agent配置
```yaml
agents:
  generation:
    use_memory: true                       # 使用记忆
    filter_failed_ideas: true             # 过滤失败创意
    failed_similarity_threshold: 0.7       # 失败相似性阈值
    max_regeneration_attempts: 2           # 最大重新生成次数
  
  evolution:
    use_memory: true                       # 进化时使用记忆
    filter_failed_ideas: true              # 过滤失败创意
```

### Long Memory配置
```yaml
memory:
  long_memory:
    enabled: true                          # 启用长期记忆
    idea_graph:
      similarity_threshold: 0.7            # 相似性阈值
    prompt_evolver:
      enabled: true                        # 启用提示词进化
      evolution_interval: 1                # 进化间隔
      num_candidates: 3                     # 候选数量
```

---

## 🎯 记忆系统优势

### 1. 避免重复错误
- Task Memory记录失败实验
- 自动过滤相似失败创意
- 减少无效实验次数

### 2. 学习成功模式
- 记录成功实验的特征
- 生成指导提示
- 提高新创意成功率

### 3. 鼓励探索
- IdeaGraph跟踪创意多样性
- 探索性评分机制
- 避免局部最优

### 4. 持续改进
- PromptEvolver基于经验进化
- 自动更新任务描述
- 提高后续轮次质量

---

## 📁 文件组织

### 记忆相关文件
```
internagent/
├── mas/
│   ├── memory/
│   │   ├── memory_manager.py         # 上下文记忆
│   │   ├── task_memory.py            # 任务记忆
│   │   ├── online_memory.py          # 在线记忆
│   │   ├── long_memory.py            # 长期记忆
│   │   ├── retriever.py              # 检索器
│   │   ├── file_system_memory_manager.py
│   │   └── in_memory_memory_manager.py
│   └── agents/
│       └── exp_analyze_agent.py      # 实验分析Agent
└── stage.py                           # 记忆系统使用

config/
└── mem_store/                         # Task Memory存储
    └── {task_name}/
        ├── memory_store.json
        ├── embedding_index.pkl
        └── metadata.json

results/
└── {task_name}/
    ├── traj_{session_id}.json         # Context Memory
    ├── chroma_db/                     # IdeaGraph
    ├── {task_name}_graph.pkl
    └── idea_graph.png
```

---

## 🔍 使用建议

### 1. 启用所有记忆层
```yaml
memory:
  task_memory:
    enabled: true
  online_memory:
    enabled: true
  long_memory:
    enabled: true
```

### 2. 配置合理的相似性阈值
```yaml
agents:
  generation:
    failed_similarity_threshold: 0.7  # 不太严格，避免过度过滤
```

### 3. 使用best聚合方法
```yaml
memory:
  online_memory:
    aggregation: "best"  # 推荐：选择最佳结果
```

### 4. 定期清理记忆
- 删除过期的session文件
- 清理IdeaGraph中的冗余节点
- 归档旧的记忆记录

---

## 💡 最佳实践

### 1. 初始阶段
- 启用Online Memory自动保存
- 收集足够的实验数据
- 建立初始经验库

### 2. 中期阶段
- 调整相似性阈值
- 优化过滤策略
- 分析记忆效果

### 3. 长期运行
- 定期检查记忆质量
- 清理无效记录
- 更新提示词进化策略

---

*文档完成时间：2024-06-09*  
*适用版本：InternAgent 1.5*  
*关键特性：四层记忆系统，自动学习，持续改进*
