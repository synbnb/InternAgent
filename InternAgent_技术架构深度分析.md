# InternAgent 技术架构深度分析

## 1. 系统概述

InternAgent 是一个基于多智能体协作的科学发现自动化系统，采用状态机驱动的编排模式，通过专业化 Agent 协作完成从假设生成到实验验证的全流程。

## 2. 核心技术架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                  │
│  launch_discovery.py | launch_qa.py | launch.py             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  编排层 (Orchestration Layer)                │
│  OrchestrationAgent (状态机驱动)                             │
│  - 会话管理                                                  │
│  - 工作流状态机                                             │
│  - Agent 调度                                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Agent 层 (Agent Layer)                       │
│  GenerationAgent | ReflectionAgent | EvolutionAgent        │
│  MethodDevelopmentAgent | RefinementAgent | RankingAgent   │
│  ScholarAgent | SurveyAgent | DRAgent | ExpAnalyzeAgent    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  模型层 (Model Layer)                         │
│  ModelFactory | OpenAIModel | AnthropicModel | DeepSeekModel │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  工具层 (Tool Layer)                           │
│  WebSearch | LiteratureSearch | MCP Server                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  记忆层 (Memory Layer)                         │
│  ContextMemory | TaskMemory | OnlineMemory | LongMemory      │
└─────────────────────────────────────────────────────────────┘
```

## 3. 工作流状态机

### 3.1 状态定义

InternAgent 的核心是一个精心设计的状态机，定义了研究的完整生命周期：

```python
class WorkflowState(Enum):
    INITIAL = "initial"                    # 初始状态
    GENERATING = "generating"              # 创意生成
    REFLECTING = "reflecting"              # 反思评估
    EVOLVING = "evolving"                # 创意进化
    RANKING = "ranking"                   # 排名筛选
    METHOD_DEVELOPMENT = "method_development"  # 方法开发
    REFINING = "refining"                 # 代码精炼
    EXTERNAL_DATA = "external_data"       # 外部数据获取
    AWAITING_FEEDBACK = "awaiting_feedback"  # 等待反馈
    COMPLETED = "completed"              # 完成
    ERROR = "error"                      # 错误
```

### 3.2 状态转换图

```
┌─────────────────────────────────────────────────────────────┐
│                        状态转换流程                            │
└─────────────────────────────────────────────────────────────┘

INITIAL
   ↓
GENERATING (GenerationAgent)
   ↓
REFLECTING (ReflectionAgent)
   ↓
EVOLVING (EvolutionAgent)
   ↓
RANKING (RankingAgent)
   ↓
METHOD_DEVELOPMENT (MethodDevelopmentAgent)
   ↓
REFINING (RefinementAgent)
   ↓
COMPLETED

可选分支：
- GENERATING → EXTERNAL_DATA (SurveyAgent 文献调研)
- 任何状态 → AWAITING_FEEDBACK (等待用户反馈)
- AWAITING_FEEDBACK → REFLECTING (处理反馈)
- 任何状态 → ERROR (错误处理)
```

## 4. 数据流分析

### 4.1 核心数据结构

#### Idea (研究创意)

```python
@dataclass
class Idea:
    id: str                          # 唯一标识
    text: str                       # 创意描述
    score: float = 0.0              # 综合评分
    rationale: str = ""              # 理论依据
    baseline_summary: str = ""       # 基线总结
    critiques: List[str]             # 批评意见
    evidence: List[Dict]             # 支持证据
    experimental_approach: str = ""  # 实验方法
    detailed_ideas: Dict             # 详细创意
    method_details: Dict             # 方法细节
    iteration: int = 0               # 迭代次数
    parent_id: Optional[str] = None  # 父创意ID
    scores: Dict[str, float]         # 各维度评分
    references: List[Dict]           # 参考文献
```

#### WorkflowSession (工作流会话)

```python
@dataclass
class WorkflowSession:
    id: str                          # 会话ID
    task: Task                       # 研究任务
    ideas: List[Idea]                # 创意列表
    state: WorkflowState             # 当前状态
    iterations_completed: int = 0    # 完成迭代数
    max_iterations: int = 4          # 最大迭代数
    top_ideas: List[str]             # 顶级创意ID
    feedback_history: List[Dict]     # 反馈历史
    method_phase: bool = False       # 方法阶段标志
    tool_usage: Dict[str, int]       # 工具使用统计
```

### 4.2 数据流动路径

#### 阶段1：初始化与背景生成

```
用户输入
   ↓
task_info.json → task_description
   ↓
Task对象创建
   ↓
DRAgent (可选) → background信息
   ↓
WorkflowSession创建
```

#### 阶段2：创意生成流程

```
WorkflowSession (GENERATING状态)
   ↓
GenerationAgent输入：
  - task.description
  - task.background
  - task.domain
  - task.constraints
   ↓
GenerationAgent输出：
  - 原始创意列表 (raw_ideas)
  - 理论依据 (rationale)
   ↓
SurveyAgent (可选)：
  - 文献搜索
  - 证据收集
   ↓
Idea对象创建并添加到session.ideas
```

#### 阶段3：反思与评估流程

```
WorkflowSession (REFLECTING状态)
   ↓
ReflectionAgent输入：
  - session.ideas (所有创意)
  - task.background
   ↓
ReflectionAgent处理：
  - 批判性分析
  - 可行性评估
  - 潜在问题识别
   ↓
Idea对象更新：
  - critiques (批评意见)
  - evidence (支持证据)
  - scores (各维度评分)
```

#### 阶段4：创意进化流程

```
WorkflowSession (EVOLVING状态)
   ↓
EvolutionAgent输入：
  - session.top_ideas (顶级创意)
  - session.ideas (所有创意)
  - TaskMemory (失败创意记忆)
   ↓
EvolutionAgent处理：
  - 创意组合
  - 变异优化
  - 失败创意过滤
   ↓
新Idea对象创建：
  - parent_id (指向父创意)
  - iteration += 1
  - refined_method_details
```

#### 阶段5：排名筛选流程

```
WorkflowSession (RANKING状态)
   ↓
RankingAgent输入：
  - session.ideas (所有创意)
  - 评分标准配置
   ↓
RankingAgent处理：
  - 多维度评分
  - 加权排名
  - Top-N选择
   ↓
session.top_ideas更新：
  - 存储Top创意ID
  - 按score排序
```

#### 阶段6：方法开发流程

```
WorkflowSession (METHOD_DEVELOPMENT状态)
   ↓
MethodDevelopmentAgent输入：
  - session.top_ideas (顶级创意)
  - task.ref_code_path (参考代码)
   ↓
MethodDevelopmentAgent处理：
  - 创意→代码转换
  - 实验设计
  - 参数配置
   ↓
Idea对象更新：
  - method_details (方法细节)
  - experimental_approach (实验方法)
```

#### 阶段7：代码精炼流程

```
WorkflowSession (REFINING状态)
   ↓
RefinementAgent输入：
  - session.top_ideas
  - method_details
   ↓
RefinementAgent处理：
  - 代码质量优化
  - 最佳实践应用
  - 性能调优
   ↓
Idea对象更新：
  - refined_method_details (精炼方法)
  - method_critiques (方法批评)
```

### 4.3 数据持久化

```
内存存储 (运行时)
   ↓
MemoryManager.store_session(session)
   ↓
磁盘存储
   - results/{task}/{timestamp}_launch/
     ├── session_*/
     │   ├── ideas.json (创意快照)
     │   └── session.json (会话状态)
     └── {timestamp}_{idea_name}/
         ├── code/ (Agent生成代码)
         ├── outputs/ (实验输出)
         └── report/ (实验报告)
```

## 5. Agent 编排机制

### 5.1 Agent 创建与管理

```python
# AgentFactory 负责创建所有Agent实例
class AgentFactory:
    def create_all_agents(self, config, model_factory):
        return {
            'generation': GenerationAgent(...),
            'reflection': ReflectionAgent(...),
            'evolution': EvolutionAgent(...),
            'ranking': RankingAgent(...),
            'method_development': MethodDevelopmentAgent(...),
            'refinement': RefinementAgent(...),
            'scholar': ScholarAgent(...),
            'survey': SurveyAgent(...),
            'dr': DRAgent(...),
            'exp_analyze': ExpAnalyzeAgent(...)
        }
```

### 5.2 Agent 协作模式

#### 模式1：顺序协作

```
GenerationAgent → ReflectionAgent → EvolutionAgent → RankingAgent
```

- **数据传递**：Idea对象在Agent间传递，每个Agent添加/修改特定字段
- **触发机制**：状态机自动转换状态，触发下一个Agent
- **协调方式**：通过WorkflowSession共享状态

#### 模式2：并行协作

```python
# 并发控制机制
async def _run_parallel_tasks(self, tasks, max_concurrent=5):
    semaphore = Semaphore(max_concurrent)
    async def bounded_task(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*[bounded_task(t) for t in tasks])
```

- **适用场景**：多个创意同时处理，多个文献搜索并行执行
- **并发限制**：`max_concurrent_tasks` 配置控制并发数
- **同步机制**：asyncio + Semaphore实现并发控制

#### 模式3：条件协作

```
GENERATING
   ↓
   ├─ do_survey=True → EXTERNAL_DATA (SurveyAgent)
   │                    ↓
   └─ do_survey=False → REFLECTING
```

- **决策依据**：配置参数、Agent输出结果
- **分支处理**：状态机根据条件选择不同路径

### 5.3 Agent 间通信机制

#### 方式1：共享状态通信

```python
# 所有Agent访问同一个WorkflowSession对象
session.ideas.append(new_idea)  # GenerationAgent
session.top_ideas = top_ids     # RankingAgent
```

#### 方式2：记忆系统通信

```python
# 跨会话记忆
TaskMemory.get_similar_ideas(new_idea)  # 查询历史
TaskMemory.store_experiment_result(result)  # 存储结果
```

#### 方式3：工具调用通信

```python
# Agent通过工具获取外部信息
WebSearch.search(query)
LiteratureSearch.search(paper_title)
```

## 6. 规划与决策机制

### 6.1 规划层次

#### 层次1：工作流级规划

**实现位置**：`OrchestrationAgent._execute_current_phase()`

```python
async def _execute_current_phase(self, session):
    # 状态机驱动的规划
    phase_handlers = {
        WorkflowState.GENERATING: self._run_generation_phase,
        WorkflowState.REFLECTING: self._run_reflection_phase,
        WorkflowState.EVOLVING: self._run_evolution_phase,
        # ... 更多阶段
    }
    
    handler = phase_handlers.get(session.state)
    await handler(session)
```

**规划内容**：
- 确定当前应该执行什么阶段
- 根据状态转换规则选择下一个阶段
- 处理异常和错误情况

#### 层次2：Agent级规划

**实现位置**：各Agent的execute()方法

```python
# GenerationAgent的规划示例
async def execute(self, task):
    # 1. 分析任务需求
    task_analysis = await self._analyze_task(task)
    
    # 2. 搜索相关文献
    if self.config.get("do_survey"):
        literature = await self.scholar_agent.search(task_analysis)
    
    # 3. 生成初始创意
    raw_ideas = await self._generate_raw_ideas(task_analysis, literature)
    
    # 4. 评估和筛选
    selected_ideas = await self._evaluate_ideas(raw_ideas)
    
    return selected_ideas
```

#### 层次3：实验级规划

**实现位置**：实验执行工具

```python
# experiments_utils_claude.py
async def perform_experiments(idea, exp_backend="claudecode"):
    # 1. 生成实验代码
    code = await coder_agent.write_code(idea.method_details)
    
    # 2. 执行实验
    for run in range(max_runs):
        result = await execute_experiment(code, run)
        
        # 3. 分析结果
        analysis = await analyzer_agent.analyze(result)
        
        # 4. 优化代码
        if run < max_runs - 1:
            code = await coder_agent.refine_code(code, analysis)
    
    return final_result
```

### 6.2 决策机制

#### 决策1：状态转换决策

```python
# 基于当前状态和条件决定下一个状态
if session.state == WorkflowState.GENERATING:
    if generation_success:
        next_state = WorkflowState.REFLECTING
    elif need_literature:
        next_state = WorkflowState.EXTERNAL_DATA
    else:
        next_state = WorkflowState.ERROR
```

#### 决策2：创意筛选决策

```python
# RankingAgent的决策逻辑
def rank_ideas(ideas, criteria):
    scores = {}
    for idea in ideas:
        # 多维度评分
        novelty_score = evaluate_novelty(idea)
        feasibility_score = evaluate_feasibility(idea)
        impact_score = evaluate_impact(idea)
        
        # 加权综合评分
        final_score = (
            novelty_score * criteria['novelty_weight'] +
            feasibility_score * criteria['feasibility_weight'] +
            impact_score * criteria['impact_weight']
        )
        scores[idea.id] = final_score
    
    # 选择Top-N
    top_ideas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_ideas
```

#### 决策3：实验优化决策

```python
# 基于反馈的迭代优化决策
if result.performance < baseline.performance * 1.1:
    # 性能提升不明显，继续优化
    continue_optimization = True
    optimization_strategy = "aggressive"
elif result.error_rate > 0.1:
    # 错误率较高，修复bug
    continue_optimization = True
    optimization_strategy = "debugging"
else:
    # 性能满足要求，停止优化
    continue_optimization = False
```

### 6.3 自适应规划

#### 记忆驱动的规划

```python
# 避免重复失败
if TaskMemory.is_similar_to_failed_idea(new_idea):
    if filter_failed_ideas:
        skip_idea(new_idea)
        generate_alternative()

# 基于成功经验优化
if TaskMemory.has_successful_pattern(task):
    apply_successful_pattern(new_idea)
```

#### 动态调整策略

```python
# 根据进度调整资源分配
if session.iterations_completed > max_iterations * 0.5:
    # 已过半程，专注于top创意
    top_ideas_count = min(3, original_count)
else:
    # 前期探索更多创意
    top_ideas_count = original_count

# 根据成功率调整并发度
if recent_success_rate > 0.8:
    max_concurrent_tasks += 1
elif recent_success_rate < 0.3:
    max_concurrent_tasks = max(1, max_concurrent_tasks - 1)
```

## 7. 信息驱动机制

### 7.1 驱动信息类型

#### 类型1：任务描述信息

```python
# 输入：task_info.json
{
    "task": "复现ProtTrans论文的核心发现",
    "background": "ProtTrans论文展示了...",
    "constraints": ["计算资源限制", "数据规模限制"]
}
```

**作用**：驱动整个研究方向，是所有Agent决策的基础

#### 类型2：配置信息

```python
# config/default_config.yaml
workflow:
    max_iterations: 4
    top_ideas_count: 5
    loop_rounds: 10

agents:
    generation:
        do_survey: true
        filter_failed_ideas: true
```

**作用**：控制工作流行为和Agent策略

#### 类型3：记忆信息

```python
# TaskMemory记忆内容
{
    "failed_ideas": [...],      # 失败创意历史
    "successful_patterns": [...], # 成功模式
    "experiment_results": [...]   # 实验结果
}
```

**作用**：避免重复错误，学习成功经验

#### 类型4：执行反馈信息

```python
# 实验执行反馈
{
    "performance": {"accuracy": 0.85, "f1": 0.82},
    "errors": ["RuntimeError: ..."],
    "warnings": ["Data size warning"],
    "logs": "[详细日志]"
}
```

**作用**：驱动迭代优化和错误修复

#### 类型5：外部数据信息

```python
# SurveyAgent获取的文献信息
{
    "papers": [
        {
            "title": "...",
            "abstract": "...",
            "key_findings": [...],
            "methods": [...]
        }
    ]
}
```

**作用**：提供最新研究进展，增强创意质量

### 7.2 信息流动机制

#### 流动路径1：任务信息流

```
用户输入 → task_info.json → Task对象 → WorkflowSession
   ↓
所有Agent读取session.task信息
   ↓
Agent根据task信息生成相应输出
```

#### 流动路径2：创意信息流

```
GenerationAgent生成原始创意
   ↓
session.ideas存储
   ↓
ReflectionAgent添加critiques
   ↓
EvolutionAgent基于critiques生成新创意
   ↓
RankingAgent评分并选择top创意
   ↓
MethodDevelopmentAgent基于top创意开发方法
```

#### 流动路径3：反馈信息流

```
实验执行 → 结果数据
   ↓
ExpAnalyzeAgent分析结果
   ↓
性能评估
   ↓
MethodDevelopmentAgent基于反馈优化代码
   ↓
再次实验执行
```

#### 流动路径4：记忆信息流

```
实验完成 → 存储到TaskMemory
   ↓
下一轮开始 → 从TaskMemory检索相关经验
   ↓
GenerationAgent避免生成相似失败创意
   ↓
EvolutionAgent应用成功模式
```

## 8. 执行流程详解

### 8.1 完整执行流程

```python
# 阶段1：系统初始化
1. 加载配置文件 (config/default_config.yaml)
2. 创建InternAgentInterface
3. 初始化所有Agent
4. 启动工具和MCP服务

# 阶段2：任务加载
1. 读取task_info.json
2. 创建Task对象
3. 可选：DRAgent生成background
4. 创建WorkflowSession (state=INITIAL)

# 阶段3：创意生成循环
for iteration in range(max_iterations):
    # 3.1 创意生成
    session.state = GENERATING
    raw_ideas = GenerationAgent.execute(task)
    
    # 3.2 文献调研（可选）
    if do_survey:
        session.state = EXTERNAL_DATA
        SurveyAgent.execute(task)
        session.state = GENERATING
    
    # 3.3 反思评估
    session.state = REFLECTING
    ReflectionAgent.evaluate(raw_ideas)
    
    # 3.4 创意进化
    session.state = EVOLVING
    evolved_ideas = EvolutionAgent.evolve(raw_ideas)
    
    # 3.5 排名筛选
    session.state = RANKING
    top_ideas = RankingAgent.rank(evolved_ideas)
    
    # 3.6 方法开发
    session.state = METHOD_DEVELOPMENT
    MethodDevelopmentAgent.develop(top_ideas)
    
    # 3.7 代码精炼
    session.state = REFINING
    RefinementAgent.refine(top_ideas)
    
    # 3.8 更新迭代计数
    session.iterations_completed += 1

# 阶段4：实验执行
for idea in top_ideas:
    # 4.1 设置实验环境
    setup_experiment_folder(idea)
    
    # 4.2 执行实验
    for run in range(max_runs):
        result = perform_experiments(idea, run)
        
        # 4.3 分析结果
        analysis = ExpAnalyzeAgent.analyze(result)
        
        # 4.4 迭代优化
        if run < max_runs - 1:
            refine_based_on_analysis(idea, analysis)

# 阶段5：结果汇总
1. 收集所有实验结果
2. 计算性能改进
3. 生成最终报告
4. 更新记忆系统
```

### 8.2 关键代码路径

#### 路径1：状态机驱动流程

```python
# orchestration_agent.py
async def run_session(self, session_id):
    session = await self._get_session(session_id)
    
    # 状态机主循环
    while session.state not in [COMPLETED, ERROR]:
        # 执行当前阶段
        await self._execute_current_phase(session)
        
        # 持久化会话状态
        await self.memory_manager.store_session(session)
        
        # 检查是否需要等待反馈
        if session.state == AWAITING_FEEDBACK:
            break
    
    return session
```

#### 路径2：创意生成流程

```python
# orchestration_agent.py
async def _run_generation_phase(self, session):
    generation_agent = self._get_agent("generation")
    
    # 调用GenerationAgent
    response = await generation_agent.execute({
        "task": session.task.description,
        "domain": session.task.domain,
        "background": session.task.background,
        "constraints": session.task.constraints
    })
    
    # 创建Idea对象
    ideas = self._create_ideas_from_response(response)
    
    # 添加到会话
    for idea in ideas:
        idea.iteration = session.iterations_completed + 1
        session.ideas.append(idea)
    
    # 转换到下一个状态
    await self._update_session_state(session, REFLECTING)
```

#### 路径3：实验执行流程

```python
# stage.py
def run_single_round(self, round_num=1):
    # 1. 生成创意
    ideas = self.idea_generator.generate_ideas()
    
    # 2. 选择顶级创意
    top_ideas = self.select_top_ideas(ideas)
    
    # 3. 执行实验
    results = []
    for idea in top_ideas:
        result = self.experiment_runner.run(idea)
        results.append(result)
    
    # 4. 分析结果
    best_result = self.find_best_result(results)
    
    # 5. 更新基线
    if best_result.performance > baseline.performance:
        update_baseline(best_result)
    
    return best_result
```

## 9. 系统特性总结

### 9.1 设计模式应用

#### 模式1：状态机模式
- **应用位置**：OrchestrationAgent
- **解决问题**：复杂工作流的状态转换管理
- **优势**：清晰的状态转换逻辑，易于扩展和维护

#### 模式2：工厂模式
- **应用位置**：AgentFactory, ModelFactory
- **解决问题**：Agent和模型的统一创建
- **优势**：解耦创建逻辑，支持动态配置

#### 模式3：策略模式
- **应用位置**：不同实验后端的选择
- **解决问题**：多种执行方式的统一接口
- **优势**：灵活切换执行策略

#### 模式4：记忆模式
- **应用位置**：MemoryManager, TaskMemory
- **解决问题**：跨会话学习和经验积累
- **优势**：避免重复错误，提升效率

### 9.2 技术优势

#### 优势1：高度模块化
- 每个Agent职责单一，易于维护
- 支持Agent的独立升级和替换
- 便于添加新的Agent类型

#### 优势2：强鲁棒性
- 状态机提供清晰的错误处理路径
- 持久化机制支持断点恢复
- 并发控制保证系统稳定性

#### 优势3：可扩展性
- 支持新Agent的无缝集成
- 支持新的工具和外部服务
- 支持多种实验后端

#### 优势4：智能化
- 记忆系统实现经验学习
- 自适应规划优化资源分配
- 多维度评估提升决策质量

### 9.3 关键技术创新

#### 创新1：状态机驱动的多Agent协作
- 传统的多Agent系统通常采用基于消息的通信
- InternAgent采用共享状态+状态机的协作模式
- 优势：状态透明、易于调试、支持断点恢复

#### 创新2：分层记忆系统
- ContextMemory：会话内的即时记忆
- TaskMemory：跨会话的实验记忆
- LongMemory：基于图结构的长期记忆
- 优势：多时间尺度的学习，不同层次的记忆互补

#### 创新3：自适应工作流
- 根据执行结果动态调整策略
- 基于记忆避免重复错误
- 支持用户反馈的实时集成
- 优势：提高效率，优化资源利用

## 10. 总结

InternAgent是一个设计精妙的科学发现自动化系统，其核心特点包括：

1. **状态机驱动的工作流**：清晰的状态转换逻辑，支持复杂的协作流程
2. **专业化Agent团队**：每个Agent专注特定任务，通过状态机协作完成全流程
3. **丰富的数据流设计**：Idea、Task、WorkflowSession等数据结构承载关键信息
4. **多层次规划机制**：工作流级、Agent级、实验级的分层规划
5. **记忆驱动的智能决策**：通过历史经验优化当前决策
6. **高度模块化架构**：支持灵活扩展和配置

这种架构设计使得InternAgent能够处理复杂的科学发现任务，同时保持系统的可维护性和可扩展性。

---

*基于 InternAgent-1.5 版本分析*
*分析时间：2024-06-03*
