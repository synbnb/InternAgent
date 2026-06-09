# InternAgent 科学论文复现技术分析报告

## 项目概述

**InternAgent-1.5** 是一个统一的智能体框架，用于长周期自主科学发现。该系统支持跨物理、生物、地球科学等领域的端到端自主科学发现任务，包括算法发现和实证发现（干湿实验室实验）。

### 核心特性

- **自主科学发现**：从假设生成到验证的全流程自动化
- **科学论文复现**：能够复现已发表科学论文的关键发现
- **多智能体协作**：专业化的Agent团队协同工作
- **记忆系统**：跨会话持久化学习，避免重复失败
- **深度研究**：复杂研究问题的结构化知识流

## 系统架构

### 1. 多智能体系统（MAS）架构

```
InternAgent System
├── InternAgentInterface（主接口）
│   ├── ModelFactory（模型工厂）
│   ├── AgentFactory（Agent工厂）
│   ├── MemoryManager（记忆管理器）
│   └── OrchestrationAgent（编排Agent）
│
├── 专业Agent团队
│   ├── GenerationAgent（创意生成）
│   ├── ReflectionAgent（反思评估）
│   ├── EvolutionAgent（进化优化）
│   ├── RankingAgent（排名筛选）
│   ├── MethodDevelopmentAgent（方法开发）
│   ├── RefinementAgent（精炼改进）
│   ├── ScholarAgent（文献调研）
│   ├── SurveyAgent（深度调研）
│   ├── DRAgent（深度研究）
│   └── ExpAnalyzeAgent（实验分析）
│
├── 记忆系统
│   ├── ContextMemory（上下文记忆）
│   ├── TaskMemory（任务记忆）
│   ├── OnlineMemory（在线记忆）
│   └── LongMemory（长期记忆）
│
└── 工具系统
    ├── WebSearch（网络搜索）
    ├── LiteratureSearch（文献搜索）
    └── MCPServer（MCP服务器工具）
```

### 2. 核心工作流程

```
Discovery Pipeline
┌─────────────────────────────────────────────────────────────┐
│                    初始化阶段                                  │
│  加载配置 → 创建Agents → 初始化记忆 → 启动工具               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    创意生成阶段                               │
│  深度研究 → 文献调研 → 创意生成 → 反思评估 → 创意进化       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    方法开发阶段                               │
│  顶级创意 → 方法开发 → 代码精炼                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    实验执行阶段                               │
│  代码执行 → 结果分析 → 报告生成 → 评分                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    迭代优化阶段                               │
│  记忆更新 → 创意进化 → 下一轮发现                            │
└─────────────────────────────────────────────────────────────┘
```

## 科学论文复现详解

### 任务类型

InternAgent支持两种主要任务类型：

1. **Auto任务**（算法发现）：通过改进算法来优化性能指标
2. **Sci任务**（科学论文复现）：复现已发表论文的实验结果

### Sci任务结构

```
sci_tasks/tasks/{Domain}_{NNN}/
├── task_info.json          # 任务描述和数据清单
├── data/                   # 数据文件
│   ├── measurements.xlsx   # 实验数据
│   ├── samples.csv        # 样本数据
│   └── ...
├── related_work/           # 相关论文（背景资料）
│   ├── paper1.pdf
│   └── paper2.pdf
└── target_study/           # 目标论文和评分标准
    ├── paper.pdf          # 需要复现的论文
    ├── checklist.json     # 评分清单
    └── images/            # 参考图表
```

### 工作流程详解

#### 第一阶段：任务理解

1. **任务分析**
   - 读取`task_info.json`了解研究目标
   - 分析`checklist.json`明确评分标准
   - 构建任务上下文和约束条件

2. **背景生成**
   - DRAgent进行深度文献调研
   - ScholarAgent分析相关论文
   - 构建领域知识背景

#### 第二阶段：创意生成

1. **假设生成**
   - GenerationAgent基于背景知识生成多个研究假设
   - 每个假设包含方法论和预期结果

2. **反思评估**
   - ReflectionAgent评估假设的可行性
   - 基于文献和领域知识进行批判性分析

3. **创意进化**
   - EvolutionAgent优化和改进假设
   - 生成更具创新性的研究方向

4. **排名筛选**
   - RankingAgent对所有假设进行排名
   - 选择top-N假设进行实验验证

#### 第三阶段：实验执行

1. **方法开发**
   - MethodDevelopmentAgent将假设转化为可执行代码
   - 编写数据分析代码和实验脚本

2. **代码精炼**
   - RefinementAgent优化代码质量
   - 确保代码符合最佳实践

3. **实验运行**
   - 执行分析代码
   - 生成中间结果和可视化图表
   - 调试和修正错误

#### 第四阶段：结果分析

1. **报告生成**
   - 撰写markdown格式的研究报告
   - 包含方法、结果、分析和结论

2. **自动评分**
   - LLM评判员对比生成的报告与原始论文
   - 基于checklist进行逐项评分
   - 生成详细的评分报告

### 评分机制

评分系统采用ResearchClawBench标准，包含两种模式：

#### Mode A（客观评分）
- 针对定量结果和具体数值
- 50分表示与论文结果相当
- >50分表示优于论文结果
- <50分表示劣于论文结果

#### Mode B（主观评分）
- 针对定性分析和机理解释
- 同样以50分为基准线
- 考虑分析的深度和准确性

### Checklist项目类型

1. **Text类型**：文本描述的发现和结论
2. **Image类型**：需要生成图表对比的结果

## 使用指南

### 环境配置

#### 1. 安装依赖

```bash
conda create -n InternAgent python=3.11
conda activate InternAgent
pip install -r requirements.txt
```

#### 2. 配置API密钥

编辑`.env`文件：

```bash
# 核心LLM配置（必需）
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE_URL=https://api.deepseek.com
OPENAI_BASE_URL=https://api.deepseek.com

# Anthropic API（实验后端必需）
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic

# DeepSeek专用（DR Agent必需）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

#### 3. 初始化子模块（如果需要）

```bash
git submodule update --init
```

### 快速开始

#### 1. 测试运行（AutoDebug任务）

```bash
python launch_discovery.py \
    --config ./config/default_config.yaml \
    --task AutoDebug \
    --exp_backend claudecode
```

#### 2. 运行科学论文复现任务

```bash
# 默认任务（Astronomy_000）
bash scripts/run_sci.sh

# 指定任务
bash scripts/run_sci.sh Chemistry_001
bash scripts/run_sci.sh Life_000
```

#### 3. 自定义任务运行

```bash
python launch_discovery.py \
    --task sci_tasks/tasks/Chemistry_001 \
    --config config/default_config.yaml \
    --exp_backend claudecode
```

### 配置调优

#### 减少运行时间（用于测试）

编辑`config/default_config.yaml`：

```yaml
workflow:
  loop_rounds: 1       # 默认10轮
  top_ideas_count: 2   # 默认5个创意

experiment:
  max_runs: 1          # 每个创意的实验轮数
```

#### 提高结果质量（生产环境）

```yaml
workflow:
  loop_rounds: 10
  top_ideas_count: 5

experiment:
  max_runs: 2
  max_parallel_experiments: 4
```

## 结果分析

### 输出目录结构

```
results/{TaskName}/{timestamp}_launch/
├── prompt.json                    # 任务配置
├── session_*/                     # 每轮发现会话
│   └── */                         # 每个实验
│       ├── INSTRUCTIONS.md        # 实验指令
│       ├── launcher.sh            # 启动脚本
│       ├── code/                  # Agent生成的代码
│       ├── outputs/               # 中间输出
│       ├── report/                # 实验报告
│       │   ├── report.md         # 主要报告文件
│       │   └── images/           # 生成的图表
│       ├── run_0/                # 基线运行
│       ├── run_1/                # 第一轮改进
│       │   ├── code/            # 代码快照
│       │   ├── report/          # 报告快照
│       │   └── final_info.json  # 评分结果
│       └── run_2/                # 第二轮改进
└── ...
```

### 评分解读

`final_info.json`格式：

```json
{
  "sci_task": {
    "means": {
      "total_score": 40.6,
      "item_0_score": 48,
      "item_1_score": 35,
      ...
    }
  }
}
```

**评分参考标准**：
- 30-40分：AI正常复现水平
- 40-50分：较好的复现结果
- 50+分：达到或超越原论文水平

## 自定义任务创建

### 创建新的Sci任务

1. **创建任务目录**

```bash
mkdir -p sci_tasks/tasks/MyDomain_001/{data,related_work,target_study/images}
```

2. **准备文件**

```bash
MyDomain_001/
├── task_info.json          # 任务描述
├── data/                   # 数据文件
├── related_work/           # 相关论文
└── target_study/          # 目标论文
    ├── paper.pdf
    ├── checklist.json
    └── images/
```

3. **编写task_info.json**

```json
{
  "task": "复现XXX论文的YYY实验结果",
  "data": [
    {
      "name": "experiment_data.csv",
      "path": "data/experiment_data.csv",
      "description": "包含实验测量数据的CSV文件"
    }
  ]
}
```

4. **编写checklist.json**

```json
[
  {
    "type": "text",
    "weight": 0.3,
    "content": "报告应包含实验设计的详细描述",
    "keywords": ["实验设计", "方法论", "数据分析"]
  },
  {
    "type": "image",
    "weight": 0.2,
    "content": "生成与论文图1相似的散点图",
    "path": "images/figure1.png"
  }
]
```

5. **运行任务**

```bash
bash scripts/run_sci.sh MyDomain_001
```

## 高级功能

### 1. 记忆系统

InternAgent包含多层记忆系统：

- **ContextMemory**：会话内的上下文管理
- **TaskMemory**：跨会话的实验结果记忆
- **OnlineMemory**：自动保存实验结果
- **LongMemory**：基于IdeaGraph的长期记忆

配置记忆系统：

```yaml
memory:
  task_memory:
    enabled: true
    memory_dir: "./config/mem_store"
    top_k: 5
  
  long_memory:
    enabled: true
    idea_graph:
      similarity_threshold: 0.7
```

### 2. 深度研究（DR）

DRAgent提供结构化的深度研究能力：

- 文献搜索和综述
- 多源信息整合
- 自动生成背景报告

### 3. 并行执行

支持多个实验并行运行：

```yaml
experiment:
  max_parallel_experiments: 4
  gpu_per_experiment: 1.0
```

## 最佳实践

### 1. 任务设计

- 明确定义研究目标和成功标准
- 提供高质量的数据文件
- 设计合理的checklist项目

### 2. 配置优化

- 根据任务复杂度调整迭代次数
- 合理设置创意生成数量
- 平衡运行时间和结果质量

### 3. 结果分析

- 仔细检查生成的代码
- 验证报告中的数值结果
- 对比生成的图表与原论文

### 4. 迭代改进

- 利用记忆系统避免重复错误
- 基于评分结果优化checklist
- 逐步改进任务描述和背景信息

## 常见问题解决

### Q1: API调用超时

**解决方案**：
- 检查`.env`中的API端点配置
- 确认网络连接稳定
- 考虑添加代理配置

### Q2: 评分结果很低

**原因分析**：
- 50分是论文水平基准线
- 30-40分是正常AI复现水平
- 检查checklist项目是否合理

### Q3: 运行时间过长

**优化方案**：
- 减少`loop_rounds`和`top_ideas_count`
- 降低`max_runs`数值
- 使用更快的模型

### Q4: 内存不足

**解决方案**：
- 减少`max_parallel_experiments`
- 禁用LongMemory功能
- 清理旧的会话数据

## 技术要点总结

### 1. Agent协作模式

InternAgent采用专业分工的Agent协作模式，每个Agent负责特定的研究环节：

- **GenerationAgent**：负责创意生成，基于背景知识产生新的研究假设
- **ReflectionAgent**：提供批判性评估，识别潜在问题
- **EvolutionAgent**：迭代优化创意，提高创新性
- **MethodDevelopmentAgent**：将假设转化为可执行代码
- **ScholarAgent**：提供文献调研支持
- **DRAgent**：进行深度研究和背景生成

### 2. 工作流编排

OrchestrationAgent负责整个工作流的编排：

- 状态机管理：INITIAL → GENERATING → REFLECTING → EVOLVING → RANKING → METHOD_DEVELOPMENT → REFINING → COMPLETED
- 并发控制：管理多个任务的并发执行
- 会话管理：跟踪每个研究会话的状态

### 3. 记忆管理

MemoryManager实现多层次的记忆系统：

- **短期记忆**：会话内的信息管理
- **中期记忆**：跨会话的实验结果记忆
- **长期记忆**：基于图结构的长期知识管理

### 4. 工具集成

系统集成了多种研究工具：

- 文献搜索工具
- 网络搜索工具
- MCP服务器工具
- 本地代码执行环境

## 结论

InternAgent提供了一个完整的科学论文复现解决方案，具有以下优势：

1. **全流程自动化**：从假设生成到报告编写的全自动化
2. **多智能体协作**：专业化的Agent团队协同工作
3. **智能记忆系统**：跨会话学习和改进
4. **标准化评分**：与ResearchClawBench对齐的评分系统
5. **灵活配置**：支持多种任务类型和运行模式

通过合理配置和使用，InternAgent可以有效地辅助科研人员进行论文复现和科学发现工作。

## 附录

### A. 支持的任务类型

- **AutoChem**：化学算法优化
- **AutoCls2D/3D**：2D/3D分类任务
- **AutoDebug**：代码调试任务
- **AutoDrug**：药物发现
- **AutoEAP**：实验分析预测
- **AutoMem**：记忆优化
- **AutoPower**：能源系统优化
- **AutoTTS**：文本转语音
- **Sci任务**：科学论文复现（多领域）

### B. 支持的实验后端

- **claudecode**：基于Claude的代码执行
- **openhands**：OpenHands执行环境
- **iflow**：iFlow工作流
- **aider**：Aider代码助手

### C. 配置文件说明

- `config/default_config.yaml`：主配置文件
- `config/debug_config.yaml`：调试配置
- `internagent/mas/agents/dr_agents/config_simple.yaml`：DR Agent简化配置
- `internagent/mas/agents/dr_agents/config_complex.yaml`：DR Agent复杂配置

### D. 日志和输出

- **日志目录**：`logs/`
- **结果目录**：`results/`
- **临时文件**：`tmp/`
- **记忆存储**：`config/mem_store/`

---

*本报告基于InternAgent-1.5版本分析生成*
*最后更新：2026-05-29*
