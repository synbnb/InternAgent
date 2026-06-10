# task_info.json 和 checklist.json 字段完整指南

## 🎯 概述

InternAgent的sci_tasks需要两个核心JSON文件：
- `task_info.json`：定义复现任务的详细信息
- `target_study/checklist.json`：定义评分标准和验证指标

---

## 📋 task_info.json 字段说明

### 文件位置
```
tasks/{domain}_XXX/task_info.json
```

### 必需字段

#### 1. `task`（必需）
**类型**：字符串  
**描述**：任务的核心描述，说明要复现什么

**示例**：
```json
{
  "task": "复现ProtTrans论文的核心发现：使用蛋白质语言模型进行二级结构预测"
}
```

**用途**：
- 被`normalize_sci_task()`函数读取
- 插入到生成的`prompt.json`中的`task_description`字段
- 用于构建完整的研究任务描述

**代码位置**：`launch_discovery.py:normalize_sci_task()`
```python
task_description = (
    f"Reproduce the findings from a scientific paper in the {domain} domain.\n\n"
    f"## Research Task\n{task_info.get('task', '')}\n\n"
    ...
)
```

#### 2. `data`（强烈推荐）
**类型**：对象数组  
**描述**：实验数据文件列表

**必需子字段**：
- `name`（必需）：文件名
- `description`（必需）：文件内容描述
- `path`（可选）：文件路径

**示例**：
```json
{
  "data": [
    {
      "name": "protein_sequences_sample.csv",
      "path": "data/protein_sequences_sample.csv",
      "description": "包含1000条蛋白质序列样本及其二级结构标签的CSV文件"
    },
    {
      "name": "pretrained_embeddings.json",
      "path": "data/pretrained_embeddings.json",
      "description": "预训练蛋白质语言模型生成的嵌入向量样本"
    }
  ]
}
```

**用途**：
- 生成数据清单
- 插入到`prompt.json`的`task_description`中
- 告诉LLM可用的数据文件

**代码位置**：`launch_discovery.py:normalize_sci_task()`
```python
data_items = task_info.get('data', [])
data_lines = [f"- {d['name']}: {d.get('description', '')}" for d in data_items]
data_manifest = "\n".join(data_lines)
```

### 可选字段

#### 3. `background`（可选）
**类型**：字符串  
**描述**：研究背景信息

**示例**：
```json
{
  "background": "ProtTrans论文展示了使用自然语言处理中的预训练语言模型来理解蛋白质序列的有效性..."
}
```

**用途**：
- 提供研究背景
- 帮助GenerationAgent理解上下文

#### 4. `research_goal`（可选）
**类型**：字符串  
**描述**：研究目标

**示例**：
```json
{
  "research_goal": "验证简化版ProtTrans方法：使用预训练蛋白质语言模型的嵌入向量进行蛋白质二级结构预测"
}
```

#### 5. `hypothesis`（可选）
**类型**：字符串  
**描述**：研究假设

**示例**：
```json
{
  "hypothesis": "预训练蛋白质语言模型的嵌入向量包含了足够的信息来进行蛋白质二级结构预测"
}
```

#### 6. `experimental_design`（可选）
**类型**：对象  
**描述**：实验设计阶段划分

**示例**：
```json
{
  "experimental_design": {
    "phase_1": {
      "name": "基线模型训练",
      "description": "使用传统蛋白质序列特征训练简单的神经网络分类器",
      "methods": ["特征工程", "神经网络"]
    },
    "phase_2": {
      "name": "语言模型嵌入方法",
      "description": "使用预训练蛋白质语言模型的嵌入向量作为特征",
      "methods": ["嵌入提取", "迁移学习"]
    }
  }
}
```

#### 7. `expected_outcomes`（可选）
**类型**：字符串数组  
**描述**：预期结果

**示例**：
```json
{
  "expected_outcomes": [
    "成功实现基于语言模型嵌入的二级结构预测流程",
    "量化比较语言模型方法与传统方法的性能差异",
    "生成性能报告和可视化图表"
  ]
}
```

#### 8. `constraints`（可选）
**类型**：字符串数组  
**描述**：实验约束条件

**示例**：
```json
{
  "constraints": [
    "由于计算资源限制，使用简化数据集（1000个蛋白质样本）",
    "使用现成的预训练模型嵌入，不重新训练语言模型"
  ]
}
```

#### 9. `success_criteria`（可选）
**类型**：字符串数组  
**描述**：成功标准

**示例**：
```json
{
  "success_criteria": [
    "代码能正常运行并生成预测结果",
    "报告包含两种方法的性能比较",
    "生成至少一张可视化图表"
  ]
}
```

---

## 📊 checklist.json 字段说明

### 文件位置
```
tasks/{domain}_XXX/target_study/checklist.json
```

### 数据结构
**类型**：对象数组  
**描述**：评分标准和验证指标的列表

### 必需字段

#### 1. `content`（必需）
**类型**：字符串  
**描述**：评分标准的具体内容

**示例**：
```json
{
  "content": "报告应详细描述实验设计，包括基线方法和语言模型方法的具体实现"
}
```

**用途**：
- LLM-as-judge的评分依据
- 插入到`prompt.json`的`constraints`字段

**代码位置**：`launch_discovery.py:normalize_sci_task()`
```python
constraints = []
for i, item in enumerate(checklist):
    w = item.get('weight', 0)
    t = item.get('type', 'text')
    preview = item.get('content', '')[:200]
    constraints.append(f"Item {i} (type={t}, weight={w:.2f}): {preview}")
```

#### 2. `weight`（强烈推荐）
**类型**：数字  
**描述**：评分权重（0-1之间）

**示例**：
```json
{
  "weight": 0.2
}
```

**用途**：
- 计算加权总分
- 影响最终评分的重要性

**代码位置**：`internagent/sci_eval.py:score_run()`
```python
w = float(item.get('weight', 1.0))
weighted_sum += sr['score'] * w
total_weight += w
```

#### 3. `type`（强烈推荐）
**类型**：字符串  
**可选值**：`"text"` 或 `"image"`  
**描述**：评分项目类型

**示例**：
```json
{
  "type": "text"
}
```

**用途**：
- 区分文本评分和图像评分
- `image`类型需要额外提供`path`字段

**代码位置**：`internagent/sci_eval.py:score_run()`
```python
if item_data.get("type", "text") == "image":
    target_rel = item_data.get("path", "")
    target_path = safe_resolve(target_base, target_rel)
```

### 可选字段

#### 4. `id`（可选）
**类型**：字符串  
**描述**：唯一标识符

**示例**：
```json
{
  "id": "item_0"
}
```

#### 5. `path`（type=image时必需）
**类型**：字符串  
**描述**：目标图像文件路径（相对于`target_study/`）

**示例**：
```json
{
  "type": "image",
  "path": "images/performance_comparison.png"
}
```

**用途**：
- 指定要对比的目标图像
- 用于图像对比评分

**注意事项**：
- 路径相对于`target_study/`目录
- 通常目标图像是论文中的原始图表
- 生成图像应该在`workspace/report/images/`中

#### 6. `keywords`（可选）
**类型**：字符串数组  
**描述**：关键词列表

**示例**：
```json
{
  "keywords": ["实验设计", "基线模型", "语言模型", "方法实现"]
}
```

**用途**：
- 帮助LLM理解评分要点
- 提供评分的关键术语

#### 7. `evaluation_criteria`（可选）
**类型**：字符串  
**描述**：评分标准说明

**示例**：
```json
{
  "evaluation_criteria": "是否清楚说明两种方法的架构、特征类型和训练过程"
}
```

**用途**：
- 为LLM-as-judge提供明确评分标准
- 指导评分的具体要求

---

## 📋 完整示例

### task_info.json 完整示例

```json
{
  "task": "复现ProtTrans论文的核心发现：使用蛋白质语言模型进行二级结构预测",
  
  "background": "ProtTrans论文（Elnaggar等，2022）展示了使用自然语言处理中的预训练语言模型来理解蛋白质序列的有效性。",
  
  "research_goal": "验证简化版ProtTrans方法：使用预训练蛋白质语言模型的嵌入向量进行蛋白质二级结构预测",
  
  "hypothesis": "预训练蛋白质语言模型的嵌入向量包含了足够的信息来进行蛋白质二级结构预测",
  
  "data": [
    {
      "name": "protein_sequences_sample.csv",
      "path": "data/protein_sequences_sample.csv",
      "description": "包含1000条蛋白质序列样本及其二级结构标签的CSV文件"
    },
    {
      "name": "pretrained_embeddings.json",
      "path": "data/pretrained_embeddings.json",
      "description": "预训练蛋白质语言模型生成的嵌入向量样本"
    }
  ],
  
  "experimental_design": {
    "phase_1": {
      "name": "基线模型训练",
      "description": "使用传统蛋白质序列特征训练简单的神经网络分类器"
    },
    "phase_2": {
      "name": "语言模型嵌入方法",
      "description": "使用预训练蛋白质语言模型的嵌入向量作为特征"
    }
  },
  
  "expected_outcomes": [
    "成功实现基于语言模型嵌入的二级结构预测流程",
    "量化比较语言模型方法与传统方法的性能差异"
  ],
  
  "constraints": [
    "使用简化数据集（1000个蛋白质样本）",
    "使用现成的预训练模型嵌入"
  ],
  
  "success_criteria": [
    "代码能正常运行并生成预测结果",
    "报告包含两种方法的性能比较"
  ]
}
```

### checklist.json 完整示例

```json
[
  {
    "id": "item_0",
    "type": "text",
    "weight": 0.2,
    "content": "报告应详细描述实验设计，包括基线方法和语言模型方法的具体实现",
    "keywords": ["实验设计", "基线模型", "语言模型"],
    "evaluation_criteria": "是否清楚说明两种方法的架构、特征类型和训练过程"
  },
  {
    "id": "item_1",
    "type": "text",
    "weight": 0.15,
    "content": "报告应包含数据预处理步骤，包括特征提取、数据分割和标准化方法",
    "keywords": ["数据预处理", "特征提取"],
    "evaluation_criteria": "是否描述了如何处理蛋白质序列数据和嵌入向量"
  },
  {
    "id": "item_2",
    "type": "text",
    "weight": 0.2,
    "content": "报告应展示量化的性能比较结果，包括准确率、精确率、召回率等指标",
    "keywords": ["性能比较", "准确率"],
    "evaluation_criteria": "是否提供具体的数值比较和统计分析"
  },
  {
    "id": "item_3",
    "type": "image",
    "weight": 0.15,
    "content": "生成性能对比图表，直观展示两种方法在不同评估指标上的差异",
    "path": "images/performance_comparison.png",
    "keywords": ["性能对比图", "可视化"],
    "evaluation_criteria": "图表是否清晰展示性能差异，包含适当的标签和图例"
  },
  {
    "id": "item_4",
    "type": "image",
    "weight": 0.1,
    "content": "生成混淆矩阵可视化图表，展示H/E/C三个二级结构类别的预测情况",
    "path": "images/confusion_matrix.png",
    "keywords": ["混淆矩阵图", "热图"],
    "evaluation_criteria": "混淆矩阵图是否清晰，准确显示各类别的预测结果"
  }
]
```

---

## 🔍 字段验证

### task_info.json 验证

#### 必需字段检查
```python
def validate_task_info(task_info):
    """验证task_info.json的必需字段"""
    
    # 必需字段
    if 'task' not in task_info:
        raise ValueError("Missing required field: 'task'")
    
    # 推荐字段
    if 'data' not in task_info:
        print("Warning: Missing 'data' field")
    
    return True
```

### checklist.json 验证

#### 必需字段检查
```python
def validate_checklist(checklist):
    """验证checklist.json的必需字段"""
    
    if not isinstance(checklist, list):
        raise ValueError("checklist must be an array")
    
    for i, item in enumerate(checklist):
        # 必需字段
        if 'content' not in item:
            raise ValueError(f"Item {i}: Missing required field: 'content'")
        
        # type=image时必需path
        if item.get('type') == 'image' and 'path' not in item:
            raise ValueError(f"Item {i}: 'image' type requires 'path' field")
        
        # 权重检查
        weight = item.get('weight', 0)
        if not isinstance(weight, (int, float)) or weight < 0:
            print(f"Warning: Item {i} has invalid weight: {weight}")
    
    return True
```

---

## 📊 字段使用流程

### task_info.json 使用流程

```
1. normalize_sci_task()读取task_info.json
    ↓
2. 提取字段
    ├─→ task → 构建task_description
    ├─→ data → 构建data_manifest
    └─→ 其他字段 → 可用于background
    ↓
3. 生成prompt.json
    ├─→ task_description包含task和data信息
    └─→ background包含补充信息
    ↓
4. GenerationAgent使用prompt生成创意
```

### checklist.json 使用流程

```
1. normalize_sci_task()读取checklist.json
    ↓
2. 构建constraints字段
    └─→ 每个item的content+weight+type
    ↓
3. 插入到prompt.json的constraints
    ↓
4. 实验完成后，sci_eval.py评分
    ├─→ 读取report.md
    ├─→ 读取checklist.json
    ├─→ LLM-as-judge逐项评分
    └─→ 计算加权总分
```

---

## 🎯 最小化模板

### task_info.json 最小模板

```json
{
  "task": "复现XXX论文的核心发现",
  
  "data": [
    {
      "name": "data.csv",
      "path": "data/data.csv",
      "description": "实验数据文件"
    }
  ]
}
```

### checklist.json 最小模板

```json
[
  {
    "content": "报告应描述实验方法",
    "weight": 0.5,
    "type": "text"
  },
  {
    "content": "报告应包含性能指标",
    "weight": 0.5,
    "type": "text"
  }
]
```

---

## 💡 最佳实践

### task_info.json 建议

1. **task字段**：清晰简洁，说明要复现什么
2. **data字段**：提供详细的文件描述，帮助LLM理解数据
3. **background字段**：提供足够的研究背景
4. **constraints字段**：明确实验限制和资源约束

### checklist.json 建议

1. **权重分配**：确保权重总和为1.0
2. **type选择**：合理使用text和image类型
3. **content描述**：明确、具体、可验证
4. **数量控制**：5-10个项目为宜

---

## 🔗 相关文件

### 代码文件
- `launch_discovery.py:normalize_sci_task()` - 读取并生成prompt.json
- `internagent/sci_eval.py:score_run()` - 基于checklist评分

### 配置文件
- `config/default_config.yaml` - 全局配置
- `tasks/{domain}_XXX/task_info.json` - 任务描述
- `tasks/{domain}_XXX/target_study/checklist.json` - 评分标准

---

*文档完成时间：2024-06-09*  
*适用版本：InternAgent 1.5*  
*关键要点：task字段和data字段必需，content字段必需*
