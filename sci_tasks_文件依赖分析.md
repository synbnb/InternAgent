# sci_tasks 文件依赖分析

## 必需文件 vs 可选文件

### ✅ 绝对必需的文件（2个）

#### 1. **task_info.json**
```
路径：sci_tasks/tasks/{TaskName}/task_info.json
```

**作用**：
- 告诉 Agent 要复现什么
- 提供数据文件清单
- 定义研究目标

**必需原因**：
```python
# launch_discovery.py: detect_task_type()
if osp.exists(osp.join(task_dir, "task_info.json")):
    return "sci"  # 没有这个文件就无法识别为sci任务
```

**最小格式**：
```json
{
  "task": "复现XXX论文",
  "data": []
}
```

#### 2. **target_study/paper.pdf**
```
路径：sci_tasks/tasks/{TaskName}/target_study/paper.pdf
```

**作用**：
- Agent 需要阅读的原始论文
- 用于理解实验方法
- 用于对比结果

**必需原因**：
- 虽然代码不会强制检查，但没有论文 Agent 无法理解要复现什么
- Agent 需要读取论文来设计实验

---

### 🔶 强烈推荐的文件（1个）

#### **target_study/checklist.json**
```
路径：sci_tasks/tasks/{TaskName}/target_study/checklist.json
```

**作用**：
- 定义评分标准
- 指导 Agent 生成报告

**是否必需**：
```python
# launch_discovery.py: normalize_sci_task()
checklist_path = osp.join(task_dir, "target_study", "checklist.json")
checklist = []
if osp.exists(checklist_path):  # ⚠️ 可选：如果不存在，使用空列表
    with open(checklist_path, 'r') as f:
        checklist = json.load(f)
```

**如果没有 checklist.json 会怎样**：
- ❌ 无法自动评分（`sci_eval.py` 会报错）
- ✅ Agent 仍然会生成代码和报告
- ⚠️ 需要手动评估结果

**最小格式**：
```json
[
  {
    "type": "text",
    "weight": 1.0,
    "content": "报告应描述实验方法"
  }
]
```

---

### 📁 data/ 目录中的文件

#### **data/ 中的文件是否必需？**

**答案：取决于 task_info.json 的配置**

```python
# normalize_sci_task() 中的逻辑
data_items = task_info.get('data', [])  # 从 task_info.json 读取
if data_items:
    data_manifest = "\n".join([f"- {d['name']}: {d['description']}" ...])
else:
    data_manifest = "No data files specified."
```

**两种情况**：

**情况1：有数据文件**
```json
// task_info.json
{
  "task": "复现ProtTrans论文",
  "data": [
    {
      "name": "protein_data.csv",
      "path": "data/protein_data.csv",
      "description": "蛋白质序列数据"
    }
  ]
}
```
→ **必需**：`data/protein_data.csv` 必须存在

**情况2：无数据文件**
```json
// task_info.json
{
  "task": "复现理论分析",
  "data": []  // 空数组
}
```
→ **不需要**：data/ 目录可以为空

---

### 📚 可选目录/文件

#### **related_work/ 目录**
```
路径：sci_tasks/tasks/{TaskName}/related_work/
```

**作用**：
- 提供背景论文
- Agent 可以参考相关方法

**是否必需**：
```python
# experiments_utils_claude.py: SCI_SYMLINK_DIRS
SCI_SYMLINK_DIRS = {'data', 'related_work', 'target_study'}

# 如果 related_work/ 不存在，软链接会失败，但不影响主流程
```

**结论**：❌ 完全可选

#### **target_study/images/ 目录**
```
路径：sci_tasks/tasks/{TaskName}/target_study/images/
```

**作用**：
- 存储论文中的参考图表
- 用于 image 类型的 checklist 项

**是否必需**：
```python
# sci_eval.py
if item.get('type') == 'image':
    target_rel = item.get('path', '')
    target_path = safe_resolve(target_base, target_rel)
    # 如果 path 不存在或找不到文件，该项评分会失败
```

**结论**：
- ✅ 如果 checklist 中有 `type: "image"` 的项目 → 必需
- ❌ 如果 checklist 都是 `type: "text"` → 不需要

---

## 📋 文件依赖总结表

| 文件/目录 | 必需程度 | 作用 | 如果缺失会怎样 |
|-----------|---------|------|---------------|
| **task_info.json** | ✅ 必需 | 任务描述 | 无法识别为 sci 任务 |
| **target_study/paper.pdf** | ✅ 必需 | 目标论文 | Agent 无法理解要复现什么 |
| **checklist.json** | 🔶 推荐 | 评分标准 | 无法自动评分 |
| **data/** 文件 | 🔶 条件必需 | 实验数据 | 取决于 task_info.json 配置 |
| **related_work/** | ❌ 可选 | 背景论文 | 无影响 |
| **target_study/images/** | ❌ 条件可选 | 参考图表 | 仅 image 类型 checklist 需要 |

---

## 🎯 最小可行配置

### 方案A：有数据的论文复现

**目录结构**：
```
sci_tasks/tasks/MyTask/
├── task_info.json              # ✅ 必需
├── data/                       # 🔶 条件必需
│   └── experiment_data.csv
└── target_study/               # ✅ 必需目录
    ├── paper.pdf              # ✅ 必需
    └── checklist.json          # 🔶 推荐
```

**文件数量**：
- 必需：2 个文件（task_info.json + paper.pdf）
- 推荐：3 个文件（+ checklist.json）
- 条件：N 个文件（data/ 中的文件，取决于 task_info.json）

### 方案B：无数据的理论分析

**目录结构**：
```
sci_tasks/tasks/MyTask/
├── task_info.json              # ✅ 必需
└── target_study/               # ✅ 必需目录
    ├── paper.pdf              # ✅ 必需
    └── checklist.json          # 🔶 推荐
```

**task_info.json 配置**：
```json
{
  "task": "分析论文中的理论框架",
  "data": []  // 空数组，表示不需要外部数据
}
```

**文件数量**：
- 必需：2 个文件
- 推荐：3 个文件

---

## ❌ 常见错误

### 错误1：缺少 task_info.json
```
❌ 错误：任务被识别为 auto 任务，无法使用 sci_task 流程
```

### 错误2：缺少 checklist.json
```
⚠️ 警告：无法自动评分，需要手动评估
```

### 错误3：task_info.json 中声明了数据但文件不存在
```
❌ 错误：Agent 尝试读取数据时失败
```

### 错误4：paper.pdf 不存在
```
⚠️ 问题：Agent 无法理解论文内容，生成的代码可能不相关
```

---

## 💡 实用建议

### 建议1：从简单开始

**第一步**：只创建必需文件
```
MyTask/
├── task_info.json
└── target_study/
    └── paper.pdf
```

**第二步**：测试运行
```bash
python launch_discovery.py \
    --task sci_tasks/tasks/MyTask \
    --exp_backend claudecode
```

**第三步**：如果需要评分，添加 checklist.json

### 建议2：使用现有任务作为模板

```bash
# 复制一个简单任务
cp -r sci_tasks/tasks/ProteinBio_001 sci_tasks/tasks/MyTask

# 修改文件
vim sci_tasks/tasks/MyTask/task_info.json
vim sci_tasks/tasks/MyTask/target_study/checklist.json
```

### 建议3：验证文件完整性

```bash
# 检查必需文件
for task in sci_tasks/tasks/*/; do
    echo "检查 $task"
    [ -f "$task/task_info.json" ] && echo "  ✅ task_info.json" || echo "  ❌ 缺少 task_info.json"
    [ -f "$task/target_study/paper.pdf" ] && echo "  ✅ paper.pdf" || echo "  ❌ 缺少 paper.pdf"
    [ -f "$task/target_study/checklist.json" ] && echo "  ✅ checklist.json" || echo "  ⚠️ 缺少 checklist.json (无法评分)"
done
```

---

## 🎯 回答您的问题

### **"只给一篇论文不行吗？"**

**答案**：不行，**至少需要 2 个文件**：

1. ✅ **paper.pdf**（论文）
2. ✅ **task_info.json**（告诉 Agent 要复现什么）

### **"必须要给 task_info.json 这些文件吗？"**

**答案**：是的，**task_info.json 是必需的**

原因：
- InternAgent 通过检查 `task_info.json` 来识别这是 sci_task
- Agent 需要知道要复现什么、有哪些数据
- 没有这个文件就无法启动 sci_task 流程

### **哪些文件是必须的？**

**绝对必需（2个）**：
- ✅ `task_info.json`
- ✅ `target_study/paper.pdf`

**强烈推荐（1个）**：
- 🔶 `target_study/checklist.json`（如果需要自动评分）

**其他都是可选的**：
- `data/` 中的文件（取决于 task_info.json 配置）
- `related_work/` 目录
- `target_study/images/` 目录

---

## 📝 快速创建模板

```bash
# 创建新任务目录
mkdir -p sci_tasks/tasks/MyTask/{data,target_study}

# 创建必需的 task_info.json
cat > sci_tasks/tasks/MyTask/task_info.json << 'EOF'
{
  "task": "复现XXX论文的核心发现",
  "data": []
}
EOF

# 复制论文
cp /path/to/paper.pdf sci_tasks/tasks/MyTask/target_study/paper.pdf

# 创建可选的 checklist.json
cat > sci_tasks/tasks/MyTask/target_study/checklist.json << 'EOF'
[
  {
    "type": "text",
    "weight": 1.0,
    "content": "报告应描述实验方法"
  }
]
EOF
```

---

*sci_tasks 文件依赖分析*
*分析时间：2024-06-03*
