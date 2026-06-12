# Paper-to-Task CLI 使用指南

## 🚀 快速开始

### 1. 交互式模式（推荐）

```bash
# 进入项目目录
cd /home/devuser/workspace/reproduction_agent/InternAgent

# 启动交互式CLI
python -m paper_to_task.interaction.cli_interface交互式处理您的论文PDF
```

### 2. 快速处理模式

```bash
python -c "
from paper_to_task import PaperToTaskPipeline
from paper_to_task.interaction.cli_interface import CLIInterface

pipeline = PaperToTaskPipeline()
cli = CLIInterface(pipeline)

# 快速处理您的PDF文件
result = cli.run_quick_process('您的论文.pdf', task_name='MyTask')
"
```

---

## 📋 交互式功能菜单

启动交互式模式后，您将看到：

```
============================================================
📚 Paper-to-Task 自动化系统
   将论文PDF转换为sci_tasks任务
============================================================

📊 生成结果:
------------------------------------------------------------
质量评分: 0.73/1.00 (评级: C)
状态: ✅ 通过
任务描述: 复现论文《ProtTrans》的核心发现...

============================================================
请选择操作:
  1. 查看详细内容
  2. 确认并继续
  3. 提供反馈并改进  ← 这是您要体验的核心功能！
  4. 取消
============================================================

请输入选项 (1-4):
```

---

## 🎯 核心体验流程

### 步骤1：上传PDF
```bash
# 准备您的测试PDF文件
cp /path/to/your/test.pdf .
```

### 步骤2：系统自动处理
```bash
python -c "
from paper_to_task import PaperToTaskPipeline
from paper_to_task.interaction.cli_interface import CLIInterface

pipeline = PaperToTaskPipeline()
cli = CLIInterface(pipeline)

# 处理您的PDF
result = cli.run_interactive('test.pdf')
"
```

### 步骤3：查看生成结果
系统会显示：
- ✅ 质量评分（0.73/1.00）
- ✅ 任务描述
- ✅ 数据文件列表
- ✅ 评分项详情
- ✅ 维度评分（完整性、准确性、清晰度、可行性）

### 步骤4：提供反馈并改进（重点！）
```
请选择操作:
  3. 提供反馈并改进

请提供改进建议: checklist的评分项需要更具体，要包含具体的模型名称和准确率指标
```

### 步骤5：查看改进效果
```
正在根据反馈改进内容...
✅ 改进完成
新评分: 0.85/1.00  ← 质量提升！
```

---

## 🔧 高级功能

### 批量处理多个PDF
```python
cli = CLIInterface(pipeline)
result = cli.run_batch(
    pdf_dir='/path/to/pdfs',
    output_dir='/path/to/output',
    pattern='*.pdf'
)
```

### 查看系统状态
```python
cli.display_status()
```

---

## 📊 体验要点

1. **自动生成质量**：系统会自动从PDF提取研究信息并生成任务
2. **质量检查**：四维度评分系统自动评估生成内容的质量
3. **迭代改进**：您可以通过自然语言反馈要求系统改进内容
4. **项目创建**：满意后可以自动创建完整的sci_tasks项目结构

---

## ⚠️ 注意事项

1. **LLM配置**：默认使用mock模式，如需真实LLM请配置API密钥
2. **PDF质量**：建议使用高质量的PDF文件以获得更好的解析效果
3. **文件权限**：确保有PDF文件的读取权限

---

## 🎉 开始体验

现在就可以尝试：
```bash
python -c "
from paper_to_task import PaperToTaskPipeline
from paper_to_task.interaction.cli_interface import CLIInterface

pipeline = PaperToTaskPipeline()
cli = CLIInterface(pipeline)

# 替换为您的PDF文件路径
result = cli.run_interactive('test.pdf')
"
```

**准备好您的测试PDF，开始体验吧！** 🚀
