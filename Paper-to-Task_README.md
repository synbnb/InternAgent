# Paper-to-Task 自动化系统

## 📖 概述

Paper-to-Task 是一个自动化系统，可以将科研论文PDF自动转换为InternAgent的sci_tasks任务。用户只需上传论文PDF，系统就会自动分析论文内容并生成所需的`task_info.json`和`checklist.json`文件。

### 🎯 核心功能

- **PDF解析**: 自动提取论文文本和结构
- **智能分析**: 使用LLM提取研究目标、方法、数据等信息
- **自动生成**: 生成符合sci_tasks标准的JSON文件
- **质量检查**: 多维度质量评分和验证
- **迭代优化**: 支持用户反馈驱动的改进
- **项目创建**: 自动创建完整的sci_tasks项目结构

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. 交互式处理（推荐）

```bash
python launch_paper_to_task.py --pdf your_paper.pdf
```

#### 2. 快速处理模式

```bash
python launch_paper_to_task.py --pdf your_paper.pdf --quick
```

#### 3. 批量处理

```bash
python launch_paper_to_task.py --pdf-dir papers/ --output output/
```

## 📋 系统架构

```
Paper-to-Task 系统架构

┌─────────────────────────────────────────────────────────────┐
│                    用户交互层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  命令行界面   │  │   Web界面     │  │  批量处理     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   核心处理引擎                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PDF解析模块   │→│ 信息提取模块   │→│ 内容生成模块   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                  ↓              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 质量检查      │←│ 迭代优化      │←│ 项目创建      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 配置选项

### LLM配置

系统支持多种LLM后端：

```bash
# 使用模拟后端（测试用）
python launch_paper_to_task.py --pdf paper.pdf --llm-backend mock

# 使用OpenAI
python launch_paper_to_task.py --pdf paper.pdf --llm-backend openai --model gpt-4

# 使用Anthropic
python launch_paper_to_task.py --pdf paper.pdf --llm-backend anthropic
```

### 配置文件

创建 `config/paper_to_task_config.yaml`:

```yaml
llm:
  backend: "openai"
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 4000
  cache_enabled: true

quality:
  min_score: 0.7
  enable_auto_improvement: true

project:
  sci_tasks_base: "sci_tasks/tasks"
```

## 📊 质量评分

系统会对生成的内容进行多维度评分：

- **完整性** (30%): 所需字段是否齐全
- **准确性** (25%): 字段类型和格式是否正确
- **清晰度** (25%): 描述是否清晰明确
- **可行性** (20%): 任务是否可行

### 评级标准

- **A级** (≥0.9): 优秀
- **B级** (≥0.8): 良好
- **C级** (≥0.7): 合格
- **D级** (≥0.6): 需要改进
- **F级** (<0.6): 不合格

## 🎨 用户交互流程

```
用户上传PDF
    ↓
自动分析论文
    ↓
显示生成结果
    ↓
用户选择操作
    ├─→ 查看详细内容
    ├─→ 确认并继续
    ├─→ 提供反馈并改进
    └─→ 取消
    ↓
创建sci_tasks项目
    ↓
显示下一步操作
```

## 📁 项目结构

```
paper_to_task/
├── core/                     # 核心处理模块
│   ├── pdf_parser.py         # PDF解析
│   ├── info_extractor.py     # 信息提取
│   ├── content_generator.py  # 内容生成
│   ├── quality_checker.py    # 质量检查
│   └── project_creator.py    # 项目创建
├── refinement/               # 迭代优化模块
│   ├── iterative_refiner.py  # 迭代优化
│   └── quality_scorer.py     # 质量评分
├── interaction/              # 用户交互模块
│   ├── cli_interface.py      # 命令行界面
│   ├── web_interface.py      # Web界面
│   └── feedback_handler.py   # 反馈处理
├── utils/                    # 工具模块
│   ├── llm_client.py         # LLM客户端
│   ├── validators.py         # 验证器
│   └── file_utils.py         # 文件工具
└── pipeline.py               # 主流程管道
```

## 🔍 使用示例

### 示例1：处理单个论文

```bash
python launch_paper_to_task.py --pdf sample_paper.pdf
```

系统会：
1. 解析PDF文件
2. 提取研究信息
3. 生成task_info和checklist
4. 进行质量检查
5. 等待用户确认
6. 创建sci_tasks项目

### 示例2：快速模式

```bash
python launch_paper_to_task.py --pdf sample_paper.pdf --quick --task-name "ProteinAnalysis"
```

自动完成所有步骤，无需用户交互。

### 示例3：批量处理

```bash
python launch_paper_to_task.py --pdf-dir papers/ --output results/ --batch
```

处理目录中的所有PDF文件，结果保存到指定目录。

## 🛠️ 故障排除

### PDF解析失败

- **问题**: 无法解析PDF文件
- **解决**: 安装 `pdfplumber` 或 `PyPDF2` 包

```bash
pip install pdfplumber PyPDF2
```

### LLM调用失败

- **问题**: LLM API调用失败
- **解决**: 检查API密钥和网络连接

### 质量评分不达标

- **问题**: 生成内容质量低于阈值
- **解决**: 使用反馈功能进行迭代改进

## 📚 API文档

### PaperToTaskPipeline

主要的处理管道类。

```python
from paper_to_task.pipeline import PaperToTaskPipeline

# 创建管道
pipeline = PaperToTaskPipeline(config)

# 处理PDF
result = pipeline.process_pdf('paper.pdf')

# 改进内容
refined = pipeline.refine_content(content, feedback)

# 创建项目
project = pipeline.create_project(task_name, task_info, checklist)
```

### CLIInterface

命令行交互界面。

```python
from paper_to_task.interaction.cli_interface import CLIInterface

cli = CLIInterface(pipeline)

# 交互式处理
result = cli.run_interactive('paper.pdf')

# 快速处理
result = cli.run_quick_process('paper.pdf', task_name='MyTask')

# 批量处理
result = cli.run_batch('papers/', 'output/')
```

## 🎯 最佳实践

1. **准备PDF**: 使用高质量的PDF文件，确保文本可以提取
2. **检查结果**: 仔细检查生成的task_info和checklist
3. **迭代改进**: 根据反馈进行多轮改进
4. **验证质量**: 确保质量评分达到C级以上
5. **备份数据**: 保留原始PDF和生成的文件

## 🔗 相关资源

- [InternAgent主项目](../README.md)
- [sci_tasks使用指南](../sci_tasks/README.md)
- [task_info字段说明](checklist和task_info字段完整指南.md)

## 📝 版本历史

### v1.0.0 (2024-06-10)

- ✨ 初始版本发布
- ✅ 支持PDF解析和信息提取
- ✅ 支持自动生成task_info和checklist
- ✅ 支持质量检查和迭代优化
- ✅ 支持项目创建
- ✅ 支持命令行界面

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。

---

*由InternAgent团队开发*
