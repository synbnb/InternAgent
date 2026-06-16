# 快速开始指南

## 1. 数据准备

```bash
# 创建数据目录（如果不存在）
mkdir -p sci_tasks/tasks/ss_001/data

# 下载数据文件并放置到data目录
# 根据DATA_README.md的说明准备数据
```

## 2. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
# 根据需要修改配置文件
```

## 3. 运行实验

```bash
# 基本运行（使用完整路径）
python launch_discovery.py --task sci_tasks/tasks/ss_001

# 指定输出目录
python launch_discovery.py --task sci_tasks/tasks/ss_001 --output results/

# 查看详细日志
python launch_discovery.py --task sci_tasks/tasks/ss_001 --verbose
```

## 4. 查看结果

```bash
# 结果目录
ls results/ss_001/

# 查看报告
cat results/ss_001/*_launch/session_*/report.md
```

## 5. 评分和评估

系统将自动根据 `checklist.json` 中的标准对结果进行评分。

## 常见问题

**Q: 数据文件在哪里下载？**
A: 请查看论文的补充材料或官方网站。

**Q: 如何修改任务配置？**
A: 编辑 `task_info.json` 文件。

**Q: 如何查看详细日志？**
A: 使用 `--verbose` 参数或查看 `results/` 目录下的日志文件。

---

如有其他问题，请参考主项目README。
