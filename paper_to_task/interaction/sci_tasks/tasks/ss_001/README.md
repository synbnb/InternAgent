# ss_001

## 任务描述

复现论文《ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning》的核心发现

## 研究背景

研究背景待补充

## 研究目标

验证论文核心发现

## 数据文件

请查看 `DATA_README.md` 获取详细的数据文件说明。

## 评分标准

评分标准定义在 `target_study/checklist.json` 中。

## 快速开始

```bash
# 启动实验
python launch_discovery.py --task sci_tasks/tasks/ss_001

# 查看结果
# 结果将保存在 results/ss_001/ 目录下
```

## 项目结构

```
ss_001/
├── task_info.json              # 任务信息
├── DATA_README.md             # 数据说明
├── TASK_STATUS.md             # 任务状态
├── data/                      # 数据文件目录
└── target_study/              # 参考研究
    ├── checklist.json         # 评分标准
    ├── images/                # 参考图表
    └── paper/                 # 参考论文
```

## 注意事项

1. 确保数据文件已正确放置
2. 检查环境配置
3. 按照评分标准完成任务

---

*由 Paper-to-Task 系统自动生成*
