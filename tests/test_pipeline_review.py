"""
流水线人工审查功能单元测试
测试结果评估和对比看板的后端数据流

使用方法:
    cd /home/devuser/workspace/reproduction_agent/InternAgent
    python -m pytest tests/test_pipeline_review.py -v

或直接运行:
    python tests/test_pipeline_review.py
"""

import os
import sys
import json
import time
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResultReview:
    """测试结果评估功能"""

    @classmethod
    def setup_class(cls):
        """创建临时测试目录"""
        cls.test_dir = tempfile.mkdtemp(prefix='pipeline_review_test_')
        cls.review_dir = os.path.join(cls.test_dir, '.human_review')
        os.makedirs(cls.review_dir, exist_ok=True)
        print(f"📁 测试目录: {cls.test_dir}")

    @classmethod
    def teardown_class(cls):
        """清理测试目录"""
        shutil.rmtree(cls.test_dir)
        print("🧹 测试目录已清理")

    def setup_method(self):
        """每个测试前清空状态文件"""
        for f in os.listdir(self.review_dir):
            os.remove(os.path.join(self.review_dir, f))

    def _create_result_review(self, results_data):
        """创建模拟的 result_review.json"""
        result_file = os.path.join(self.review_dir, 'result_review.json')
        with open(result_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        return result_file

    def _create_feedback(self, feedback_data):
        """创建模拟的 result_feedback.json"""
        fb_file = os.path.join(self.review_dir, 'result_feedback.json')
        with open(fb_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        return fb_file

    def test_1_write_result_review(self):
        """测试: 写入待审查的实验结果"""
        from paper_to_task.interaction import pipeline_hooks

        result_file = pipeline_hooks.write_result_for_review(
            output_dir=self.test_dir,
            idea_name="ProtT5 + SVM 二级结构预测",
            run_num=0,
            scores={"completeness": 80, "accuracy": 65, "clarity": 70, "feasibility": 60},
            report_text="# ProtTrans 复现报告\n## 摘要\n本实验成功复现了ProtT5模型...",
            report_images=["report/images/fig1.png", "report/images/fig2.png"]
        )

        assert os.path.exists(result_file), "❌ 结果审查文件未创建"
        with open(result_file) as f:
            state = json.load(f)

        assert state['status'] == 'waiting', f"❌ 状态应为waiting, 实际为{state['status']}"
        assert len(state['results']) == 1, f"❌ 应有1个结果, 实际{len(state['results'])}"
        assert state['results'][0]['idea_name'] == "ProtT5 + SVM 二级结构预测"
        print("✅ test_1_write_result_review 通过")

    def test_2_read_result_review(self):
        """测试: 读取待审查的实验结果"""
        from paper_to_task.interaction import pipeline_hooks

        # 先写入一个结果
        self._create_result_review({
            "status": "waiting",
            "results": [
                {
                    "idea_name": "ProtT5 二级结构预测",
                    "run_num": 0,
                    "timestamp": time.time(),
                    "scores": {"completeness": 85, "accuracy": 72},
                    "report_preview": "# 报告预览",
                    "report_images": ["fig1.png"]
                }
            ]
        })

        # 通过 hooks 读取
        review_state = pipeline_hooks.read_state(
            os.path.join(self.review_dir, 'result_review.json')
        )

        assert review_state is not None, "❌ 读取状态失败"
        assert review_state['status'] == 'waiting'
        assert len(review_state['results']) == 1
        assert review_state['results'][0]['scores']['completeness'] == 85
        print("✅ test_2_read_result_review 通过")

    def test_3_submit_result_feedback_approve(self):
        """测试: 提交结果反馈（确认评分）"""
        from paper_to_task.interaction import pipeline_hooks

        # 先创建要审查的结果
        self._create_result_review({
            "status": "waiting",
            "results": [{
                "idea_name": "XLNet 蛋白质分类",
                "run_num": 0,
                "timestamp": time.time(),
                "scores": {"completeness": 75},
                "report_preview": "报告内容"
            }]
        })

        # 提交反馈
        feedback = {
            "action": "approve",
            "overrides": {"completeness": 85},
            "comments": "实验结果合理，确认通过"
        }
        result = pipeline_hooks.submit_result_feedback(self.test_dir, feedback)

        assert result['status'] == 'approved', f"❌ 状态应为approved, 实际{result['status']}"
        assert result['feedback']['overrides']['completeness'] == 85
        print("✅ test_3_submit_result_feedback_approve 通过")

    def test_4_mulitple_results_in_review(self):
        """测试: 多个实验结果待审查"""
        from paper_to_task.interaction import pipeline_hooks

        # 写入3个实验结果
        results = []
        for i, name in enumerate(["BERT 嵌入", "T5 模型", "XLNet 分类"]):
            pipeline_hooks.write_result_for_review(
                output_dir=self.test_dir,
                idea_name=name,
                run_num=i,
                scores={"total": 70 + i * 5},
                report_text=f"#{name}的实验报告",
                report_images=[]
            )
            results.append(name)

        # 读取审查状态
        state = pipeline_hooks.read_state(
            os.path.join(self.review_dir, 'result_review.json')
        )

        assert len(state['results']) == 3, f"❌ 应有3个结果, 实际{len(state['results'])}"
        names = [r['idea_name'] for r in state['results']]
        assert names == ["BERT 嵌入", "T5 模型", "XLNet 分类"]
        print("✅ test_4_mulitple_results_in_review 通过")

    def test_5_result_feedback_overrides_scores(self):
        """测试: 用户修改评分后提交"""
        from paper_to_task.interaction import pipeline_hooks

        # 创建待审查结果
        self._create_result_review({
            "status": "waiting",
            "results": [{
                "idea_name": "测试实验",
                "run_num": 0,
                "timestamp": time.time(),
                "scores": {"completeness": 60, "accuracy": 55},
                "report_preview": "报告内容"
            }]
        })

        # 用户修改评分并提交
        feedback = {
            "action": "approve",
            "user_scores": {"completeness": 90, "accuracy": 85},
            "comments": "人工确认，评分应更高"
        }
        result = pipeline_hooks.submit_result_feedback(self.test_dir, feedback)
        assert result['status'] == 'approved'
        assert result['feedback']['user_scores']['completeness'] == 90
        print("✅ test_5_result_feedback_overrides_scores 通过")


class TestPipelineHooks:
    """测试 pipeline_hooks 工具函数"""

    def test_read_write_state(self):
        """测试状态文件读写"""
        from paper_to_task.interaction import pipeline_hooks

        test_file = os.path.join(tempfile.gettempdir(), 'test_state.json')
        test_state = {"status": "running", "value": 42}

        pipeline_hooks.write_state(test_file, test_state)
        read_back = pipeline_hooks.read_state(test_file)

        assert read_back['status'] == 'running'
        assert read_back['value'] == 42
        os.remove(test_file)
        print("✅ test_read_write_state 通过")

    def test_read_nonexistent_state(self):
        """测试读取不存在的状态文件"""
        from paper_to_task.interaction import pipeline_hooks

        result = pipeline_hooks.read_state("/nonexistent/path/state.json")
        assert result == {}, f"❌ 不存在的文件应返回空dict, 实际{result}"
        print("✅ test_read_nonexistent_state 通过")

    def test_ensure_review_dir(self):
        """测试创建审查目录"""
        from paper_to_task.interaction import pipeline_hooks

        with tempfile.TemporaryDirectory() as tmpdir:
            review_dir = pipeline_hooks.ensure_review_dir(tmpdir)
            expected = os.path.join(tmpdir, '.human_review')
            assert review_dir == expected
            assert os.path.exists(expected)
        print("✅ test_ensure_review_dir 通过")

    def test_get_review_dir(self):
        """测试获取审查目录路径"""
        from paper_to_task.interaction import pipeline_hooks

        review_dir = pipeline_hooks.get_review_dir("/tmp/test_output")
        assert review_dir == "/tmp/test_output/.human_review"
        print("✅ test_get_review_dir 通过")


class TestComparisonDashboard:
    """测试对比看板数据"""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp(prefix='comparison_test_')

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def _create_session_dir(self, session_id):
        """创建模拟的 session 目录"""
        session_dir = os.path.join(self.test_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _create_ideas_file(self, session_dir, ideas):
        """创建模拟的 ideas.json"""
        with open(os.path.join(session_dir, 'ideas.json'), 'w') as f:
            json.dump(ideas, f, indent=2)

    def _create_experiment_result(self, session_dir, exp_name, run_num=1, success=True):
        """创建模拟的实验结果目录"""
        exp_dir = os.path.join(session_dir, f"20260618_{exp_name}")
        os.makedirs(exp_dir, exist_ok=True)

        run_dir = os.path.join(exp_dir, f"run_{run_num}")
        os.makedirs(run_dir, exist_ok=True)

        # 创建 final_info.json
        final_info = {
            "success": success,
            "performance": {
                "overall_improvement_rate": 15.0 if success else 0,
                "sci_task": {
                    "means": {
                        "total_score": 72.5 if success else 0,
                        "item_0_score": 80,
                        "item_1_score": 65
                    }
                }
            }
        }
        with open(os.path.join(run_dir, 'final_info.json'), 'w') as f:
            json.dump(final_info, f, indent=2)

        # 创建报告
        report_dir = os.path.join(exp_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "report.md"), 'w') as f:
            f.write(f"# {exp_name} 实验报告\n\n## 结果\n实验{'成功' if success else '失败'}。")

        return exp_dir

    def test_6_compare_multiple_results(self):
        """测试: 对比多个候选结果的数据结构"""
        from paper_to_task.interaction import pipeline_hooks

        session_dir = self._create_session_dir("session_001")

        # 创建多个实验结果
        results_data = []
        for i, name in enumerate(["ProtT5_SVM", "XLNet_Classifier", "BERT_Embedding"]):
            scores = {
                "item_0_score": 70 + i * 5,
                "item_1_score": 60 + i * 8,
                "total_score": 65 + i * 7
            }
            entry = {
                "idea_name": name,
                "run_num": 0,
                "timestamp": time.time(),
                "scores": scores,
                "report_preview": f"#{name}实验的简要报告内容..."
            }
            results_data.append(entry)

        # 写入审查状态
        pipeline_hooks.write_result_for_review(
            output_dir=self.test_dir,
            idea_name="ProtT5_SVM",
            run_num=0,
            scores=results_data[0]["scores"],
            report_text=results_data[0]["report_preview"],
            report_images=[]
        )

        # 检查写入的数据
        state = pipeline_hooks.read_state(
            os.path.join(self.test_dir, '.human_review', 'result_review.json')
        )

        assert state['status'] == 'waiting'
        assert len(state['results']) >= 1
        result = state['results'][0]
        assert 'scores' in result, "❌ 结果中缺少scores字段"
        assert 'idea_name' in result, "❌ 结果中缺少idea_name字段"

        # 验证分数可以被对比
        scores = result['scores']
        total = sum(scores.values()) / len(scores)
        assert total > 0, "❌ 平均分应为正数"
        print(f"✅ test_6_compare_multiple_results 通过 (平均分: {total:.1f})")

    def test_7_result_review_approve_flow(self):
        """测试: 完整的 待审查 → 确认 → 已审批 流程"""
        from paper_to_task.interaction import pipeline_hooks

        # 阶段1: 写入待审查结果
        pipeline_hooks.write_result_for_review(
            output_dir=self.test_dir,
            idea_name="ProtT5 二级结构预测",
            run_num=0,
            scores={"completeness": 85, "accuracy": 78},
            report_text="# 实验报告",
            report_images=[]
        )

        state = pipeline_hooks.read_state(
            os.path.join(self.test_dir, '.human_review', 'result_review.json')
        )
        assert state['status'] == 'waiting', "❌ 刚写入的状态应为waiting"

        # 阶段2: 用户确认审批
        pipeline_hooks.submit_result_feedback(self.test_dir, {
            "action": "approve",
            "comments": "确认通过"
        })

        # 阶段3: 验证状态已更新
        fb_state = pipeline_hooks.read_state(
            os.path.join(self.test_dir, '.human_review', 'result_feedback.json')
        )
        assert fb_state['status'] == 'approved', "❌ 审批后状态应为approved"

        result_state = pipeline_hooks.read_state(
            os.path.join(self.test_dir, '.human_review', 'result_review.json')
        )
        assert result_state['status'] == 'approved', "❌ 审批后结果状态应为approved"
        print("✅ test_7_result_review_approve_flow 通过")


if __name__ == '__main__':
    # 手动运行测试
    tests = [
        TestPipelineHooks(),
        TestResultReview(),
        TestComparisonDashboard(),
    ]

    for test_class in tests:
        print(f"\n{'='*60}")
        print(f"测试类: {test_class.__class__.__name__}")
        print(f"{'='*60}")

        # 调用 setup_class 如果存在
        if hasattr(test_class, 'setup_class'):
            test_class.setup_class()

        for name in sorted(dir(test_class)):
            if name.startswith('test_'):
                if hasattr(test_class, 'setup_method'):
                    test_class.setup_method()
                try:
                    getattr(test_class, name)()
                except Exception as e:
                    import traceback
                    print(f"❌ {name} 失败: {e}")
                    traceback.print_exc()
                finally:
                    if hasattr(test_class, 'teardown_method'):
                        test_class.teardown_method()

        # 调用 teardown_class 如果存在
        if hasattr(test_class, 'teardown_class'):
            test_class.teardown_class()

    print(f"\n{'='*60}")
    print("所有测试完成！")
