"""
在InternAgent环境中测试Web API功能
"""

import requests
import json
import os
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:5000"

def test_upload_api():
    """测试PDF上传和处理API"""
    print("=" * 60)
    print("🧪 在InternAgent环境中测试Web API功能")
    print("=" * 60)

    # 选择测试PDF文件
    test_pdf = "/home/devuser/workspace/reproduction_agent/InternAgent/ProtTrans_Toward_Understanding_the_Language_of_Life_Through_Self-Supervised_Learning.pdf"

    if not os.path.exists(test_pdf):
        print(f"❌ 测试PDF文件不存在: {test_pdf}")
        return

    print(f"\n📄 测试PDF: {Path(test_pdf).name}")

    # 第一步：测试上传API
    print("\n[1] 测试上传API...")

    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': ('test.pdf', f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files, timeout=60)

        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ 上传成功")
            print(f"调试 - 完整响应: {json.dumps(data, indent=2, ensure_ascii=False)[:800]}...")

            if data.get('success'):
                task_id = data.get('task_id')
                print(f"任务ID: {task_id}")

                result = data.get('result', {})

                if not result.get('success'):
                    print(f"❌ PDF处理失败: {result.get('error')}")
                    return

                quality = result.get('quality', {})

                print(f"\n[2] 处理结果:")
                print(f"质量评分: {quality.get('score', 0):.2f}")
                print(f"等级: {quality.get('grade', 'N/A')}")
                print(f"状态: {'✅ 通过' if quality.get('passed') else '❌ 未达标'}")

                print(f"\n[3] 维度评分:")
                dimension_scores = quality.get('dimension_scores', {})
                for dim, score in dimension_scores.items():
                    dim_names = {
                        'completeness': '完整性',
                        'accuracy': '准确性',
                        'clarity': '清晰度',
                        'feasibility': '可行性'
                    }
                    print(f"  {dim_names.get(dim, dim)}: {score:.2f}")

                print(f"\n[4] 任务信息:")
                task_info = result.get('task_info', {})
                print(f"任务: {task_info.get('task', '')}")
                data_count = len(task_info.get('data', []))
                print(f"数据文件: {data_count} 个")

                if data_count > 0:
                    print(f"\n数据文件详情:")
                    for i, data_item in enumerate(task_info.get('data', []), 1):
                        print(f"  [{i}] {data_item.get('name', '')}")
                        print(f"      路径: {data_item.get('path', '')}")
                        print(f"      描述: {data_item.get('description', '')[:60]}...")

                print(f"\n[5] 评分项:")
                checklist = result.get('checklist', [])
                print(f"评分项数量: {len(checklist)} 个")

                if checklist:
                    print("\n所有评分项:")
                    for i, item in enumerate(checklist, 1):
                        print(f"\n  [{i}] 权重: {item.get('weight', 0)}")
                        print(f"      内容: {item.get('content', '')}")

                        print(f"      评估标准: {item.get('evaluation_criteria', '')[:60]}...")

                # 测试反馈改进API
                print(f"\n[6] 测试反馈改进API...")

                feedback = "checklist的评分项需要更具体，要包含ProtTrans模型的具体细节和准确率指标"

                refine_response = requests.post(
                    f"{BASE_URL}/refine",
                    json={
                        'task_id': task_id,
                        'feedback': feedback
                    },
                    timeout=60
                )

                print(f"状态码: {refine_response.status_code}")

                if refine_response.status_code == 200:
                    refine_data = refine_response.json()
                    if refine_data.get('success'):
                        print(f"✅ 改进成功")

                        refined_result = refine_data.get('result', {})
                        refined_quality = refined_result.get('quality', {})

                        print(f"\n[7] 改进后质量评分:")
                        print(f"新评分: {refined_quality.get('score', 0):.2f}")
                        print(f"新等级: {refined_quality.get('grade', 'N/A')}")

                        # 检查是否有改进
                        original_score = quality.get('score', 0)
                        new_score = refined_quality.get('score', 0)

                        if new_score > original_score:
                            improvement = new_score - original_score
                            print(f"✅ 质量提升: +{improvement:.2f}")
                        else:
                            print(f"⚠️ 质量未提升")

                    else:
                        print(f"❌ 改进失败: {refine_data.get('error')}")
                else:
                    print(f"❌ 改进请求失败: {refine_response.status_code}")

            else:
                print(f"❌ 处理失败: {data.get('error')}")

        else:
            print(f"❌ 上传失败: HTTP {response.status_code}")

    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器，请确保Web应用正在运行")
        print(f"   访问地址: {BASE_URL}")
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("\n" + "=" * 60)
    print("🎯 API测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_upload_api()
