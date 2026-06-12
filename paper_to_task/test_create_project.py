"""
完整测试：从PDF上传到项目创建
"""

import requests
import json
import os
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:5000"

def test_full_workflow():
    """测试完整工作流：上传 -> 处理 -> 创建项目"""
    print("=" * 60)
    print("🧪 完整工作流测试：从PDF到项目创建")
    print("=" * 60)

    # 选择测试PDF文件
    test_pdf = "/home/devuser/workspace/reproduction_agent/InternAgent/ProtTrans_Toward_Understanding_the_Language_of_Life_Through_Self-Supervised_Learning.pdf"

    if not os.path.exists(test_pdf):
        print(f"❌ 测试PDF文件不存在: {test_pdf}")
        return

    print(f"\n📄 测试PDF: {Path(test_pdf).name}")

    # 第一步：上传PDF
    print("\n[1] 上传PDF...")
    with open(test_pdf, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=60)

    if response.status_code != 200:
        print(f"❌ 上传失败: {response.status_code}")
        return

    data = response.json()
    if not data.get('success'):
        print(f"❌ 处理失败: {data.get('result', {}).get('error')}")
        return

    task_id = data.get('task_id')
    result = data.get('result', {})
    print(f"✅ 上传成功，任务ID: {task_id}")

    # 显示处理结果摘要
    quality = result.get('quality', {})
    task_info = result.get('task_info', {})
    checklist = result.get('checklist', [])

    print(f"\n[2] 处理结果摘要:")
    print(f"质量评分: {quality.get('score', 0):.2f} ({quality.get('grade', 'N/A')})")
    print(f"任务: {task_info.get('task', '')[:60]}...")
    print(f"数据文件: {len(task_info.get('data', []))} 个")
    print(f"评分项: {len(checklist)} 个")

    # 第二步：创建项目
    print(f"\n[3] 创建项目...")
    task_name = "Science_Test_001"

    create_response = requests.post(
        f"{BASE_URL}/create_project",
        json={
            'task_id': task_id,
            'task_name': task_name
        },
        timeout=60
    )

    if create_response.status_code != 200:
        print(f"❌ 创建请求失败: {create_response.status_code}")
        return

    create_data = create_response.json()
    if not create_data.get('success'):
        print(f"❌ 创建失败: {create_data.get('error')}")
        return

    creation_result = create_data.get('creation_result', {})

    print(f"✅ 项目创建成功！")
    print(f"\n[4] 项目详情:")
    print(f"项目路径: {creation_result.get('task_path', '')}")
    print(f"任务ID: {creation_result.get('task_id', '')}")

    # 显示生成的文件
    task_path = creation_result.get('task_path', '')
    if task_path and os.path.exists(task_path):
        print(f"\n[5] 生成的文件:")

        # 列出所有文件
        for root, dirs, files in os.walk(task_path):
            level = root.replace(task_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")

        # 显示关键文件内容
        print(f"\n[6] 关键文件内容:")

        # task_info.json
        task_info_path = os.path.join(task_path, 'task_info.json')
        if os.path.exists(task_info_path):
            with open(task_info_path, 'r', encoding='utf-8') as f:
                task_info_content = json.load(f)
            print(f"\n📋 task_info.json:")
            print(json.dumps(task_info_content, indent=2, ensure_ascii=False))

        # checklist.json
        checklist_path = os.path.join(task_path, 'target_study', 'checklist.json')
        if os.path.exists(checklist_path):
            with open(checklist_path, 'r', encoding='utf-8') as f:
                checklist_content = json.load(f)
            print(f"\n📊 checklist.json (共{len(checklist_content)}个评分项):")
            for i, item in enumerate(checklist_content, 1):
                print(f"\n  [{i}] 权重: {item.get('weight', 0)}")
                print(f"      内容: {item.get('content', '')[:80]}...")

    print("\n" + "=" * 60)
    print("🎉 完整工作流测试完成！")
    print(f"📁 项目已创建在: {task_path}")
    print("=" * 60)

if __name__ == "__main__":
    test_full_workflow()
