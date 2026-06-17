"""
Paper-to-Task Web应用
提供Web界面让用户上传PDF并体验自动化生成流程
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import json
import tempfile
import shutil
from pathlib import Path
import sys
from dotenv import load_dotenv
import yaml
import copy

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_to_task.pipeline import PaperToTaskPipeline
from paper_to_task.interaction.cli_interface import CLIInterface
from paper_to_task.interaction import pipeline_hooks

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() + '/paper_to_task_uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 确保上传目录存在
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# 直接使用DeepSeek API密钥
API_KEY = "sk-160bc446a77248b8a9220ef211650e19"

config = {
    'llm': {
        'backend': 'deepseek',
        'api_key': API_KEY,
        'base_url': 'https://api.deepseek.com',
        'model': 'deepseek-chat',
        'temperature': 0.3,
        'max_tokens': 4000,
        'cache_enabled': True
    },
    'project': {
        'sci_tasks_base': str((Path(__file__).parent.parent.parent / 'sci_tasks' / 'tasks').resolve())
    },
    'quality': {
        'min_score': 0.7,
        'enable_auto_improvement': True
    }
}

print(f"🔧 LLM配置: backend={config['llm']['backend']}")
print(f"🔧 API Key: {API_KEY[:10]}...{API_KEY[-4:]}")

pipeline = PaperToTaskPipeline(config)
cli = CLIInterface(pipeline)

# 存储处理结果
processing_results = {}

# 存储已上传的数据文件
uploaded_files = {}


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

# 服务 interaction/static/ 目录下的静态文件（如 pipeline_review.js）
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和处理"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件上传'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})

    if not file.filename.endswith('.pdf'):
        return jsonify({'success': False, 'error': '只支持PDF文件'})

    try:
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 生成任务ID
        task_id = f"task_{len(processing_results) + 1}"

        # 处理PDF
        result = pipeline.process_pdf(file_path, auto_improve=True)

        # 保存结果
        processing_results[task_id] = {
            'result': result,
            'file_path': file_path,
            'filename': file.filename,
            'polished_markdown': None  # 论文润色缓存
        }

        return jsonify({
            'success': True,
            'task_id': task_id,
            'result': format_result_for_frontend(result)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/refine', methods=['POST'])
def refine_content():
    """根据用户反馈改进内容"""
    data = request.json
    task_id = data.get('task_id')
    feedback = data.get('feedback')

    if not task_id or not feedback:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    if task_id not in processing_results:
        return jsonify({'success': False, 'error': '任务不存在'})

    try:
        # 获取当前内容
        current_result = processing_results[task_id]['result']
        current_content = {
            'task_info': current_result['task_info'],
            'checklist': current_result['checklist']
        }

        # 执行改进
        refinement_result = pipeline.refine_content(current_content, feedback)

        if refinement_result['success']:
            # 更新结果
            processing_results[task_id]['result'] = {
                **current_result,
                'task_info': refinement_result['task_info'],
                'checklist': refinement_result['checklist']
            }

            return jsonify({
                'success': True,
                'result': format_result_for_frontend(
                    processing_results[task_id]['result']
                ),
                'improvements': refinement_result.get('improvements', [])
            })
        else:
            return jsonify({
                'success': False,
                'error': refinement_result.get('error', '改进失败')
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/create_project', methods=['POST'])
def create_project():
    """创建项目"""
    data = request.json
    task_id = data.get('task_id')
    task_name = data.get('task_name')

    if not task_id:
        return jsonify({'success': False, 'error': '缺少任务ID'})

    if task_id not in processing_results:
        return jsonify({'success': False, 'error': '任务不存在'})

    try:
        task_data = processing_results[task_id]
        result = task_data['result']
        file_path = task_data['file_path']

        # 如果没有提供任务名称，使用默认名称
        if not task_name:
            task_name = f"Science_{len(processing_results) + 8:03d}"

        # 创建项目
        creation_result = pipeline.create_project(
            task_name=task_name,
            task_info=result['task_info'],
            checklist=result['checklist'],
            pdf_path=file_path,
            research_doc=result.get('research_doc', ''),
            domain='Science'
        )

        return jsonify({
            'success': True,
            'creation_result': creation_result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/result/<task_id>')
def view_result(task_id):
    """查看详细结果"""
    if task_id not in processing_results:
        return "任务不存在", 404

    task_data = processing_results[task_id]
    result = task_data['result']

    return render_template('result.html',
                          task_id=task_id,
                          result=result,
                          filename=task_data['filename'])


def format_result_for_frontend(result):
    """格式化结果用于前端显示"""
    if not result.get('success'):
        return {
            'success': False,
            'error': result.get('error')
        }

    quality = result.get('quality', {})

    return {
        'success': True,
        'quality': {
            'score': quality.get('score', 0),
            'grade': quality.get('grade', 'N/A'),
            'passed': quality.get('passed', False),
            'dimension_scores': quality.get('dimension_scores', {})
        },
        'task_info': result.get('task_info', {}),
        'checklist': result.get('checklist', []),
        'research_doc': result.get('research_doc', ''),
        'research_info': result.get('research_info', {}),
        'paper_sources': result.get('paper_sources', {}),
        'raw_markdown': result.get('raw_markdown', ''),
        'processing_time': result.get('processing_time', 0),
        'next_steps': result.get('next_steps', [])
    }


@app.route('/upload_data_file', methods=['POST'])
def upload_data_file():
    """上传数据文件"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件上传'})

    file = request.files['file']
    task_id = request.form.get('task_id')
    dataset_name = request.form.get('dataset_name', '')

    if not task_id or not dataset_name:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})

    try:
        # 保存文件到临时目录
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{dataset_name}_{file.filename}")
        file.save(file_path)

        # 记录上传的文件
        if task_id not in uploaded_files:
            uploaded_files[task_id] = {}

        uploaded_files[task_id][dataset_name] = {
            'filename': file.filename,
            'path': file_path,
            'size': os.path.getsize(file_path)
        }

        return jsonify({
            'success': True,
            'message': f'文件 {file.filename} 上传成功',
            'dataset_name': dataset_name,
            'file_info': {
                'filename': file.filename,
                'size': os.path.getsize(file_path)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'文件上传失败: {str(e)}'
        })


@app.route('/list_tasks', methods=['GET'])
def list_tasks():
    """列出sci_tasks/tasks目录下的Science_XXX任务目录"""
    project_root = Path(__file__).parent.parent.parent
    tasks_dir = project_root / 'sci_tasks' / 'tasks'
    task_dirs = []

    if tasks_dir.exists():
        for d in sorted(tasks_dir.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                # 检查目录是否包含任务标志文件
                has_task_info = (d / 'task_info.json').exists()
                task_dirs.append({
                    'name': d.name,
                    'path': str(d),
                    'relative_path': f'sci_tasks/tasks/{d.name}',
                    'has_task_info': has_task_info
                })

    return jsonify({'success': True, 'tasks': task_dirs})


# ============================================================================
# sci_tasks 流水线启动与监视
# ============================================================================
import subprocess as subprocess_module
import signal as signal_module
import threading
import time as time_module

# 存储后台运行的任务状态
pipeline_tasks = {}  # task_id -> {'process': Popen, 'output': [], 'status': 'running'|'done'|'error'}

# 持久化存储流水线记录
PIPELINE_DATA_DIR = Path(__file__).parent / 'pipeline_data'
PIPELINE_TASKS_FILE = PIPELINE_DATA_DIR / 'tasks.json'
PIPELINE_LOGS_DIR = PIPELINE_DATA_DIR / 'logs'

PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _load_pipeline_tasks_from_disk():
    """从磁盘加载历史流水线记录"""
    if not PIPELINE_TASKS_FILE.exists():
        return {}
    try:
        with open(PIPELINE_TASKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[pipeline] 加载历史记录失败: {e}")
        return {}

def _save_pipeline_tasks_to_disk(tasks_dict):
    """将流水线记录写入磁盘（不含进程句柄）"""
    try:
        serializable = {}
        for uid, task in tasks_dict.items():
            entry = {k: v for k, v in task.items() if k != 'process'}
            serializable[uid] = entry
        with open(PIPELINE_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[pipeline] 保存历史记录失败: {e}")

def _append_pipeline_log(uuid, line):
    """追加一行日志到文件"""
    try:
        log_file = PIPELINE_LOGS_DIR / f"{uuid}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception as e:
        print(f"[pipeline] 写入日志失败: {e}")

def _get_pipeline_log_lines(uuid, max_lines=0):
    """读取日志文件，max_lines=0 返回全部"""
    log_file = PIPELINE_LOGS_DIR / f"{uuid}.log"
    if not log_file.exists():
        return []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        lines = [l.rstrip('\n\r') for l in lines]
        if max_lines > 0 and len(lines) > max_lines:
            lines = lines[-max_lines:]
        return lines
    except Exception as e:
        return [f"[读取日志失败: {e}]"]

# 启动时恢复已完成/出错的流水线记录
_persisted_tasks = _load_pipeline_tasks_from_disk()


@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    """启动sci_tasks流水线"""
    data = request.json
    task_path = data.get('task_path')
    task_id = data.get('task_id')
    extra_args = data.get('extra_args', [])

    if not task_path:
        return jsonify({'success': False, 'error': '缺少任务路径'})

    # 构建launch_discovery.py的路径
    project_root = Path(__file__).parent.parent.parent
    launch_script = project_root / 'launch_discovery.py'

    if not launch_script.exists():
        return jsonify({'success': False, 'error': f'启动脚本不存在: {launch_script}'})

    try:
        # 构建命令 — 使用InternAgent conda环境
        python_bin = '/opt/conda/envs/InternAgent/bin/python3'
        if not os.path.exists(python_bin):
            python_bin = sys.executable  # fallback
        cmd = [python_bin, str(launch_script), '--task', task_path]

        if extra_args:
            if isinstance(extra_args, list):
                cmd.extend(extra_args)
            elif isinstance(extra_args, str):
                cmd.extend(extra_args.split())

        # 启动进程
        process = subprocess_module.Popen(
            cmd,
            stdout=subprocess_module.PIPE,
            stderr=subprocess_module.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root)
        )

        # 记录任务
        task_uuid = task_id or f"pipeline_{int(time_module.time())}"
        pipeline_tasks[task_uuid] = {
            'process': process,
            'output': [],
            'stored_output': [],
            'status': 'running',
            'task_path': task_path,  # 记录任务路径，供审批API反查launch_dir
            'start_time': time_module.time(),
            'cmd': ' '.join(cmd),
            'task_path': task_path
        }

        # 持久化保存
        _save_pipeline_tasks_to_disk(pipeline_tasks)

        # 启动后台线程读取输出
        def read_output(task_uuid):
            task = pipeline_tasks.get(task_uuid)
            if not task:
                return
            process = task['process']
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    # 清空ANSI颜色转义码
                    import re
                    clean_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
                    clean_line = clean_line.rstrip('\n\r')
                    if clean_line:
                        task['output'].append(clean_line)
                        task['stored_output'].append(clean_line)
                        _append_pipeline_log(task_uuid, clean_line)
                process.wait()
                if process.returncode == 0:
                    task['status'] = 'completed'
                else:
                    task['status'] = 'error'
                    err_line = f'进程退出码: {process.returncode}'
                    task['output'].append(err_line)
                    _append_pipeline_log(task_uuid, err_line)
            except Exception as e:
                task['status'] = 'error'
                err_line = f'读取输出失败: {str(e)}'
                task['output'].append(err_line)
                _append_pipeline_log(task_uuid, err_line)
            task['end_time'] = time_module.time()
            _save_pipeline_tasks_to_disk(pipeline_tasks)

        thread = threading.Thread(target=read_output, args=(task_uuid,), daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'pipeline_uuid': task_uuid,
            'message': '流水线已启动',
            'cmd': ' '.join(cmd)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'启动流水线失败: {str(e)}'
        })


@app.route('/pipeline_status', methods=['POST'])
def pipeline_status():
    """获取流水线状态（支持活跃和历史任务）"""
    data = request.json
    pipeline_uuid = data.get('pipeline_uuid')

    # 活跃任务
    task = pipeline_tasks.get(pipeline_uuid)
    is_historical = False
    if not task:
        # 尝试从历史记录恢复
        task = _persisted_tasks.get(pipeline_uuid)
        if not task:
            return jsonify({'success': False, 'error': '流水线任务不存在'})
        is_historical = True

    # 获取输出：活跃任务从内存取，历史任务从文件读
    if not is_historical:
        output = task['output'][-10000:]
        elapsed = time_module.time() - task['start_time']
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    else:
        output = _get_pipeline_log_lines(pipeline_uuid, max_lines=10000)
        elapsed = (task.get('end_time', task['start_time']) if task.get('end_time') else task['start_time']) - task['start_time']
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    status_info = {
        'success': True,
        'pipeline_uuid': pipeline_uuid,
        'status': task['status'],
        'output': output,
        'elapsed': elapsed_str,
        'elapsed_seconds': int(elapsed),
        'cmd': task.get('cmd', ''),
        'task_path': task.get('task_path', ''),
        'start_time': time_module.strftime('%H:%M:%S', time_module.localtime(task['start_time'])),
        'is_historical': is_historical
    }

    if task.get('end_time'):
        status_info['end_time'] = time_module.strftime('%H:%M:%S', time_module.localtime(task['end_time']))

    return jsonify(status_info)


@app.route('/pipeline_history', methods=['GET'])
def pipeline_history():
    """获取历史流水线记录列表"""
    # 合并活跃任务 + 已持久化的历史任务
    seen = set()
    records = []

    # 先把持久化的历史记录加进来
    for uid, t in _persisted_tasks.items():
        seen.add(uid)
        end_time = t.get('end_time', t['start_time'])
        if t.get('status') in ('running',):
            continue  # 历史记录中跳过 running（如果有残留）
        records.append({
            'uuid': uid,
            'status': t.get('status', 'unknown'),
            'task_path': t.get('task_path', ''),
            'cmd': t.get('cmd', ''),
            'start_time': time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(t['start_time'])),
            'start_ts': t['start_time'],
            'end_time': time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(end_time)),
            'end_ts': end_time,
            'elapsed': int(end_time - t['start_time']),
            'is_active': False
        })

    # 再把活跃任务加进来（可能覆盖历史中的 running 残留）
    for uid, t in pipeline_tasks.items():
        seen.add(uid)
        now = time_module.time()
        records.append({
            'uuid': uid,
            'status': t['status'],
            'task_path': t.get('task_path', ''),
            'cmd': t.get('cmd', ''),
            'start_time': time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(t['start_time'])),
            'start_ts': t['start_time'],
            'end_time': time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(t.get('end_time', now))),
            'end_ts': t.get('end_time', now),
            'elapsed': int((t.get('end_time', now)) - t['start_time']),
            'is_active': t['status'] == 'running'
        })

    # 按启动时间倒序排列
    records.sort(key=lambda r: r['start_ts'], reverse=True)

    return jsonify({'success': True, 'records': records})


@app.route('/pipeline_log_detail', methods=['POST'])
def pipeline_log_detail():
    """获取流水线完整日志"""
    data = request.json
    pipeline_uuid = data.get('pipeline_uuid')
    max_lines = data.get('max_lines', 0)

    lines = _get_pipeline_log_lines(pipeline_uuid, max_lines=max_lines)

    # 也尝试从活跃任务取
    task = pipeline_tasks.get(pipeline_uuid)
    if not task:
        task = _persisted_tasks.get(pipeline_uuid)

    return jsonify({
        'success': True,
        'lines': lines,
        'count': len(lines),
        'status': task.get('status', 'unknown') if task else 'unknown'
    })

@app.route('/stop_pipeline', methods=['POST'])
def stop_pipeline():
    """停止流水线"""
    data = request.json
    pipeline_uuid = data.get('pipeline_uuid')

    if not pipeline_uuid or pipeline_uuid not in pipeline_tasks:
        return jsonify({'success': False, 'error': '流水线任务不存在'})

    task = pipeline_tasks[pipeline_uuid]
    if task['status'] in ('running', 'paused'):
        task['process'].terminate()
        task['status'] = 'stopped'
        stop_line = '🛑 用户手动停止'
        task['output'].append(stop_line)
        _append_pipeline_log(pipeline_uuid, stop_line)
    task['end_time'] = time_module.time()
    _save_pipeline_tasks_to_disk(pipeline_tasks)

    return jsonify({
        'success': True,
        'message': '流水线已停止'
    })


@app.route('/pause_pipeline', methods=['POST'])
def pause_pipeline():
    """暂停流水线（发送 SIGSTOP）"""
    data = request.json
    pipeline_uuid = data.get('pipeline_uuid')

    if not pipeline_uuid or pipeline_uuid not in pipeline_tasks:
        return jsonify({'success': False, 'error': '流水线任务不存在'})

    task = pipeline_tasks[pipeline_uuid]
    if task['status'] != 'running':
        return jsonify({'success': False, 'error': f'当前状态为 {task["status"]}，无法暂停'})

    try:
        task['process'].send_signal(signal_module.SIGSTOP)
        task['status'] = 'paused'
        pause_line = '⏸️ 流水线已暂停'
        task['output'].append(pause_line)
        _append_pipeline_log(pipeline_uuid, pause_line)
        _save_pipeline_tasks_to_disk(pipeline_tasks)
        return jsonify({'success': True, 'message': '流水线已暂停'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'暂停失败: {str(e)}'})


@app.route('/resume_pipeline', methods=['POST'])
def resume_pipeline():
    """继续流水线（发送 SIGCONT）"""
    data = request.json
    pipeline_uuid = data.get('pipeline_uuid')

    if not pipeline_uuid or pipeline_uuid not in pipeline_tasks:
        return jsonify({'success': False, 'error': '流水线任务不存在'})

    task = pipeline_tasks[pipeline_uuid]
    if task['status'] != 'paused':
        return jsonify({'success': False, 'error': f'当前状态为 {task["status"]}，无法继续'})

    try:
        task['process'].send_signal(signal_module.SIGCONT)
        task['status'] = 'running'
        resume_line = '▶️ 流水线已继续'
        task['output'].append(resume_line)
        _append_pipeline_log(pipeline_uuid, resume_line)
        _save_pipeline_tasks_to_disk(pipeline_tasks)
        return jsonify({'success': True, 'message': '流水线已继续'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'继续失败: {str(e)}'})


@app.route('/get_data_guidance', methods=['POST'])
def get_data_guidance():
    """获取数据准备指导"""
    data = request.json
    task_id = data.get('task_id')

    if not task_id or task_id not in processing_results:
        return jsonify({'success': False, 'error': '任务不存在'})

    try:
        result = processing_results[task_id]['result']
        task_info = result.get('task_info', {})
        datasets = task_info.get('data', [])

        guidance = {
            'datasets': [],
            'general_instructions': []
        }

        # 为每个数据集生成指导
        for dataset in datasets:
            dataset_guidance = {
                'name': dataset.get('name', ''),
                'description': dataset.get('description', ''),
                'path': dataset.get('path', ''),
                'requirements': [],
                'alternatives': []
            }

            # 根据数据集名称生成特定指导
            dataset_name = dataset.get('name', '').lower()

            if 'uniref50' in dataset_name:
                dataset_guidance['requirements'] = [
                    'UniRef50数据库文件（约50GB）',
                    '可以从UniProt官网下载',
                    '或使用提供的预处理子集'
                ]
                dataset_guidance['alternatives'] = [
                    'UniProt FTP: ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniref100/',
                    '使用较小的测试数据集进行初步验证'
                ]

            elif 'uniref100' in dataset_name:
                dataset_guidance['requirements'] = [
                    'UniRef100完整数据库（约200GB）',
                    '用于最终模型评估'
                ]
                dataset_guidance['alternatives'] = [
                    '如果资源有限，可以使用UniRef50进行训练',
                    '使用代表性子集进行功能验证'
                ]

            elif 'bfd' in dataset_name:
                dataset_guidance['requirements'] = [
                    'BFD (Big Fantastic Database) 文件',
                    '结合UniProt和宏基因组数据的大规模数据集'
                ]
                dataset_guidance['alternatives'] = [
                    '可以使用UniRef作为替代',
                    '或使用项目提供的预处理样本'
                ]

            else:
                dataset_guidance['requirements'] = [
                    f'{dataset.get("name")} 数据文件',
                    '请根据论文中的数据准备部分获取相应数据'
                ]
                dataset_guidance['alternatives'] = [
                    '查看论文的实验部分了解数据来源',
                    '联系论文作者获取数据访问权限'
                ]

            guidance['datasets'].append(dataset_guidance)

        # 添加通用指导
        guidance['general_instructions'] = [
            '1. 确认数据文件格式正确（通常是FASTA格式）',
            '2. 检查数据文件的完整性（使用MD5校验）',
            '3. 确保有足够的磁盘空间存储数据',
            '4. 考虑使用较小的测试集进行初步验证',
            '5. 在使用数据前，请确保遵守相应的数据使用许可'
        ]

        return jsonify({
            'success': True,
            'guidance': guidance
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取指导失败: {str(e)}'
        })


# ============================================================================
# 配置管理
# ============================================================================

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'default_config.yaml'

# 配置项的UI元数据：定义每个字段的前端展示方式
CONFIG_SCHEMA = {
    'workflow': {
        'label': '工作流',
        'fields': {
            'max_iterations': {'label': '最大迭代次数', 'type': 'int', 'min': 1, 'max': 20, 'desc': '每轮实验的最大迭代次数'},
            'top_ideas_count': {'label': 'Top Ideas 数量', 'type': 'int', 'min': 1, 'max': 10, 'desc': '每一轮保留的最佳想法数量'},
            'top_ideas_evo': {'label': 'Top Ideas 进化', 'type': 'bool', 'desc': '是否对入选的Top Ideas进行进化优化'},
            'max_concurrent_tasks': {'label': '最大并行任务数', 'type': 'int', 'min': 1, 'max': 8, 'desc': '同时执行的最大任务数'},
            'loop_rounds': {'label': '循环轮数', 'type': 'int', 'min': 1, 'max': 50, 'desc': '完整的发现循环轮数'},
            'loop_mode': {'label': '循环模式', 'type': 'select', 'options': ['incremental', 'fresh'], 'desc': 'fresh=每轮从基线开始; incremental=每轮从最好结果继续'},
        }
    },
    'agents.generation': {
        'label': '想法生成',
        'fields': {
            'generation_count': {'label': '生成数量', 'type': 'int', 'min': 1, 'max': 50, 'desc': '每轮生成的初始想法数量'},
            'creativity': {'label': '创造力', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '创造力的随机性程度'},
            'do_survey': {'label': '启用文献调研', 'type': 'bool', 'desc': '生成想法前是否进行文献调研'},
            'use_memory': {'label': '启用记忆系统', 'type': 'bool', 'desc': '是否利用历史经验辅助生成'},
            'filter_failed_ideas': {'label': '过滤失败想法', 'type': 'bool', 'desc': '过滤与历史失败记录相似的想法'},
            'failed_similarity_threshold': {'label': '失败相似度阈值', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '判断与历史失败相似的阈值'},
            'max_regeneration_attempts': {'label': '最大重生成次数', 'type': 'int', 'min': 0, 'max': 10, 'desc': '被过滤后的最大重生成尝试次数'},
        }
    },
    'agents.reflection': {
        'label': '反思',
        'fields': {
            'count': {'label': '反思数量', 'type': 'int', 'min': 1, 'max': 10, 'desc': '每次实验后产生的反思数量'},
            'detail_level': {'label': '详细程度', 'type': 'select', 'options': ['low', 'medium', 'high'], 'desc': '反思内容的详细程度'},
        }
    },
    'agents.evolution': {
        'label': '进化优化',
        'fields': {
            'evolution_count': {'label': '进化数量', 'type': 'int', 'min': 1, 'max': 10, 'desc': '每个想法进化出的变体数量'},
            'creativity_level': {'label': '创造力水平', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '进化时的创造性程度'},
            'temperature': {'label': '温度参数', 'type': 'float', 'min': 0.0, 'max': 2.0, 'step': 0.1, 'desc': 'LLM采样温度'},
            'use_memory': {'label': '启用记忆', 'type': 'bool', 'desc': '进化时是否利用记忆系统'},
            'filter_failed_ideas': {'label': '过滤失败', 'type': 'bool', 'desc': '进化时过滤失败方向'},
            'failed_similarity_threshold': {'label': '失败相似度阈值', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '判断与失败相似的阈值'},
            'max_regeneration_attempts': {'label': '最大重生成次数', 'type': 'int', 'min': 0, 'max': 10, 'desc': '最大重生成尝试次数'},
        }
    },
    'agents.ranking': {
        'label': '排序',
        'fields': {
            'criteria.novelty': {'label': '新颖性权重', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '排序标准：新颖性权重'},
            'criteria.plausibility': {'label': '合理性权重', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '排序标准：合理性权重'},
            'criteria.testability': {'label': '可测试性权重', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '排序标准：可测试性权重'},
            'criteria.alignment': {'label': '对齐度权重', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '排序标准：与目标对齐度权重'},
        }
    },
    'experiment': {
        'label': '实验配置',
        'fields': {
            'max_runs': {'label': '最大运行次数', 'type': 'int', 'min': 1, 'max': 20, 'desc': '总体实验运行次数'},
            'max_parallel_experiments': {'label': '最大并行实验数', 'type': 'int', 'min': 1, 'max': 8, 'desc': '并行执行的实验数量'},
            'use_mcts': {'label': '启用MCTS', 'type': 'bool', 'desc': '是否使用蒙特卡洛树搜索'},
        }
    },
    'human_review': {
        'label': '人工审批',
        'fields': {
            'idea_review': {'label': '想法审批', 'type': 'bool', 'desc': 'MAS生成想法后暂停，等待人工选择研究方向'},
            'idea_timeout': {'label': '想法审批超时(秒)', 'type': 'int', 'min': 60, 'max': 86400, 'desc': '等待人工审批的最长时间'},
            'result_review': {'label': '结果审查', 'type': 'bool', 'desc': '实验完成后暂停，展示结果供人工审阅'},
            'result_timeout': {'label': '结果审查超时(秒)', 'type': 'int', 'min': 60, 'max': 86400, 'desc': '等待结果审查的最长时间'},
            'poll_interval': {'label': '轮询间隔(秒)', 'type': 'int', 'min': 1, 'max': 60, 'desc': '前端轮询审批状态的间隔'},
        }
    },
    'memory.task_memory': {
        'label': '任务记忆',
        'fields': {
            'enabled': {'label': '启用', 'type': 'bool', 'desc': '是否启用任务记忆'},
            'top_k': {'label': 'Top-K 检索', 'type': 'int', 'min': 1, 'max': 20, 'desc': '检索相似历史记录的数量'},
            'alpha': {'label': '混合检索权重', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'desc': '关键词vs语义检索的权重混合比'},
            'include_details': {'label': '包含详情', 'type': 'bool', 'desc': '检索结果是否包含详细内容'},
            'embedding_mode': {'label': '嵌入模式', 'type': 'select', 'options': ['title', 'description', 'method', 'full'], 'desc': '用于检索的文本嵌入模式'},
        }
    },
    'scholar': {
        'label': '学者搜索',
        'fields': {
            'search_depth': {'label': '搜索深度', 'type': 'select', 'options': ['light', 'moderate', 'deep'], 'desc': '学术文献搜索的深度'},
        }
    },
    'survey': {
        'label': '文献调研',
        'fields': {
            'max_papers': {'label': '最大论文数', 'type': 'int', 'min': 5, 'max': 200, 'desc': '文献调研时检索的最大论文数量'},
        }
    },
}


def _get_nested_key(d, key_path):
    """通过点分隔的路径获取嵌套字典值"""
    keys = key_path.split('.')
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, {})
        else:
            return None
    return val


def _set_nested_key(d, key_path, value):
    """通过点分隔的路径设置嵌套字典值"""
    keys = key_path.split('.')
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    # 类型转换
    raw = d.get(keys[-1])
    if isinstance(raw, bool):
        value = value in (True, 'true', 'True', 1, '1')
    elif isinstance(raw, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            pass
    elif isinstance(raw, float):
        try:
            value = float(value)
        except (ValueError, TypeError):
            pass
    d[keys[-1]] = value


def _flatten_config(cfg, prefix=''):
    """将嵌套配置展平为点分隔键值对"""
    result = {}
    for key, val in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(_flatten_config(val, path))
        else:
            result[path] = val
    return result


def _unflatten_config(flat):
    """将展平的键值对恢复为嵌套字典"""
    result = {}
    for path, val in flat.items():
        _set_nested_key(result, path, val)
    return result


@app.route('/get_config', methods=['GET'])
def get_config():
    """读取当前配置文件"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
        else:
            raw_config = {}

        # 展平并构建前端友好的数据
        flat = _flatten_config(raw_config)

        # 只返回在 schema 中定义的字段
        config_data = {}
        for schema_key, schema_group in CONFIG_SCHEMA.items():
            for field_key, field_meta in schema_group['fields'].items():
                full_key = f"{schema_key}.{field_key}" if '.' in schema_key else f"{schema_key}.{field_key}"
                # 对于像 criteria.novelty 这样的三级键
                if '.' in field_key:
                    full_key = f"{schema_key}.{field_key}"
                else:
                    full_key = f"{schema_key}.{field_key}"

                # 尝试获取值
                val = _get_nested_key(raw_config, full_key)
                if val is not None:
                    config_data[full_key] = val

        return jsonify({
            'success': True,
            'config': config_data,
            'schema': CONFIG_SCHEMA,
            'file_path': str(CONFIG_PATH)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'读取配置失败: {str(e)}'
        })


@app.route('/save_config', methods=['POST'])
def save_config():
    """保存配置到文件"""
    try:
        data = request.json
        updates = data.get('updates', {})

        if not CONFIG_PATH.exists():
            return jsonify({'success': False, 'error': '配置文件不存在'})

        # 读取当前配置
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        # 应用更新
        for full_key, value in updates.items():
            _set_nested_key(config, full_key, value)

        # 写回文件
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return jsonify({
            'success': True,
            'message': '配置保存成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'保存配置失败: {str(e)}'
        })


@app.route('/polish_paper', methods=['POST'])
def polish_paper():
    """使用LLM润色论文原始Markdown文本（分段处理，避免截断）"""
    data = request.json
    task_id = data.get('task_id')

    if not task_id or task_id not in processing_results:
        return jsonify({'success': False, 'error': '任务不存在'})

    try:
        result_data = processing_results[task_id]['result']
        raw_md = result_data.get('raw_markdown', '')

        if not raw_md:
            return jsonify({'success': False, 'error': '没有找到原始论文内容'})

        # 检查缓存
        cached = processing_results[task_id].get('polished_markdown')
        if cached:
            return jsonify({
                'success': True,
                'polished': cached,
                'cached': True
            })

        # 直接使用底层API调用，跳过_optimize_prompt()的截断
        # 使用更大的输出窗口
        llm_config = pipeline.llm_client.config
        api_key = llm_config.get('api_key', '')
        base_url = llm_config.get('base_url', 'https://api.deepseek.com')
        model = llm_config.get('model', 'deepseek-chat')

        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        system_prompt = """你是一个专业的学术论文排版与润色助手。你的任务是将论文的原始Markdown文本优化为排版精美的研究文档。

要求：
1. **保留所有学术内容**：不能删减任何研究信息、数据、公式、引用，必须完整保留原文
2. **优化Markdown排版**：
   - 使用合适的标题层级（# 标题 → ## 小节 → ### 子节）
   - 用 `---` 分隔明显不同的章节
   - 为表格添加对齐格式
   - 使用 `> 引用` 块突出重要结论
3. **美化代码和公式**：
   - 代码块使用 ```language 标记
   - 数学公式保持原始格式
4. **保留原文语言**：不翻译任何内容，保持原始语言
5. **高亮关键信息**：用 **粗体** 标记重要的术语、数据、结论
6. **完整输出**：必须输出全文，不允许省略任何部分"""

        # 长论文：必须按固定大小分块处理，确保每块输出完整
        # DeepSeek 服务端 max_tokens 上限约 8192，每块输入控制在 2500 字符以内
        # 这样每块输出约 6000-8192 tokens，足够完整输出
        CHUNK_CHARS = 2500
        MAX_TOKENS = 16384

        if len(raw_md) <= CHUNK_CHARS:
            # 短论文：直接一次润色
            prompt = f"请将以下论文原始Markdown内容润色为排版精美的学术文档：\n\n{raw_md}"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=MAX_TOKENS
            )
            polished = response.choices[0].message.content

        else:
            # 按固定字符数分块（每块 3000 字符，保证输出不截断）
            chunks = []
            start = 0
            while start < len(raw_md):
                end = min(start + CHUNK_CHARS, len(raw_md))
                # 如果不是最后一块，在段落边界截断
                if end < len(raw_md):
                    # 往回找最近的换行
                    newline_pos = raw_md.rfind('\n', start, end)
                    if newline_pos > start + CHUNK_CHARS // 2:
                        end = newline_pos + 1
                chunks.append(raw_md[start:end])
                start = end

            polished_parts = []
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                chunk_prompt = f"这是论文的第 {i+1}/{total_chunks} 部分，请润色为排版精美的学术文档。只输出润色后的内容，不要标注\"第X部分\"。\n\n{chunk}"
                chunk_system = system_prompt + f"\n\n这是论文的第 {i+1}/{total_chunks} 部分，请完整输出此部分的所有内容，不要省略。"

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": chunk_system},
                        {"role": "user", "content": chunk_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=MAX_TOKENS
                )
                polished_parts.append(response.choices[0].message.content)

            polished = "\n\n---\n\n".join(polished_parts)

        # 缓存结果
        processing_results[task_id]['polished_markdown'] = polished

        return jsonify({
            'success': True,
            'polished': polished,
            'cached': False
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'润色失败: {str(e)}'
        })


# ============================================================================
# 流水线人工审批 API
# ============================================================================


def _find_latest_launch_dir(task_path):
    """从任务路径自动查找最新的流水线启动目录"""
    project_root = Path(__file__).parent.parent.parent
    task_name = Path(task_path).name
    results_dir = project_root / 'results' / task_name
    if not results_dir.exists():
        return ''
    launches = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith('_launch')], reverse=True)
    return str(launches[0]) if launches else ''


@app.route('/pipeline/pending_ideas', methods=['POST'])
def api_pending_ideas():
    """获取待审批的想法列表"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    # 支持通过 task_path 自动查找
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    pending = pipeline_hooks.read_pending_ideas(launch_dir)

    # 检查是否有待审批的想法
    if not pending or pending.get('status') != pipeline_hooks.STATUS_WAITING:
        return jsonify({
            'success': True,
            'status': 'none',
            'ideas': [],
            'message': '当前没有待审批的想法'
        })

    return jsonify({
        'success': True,
        'status': pipeline_hooks.STATUS_WAITING,
        'session_id': pending.get('session_id'),
        'total_count': pending.get('total_count', 0),
        'ideas': pending.get('ideas', []),
        'created_at': pending.get('created_at', 0),
    })


@app.route('/pipeline/approve_ideas', methods=['POST'])
def api_approve_ideas():
    """审批想法：选择/修改/否决"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    # 支持通过 task_path 自动查找
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))
    action = data.get('action', 'approve')  # approve / reject
    selected_ids = data.get('selected_ids')  # List[str] 或 None
    modifications = data.get('modifications')  # Dict[str, Dict] 或 None

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    try:
        if action == 'reject':
            result = pipeline_hooks.reject_ideas(
                launch_dir,
                reason=data.get('reason', '用户否决')
            )
            return jsonify({
                'success': True,
                'action': 'rejected',
                'result': result
            })

        # approve
        result = pipeline_hooks.approve_ideas(
            launch_dir,
            selected_ids=selected_ids,
            modifications=modifications
        )

        if result.get('status') == 'error':
            return jsonify({'success': False, 'error': result.get('message', '审批失败')})

        return jsonify({
            'success': True,
            'action': 'approved',
            'selected_count': result.get('selected_count', 0),
            'total_count': result.get('total_count', 0),
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'审批操作失败: {str(e)}'
        })


@app.route('/pipeline/status', methods=['POST'])
def api_pipeline_status():
    """获取流水线审批状态（与原有的 pipeline_status 区分）"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    status = pipeline_hooks.get_pipeline_status(launch_dir)

    return jsonify({
        'success': True,
        'status': status.get('status', 'unknown'),
        'reviews': status.get('reviews', {}),
        'message': status.get('message', ''),
    })


@app.route('/pipeline/result_review', methods=['POST'])
def api_result_review():
    """获取实验结果审查信息"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    result_index = data.get('result_index', -1)
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    review_dir = pipeline_hooks.get_review_dir(launch_dir)
    result_file = os.path.join(review_dir, pipeline_hooks.RESULT_REVIEW_FILE)

    state = pipeline_hooks.read_state(result_file)
    if not state:
        return jsonify({'success': True, 'has_results': False, 'results': []})

    results = state.get('results', [])

    # 如果指定了索引，只返回单个结果
    if result_index >= 0 and result_index < len(results):
        return jsonify({
            'success': True,
            'has_results': True,
            'result': results[result_index],
            'total': len(results),
            'current_index': result_index,
        })

    return jsonify({
        'success': True,
        'has_results': len(results) > 0,
        'results': results,
        'total': len(results),
    })


@app.route('/pipeline/result_feedback', methods=['POST'])
def api_result_feedback():
    """提交实验结果反馈"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    feedback = data.get('feedback', {})
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    try:
        pipeline_hooks.submit_result_feedback(launch_dir, feedback)
        return jsonify({
            'success': True,
            'message': '反馈已提交'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'提交反馈失败: {str(e)}'
        })


@app.route('/pipeline/list_launches', methods=['POST'])
def api_list_launches():
    """列出指定任务的所有流水线启动目录（用于选择审批准入点）"""
    data = request.json
    task_name = data.get('task_name', '')

    if not task_name:
        return jsonify({'success': False, 'error': '缺少任务名称'})

    results_base = os.path.join(
        Path(__file__).parent.parent.parent,
        'results',
        task_name
    )

    if not os.path.exists(results_base):
        return jsonify({'success': True, 'launches': []})

    launches = []
    for entry in sorted(os.listdir(results_base)):
        launch_dir = os.path.join(results_base, entry)
        if os.path.isdir(launch_dir) and entry.endswith('_launch'):
            review_dir = pipeline_hooks.get_review_dir(launch_dir)
            status = pipeline_hooks.get_pipeline_status(launch_dir)
            launches.append({
                'launch_id': entry,
                'path': launch_dir,
                'has_human_review': os.path.exists(review_dir),
                'status': status.get('status', 'unknown'),
            })

    return jsonify({
        'success': True,
        'launches': launches,
    })


@app.route('/pipeline/list_experiment_results', methods=['POST'])
def api_list_experiment_results():
    """列出某个启动目录下所有实验的结果（用于结果审查 Dashboard）"""
    data = request.json
    launch_dir = data.get('launch_dir', '')
    if not launch_dir or not os.path.exists(launch_dir):
        launch_dir = _find_latest_launch_dir(data.get('task_path', ''))

    if not launch_dir or not os.path.exists(launch_dir):
        return jsonify({'success': False, 'error': '无效的启动目录'})

    experiments = []

    # 扫描 launch_dir 下的 session_* 目录
    for entry in sorted(os.listdir(launch_dir)):
        entry_path = os.path.join(launch_dir, entry)
        if not os.path.isdir(entry_path) or not entry.startswith('session_'):
            continue

        # 查找实验运行目录
        run_dirs = sorted([
            d for d in os.listdir(entry_path)
            if d.startswith('run_') and os.path.isdir(os.path.join(entry_path, d))
        ])

        for run_dir_name in run_dirs:
            run_path = os.path.join(entry_path, run_dir_name)
            final_info_path = os.path.join(run_path, 'final_info.json')
            report_path = os.path.join(run_path, 'report', 'report.md')

            if not os.path.exists(final_info_path):
                continue

            try:
                with open(final_info_path, 'r') as f:
                    final_info = json.load(f)

                scores = final_info.get('sci_task', {}).get('means', {})

                # 尝试获取 idea 名称
                idea_name = entry  # fallback: session name
                ideas_file = os.path.join(launch_dir, 'run_0', 'final_info.json')
                if os.path.exists(ideas_file):
                    pass  # 保持现状

                report_preview = ''
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_text = f.read()
                    report_preview = report_text[:500]

                # 查找图片
                images_dir = os.path.join(run_path, 'report', 'images')
                images = []
                if os.path.exists(images_dir):
                    for img in sorted(os.listdir(images_dir))[:5]:
                        img_path = os.path.join(images_dir, img)
                        if os.path.isfile(img_path):
                            images.append(img)

                experiments.append({
                    'session_id': entry,
                    'run_id': run_dir_name,
                    'path': run_path,
                    'scores': scores,
                    'has_report': os.path.exists(report_path),
                    'report_preview': report_preview,
                    'images': images,
                    'has_final_info': True,
                })
            except Exception as e:
                experiments.append({
                    'session_id': entry,
                    'run_id': run_dir_name,
                    'path': run_path,
                    'error': str(e),
                })

    return jsonify({
        'success': True,
        'experiments': experiments,
        'total': len(experiments),
    })


@app.route('/pipeline/list_results_dir', methods=['POST'])
def api_list_results_dir():
    """列出 results/ 目录结构"""
    data = request.json
    task_name = data.get('task_name', '')

    results_base = os.path.join(
        Path(__file__).parent.parent.parent,
        'results'
    )

    if not os.path.exists(results_base):
        return jsonify({'success': False, 'error': 'results 目录不存在'})

    # 如果指定了 task_name，列出该任务的结构
    if task_name:
        task_dir = os.path.join(results_base, task_name)
        if not os.path.exists(task_dir):
            return jsonify({'success': False, 'error': f'任务 {task_name} 不存在'})
        tree = _build_dir_tree(task_dir, max_depth=6, max_files=200)
        return jsonify({'success': True, 'task_name': task_name, 'tree': tree})

    # 否则列出所有任务
    tasks = []
    for entry in sorted(os.listdir(results_base)):
        entry_path = os.path.join(results_base, entry)
        if entry.startswith('.'):
            continue
        size = _get_dir_size(entry_path) if os.path.isdir(entry_path) else os.path.getsize(entry_path)
        modified = os.path.getmtime(entry_path)
        tasks.append({
            'name': entry,
            'is_dir': os.path.isdir(entry_path),
            'size': size,
            'size_display': _format_size(size),
            'modified': modified,
            'modified_display': time_module.strftime('%Y-%m-%d %H:%M', time_module.localtime(modified)),
        })

    return jsonify({'success': True, 'tasks': tasks})


@app.route('/pipeline/read_result_file', methods=['POST'])
def api_read_result_file():
    """读取 results/ 下的某个文件内容"""
    data = request.json
    file_path = data.get('file_path', '')

    if not file_path:
        return jsonify({'success': False, 'error': '文件路径不能为空'})

    # 安全检查：确保文件在 results 目录下
    results_base = os.path.join(Path(__file__).parent.parent.parent, 'results')
    real_path = os.path.realpath(file_path)
    real_base = os.path.realpath(results_base)
    if not real_path.startswith(real_base):
        return jsonify({'success': False, 'error': '不允许访问 results 目录以外的文件'})

    if not os.path.exists(real_path) or not os.path.isfile(real_path):
        return jsonify({'success': False, 'error': '文件不存在'})

    try:
        size = os.path.getsize(real_path)
        # 限制读取大小（5MB）
        MAX_SIZE = 5 * 1024 * 1024
        if size > MAX_SIZE:
            return jsonify({
                'success': False,
                'error': f'文件过大 ({_format_size(size)})，不支持在线预览',
                'too_large': True,
                'size': size,
                'size_display': _format_size(size)
            })

        ext = os.path.splitext(real_path)[1].lower()
        text_exts = {'.json', '.md', '.txt', '.py', '.yaml', '.yml', '.cfg', '.ini',
                     '.log', '.csv', '.tsv', '.html', '.css', '.js', '.sh', '.toml',
                     '.xml', '.tex', '.rst', '.env', '.gitignore'}

        if ext in text_exts:
            with open(real_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return jsonify({
                'success': True,
                'content': content,
                'type': 'text',
                'size': size,
                'size_display': _format_size(size),
                'filename': os.path.basename(real_path),
            })
        else:
            return jsonify({
                'success': True,
                'content': None,
                'type': 'binary',
                'size': size,
                'size_display': _format_size(size),
                'filename': os.path.basename(real_path),
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'读取文件失败: {str(e)}'
        })


def _build_dir_tree(path, max_depth=4, max_files=100, current_depth=0):
    """构建目录树（递归）"""
    if current_depth > max_depth:
        return {'name': os.path.basename(path), 'type': 'truncated', 'children': []}

    entries = []
    try:
        for entry in sorted(os.listdir(path)):
            if entry.startswith('.') or entry.startswith('__pycache__'):
                continue
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                children = _build_dir_tree(full_path, max_depth, max_files, current_depth + 1)
                if children is not None:
                    entries.append({
                        'name': entry,
                        'type': 'dir',
                        'path': full_path,
                        'children': children.get('children', []),
                        'size': children.get('size', 0),
                    })
            else:
                size = os.path.getsize(full_path)
                entries.append({
                    'name': entry,
                    'type': 'file',
                    'path': full_path,
                    'size': size,
                    'size_display': _format_size(size),
                })
    except PermissionError:
        return None

    # 限制条目数
    if len(entries) > max_files:
        entries = entries[:max_files]

    total_size = sum(e.get('size', 0) for e in entries)
    return {
        'name': os.path.basename(path),
        'type': 'dir',
        'path': path,
        'children': entries,
        'size': total_size,
        'size_display': _format_size(total_size),
    }


def _get_dir_size(path):
    """计算目录总大小"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except (PermissionError, OSError):
        pass
    return total


def _format_size(size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@app.route('/pipeline/chat', methods=['POST'])
def api_pipeline_chat():
    """想法审批中的AI对话聊天"""
    data = request.json
    message = data.get('message', '')
    context = data.get('context', '')

    if not message:
        return jsonify({'success': False, 'error': '消息不能为空'})

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=API_KEY,
            base_url='https://api.deepseek.com'
        )

        system_prompt = '你是一个科研助手，帮助研究人员讨论和完善实验想法。'
        if context:
            system_prompt += f'\n当前正在讨论的想法：\n{context}\n请根据用户的问题提供具体的修改建议，帮助完善这个想法。'

        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': message}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        reply = response.choices[0].message.content
        return jsonify({'success': True, 'reply': reply})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/pipeline/list_files', methods=['POST'])
def api_list_files():
    """列出结果目录下的所有文件"""
    data = request.json
    path = data.get('path', '')

    if not path or not os.path.exists(path):
        # Try to find results dir relative to project root
        project_root = Path(__file__).parent.parent.parent
        results_dir = project_root / 'results'
        if results_dir.exists():
            path = str(results_dir)
        else:
            return jsonify({'success': False, 'error': '路径不存在'})

    try:
        path_obj = Path(path).resolve()
        items = []
        for item in sorted(path_obj.iterdir()):
            items.append({
                'name': item.name,
                'path': str(item),
                'is_dir': item.is_dir(),
                'size': item.stat().st_size if item.is_file() else 0,
                'mtime': item.stat().st_mtime
            })
        return jsonify({
            'success': True,
            'current_path': str(path_obj),
            'items': items,
            'parent_path': str(path_obj.parent) if str(path_obj.parent) != str(path_obj) else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/pipeline/read_file', methods=['POST'])
def api_read_file():
    """读取文件内容"""
    data = request.json
    file_path = data.get('path', '')

    if not file_path or not os.path.exists(file_path) or os.path.isdir(file_path):
        return jsonify({'success': False, 'error': '文件不存在'})

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content, 'path': file_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/pipeline/save_file', methods=['POST'])
def api_save_file():
    """保存文件内容"""
    data = request.json
    file_path = data.get('path', '')
    content = data.get('content', '')

    if not file_path:
        return jsonify({'success': False, 'error': '路径不能为空'})

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'message': '保存成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



if __name__ == '__main__':
    print("=" * 60)
    print("🌐 Paper-to-Task Web应用启动")
    print("=" * 60)
    print("📱 访问地址: http://localhost:5000")
    print("📄 上传您的PDF文件开始体验")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)

