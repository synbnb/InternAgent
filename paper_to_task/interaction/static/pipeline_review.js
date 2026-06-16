/**
 * 流水线人工审批前端控制模块
 *
 * 在流水线标签页中动态添加子标签页：
 *   - 💡 想法审批 (Idea Review)
 *   - 🔬 结果审查 (Result Review)
 *
 * 设计原则：不修改 index.html 核心结构，通过 JS 动态注入 UI
 */

const PipelineReview = (() => {
  // ======================================================================
  // 状态
  // ======================================================================
  let state = {
    currentLaunchDir: '',
    ideas: [],
    selectedIdeas: new Set(),
    results: [],
    pollTimer: null,
    isPolling: false,
  };

  // ======================================================================
  // 注入 UI
  // ======================================================================
  // 根据流水线UUID获取对应的启动目录
  async function resolveLaunchDirFromPipeline() {
    const pipelineTaskPath = document.getElementById('pipelineTaskPath');
    if (!pipelineTaskPath) return '';

    const val = pipelineTaskPath.value;
    if (!val) return '';

    const taskName = val.split('/').pop() || val.split('\\').pop();
    if (!taskName) return '';

    try {
      const resp = await fetch('/pipeline/list_launches', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_name: taskName }),
      });
      const data = await resp.json();
      if (!data.success || !data.launches.length) return '';

      // 优先选 waiting_approval 状态的启动目录，其次是最新的
      const waiting = data.launches.find(l => l.status === 'waiting_approval');
      if (waiting) return waiting.path;
      return data.launches[data.launches.length - 1].path;
    } catch (e) {
      console.error('Failed to resolve launch dir:', e);
      return '';
    }
  }

  // 更新连接状态
  async function updateConnectionStatus() {
    const statusEl = document.getElementById('reviewGlobalStatus');
    const statusText = document.getElementById('reviewConnectionText');
    if (!statusEl) return;

    const launchDir = await resolveLaunchDirFromPipeline();
    if (launchDir) {
      state.currentLaunchDir = launchDir;
      statusEl.textContent = '已连接';
      statusEl.style.background = '#4caf50';
      if (statusText) statusText.textContent = '已关联流水线：' + launchDir.split('/').pop();
      return true;
    } else {
      // 检查当前是否有运行中的流水线
      const pipelineStatusBadge = document.getElementById('pipelineStatusBadge');
      if (pipelineStatusBadge && pipelineStatusBadge.textContent.includes('运行')) {
        statusEl.textContent = '已连接';
        statusEl.style.background = '#4caf50';
        if (statusText) statusText.textContent = '流水线运行中';
        return true;
      }
      statusEl.textContent = '未连接';
      statusEl.style.background = '#666';
      if (statusText) statusText.textContent = '请先启动流水线';
      return false;
    }
  }

  function injectUI() {
    let pipelineTab = document.getElementById('pipeline');
    if (!pipelineTab) pipelineTab = document.getElementById('page-pipeline');
    if (!pipelineTab) return;

    if (document.getElementById('pipelineReviewNav')) return;

    const container = document.createElement('div');
    container.id = 'pipelineReviewContainer';
    container.style.cssText = 'margin-top: 15px; border-top: 2px solid var(--border-light); padding-top: 15px;';

    container.innerHTML = `
      <div class="section-title" style="margin-bottom: 10px; font-size: 14px;">
        🤝 人工审批
        <span id="reviewGlobalStatus" style="font-size: 11px; padding: 2px 8px; border-radius: 8px; background: #666; color: #fff; margin-left: 8px;">未连接</span>
        <span id="reviewConnectionText" style="font-size: 11px; color: var(--text-tertiary); margin-left: 8px;"></span>
      </div>

      <!-- 子标签导航 (统一) -->
      <div class="tab-nav" id="pipelineReviewNav" style="display: flex; gap: 4px; margin-bottom: 12px; border-bottom: 2px solid var(--border-light); padding-bottom: 0;">
        <button class="tab-btn review-tab-btn active" data-review-tab="idea" style="padding: 6px 14px; font-size: 12px;">💡 想法审批</button>
        <button class="tab-btn review-tab-btn" data-review-tab="result" style="padding: 6px 14px; font-size: 12px;">🔬 结果审查</button>
        <button class="tab-btn review-tab-btn" data-review-tab="comparison" style="padding: 6px 14px; font-size: 12px;">📊 对比看板</button>
        <button class="tab-btn review-tab-btn" data-review-tab="browser" style="padding: 6px 14px; font-size: 12px;">📁 结果目录</button>
        <div style="flex: 1;"></div>
        <button id="reviewRefreshBtn" style="padding: 4px 10px; font-size: 11px; background: none; border: 1px solid var(--border-light); border-radius: 6px; cursor: pointer;"> 刷新</button>
      </div>

      <!-- 💡 想法审批面板 -->
      <div class="review-tab-content" id="reviewTabIdea" style="display: block;">
        <div id="ideaReviewStatus" style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">
          等待流水线运行到想法生成阶段。
        </div>
        <div id="ideaReviewCards" style="display: flex; flex-direction: column; gap: 10px; max-height: 600px; overflow-y: auto;"></div>
        <div class="edit-controls" id="ideaReviewActions" style="margin-top: 12px; display: none; justify-content: space-between;">
          <div style="font-size: 12px; color: var(--text-tertiary);">
            已选: <span id="ideaSelCount">0</span> / <span id="ideaTotalCount">0</span>
          </div>
          <div style="display: flex; gap: 8px;">
            <button id="ideaRejectBtn" class="btn" style="width: auto; padding: 6px 16px; background: #f44336; font-size: 12px;"> 否决</button>
            <button id="ideaApproveBtn" class="btn" style="width: auto; padding: 6px 16px; font-size: 12px;"> 确认选择</button>
          </div>
        </div>
      </div>

      <!-- 🔬 结果审查面板 -->
      <div class="review-tab-content" id="reviewTabResult" style="display: none;">
        <div id="resultReviewStatus" style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">
          等待流水线运行到实验阶段。
        </div>
        <div id="resultReviewCards" style="display: flex; flex-direction: column; gap: 10px; max-height: 600px; overflow-y: auto;"></div>
        <div class="edit-controls" id="resultReviewActions" style="margin-top: 12px; display: none;">
          <button id="resultApproveBtn" class="btn" style="width: auto; padding: 6px 16px; font-size: 12px;"> 确认结果并继续</button>
          <button id="resultRefreshBtn" class="btn" style="width: auto; padding: 6px 16px; background: var(--text-secondary); font-size: 12px;"> 刷新结果</button>
        </div>
      </div>

      <!-- 📊 对比看板面板 -->
      <div class="review-tab-content" id="reviewTabComparison" style="display: none;">
        <div id="comparisonStatus" style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">
          查看各候选方向的实验评分对比，选择最佳方向继续迭代。
        </div>
        <div id="comparisonTableContainer"></div>
        <div id="comparisonActions" style="display: none; margin-top: 12px; justify-content: center; gap: 12px;">
          <button id="selectBestBtn" class="btn" style="width: auto; padding: 10px 28px;"> 确认最佳方向</button>
          <button id="rejectAllComparisonBtn" class="btn" style="width: auto; padding: 10px 28px; background: var(--text-secondary);"> 继续查看</button>
          <span id="comparisonFeedback" style="font-size: 12px; color: var(--text-tertiary); align-self: center;"></span>
        </div>
      </div>

      <!-- 📁 结果目录面板 -->
      <div class="review-tab-content" id="reviewTabBrowser" style="display: none;">
        <div id="browserStatus" style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">
          浏览 results/ 目录下的任务输出文件和目录结构。
        </div>
        <div id="browserTaskSelector" style="margin-bottom: 12px;">
          <select id="browserTaskSelect" class="feedback-input" style="min-height: auto; font-size: 12px; margin-bottom: 0;">
            <option value="">-- 选择任务 --</option>
          </select>
        </div>
        <div id="browserContainer" style="display: flex; gap: 12px; min-height: 300px;">
          <div id="browserTreePanel" style="flex: 1; min-width: 0; background: var(--bg-canvas); border-radius: var(--radius-md); padding: 10px; border: 1px solid var(--border-light); overflow-y: auto; max-height: 520px; font-size: 12px;"></div>
          <div id="browserPreviewPanel" style="flex: 2; min-width: 0; background: var(--bg-canvas); border-radius: var(--radius-md); padding: 12px 16px; border: 1px solid var(--border-light); overflow-y: auto; max-height: 520px; font-family: var(--font-mono); font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; color: var(--text-primary);">
            <div style="text-align: center; color: var(--text-tertiary); padding: 40px 0;">选择左侧文件查看内容</div>
          </div>
        </div>
      </div>
    `;

    pipelineTab.appendChild(container);
    bindEvents();
  }

  // ======================================================================
  // 事件绑定
  // ======================================================================
  function bindEvents() {
    // 子标签切换 — 切换时自动刷新内容
    document.querySelectorAll('.review-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.review-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.review-tab-content').forEach(c => c.style.display = 'none');
        btn.classList.add('active');
        const tab = btn.dataset.reviewTab;
        const content = document.getElementById(`reviewTab${tab.charAt(0).toUpperCase() + tab.slice(1)}`);
        if (content) content.style.display = 'block';
        // 切换子标签时刷新内容
        refreshCurrentTab();
      });
    });

    // 刷新按钮
    document.getElementById('reviewRefreshBtn')?.addEventListener('click', () => {
      refreshCurrentTab();
    });

    // 结果目录任务选择变化
    document.getElementById('browserTaskSelect')?.addEventListener('change', function() {
      if (this.value) loadBrowserTree(this.value);
      else document.getElementById('browserTreePanel').innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 20px;">请选择一个任务</div>';
    });

    // 监听流水线任务选择下拉框变化
    document.getElementById('pipelineTaskPath')?.addEventListener('change', () => {
      updateConnectionStatus().then(connected => {
        if (connected) refreshCurrentTab();
      });
    });

    // 想法审批
    document.getElementById('ideaApproveBtn')?.addEventListener('click', approveIdeas);
    document.getElementById('ideaRejectBtn')?.addEventListener('click', rejectIdeas);

    // 结果审查
    document.getElementById('resultApproveBtn')?.addEventListener('click', approveResults);
    document.getElementById('resultRefreshBtn')?.addEventListener('click', () => {
      if (state.currentLaunchDir) loadExperimentResults(state.currentLaunchDir);
    });

    // 对比看板按钮
    document.getElementById('selectBestBtn')?.addEventListener('click', selectBestDirection);
    document.getElementById('rejectAllComparisonBtn')?.addEventListener('click', () => {
      document.getElementById('comparisonActions').style.display = 'none';
      document.getElementById('comparisonStatus').textContent = ' 已跳过选择，继续迭代。';
    });
  }

  // ======================================================================
  // 想法审批
  // ======================================================================
  async function loadPendingIdeas(launchDir) {
    const statusDiv = document.getElementById('ideaReviewStatus');
    const cardsDiv = document.getElementById('ideaReviewCards');
    const actionsDiv = document.getElementById('ideaReviewActions');

    try {
      const resp = await fetch('/pipeline/pending_ideas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ launch_dir: launchDir }),
      });
      const data = await resp.json();
      if (!data.success) {
        statusDiv.textContent = `加载失败: ${data.error}`;
        cardsDiv.innerHTML = '';
        actionsDiv.style.display = 'none';
        return;
      }

      if (data.status !== 'waiting' || !data.ideas.length) {
        statusDiv.textContent = '当前没有待审批的想法。';
        cardsDiv.innerHTML = '';
        actionsDiv.style.display = 'none';
        return;
      }

      statusDiv.textContent = `共生成 ${data.total_count} 个候选方向，请选择 1-3 个进行实验。`;
      state.ideas = data.ideas;
      state.selectedIdeas = new Set();
      state.selectedIdeas.add(data.ideas[0]?.id); // 默认选第一个

      renderIdeaCards(data.ideas);
      actionsDiv.style.display = 'flex';
      updateIdeaCounts();
    } catch (e) {
      statusDiv.textContent = `加载失败: ${e.message}`;
      cardsDiv.innerHTML = '';
    }
  }

  function renderIdeaCards(ideas) {
    const container = document.getElementById('ideaReviewCards');
    container.innerHTML = ideas.map((idea, idx) => {
      const isSelected = state.selectedIdeas.has(idea.id);
      const title = idea.title || idea.description?.slice(0, 80) || `候选 ${idx + 1}`;
      const method = idea.method || '暂无方法描述';
      const score = idea.score || 0;
      const rationale = idea.rationale || '';

      return `
        <div class="idea-card" data-id="${idea.id}"
             style="border: 2px solid ${isSelected ? '#667eea' : 'var(--border-light)'};
                    border-radius: 10px; padding: 12px; cursor: pointer;
                    background: ${isSelected ? 'rgba(102,126,234,0.08)' : 'var(--bg-surface)'};
                    transition: all 0.2s;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
              <div style="font-weight: bold; font-size: 13px; margin-bottom: 4px;">
                ${isSelected ? '✅' : '⬜'} ${escapeHtml(title)}
              </div>
              <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; line-height: 1.5;">
                ${escapeHtml(method.slice(0, 200))}${method.length > 200 ? '...' : ''}
              </div>
              ${rationale ? `<div style="font-size: 10px; color: var(--text-tertiary); font-style: italic;">📌 ${escapeHtml(rationale.slice(0, 100))}</div>` : ''}
            </div>
            <div style="text-align: right; flex-shrink: 0; margin-left: 10px;">
              <div style="font-size: 18px; font-weight: bold; color: #667eea;">${(score * 10).toFixed(0)}</div>
              <div style="font-size: 9px; color: var(--text-tertiary);">分数</div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    // 绑定点击事件
    container.querySelectorAll('.idea-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.dataset.id;
        toggleIdeaSelection(id);
      });
    });
  }

  function toggleIdeaSelection(id) {
    if (state.selectedIdeas.has(id)) {
      state.selectedIdeas.delete(id);
    } else {
      if (state.selectedIdeas.size >= 3) {
        showToast('最多选择 3 个方向', 'warning');
        return;
      }
      state.selectedIdeas.add(id);
    }
    renderIdeaCards(state.ideas);
    updateIdeaCounts();
  }

  function updateIdeaCounts() {
    document.getElementById('ideaSelCount').textContent = state.selectedIdeas.size;
    document.getElementById('ideaTotalCount').textContent = state.ideas.length;
  }

  async function approveIdeas() {
    if (!state.selectedIdeas.size) {
      showToast('请至少选择一个方向', 'error');
      return;
    }

    try {
      const resp = await fetch('/pipeline/approve_ideas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          launch_dir: state.currentLaunchDir,
          action: 'approve',
          selected_ids: Array.from(state.selectedIdeas),
        }),
      });
      const data = await resp.json();
      if (data.success) {
        showToast(`已批准 ${data.selected_count} 个方向，流水线继续执行`, 'success');
        document.getElementById('ideaReviewActions').style.display = 'none';
        document.getElementById('ideaReviewCards').innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 20px;">✅ 已审批，等待流水线继续执行...</div>';
      } else {
        showToast(`审批失败: ${data.error}`, 'error');
      }
    } catch (e) {
      showToast(`审批失败: ${e.message}`, 'error');
    }
  }

  async function rejectIdeas() {
    if (!confirm('确定否决所有候选方向？流水线将使用系统默认排名继续。')) return;

    try {
      const resp = await fetch('/pipeline/approve_ideas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          launch_dir: state.currentLaunchDir,
          action: 'reject',
          reason: '用户否决所有方向',
        }),
      });
      const data = await resp.json();
      if (data.success) {
        showToast('已否决，使用系统默认', 'info');
        document.getElementById('ideaReviewActions').style.display = 'none';
        document.getElementById('ideaReviewCards').innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 20px;">⛔ 已否决</div>';
      }
    } catch (e) {
      showToast(`操作失败: ${e.message}`, 'error');
    }
  }

  // ======================================================================
  // 结果审查
  // ======================================================================
  async function loadExperimentResults(launchDir) {
    const statusDiv = document.getElementById('resultReviewStatus');
    const cardsDiv = document.getElementById('resultReviewCards');
    const actionsDiv = document.getElementById('resultReviewActions');

    try {
      const resp = await fetch('/pipeline/list_experiment_results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ launch_dir: launchDir }),
      });
      const data = await resp.json();
      if (!data.success) {
        statusDiv.textContent = `加载失败: ${data.error}`;
        cardsDiv.innerHTML = '';
        actionsDiv.style.display = 'none';
        return;
      }

      if (!data.experiments.length) {
        statusDiv.textContent = '暂无实验结果。流水线可能还在运行中或尚未启动实验。';
        cardsDiv.innerHTML = '';
        actionsDiv.style.display = 'none';
        return;
      }

      statusDiv.textContent = `找到 ${data.total} 个实验结果，请审阅。`;
      state.results = data.experiments;
      renderResultCards(data.experiments);
      actionsDiv.style.display = 'flex';

      // 也刷新对比看板
      renderComparisonTable(data.experiments);
    } catch (e) {
      statusDiv.textContent = `加载失败: ${e.message}`;
      cardsDiv.innerHTML = '';
    }
  }

  function renderResultCards(results) {
    const container = document.getElementById('resultReviewCards');
    container.innerHTML = results.map((r, idx) => {
      const scores = r.scores || {};
      const totalScore = scores.total_score || 0;
      const ideaName = r.session_id || `实验 ${idx + 1}`;
      const hasReport = r.has_report ? '✅' : '❌';
      const reportPreview = r.report_preview || '';

      // 提取非 reasoning 的子分数
      const subScores = Object.entries(scores)
        .filter(([k]) => !k.endsWith('_reasoning') && k !== 'total_score')
        .slice(0, 6);

      return `
        <div class="result-card" style="border: 1px solid var(--border-light); border-radius: 10px; padding: 12px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-weight: bold; font-size: 13px;">${escapeHtml(ideaName)}</div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-size: 20px; font-weight: bold; color: ${totalScore >= 70 ? '#4caf50' : totalScore >= 50 ? '#ff9800' : '#f44336'};">${totalScore}</span>
              <span style="font-size: 11px; color: var(--text-tertiary);">报告 ${hasReport}</span>
            </div>
          </div>
          ${subScores.length ? `
          <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px;">
            ${subScores.map(([k, v]) => `
              <span style="font-size: 10px; padding: 2px 6px; background: var(--bg-subtle); border-radius: 4px;">
                ${k.replace('item_', '').replace('_score', '')}: ${v}
              </span>
            `).join('')}
          </div>` : ''}
          ${reportPreview ? `
          <div style="font-size: 10px; color: var(--text-tertiary); background: var(--bg-subtle); padding: 6px 8px; border-radius: 6px; max-height: 60px; overflow: hidden; line-height: 1.5;">
            ${escapeHtml(reportPreview.slice(0, 200))}
          </div>` : ''}
        </div>
      `;
    }).join('');
  }

  // ======================================================================
  // 对比看板
  // ======================================================================
  function renderComparisonTable(results) {
    const container = document.getElementById('comparisonTableContainer');
    if (!results.length) {
      container.innerHTML = '<div style="font-size: 12px; color: var(--text-tertiary);">暂无实验数据。</div>';
      return;
    }

    // 提取所有评分维度
    const allDims = new Set();
    results.forEach(r => {
      Object.keys(r.scores || {}).forEach(k => {
        if (!k.endsWith('_reasoning') && k !== 'total_score') allDims.add(k);
      });
    });
    const dimList = Array.from(allDims).slice(0, 8);
    const sorted = [...results].sort((a, b) => (b.scores?.total_score || 0) - (a.scores?.total_score || 0));

    // 可视化布局：雷达图 + 柱状图 + 表格
    let html = '';

    // === 图表区域 ===
    html += '<div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;">';

    // 雷达图
    if (dimList.length >= 3) {
      html += '<div style="flex: 1; min-width: 320px; max-width: 480px;">';
      html += '<div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;"> 各维度评分雷达图</div>';
      html += '<canvas id="radarChart" width="380" height="340" style="width:100%;max-width:380px;height:auto;"></canvas>';
      html += '</div>';
    }

    // 总分柱状图
    html += '<div style="flex: 1; min-width: 280px; max-width: 400px;">';
    html += '<div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;"> 总分对比</div>';
    html += '<canvas id="barChart" width="360" height="280" style="width:100%;max-width:360px;height:auto;"></canvas>';
    html += '</div>';

    // 趋势图（多轮数据）
    const multiRunResults = results.filter(r => r.run_id && parseInt(r.run_id.replace('run_','')) > 0);
    if (multiRunResults.length >= 2) {
      html += '<div style="flex: 1; min-width: 280px; max-width: 400px;">';
      html += '<div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;"> 多轮趋势</div>';
      html += '<canvas id="trendChart" width="360" height="280" style="width:100%;max-width:360px;height:auto;"></canvas>';
      html += '</div>';
    }

    html += '</div>';

    // === 表格区域 ===
    html += '<div style="overflow-x: auto; font-size: 11px;">';
    html += '<table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr style="background: var(--bg-subtle);">';
    html += '<th style="padding: 8px 10px; text-align: left; border-bottom: 2px solid var(--border-light);">候选方向</th>';
    html += '<th style="padding: 8px 10px; text-align: center; border-bottom: 2px solid var(--border-light);">总分</th>';
    dimList.forEach(d => {
      html += '<th style="padding: 8px 6px; text-align: center; border-bottom: 2px solid var(--border-light); font-size: 10px;">' + d.replace('item_', '').replace('_score', '') + '</th>';
    });
    html += '<th style="padding: 8px 10px; text-align: center; border-bottom: 2px solid var(--border-light);">报告</th>';
    html += '</tr></thead><tbody>';

    sorted.forEach((r, idx) => {
      const total = r.scores?.total_score || 0;
      const isBest = idx === 0;
      html += '<tr style="' + (isBest ? 'background: rgba(76, 175, 80, 0.08);' : idx % 2 === 0 ? 'background: var(--bg-surface);' : '') + '">';
      html += '<td style="padding: 6px 10px; border-bottom: 1px solid var(--border-light);">' + (isBest ? '🏆 ' : '') + escapeHtml(r.session_id || '方向 ' + (idx+1)) + '</td>';
      html += '<td style="padding: 6px 10px; text-align: center; border-bottom: 1px solid var(--border-light); font-weight: bold; color: ' + (total >= 70 ? '#4caf50' : total >= 50 ? '#ff9800' : '#f44336') + ';">' + total + '</td>';
      dimList.forEach(d => {
        const val = r.scores?.[d];
        html += '<td style="padding: 6px 4px; text-align: center; border-bottom: 1px solid var(--border-light); font-size: 10px;">' + (val !== undefined ? val : '-') + '</td>';
      });
      html += '<td style="padding: 6px 10px; text-align: center; border-bottom: 1px solid var(--border-light);">' + (r.has_report ? '✅' : '❌') + '</td>';
      html += '</tr>';
    });

    html += '</tbody></table></div>';
    html += '<div style="margin-top: 8px; font-size: 10px; color: var(--text-tertiary);">🏆 标记表示当前总分最高的候选方向</div>';

    container.innerHTML = html;

    // 绘制图表（需要等 canvas 渲染完成）
    setTimeout(function() {
      drawRadarChart(sorted, dimList);
      drawBarChart(sorted);
      if (multiRunResults.length >= 2) drawTrendChart(multiRunResults);
    }, 100);
  }

    // ======================================================================
  // 对比看板 - 选择最佳方向
  // ======================================================================
  function selectBestDirection() {
    if (!state.results.length) return;

    // 取总分最高的
    const sorted = [...state.results].sort((a, b) => (b.scores?.total_score || 0) - (a.scores?.total_score || 0));
    const best = sorted[0];

    if (!confirm(`确认选择 "${best.session_id || '当前最佳方向'}" (总分: ${best.scores?.total_score || 0}) 作为最佳方向？`)) return;

    const feedbackEl = document.getElementById('comparisonFeedback');
    feedbackEl.textContent = ' 正在提交...';

    // 通过 result_feedback 告知后端选择结果
    fetch('/pipeline/result_feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        launch_dir: state.currentLaunchDir,
        feedback: {
          approved: true,
          selected_session: best.session_id,
          selected_score: best.scores?.total_score || 0,
          submitted_at: Date.now(),
          note: '用户从对比看板选择了最佳方向',
        }
      }),
    })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        feedbackEl.textContent = ' 已确认！';
        document.getElementById('comparisonActions').style.display = 'none';
        showToast('已确认最佳方向: ' + (best.session_id || ''), 'success');
      } else {
        feedbackEl.textContent = ' 提交失败: ' + (data.error || '');
      }
    })
    .catch(e => {
      feedbackEl.textContent = ' 提交失败: ' + e.message;
    });
  }

  // 绑定对比看板按钮事件
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
      document.getElementById('selectBestBtn')?.addEventListener('click', selectBestDirection);
      document.getElementById('rejectAllComparisonBtn')?.addEventListener('click', function() {
        document.getElementById('comparisonActions').style.display = 'none';
        document.getElementById('comparisonStatus').textContent = ' 已跳过选择，继续迭代。';
      });
    }, 1000);
  });

// ======================================================================
  // 雷达图绘制
  // ======================================================================
  function drawRadarChart(results, dimList) {
    const canvas = document.getElementById('radarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2 + 10;
    const radius = Math.min(cx, cy) - 40;
    const levels = 5;
    const colors = ['#667eea', '#f44336', '#4caf50', '#ff9800', '#9c27b0', '#00bcd4', '#e91e63', '#3f51b5'];
    const angleStep = (Math.PI * 2) / dimList.length;

    ctx.clearRect(0, 0, W, H);

    // 绘制网格
    for (let l = 1; l <= levels; l++) {
      const r = (radius / levels) * l;
      ctx.beginPath();
      for (let i = 0; i <= dimList.length; i++) {
        const angle = angleStep * i - Math.PI / 2;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(200,200,200,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // 绘制轴线
    for (let i = 0; i < dimList.length; i++) {
      const angle = angleStep * i - Math.PI / 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
      ctx.strokeStyle = 'rgba(200,200,200,0.4)';
      ctx.stroke();

      // 维度标签
      const labelR = radius + 18;
      const lx = cx + labelR * Math.cos(angle);
      const ly = cy + labelR * Math.sin(angle);
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#666';
      ctx.textAlign = angle > Math.PI / 2 && angle < Math.PI * 1.5 ? 'right' : angle > -Math.PI / 6 && angle < Math.PI / 6 ? 'center' : 'left';
      ctx.textBaseline = 'middle';
      const label = dimList[i].replace('item_', '').replace('_score', '').slice(0, 12);
      ctx.fillText(label, lx, ly);
    }

    // 绘制每个候选的数据
    const maxResultsToShow = Math.min(results.length, 6);
    results.slice(0, maxResultsToShow).forEach((r, ri) => {
      const color = colors[ri % colors.length];
      ctx.beginPath();
      const vals = dimList.map(d => r.scores?.[d] || 0);

      // 透明度以体现排名
      ctx.globalAlpha = 1 - ri * 0.12;

      for (let i = 0; i <= dimList.length; i++) {
        const idx = i % dimList.length;
        const val = Math.max(0, Math.min(100, vals[idx] || 0));
        const r2 = (val / 100) * radius;
        const angle = angleStep * idx - Math.PI / 2;
        const x = cx + r2 * Math.cos(angle);
        const y = cy + r2 * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // 图例
      const lx = 15;
      const ly = 15 + ri * 18;
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, 10, 10);
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#333';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(escapeHtml(r.session_id || '候选 ' + (ri+1)) + ' (' + (r.scores?.total_score || 0) + ')', lx + 15, ly + 5);
    });
  }

  // ======================================================================
  // 柱状图绘制
  // ======================================================================
  function drawBarChart(results) {
    const canvas = document.getElementById('barChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width, H = canvas.height;
    const padding = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;
    const colors = ['#667eea', '#f44336', '#4caf50', '#ff9800', '#9c27b0', '#00bcd4'];

    ctx.clearRect(0, 0, W, H);

    const maxResults = Math.min(results.length, 6);
    const barWidth = Math.min(chartW / maxResults - 8, 40);
    const gap = (chartW - barWidth * maxResults) / (maxResults + 1);

    // Y轴刻度
    ctx.strokeStyle = 'rgba(200,200,200,0.3)';
    ctx.lineWidth = 1;
    for (let v = 0; v <= 100; v += 20) {
      const y = padding.top + chartH - (v / 100) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(W - padding.right, y);
      ctx.stroke();
      ctx.font = '9px sans-serif';
      ctx.fillStyle = '#999';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(v, padding.left - 5, y);
    }

    // Y轴标签
    ctx.font = '9px sans-serif';
    ctx.fillStyle = '#999';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('分数', 12, padding.top + 10);

    // 绘制柱状
    results.slice(0, maxResults).forEach((r, i) => {
      const total = r.scores?.total_score || 0;
      const x = padding.left + gap + i * (barWidth + gap);
      const barH = (total / 100) * chartH;
      const y = padding.top + chartH - barH;

      // 柱体
      const grad = ctx.createLinearGradient(x, y, x, padding.top + chartH);
      grad.addColorStop(0, colors[i % colors.length]);
      grad.addColorStop(1, colors[i % colors.length] + '60');
      ctx.fillStyle = grad;
      ctx.beginPath();
      roundRect(ctx, x, y, barWidth, barH, 3);
      ctx.fill();

      // 数值
      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = colors[i % colors.length];
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(total, x + barWidth / 2, y - 3);

      // 标签
      ctx.font = '9px sans-serif';
      ctx.fillStyle = '#666';
      ctx.textBaseline = 'top';
      const label = (r.session_id || '候选' + (i+1)).slice(0, 10);
      ctx.fillText(label, x + barWidth / 2, padding.top + chartH + 5);
    });
  }

  // 圆角矩形辅助
  function roundRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ======================================================================
  // 多轮趋势折线图
  // ======================================================================
  function drawTrendChart(results) {
    const canvas = document.getElementById('trendChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width, H = canvas.height;
    const padding = { top: 20, right: 15, bottom: 35, left: 35 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;

    ctx.clearRect(0, 0, W, H);

    // 按 run_id 排序
    const sorted = [...results].sort((a, b) => {
      const na = parseInt((a.run_id || '').replace('run_', '')) || 0;
      const nb = parseInt((b.run_id || '').replace('run_', '')) || 0;
      return na - nb;
    });

    const scores = sorted.map(r => r.scores?.total_score || 0);
    const labels = sorted.map(r => (r.run_id || '').replace('run_', 'R') || 'R' + (r.session_id || '').slice(-3));
    const n = scores.length;
    if (n < 2) return;

    const minScore = Math.max(0, Math.min(...scores) - 10);
    const maxScore = Math.min(100, Math.max(...scores) + 10);
    const range = maxScore - minScore || 50;

    // Y轴刻度
    ctx.strokeStyle = 'rgba(200,200,200,0.3)';
    ctx.lineWidth = 1;
    for (let v = 0; v <= 5; v++) {
      const sv = minScore + (v / 5) * range;
      const y = padding.top + chartH - ((sv - minScore) / range) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(W - padding.right, y);
      ctx.stroke();
      ctx.font = '9px sans-serif';
      ctx.fillStyle = '#999';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(Math.round(sv), padding.left - 5, y);
    }

    // 折线
    ctx.beginPath();
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    scores.forEach((s, i) => {
      const x = padding.left + (i / (n - 1)) * chartW;
      const y = padding.top + chartH - ((s - minScore) / range) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 面积填充
    ctx.lineTo(padding.left + chartW, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    grad.addColorStop(0, 'rgba(102,126,234,0.3)');
    grad.addColorStop(1, 'rgba(102,126,234,0.05)');
    ctx.fillStyle = grad;
    ctx.fill();

    // 数据点 + 数值
    scores.forEach((s, i) => {
      const x = padding.left + (i / (n - 1)) * chartW;
      const y = padding.top + chartH - ((s - minScore) / range) * chartH;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#667eea';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.font = 'bold 10px sans-serif';
      ctx.fillStyle = '#667eea';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(s, x, y - 8);

      // X轴标签
      ctx.font = '9px sans-serif';
      ctx.fillStyle = '#666';
      ctx.textBaseline = 'top';
      ctx.fillText(labels[i], x, padding.top + chartH + 5);
    });

    // 趋势箭头标注
    const lastVal = scores[scores.length - 1];
    const firstVal = scores[0];
    const diff = lastVal - firstVal;
    ctx.font = '10px sans-serif';
    ctx.fillStyle = diff >= 0 ? '#4caf50' : '#f44336';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText((diff >= 0 ? '↑' : '↓') + ' ' + Math.abs(diff).toFixed(1), W - padding.right, padding.top + 12);
  }

  // ======================================================================
  // 结果确认
  // ======================================================================
  async function approveResults() {
    if (!state.currentLaunchDir) return;
    try {
      const resp = await fetch('/pipeline/result_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          launch_dir: state.currentLaunchDir,
          feedback: {
            approved: true,
            submitted_at: Date.now(),
            note: '用户已确认实验结果',
          }
        }),
      });
      const data = await resp.json();
      if (data.success) {
        showToast('结果已确认，流水线将继续执行', 'success');
        document.getElementById('resultReviewActions').style.display = 'none';
      }
    } catch (e) {
      showToast(`提交失败: ${e.message}`, 'error');
    }
  }

  // ======================================================================
  // 结果目录浏览器
  // ======================================================================
  let currentBrowserPath = null;

  async function loadBrowserTasks() {
    const select = document.getElementById('browserTaskSelect');
    if (!select) return;
    select.innerHTML = '<option value="">-- 加载中... --</option>';
    try {
      const resp = await fetch('/pipeline/list_results_dir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await resp.json();
      select.innerHTML = '<option value="">-- 选择任务 --</option>';
      if (data.success && data.tasks) {
        data.tasks.filter(t => t.is_dir).forEach(task => {
          const opt = document.createElement('option');
          opt.value = task.name;
          opt.textContent = task.name + ' (' + task.size_display + ')';
          select.appendChild(opt);
        });
      }
    } catch (e) {
      select.innerHTML = '<option value="">-- 加载失败 --</option>';
    }
  }

  async function loadBrowserTree(taskName) {
    const treePanel = document.getElementById('browserTreePanel');
    const previewPanel = document.getElementById('browserPreviewPanel');
    if (!taskName) {
      treePanel.innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 20px;">请选择一个任务</div>';
      return;
    }
    treePanel.innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 20px;">⏳ 加载目录结构...</div>';
    try {
      const resp = await fetch('/pipeline/list_results_dir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_name: taskName }),
      });
      const data = await resp.json();
      if (!data.success) {
        treePanel.innerHTML = '<div style="color: var(--danger); padding: 10px;">加载失败: ' + data.error + '</div>';
        return;
      }
      const tree = data.tree;
      currentBrowserPath = null;
      previewPanel.innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 40px 0;">选择左侧文件查看内容</div>';
      treePanel.innerHTML = renderBrowserTree(tree, 0);
      bindBrowserTreeClicks(treePanel);
    } catch (e) {
      treePanel.innerHTML = '<div style="color: var(--danger);">加载失败: ' + e.message + '</div>';
    }
  }

  function renderBrowserTree(node, depth) {
    if (!node || !node.children) return '';
    const indent = depth * 16;
    let html = '';
    node.children.forEach(child => {
      if (child.type === 'dir') {
        html += '<div class="browser-dir" data-path="' + escapeHtml(child.path) + '" style="padding: 4px 8px 4px ' + (indent + 8) + 'px; cursor: pointer; border-radius: 4px; margin: 1px 0; display: flex; align-items: center; gap: 4px;">' +
          '📁 <span style="font-weight: 500; color: var(--text-primary);">' + escapeHtml(child.name) + '</span>' +
          '<span style="font-size: 10px; color: var(--text-tertiary); margin-left: auto;">' + child.size_display + '</span></div>';
        html += renderBrowserTree(child, depth + 1);
      } else {
        const icon = getFileIcon(child.name);
        html += '<div class="browser-file" data-path="' + escapeHtml(child.path) + '" style="padding: 3px 8px 3px ' + (indent + 24) + 'px; cursor: pointer; border-radius: 4px; margin: 1px 0; display: flex; align-items: center; gap: 4px;">' +
          icon + ' <span style="color: var(--text-secondary);">' + escapeHtml(child.name) + '</span>' +
          '<span style="font-size: 10px; color: var(--text-tertiary); margin-left: auto;">' + child.size_display + '</span></div>';
      }
    });
    return html;
  }

  function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
      'json': '📋', 'md': '📝', 'txt': '📄', 'py': '🐍',
      'yaml': '⚙️', 'yml': '⚙️', 'log': '📜', 'html': '🌐',
      'css': '🎨', 'js': '⚡', 'csv': '📊', 'png': '🖼️',
      'jpg': '🖼️', 'jpeg': '🖼️', 'pdf': '📕', 'sh': '💻',
    };
    return icons[ext] || '📄';
  }

  function bindBrowserTreeClicks(container) {
    container.querySelectorAll('.browser-file').forEach(el => {
      el.addEventListener('click', function() {
        container.querySelectorAll('.browser-file, .browser-dir').forEach(e => e.style.background = '');
        this.style.background = 'var(--bg-hover)';
        readBrowserFile(this.dataset.path);
      });
    });
    container.querySelectorAll('.browser-dir').forEach(el => {
      el.addEventListener('click', function() {
        const next = this.nextElementSibling;
        if (next && next.classList.contains('browser-file') || (next && next.classList.contains('browser-dir') && parseInt(next.style.paddingLeft) > parseInt(this.style.paddingLeft))) {
          // toggle visibility of children
          let sib = this.nextElementSibling;
          const basePad = parseInt(this.style.paddingLeft) || 0;
          const show = this.dataset.collapsed !== 'true';
          this.dataset.collapsed = show ? 'true' : 'false';
          this.innerHTML = this.innerHTML.replace(show ? '📁' : '📂', show ? '📂' : '📁');
          while (sib) {
            const pad = parseInt(sib.style.paddingLeft) || 0;
            if (pad <= basePad) break;
            sib.style.display = show ? 'none' : '';
            sib = sib.nextElementSibling;
          }
        }
      });
    });
  }

  async function readBrowserFile(filePath) {
    const previewPanel = document.getElementById('browserPreviewPanel');
    previewPanel.innerHTML = '<div style="text-align: center; color: var(--text-tertiary);">⏳ 加载文件...</div>';
    try {
      const resp = await fetch('/pipeline/read_result_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath }),
      });
      const data = await resp.json();
      if (!data.success) {
        if (data.too_large) {
          previewPanel.innerHTML = '<div style="color: var(--warning); padding: 20px; text-align: center;">⚠️ 文件过大 (' + data.size_display + ')，不支持在线预览</div>';
        } else {
          previewPanel.innerHTML = '<div style="color: var(--danger);">读取失败: ' + data.error + '</div>';
        }
        return;
      }
      if (data.type === 'binary') {
        previewPanel.innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: 40px 0;">🖼️ 二进制文件 (' + data.size_display + ')，不支持文本预览</div>';
        return;
      }
      previewPanel.innerHTML = '<div style="color: var(--text-primary);">' + escapeHtml(data.content) + '</div>';
    } catch (e) {
      previewPanel.innerHTML = '<div style="color: var(--danger);">读取失败: ' + e.message + '</div>';
    }
  }

  // ======================================================================
  // 当前标签刷新
  // ======================================================================
  async function refreshCurrentTab() {
    const activeBtn = document.querySelector('.review-tab-btn.active');
    if (!activeBtn) return;

    const tab = activeBtn.dataset.reviewTab;
    if (tab === 'idea') {
      if (!state.currentLaunchDir) {
        const connected = await updateConnectionStatus();
        if (!connected) return;
      }
      loadPendingIdeas(state.currentLaunchDir);
    } else if (tab === 'result') {
      if (!state.currentLaunchDir) {
        const connected = await updateConnectionStatus();
        if (!connected) return;
      }
      loadExperimentResults(state.currentLaunchDir);
    } else if (tab === 'comparison') {
      if (!state.currentLaunchDir) {
        const connected = await updateConnectionStatus();
        if (!connected) return;
      }
      loadExperimentResults(state.currentLaunchDir);
    } else if (tab === 'browser') {
      const select = document.getElementById('browserTaskSelect');
      if (select) {
        const val = select.value;
        if (!val) loadBrowserTasks();
        else loadBrowserTree(val);
      }
    }
  }

  // ======================================================================
  // 轮询检测是否有待审批
  // ======================================================================
  function startPolling() {
    if (state.isPolling) return;
    state.isPolling = true;

    async function poll() {
      if (!state.isPolling) return;

      // 更新连接状态
      await updateConnectionStatus();

      // 检查流水线输出中是否有 "HUMAN REVIEW" 等待标记
      if (state.currentLaunchDir) {
        try {
          const resp = await fetch('/pipeline/status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ launch_dir: state.currentLaunchDir }),
          });
          const data = await resp.json();
          if (data.success) {
            const status = data.status;
            if (status === 'waiting_idea' || status === 'waiting_result') {
              refreshCurrentTab();
            }
          }
        } catch (e) {
          // 忽略
        }
      }

      setTimeout(poll, 10000);
    }

    setTimeout(poll, 5000);
  }

  // ======================================================================
  // Toast 通知
  // ======================================================================
  function showToast(message, type = 'info') {
    const colors = { success: '#4caf50', error: '#f44336', warning: '#ff9800', info: '#2196f3' };
    const toast = document.createElement('div');
    toast.style.cssText = `
      position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
      background: ${colors[type] || '#333'}; color: #fff;
      padding: 10px 24px; border-radius: 8px; font-size: 13px;
      z-index: 9999; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      transition: opacity 0.3s;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
  }

  // ======================================================================
  // HTML 转义
  // ======================================================================
  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ======================================================================
  // 初始化
  // ======================================================================
  function init() {
    // 等 DOM 加载完成
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        setTimeout(doInit, 500);
      });
    } else {
      setTimeout(doInit, 500);
    }
  }

  function doInit() {
    injectUI();

    // 初始连接检测
    setTimeout(async () => {
      await updateConnectionStatus();
      refreshCurrentTab();
    }, 500);

    // 监听侧边栏导航点击，切换到流水线时刷新
    document.querySelectorAll('.sidebar-nav-item[data-page="pipeline"]').forEach(item => {
      item.addEventListener('click', () => {
        setTimeout(async () => {
          await updateConnectionStatus();
          refreshCurrentTab();
        }, 300);
      });
    });

    startPolling();
  }

  return { init, loadLaunchDirs, loadPendingIdeas, loadExperimentResults };
})();

// 页面加载后初始化
if (typeof PipelineReview !== 'undefined') {
  PipelineReview.init();
}
