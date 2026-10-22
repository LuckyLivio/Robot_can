"""Build the current technical report/navigation only after verified results."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
REL = RUN.relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    summary = read(RUN / 'analysis/confirmation/summary.json')
    required = ['analysis/confirmation_trace_verification.json', 'analysis/confirmation_physics_replay.json', 'analysis/io_recovery_verification.json',
                'preservation_check.json']
    checks = {p: read(RUN / p) for p in required}
    if summary['status'] != 'complete' or summary['executions'] != 1680 or any(not x.get('passed') for x in checks.values()):
        raise ValueError('Delivery requires complete confirmation, trace/physics and preservation checks')
    if not (RUN / 'figures/manifest.json').is_file():
        raise ValueError('Verified figures required')
    cells, effects = summary['cells'], summary['paired_effects']

    def counts(controller, method, noise, *, arm='recovery_factorial', semantics='legacy'):
        chosen = [c for c in cells if c['controller'] == controller and c['method'] == method
                  and c['noise'] == noise and c['study_arm'] == arm
                  and c['observation_semantics'] == semantics]
        return sum(c['successes'] for c in chosen), sum(c['executions'] for c in chosen)

    def cell_text(controller, method, **kw):
        pairs = [counts(controller, method, n, **kw) for n in (0., .05)]
        return ' / '.join(f'{a}/{b}' for a, b in pairs)

    def effect(method, comparison, *, arm='recovery_factorial'):
        return next(e for e in effects if e['method'] == method and e['comparison'] == comparison
                    and e['noise'] == 'equal_noise_joint' and e['metric'] == 'success'
                    and e['study_arm'] == arm)

    def effect_text(item):
        return f"{item['effect']:+.2f}个百分点，95%区间[{item['ci95_low']:.2f}, {item['ci95_high']:.2f}]"

    bc_recovery = effect('bc', 'v5 - none')
    full_recovery = effect('full', 'v5 - none')
    wm_with_recovery = effect('v5', 'Full - BC with v5')
    bc_budget = effect('bc', 'queue_budget - v5')
    full_budget = effect('full', 'queue_budget - v5')
    semantic = effect('bc', 'matched newly trained canonical - matched newly trained legacy', arm='observation_confirmation')
    table = '\n'.join(f"| {'BC固定权重' if method == 'bc' else 'Full三个固定WM种子'} | {cell_text('none', method)} | {cell_text('v5', method)} | {cell_text('queue_budget', method)} |" for method in ('bc', 'full'))
    semantic_table = '\n'.join(f"| {semantics}匹配重训BC | {cell_text('none', 'bc', arm='observation_confirmation', semantics=semantics)} |" for semantics in ('legacy', 'canonical'))
    cleanup = read(RUN / 'cleanup_manifest.json')
    preserved = checks['preservation_check.json']['verified_original_files']
    budget_null = bc_budget['effect'] == 0 and full_budget['effect'] == 0
    budget_status = ('本批两种策略成功数均未改变，未证明新增性能收益，预算模块保留为未采用候选。' if budget_null else
                     '预算候选存在一次救回和一次退步，具体配对区间见本轮完整报告；本轮保留为未采用候选。')
    report = f'''# 高优先级修复、配对实验与创新候选报告

日期：2026-10-22。本轮为软件修复与实验交付，没有撰写论文。

## 已完成

新活动实现位于robot_can_v2/，完成同刻观测契约、安全输出、prediction_only训练/选点/部署隔离、延迟预算契约、模型输入验证和显式平局指标。原源码、权重、数据、协议与实验结果保留；{preserved}个原路径校验通过，活动导航旧字节在before/。58项回归与契约测试通过，四方法各2更新原生训练验收通过；新legacy/canonical BC各15000更新，初始化与批次哈希完全一致。

本轮完成120次观测开发、240次预算开发及1680次冻结后确认，共2040次登记研究执行。两项Windows记录写入故障导致另60次完整前缀重复执行，全部保留、逐步精确匹配，不计入统计分母；因此登记执行加重复前缀实际为2100次。两个单回合工程预检、四条物理回放、20步稳定性附加观察另行解释，不增加独立样本。确认只有60个全新独立原生起点，已查重、逆序精确重生成；固定策略和WM种子在这些起点上重复执行。

## 最终同批四组恢复对照

下面每格依次为干净 / 噪声0.05的成功执行数。Full每条件180次来自60起点×三个固定WM种子，不是180个独立起点。

| 冻结策略 | 无恢复 | v5共享恢复 | 队列预算候选 |
|---|---|---|---|
{table}

- v5对BC的同批恢复作用：{effect_text(bc_recovery)}。
- v5对Full的同批恢复作用：{effect_text(full_recovery)}。
- v5下Full相对BC：{effect_text(wm_with_recovery)}。

其中960次是BC/Full×有无恢复的主要对照，另480次是预算候选；世界模型种子先在每个起点内平均，再等权两噪声。不得用池化1440次的成功率掩盖3:1的Full/BC执行数量差异，也不能把恢复作用全部归给世界模型。交互效应与逐条件结果见[配对表](analysis/confirmation/paired_effects.csv)。

## 观测修复的独立确认

两份新BC均使用预定final.pt，无恢复，与上表历史BC权重分别统计。开发为47/60→51/60；该开发区间跨零，未用来挑确认checkpoint。

| 新策略 | 干净 / 噪声0.05 |
|---|---|
{semantic_table}

canonical相对匹配legacy：{effect_text(semantic)}。输入几何一致性与性能改善是两个结论；不能仅凭修复输入就假定成功率一定提高。当前只配对训练了seed17，区间仅覆盖起点变化，不覆盖任意重新训练。旧BC及其结果没有替换或重新贴标签。

## 创新候选结论

新增队列与阶段进度感知的剩余预算估计，使用真实在途D2动作、当前阶段与末端历史速度，保留原几何、动作速度规则、确认帧数、尝试上限和400步预算；没有新增奖励或接触标签输入。与原v5单因素对照。

- 开发119/120→119/120，0救回、0退步；门槛通过只表示可进入确认。
- 确认BC的候选减v5：{effect_text(bc_budget)}。
- 确认Full的候选减v5：{effect_text(full_budget)}。
- {budget_status}

若差值为零，退化bootstrap[0,0]仍不能证明总体等效。候选实际触发和逐轨迹分歧见[机制核对](analysis/budget_mechanisms.json)。本轮没有宣称全球首创、稳定领先强BC、视觉泛化或真机安全。

## 额外评价与核验

补充了成功后的20控制稳定性诊断，含在途队列与相同噪声。它要求持续满足原成功判据且最后10帧物体平移差分速度小于0.02m/s，独立于原400步基准。逐cell稳定成功数及分母见[cells.csv](analysis/confirmation/cells.csv)。compute_ms包含BC、WM与恢复计算，物理与日志在计时外；并行计时和单个串行预检均不能证明真机端到端实时性。

全部研究轨迹已做FIFO、噪声、连续性、首次成功、来源和分母检查；确认预定四个已有回合已物理回放。旧资产校验通过。只清理{len(cleanup['deleted_files'])}个未跟踪可再生Python缓存，回收{cleanup['bytes_removed']}字节；数据、失败日志、权重、快照、交付ZIP、虚拟环境与Git对象均未删除。

## 入口

- [新代码与命令](../../robot_can_v2/README.md)
- [完整方法与信息边界](METHODS.md)
- [执行前计划](PLAN.md) / [确认前补充登记](CONFIRMATION_AMENDMENT.md)
- [确认输入锁](confirmation_lock.json) / [新起点验证](confirmation_state_validation.json)
- [确认汇总](analysis/confirmation/summary.json) / [结果图与来源](figures/manifest.json)
- [全部动作审计](analysis/confirmation_trace_verification.json) / [四回合物理回放](analysis/confirmation_physics_replay.json)
- [记录写入恢复与60次重复前缀](analysis/io_recovery_verification.json) / [基础设施恢复说明](IO_RECOVERY_AMENDMENT.md)
- [旧资产保护](preservation_check.json) / [清理清单](cleanup_manifest.json)
- [原正式v5评估](../recovery_v5_20261012_01/ASSESSMENT.md) / [历史v21探索](../improvement_v21_20261019_01/REPORT.md)

GitHub分支：codex/graduation-repairs-20261022。各阶段已提交并推送；不改写历史Git提交。本报告不是论文，也不是学校毕业资格认定。
'''
    report_path = RUN / 'REPORT.md'
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(report, encoding='utf-8')
    cards = ''.join(f'<a class="card" href="{escape(url)}"><b>{escape(title)}</b><span>{escape(desc)}</span></a>' for title, url, desc in [
        ('完整技术报告', 'REPORT.md', '修复、效果与结论边界'), ('新活动代码', '../../robot_can_v2/README.md', '安全入口与运行命令'),
        ('正式冻结v5', '../recovery_v5_20261012_01/ASSESSMENT.md', '保留原有研究主线'), ('历史v21探索', '../improvement_v21_20261019_01/index.html', '失败与负结果保留')])
    rows_html = ''.join('<tr><td>' + '</td><td>'.join(escape(s) for s in [label, cell_text('none', m), cell_text('v5', m), cell_text('queue_budget', m)]) + '</td></tr>' for label, m in [('BC固定权重', 'bc'), ('Full三个固定种子', 'full')])
    images = ''.join(f'<figure><img src="figures/{name}.png" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>' for name, label in [
        ('recovery_factorial_success', '同批冻结策略恢复对照：60个独立起点'), ('recovery_factorial_effects', '恢复作用与世界模型作用：按起点配对区间'),
        ('observation_confirmation_success', '匹配重训BC的观测语义确认，与旧BC分别统计'), ('queue_budget_candidate_effects', '预算候选：保留零变化与负结果')])
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAN LAB · 修复与受控实验</title><style>
body{{margin:0;background:#f4f6f5;color:#183029;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:54px 24px}}h1{{font-size:36px;line-height:1.2}}.eyebrow{{color:#28775d;font-weight:700;letter-spacing:.14em}}.lead{{max-width:850px;color:#455d54}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:16px;margin:30px 0}}.card{{background:white;padding:22px;border:1px solid #dbe4df;border-radius:14px;text-decoration:none;color:inherit}}.card span{{display:block;color:#62776e;font-size:14px;margin-top:7px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #dbe4df}}.note{{border-left:4px solid #b78535;padding:12px 18px;background:#fff7e9;margin:24px 0}}figure{{margin:28px 0;background:white;padding:12px;border-radius:12px}}img{{width:100%;height:auto}}figcaption{{font-size:14px;color:#50665b}}a{{color:#176244}}footer{{padding-top:24px;border-top:1px solid #cddbd3;color:#62776e}}</style>
<main><div class="eyebrow">CAN LAB / 2026.10.22</div><h1>修复有证据，改进有对照</h1><p class="lead">统一离线与在线观测，保护历史代码、权重与结果，补齐同批恢复对照。360次开发、1680次确认；确认只有60个独立原生起点。本轮未写论文。</p>{cards}
<h2>冻结策略的同批恢复实验</h2><p>每格依次为干净 / 噪声0.05。Full每条件180次来自60起点×三个固定种子。</p><table><thead><tr><th>策略</th><th>无恢复</th><th>v5恢复</th><th>队列预算候选</th></tr></thead><tbody>{rows_html}</tbody></table>
<div class="note">{escape(budget_status)}<br>观测修复的独立差值：{escape(effect_text(semantic))}。完整区间、稳定性指标和限制见技术报告。</div>{images}<footer>原始结果与验证：<a href="analysis/confirmation/summary.json">汇总JSON</a> · <a href="analysis/confirmation/paired_effects.csv">配对CSV</a> · <a href="confirmation_lock.json">输入锁</a> · <a href="METHODS.md">方法与边界</a>。当前仍为特权状态仿真，未新增视觉或真机验证。</footer></main></html>'''
    (RUN / 'index.html').write_text(html, encoding='utf-8')
    # Every replaced navigation document has an exact original byte copy in before/.
    navigation = {
        'README.md': f'''# CAN LAB · 机器人抓取与恢复控制\n\n2026-10-22：完成高优先级修复、新BC配对训练、360次开发与1680次冻结后确认。本轮不写论文。\n\n[本轮完整报告]({REL}/REPORT.md) · [结果前端]({REL}/index.html) · [新活动代码与命令](robot_can_v2/README.md)\n\n新实现统一观测时序，修复prediction_only与评估契约，拒绝覆盖旧产物。旧源码、权重、数据和实验结果保留；原正式v5仍是冻结参考，预算候选不因开发门槛通过就自动采用。\n\n| 同批冻结策略 | 无恢复，干净/噪声 | v5恢复，干净/噪声 |\n|---|---|---|\n| BC | {cell_text('none', 'bc')} | {cell_text('v5', 'bc')} |\n| Full（三固定种子） | {cell_text('none', 'full')} | {cell_text('v5', 'full')} |\n\n确认共60个独立起点。Full重复种子与噪声执行不增加起点数。新BC观测确认另计240次，不能与旧BC混合。{budget_status}\n\n启动展示：\n\n```powershell\n& '.\\.venv\\Scripts\\python.exe' -m http.server 8766 --bind 127.0.0.1\n```\n\n浏览 http://127.0.0.1:8766/ 。已有同目录服务时直接刷新。\n\n[分类导航](PROJECT_NAVIGATION.md) · [运行指南](PROGRAM_GUIDE.md) · [结果](RESULTS.md) · [证据](EVIDENCE_INDEX.md) · [此前README原字节]({REL}/before/README.md)\n''',
        'STATUS.md': f'# 当前状态 · 2026-10-22\n\n本轮修复与实验已完成，未写论文。58项测试、全部动作记录审计、预定四回合物理回放、旧资产保护检查通过。360开发+1680确认，确认60独立起点。\n\n{budget_status}\n\n[本轮报告]({REL}/REPORT.md) · [统计与原始结果]({REL}/analysis/confirmation/summary.json) · [此前状态]({REL}/before/STATUS.md)\n',
        'RESULTS.md': f'# 本轮结果入口 · 2026-10-22\n\n[完整表、区间与边界]({REL}/REPORT.md) · [机器可读汇总]({REL}/analysis/confirmation/summary.json) · [配对效应CSV]({REL}/analysis/confirmation/paired_effects.csv)\n\n原冻结BC恢复作用：{effect_text(bc_recovery)}。原Full恢复作用：{effect_text(full_recovery)}。新重训BC观测语义作用：{effect_text(semantic)}。不同策略权重、任务因素和不同确认批次均分别统计。\n\n{budget_status}\n\n[此前全部结果索引原字节]({REL}/before/RESULTS.md)。旧v5与世界模型主结果不被新起点数值改写；历史探索、失败和确认均保留。\n',
        'PROJECT_NAVIGATION.md': f'# 项目分类导航\n\n- [新活动实现与CLI](robot_can_v2/README.md)\n- [本轮报告与前端]({REL}/REPORT.md) / [前端]({REL}/index.html)\n- [方法、输入与统计契约]({REL}/METHODS.md)\n- [新实验原始结果]({REL}/confirmation/) / [图表来源]({REL}/figures/manifest.json)\n- [正式冻结v5](reproduction/recovery_v5_20261012_01/ASSESSMENT.md)\n- [原世界模型确认](reproduction/confirmation_20261012_01/analysis/summary.csv)\n- [历史v21探索](reproduction/improvement_v21_20261019_01/REPORT.md)\n- [此前完整导航]({REL}/before/PROJECT_NAVIGATION.md)\n\n根目录旧Python源码、旧模型与数据均保留；新实验使用robot_can_v2入口。\n',
        'PROGRAM_GUIDE.md': f'# 运行与复核指南\n\n解释器：.venv/Scripts/python.exe。\n\n[新训练、评估和复核命令](robot_can_v2/README.md) · [本轮报告]({REL}/REPORT.md)\n\n新训练/评估必须选择不存在的输出目录。既有确认起点重放不算新的独立确认；锁定输入漂移会直接报错。不要用根目录历史训练入口覆盖已有模型。\n\n[此前指南原字节]({REL}/before/PROGRAM_GUIDE.md)\n',
        'DEMO_GUIDE.md': f'# 展示指南\n\n项目根目录用.venv解释器运行python -m http.server 8766 --bind 127.0.0.1，浏览http://127.0.0.1:8766/。\n\n当前首页展示修复、同批恢复对照、新BC观测确认及预算候选；不是新的机器人视频。历史视频按原版本使用。\n\n先解释D2与400步，再展示有无恢复四组，随后单列观测修复和预算候选的正负结果，最后说明60个独立起点及稳定性附加指标。\n\n[当前结果页]({REL}/index.html) · [历史演示指南]({REL}/before/DEMO_GUIDE.md)\n',
        'EVIDENCE_INDEX.md': f'# 本轮证据索引\n\n- [执行前计划]({REL}/PLAN.md) / [确认前补充登记]({REL}/CONFIRMATION_AMENDMENT.md)\n- [匹配训练检查]({REL}/policies/paired_training_check.json)\n- [原v5逐步精确回归]({REL}/preflight_eval_v5/legacy_v5_regression_comparison.json)\n- [确认输入锁]({REL}/confirmation_lock.json) / [新起点重生成]({REL}/confirmation_state_validation.json)\n- [360次开发的动作审计]({REL}/analysis/dev_budget_trace_verification.json) / [观测对照审计]({REL}/analysis/dev_observations_trace_verification.json)\n- [1680次确认动作审计]({REL}/analysis/confirmation_trace_verification.json) / [四回合物理回放]({REL}/analysis/confirmation_physics_replay.json)\n- [独立分析与配对区间]({REL}/analysis/confirmation/summary.json) / [机制变化]({REL}/analysis/budget_mechanisms.json)\n- [图表与来源]({REL}/figures/manifest.json) / [旧资产保护]({REL}/preservation_check.json) / [缓存清理]({REL}/cleanup_manifest.json)\n- [旧证据导航原字节]({REL}/before/EVIDENCE_INDEX.md)\n',
        'PROJECT_AUDIT.md': f'# 本轮整改状态 · 2026-10-22\n\n高优先级问题已在独立活动实现中修复。输入几何契约、评分头隔离、配置验证、来源锁和拒绝覆盖均有检查；旧源代码、权重和结果保留。\n\n[整改与实验报告]({REL}/REPORT.md) · [此前审查]({REL}/before/PROJECT_AUDIT.md)\n\n当前限制仍包括单任务、特权状态、固定D2、无实机；新BC仅一个配对训练种子。{budget_status} 本轮没有写论文，也不把本次整改当作学校正式毕业审核。\n',
        'ENVIRONMENT_GUIDE.md': f'# 当前环境与迁移说明\n\n本轮实际环境：Python3.10.9、torch2.14.1+cu130、NumPy1.26.4、robosuite1.5.1、MuJoCo3.3.0。各阶段started.json含实际包版本。\n\n继续使用项目.venv；没有重装或降级环境。requirements-repro.txt仍是历史安装建议，不等同新电脑可安装锁；不要照旧torch注释降级。目标演示电脑的干净安装仍需单独实测，不能以本机试验替代。\n\n[当前命令](robot_can_v2/README.md) · [原环境说明]({REL}/before/ENVIRONMENT_GUIDE.md)\n',
        'PROTOCOL.md': f'# 协议与版本入口\n\n本轮[执行前计划]({REL}/PLAN.md)、[确认前补充登记]({REL}/CONFIRMATION_AMENDMENT.md)、[确认设计]({REL}/confirmation_design.json)、[最终输入锁]({REL}/confirmation_lock.json)保留实际登记时点与哈希。\n\n30个已知dev起点与60个全新确认起点分开；独立统计单位是起点。正式四组960执行、预算候选480执行和新BC观测240执行分别解释。400步、D2、成功口径不变；20步稳定诊断单独计数。\n\n原协议均未改写。[此前协议索引原字节]({REL}/before/PROTOCOL.md) · [本轮方法与约束]({REL}/METHODS.md)\n',
        'index.html': f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={REL}/index.html"><title>CAN LAB</title><a href="{REL}/index.html">查看当前修复与对照结果</a>'}
    for name, content in navigation.items():
        if not (RUN / 'before' / name).is_file():
            raise ValueError('Missing original navigation backup: ' + name)
        (ROOT / name).write_text(content, encoding='utf-8')
    (RUN / 'delivery_status.json').write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
        'navigation_updated': list(navigation), 'original_navigation_preserved': True,
        'research_executions': 2040, 'confirmation_executions': 1680, 'independent_confirmation_starts': 60,
        'extra_complete_duplicate_episodes': 60, 'registered_plus_duplicate_executions': 2100,
        'paper_written': False, 'budget_candidate_auto_adopted': False}, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'updated_navigation': len(navigation)}))


if __name__ == '__main__':
    main()
