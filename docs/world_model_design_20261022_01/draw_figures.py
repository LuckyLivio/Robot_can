"""Draw compact design diagrams from this project's actual control contracts.

These are original code-drawn figures; no reference-PDF pixels are reused.
All figure labels are >= 38 px at 1900 px width (>= 9 pt at 16 cm).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
WIDTH = 1900
INK = "#1b2c3a"
EDGE = "#5b7282"
BLUE = "#eaf2f8"
GREEN = "#edf5f0"
GOLD = "#a16e25"
GOLD_BG = "#faf3e8"
GREY = "#f5f7f8"
FONT = "C:/Windows/Fonts/msyh.ttc"
BOLD_FONT = "C:/Windows/Fonts/msyhbd.ttc"


def font(size=40, bold=False):
    return ImageFont.truetype(BOLD_FONT if bold else FONT, size)


def canvas(height):
    im = Image.new("RGB", (WIDTH, height), "white")
    return im, ImageDraw.Draw(im)


def text(draw, value, center, size=40, bold=False, fill=INK):
    lines = str(value).split("\n")
    leading = size + 13
    for n, line in enumerate(lines):
        bounds = draw.textbbox((0, 0), line, font=font(size, bold))
        width = bounds[2] - bounds[0]
        y = center[1] + (n - (len(lines) - 1) / 2) * leading
        draw.text((center[0] - width / 2, y - size / 2 - 6), line,
                  font=font(size, bold), fill=fill)


def label(draw, value, center, size=38, fill=INK):
    bounds = draw.textbbox((0, 0), value, font=font(size))
    w, h = bounds[2] - bounds[0], size + 10
    draw.rectangle((center[0]-w/2-10, center[1]-h/2,
                    center[0]+w/2+10, center[1]+h/2), fill="white")
    text(draw, value, center, size=size, fill=fill)


def box(draw, bounds, title, subtitle="", fill=BLUE, title_size=42):
    draw.rounded_rectangle(bounds, radius=15, fill=fill, outline=EDGE, width=3)
    x, y = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
    if subtitle:
        text(draw, title, (x, y-30), size=title_size, bold=True)
        text(draw, subtitle, (x, y+31), size=38)
    else:
        text(draw, title, (x, y), size=title_size, bold=True)


def arrow(draw, points, color=EDGE, width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    a, b = points[-2:]
    theta = math.atan2(b[1]-a[1], b[0]-a[0])
    tips = [(b[0]-20*math.cos(theta-.46), b[1]-20*math.sin(theta-.46)),
            (b[0]-20*math.cos(theta+.46), b[1]-20*math.sin(theta+.46))]
    draw.polygon([b, *tips], fill=color)


def band(draw, bounds, title):
    draw.rounded_rectangle(bounds, radius=10, fill=GREY,
                           outline="#bac8d1", width=2)
    text(draw, title, ((bounds[0]+bounds[2])/2,
                       (bounds[1]+bounds[3])/2), size=42, bold=True)


def save(im, name):
    path = ASSETS / name
    im.save(path, dpi=(300, 300), optimize=True)
    return {"file": name, "width": im.width, "height": im.height,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def architecture():
    im, d = canvas(900)
    band(d, (45, 25, 1855, 95), "交互展示层：命令行配置 · 结果与轨迹浏览")
    text(d, "模型与控制层", (950, 167), 42, True)
    boxes = [((65, 320, 365, 465), "行为克隆", "8 步动作块"),
             ((445, 320, 775, 465), "世界模型", "未来状态与读出"),
             ((855, 320, 1185, 465), "候选选择", "评分与保守回退"),
             ((1265, 320, 1575, 465), "恢复监督", "观测判据 / v5"),
             ((1655, 320, 1855, 465), "FIFO", "D2 队列")]
    for bounds, title, sub in boxes:
        box(d, bounds, title, sub, fill=GREEN if title == "世界模型" else BLUE)
    for n in range(len(boxes)-1):
        a, b = boxes[n][0], boxes[n+1][0]
        arrow(d, [(a[2], 392), (b[0], 392)])
    # Q is an independent feedback input to prediction, not an intermediate
    # control action or a queue reset.
    arrow(d, [(1755, 320), (1755, 247), (610, 247), (610, 320)], GOLD, 5)
    label(d, "真实在途队列 Q", (1180, 245), fill=GOLD)
    text(d, "当前状态供策略、预测与恢复共享", (950, 555), 38)
    band(d, (45, 620, 1855, 680), "数据与仿真层")
    box(d, (65, 710, 480, 845), "同刻状态构建", "23 维 / 语义匹配")
    box(d, (670, 710, 1230, 845), "示范 · 分支 · 模型 · 轨迹", "HDF5 / NPZ / PT / JSONL", title_size=40)
    box(d, (1430, 710, 1855, 845), "机器人仿真环境", "出队后加噪与限幅")
    arrow(d, [(1755, 465), (1755, 710)])
    label(d, "实际执行", (1755, 555))
    arrow(d, [(1640, 845), (1640, 878), (270, 878), (270, 845)])
    arrow(d, [(215, 710), (215, 465)])
    label(d, "观测反馈", (410, 877))
    return save(im, "architecture.png")


def world_model():
    im, d = canvas(870)
    band(d, (45, 25, 1855, 95), "离线学习：同一锚点的多候选分支监督")
    box(d, (70, 150, 455, 315), "同起点分支", "状态 / 动作 / 标签")
    box(d, (550, 150, 1050, 315), "GRU 状态递推", "23 + 7 → 128 → 23", GREEN)
    box(d, (1145, 150, 1830, 315), "任务读出与候选排序", "预测状态 → 进展 / 失持", GREEN)
    arrow(d, [(455, 232), (550, 232)])
    arrow(d, [(1050, 232), (1145, 232)])
    text(d, "联合损失：状态误差 + 进展监督 + 失持监督 + 组内排序", (950, 375), 38)
    band(d, (45, 450, 1855, 520), "在线控制：冻结权重与标准化量，逐周期重新预测")
    box(d, (70, 595, 455, 755), "状态与候选命令", "含真实 D2 队列")
    box(d, (550, 595, 1050, 755), "固定权重递推", "8 步预测 / 不注入真值", GREEN)
    box(d, (1145, 595, 1830, 755), "预测读出与候选评分", "本轮只提交 1 条新命令", GREEN)
    arrow(d, [(455, 675), (550, 675)])
    arrow(d, [(1050, 675), (1145, 675)])
    text(d, "读出头输入为标准化预测状态；隐藏状态参与动态递推", (950, 825), 38)
    return save(im, "world_model.png")


def candidate_flow():
    im, d = canvas(890)
    box(d, (70, 65, 540, 215), "BC 动作块", "8 条七维命令")
    box(d, (700, 65, 1830, 215), "取索引 0、1、2、3", "得到 4 条候选命令 c0、c1、c2、c3")
    arrow(d, [(540, 140), (700, 140)])
    box(d, (70, 335, 540, 510), "当前真实队列", "Q = [q0, q1]", GOLD_BG)
    d.rounded_rectangle((700, 315, 1830, 535), radius=15, fill=GREEN, outline=EDGE, width=3)
    text(d, "每个候选构造 8 步输入", (1265, 355), 42, True)
    entries = ["q0", "q1", "ck", "ck", "ck", "ck", "ck", "ck"]
    for n, entry in enumerate(entries):
        x = 755+n*129
        d.rounded_rectangle((x, 405, x+108, 490), radius=8,
                            fill=GOLD_BG if n < 2 else "white",
                            outline=GOLD if n < 2 else EDGE, width=2)
        text(d, entry, (x+54, 448), 42)
    arrow(d, [(540, 420), (700, 420)], GOLD)
    arrow(d, [(1265, 215), (1265, 315)])
    box(d, (70, 675, 540, 825), "世界模型预测", "当前 23 维状态")
    box(d, (700, 675, 1180, 825), "计算候选评分", "末步 / 均值 / 失持")
    box(d, (1340, 675, 1830, 825), "默认索引 2 回退", "非默认且提升 < 0.02")
    arrow(d, [(1000, 535), (1000, 600), (305, 600), (305, 675)])
    label(d, "公共前缀 2 步 + 每个候选重复 6 步", (1100, 600))
    arrow(d, [(540, 750), (700, 750)])
    arrow(d, [(1180, 750), (1340, 750)])
    return save(im, "candidate_flow.png")


def recovery_flow():
    im, d = canvas(1085)
    coords = []
    stages = [("正常监测", "透传策略建议"), ("回撤", "RETRACT"),
              ("对齐", "ALIGN"), ("下降", "DESCEND"),
              ("闭爪", "CLOSE"), ("试抬确认", "TEST_LIFT"),
              ("抬升", "LIFT"), ("搬运", "CARRY"),
              ("放置", "PLACE"), ("释放", "RELEASE"),
              ("退出", "RETREAT"), ("正常监测", "继续闭环")]
    for n, (title, sub) in enumerate(stages):
        row, col = divmod(n, 4)
        if row == 1:
            col = 3-col
        bounds = (60+col*460, 60+row*210, 440+col*460, 190+row*210)
        coords.append(bounds)
        box(d, bounds, title, sub, title_size=40)
    for n in range(1, 11):
        a, b = coords[n:n+2]
        if n in (3, 7):
            arrow(d, [((a[0]+a[2])/2, a[3]), ((b[0]+b[2])/2, b[1])])
        elif n < 3 or n > 7:
            arrow(d, [(a[2], (a[1]+a[3])/2), (b[0], (b[1]+b[3])/2)])
        else:
            arrow(d, [(a[0], (a[1]+a[3])/2), (b[2], (b[1]+b[3])/2)])
    # This branch is only the ordinary unheld attempt.  The holding=True
    # bypass is stated separately, rather than conflated with an observation.
    arrow(d, [(250, 60), (250, 25), (710, 25), (710, 60)])
    label(d, "触发且未持物", (480, 25))
    box(d, (70, 715, 590, 835), "试抬失败 / 可重试超时", "调用 start_attempt", GOLD_BG, 40)
    box(d, (700, 715, 1190, 835), "剩余尝试次数 > 0？", "每回合最多 2 次", GOLD_BG, 40)
    box(d, (1310, 715, 1830, 835), "EXHAUSTED", "停止覆盖，透传策略", GOLD_BG, 40)
    # TEST_LIFT failure and retryable timeouts use the default holding=False.
    # Workspace failure and loss inside the target area have direct bypasses.
    arrow(d, [(1170, 400), (1170, 435), (1870, 435),
              (1870, 665), (330, 665), (330, 715)], GOLD, 4)
    label(d, "试抬未确认 / 可重试超时", (1360, 665), fill=GOLD)
    arrow(d, [(590, 775), (700, 775)], GOLD, 4)
    arrow(d, [(1190, 775), (1310, 775)], GOLD, 4)
    label(d, "否", (1250, 745), fill=GOLD)
    arrow(d, [(945, 835), (945, 870), (20, 870),
              (20, 225), (710, 225), (710, 190)], GOLD, 4)
    label(d, "是：默认 holding=False，重回 RETRACT", (500, 870), fill=GOLD)
    text(d, "持物旁路：NORMAL 持物停滞 → start_attempt(holding=True) → LIFT", (950, 940), 38)
    text(d, "工作区旁路：活动恢复阶段越界 → EXHAUSTED", (950, 995), 38)
    text(d, "释放旁路：LIFT / CARRY / PLACE 失持且位于目标 XY 区 → RELEASE", (950, 1050), 38)
    return save(im, "recovery_flow.png")


def data_entities():
    im, d = canvas(870)
    box(d, (70, 100, 490, 270), "示范 / 物理起点", "HDF5 · demo")
    box(d, (700, 100, 1120, 270), "同起点候选分支", "NPZ · parent / point")
    box(d, (1330, 100, 1830, 270), "BC / WM 检查点", "PT · config / SHA")
    arrow(d, [(490, 185), (700, 185)])
    label(d, "采集", (595, 147))
    arrow(d, [(1120, 185), (1330, 185)])
    label(d, "WM 训练", (1225, 147))
    arrow(d, [(280, 100), (280, 35), (1580, 35), (1580, 100)])
    label(d, "BC 训练", (940, 35))
    box(d, (700, 415, 1220, 575), "实验作业", "JSON · job id / 条件")
    arrow(d, [(280, 270), (280, 495), (700, 495)])
    label(d, "登记起点", (488, 495))
    arrow(d, [(1580, 270), (1580, 495), (1220, 495)])
    label(d, "装载冻结模型", (1545, 370))
    box(d, (300, 700, 850, 850), "回合结果", "JSON · job / demo")
    box(d, (1110, 700, 1740, 850), "逐步控制轨迹", "JSONL · demo / step")
    arrow(d, [(960, 575), (960, 635), (575, 635), (575, 700)])
    label(d, "每个作业包含多个回合", (1010, 635))
    arrow(d, [(850, 775), (1110, 775)])
    label(d, "留痕", (980, 735))
    return save(im, "data_entities.png")


def control_sequence():
    im, d = canvas(900)
    centers = [150, 470, 790, 1110, 1430, 1750]
    titles = ["观测构建", "行为克隆", "世界模型", "恢复监督", "D2 队列", "仿真环境"]
    for x, title in zip(centers, titles):
        box(d, (x-130, 35, x+130, 130), title, title_size=40)
        for y in range(130, 860, 24):
            d.line((x, y, x, min(y+12, 860)), fill="#bcc9d2", width=3)
    events = [(1750, 150, 200, "当前观测"),
              (150, 470, 285, "pack：23 维状态"),
              (470, 790, 370, "8 步动作块"),
              (1430, 790, 455, "真实在途队列 Q"),
              (790, 1110, 540, "候选建议"),
              (1110, 1430, 625, "choose：提交命令"),
              (1430, 1750, 710, "出队 → 加噪 → 执行"),
              (1750, 150, 795, "新观测：进入下一控制周期")]
    for x1, x2, y, caption in events:
        arrow(d, [(x1, y), (x2, y)])
        label(d, caption, ((x1+x2)/2, y-32))
    text(d, "prev 保存实际执行动作供下一次 choose 使用；成功标志仅用于评价", (950, 870), 38)
    return save(im, "control_sequence.png")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    figures = [architecture(), world_model(), candidate_flow(),
               recovery_flow(), data_entities(), control_sequence()]
    manifest = {"description": "Original project-specific PIL diagrams; not copied from reference thesis",
                "width_px": WIDTH, "minimum_font_px": 38,
                "minimum_font_pt_at_16cm": round(38/WIDTH*(16/2.54*72), 3),
                "figures": figures}
    (HERE / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figures": len(figures), "assets": str(ASSETS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
