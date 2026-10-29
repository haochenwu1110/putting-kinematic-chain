# -*- coding: utf-8 -*-
"""全局配置。改路径只改这里。"""
from pathlib import Path

# ---------- Paths (relative to the repository root) ----------
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "raw"            # de-identified motion-capture files (take***_P##_###_chr01.csv)
FACE_XLSX = REPO / "data" / "gcquad.csv"     # launch-monitor table (tidy CSV) or an original GCQuad .xlsx
OUT_DIR = REPO / "data" / "processed_rerun"  # output folder (created automatically)

# ---------- 采集参数 ----------
FS = 60.0                  # Hz，已由 位置/速度、四元数/角速度 两种方法反推验证
FILE_PATTERN = "*take*_chr*.csv"   # 允许前缀，如 大三take001chu001_chr01.csv

# 真实挂了传感器的 10 个节点（其余节点 Sensor-Lost 全程 = 1，是软件推算的）
REAL_SEGMENTS = ["Hips", "RightUpLeg", "RightLeg", "LeftUpLeg", "LeftLeg",
                 "Spine2", "RightArm", "RightForeArm", "LeftArm", "LeftForeArm"]

# ---------- 预处理参数 ----------
LOWPASS_HZ = 6.0           # 4 阶零相位 Butterworth 截止频率（推杆为低频动作）
MAX_GAP_FRAMES = 20        # 顶点→击球附近连续丢帧超过此数（0.33 s）→ CHECK；其它位置只记 WARN
                           # （200 杆验证：插值后丢帧杆与杆头速度的残差不比无丢帧杆大）
ADDRESS_WIN_S = (0.40, 0.10)   # 准备姿态窗口：起杆前 0.40 s 到 0.10 s 的平均
TAKEAWAY_SPEED_FRAC = 0.10     # 起杆/收杆判定：握把速度 < 峰值速度的 10%
MULTI_STROKE_FRAC = 0.40       # 文件内其它挥杆峰 > 主峰 40% 视为"多次挥杆"→ QC 标记
IMPACT_FALLBACK_TOL = 5        # 击球点两种估计相差 > 5 帧 → QC 标记
N_NORM = 101                   # 各阶段时间归一化点数（0–100%）

# ---------- 运动链分解 ----------
CHAIN_DECOMPOSE = True         # 计算握把运动 / 前臂滚转的分关节贡献，输出 chain_features.csv、chain_curves.csv
CHAIN_RESID_WARN_PCT = 15.0    # 目标轴方向残差 RMS 超过实测握把速度 RMS 的此百分比 → WARN（只提示，不剔除）

# 右手球员默认；左手球员在这里列出受试者名（与杆面表一致），程序做 X 轴镜像
LEFT_HANDED = []           # e.g. ["P17"]; all participants in this study were right-handed

# 同名同时存在 name 与 name重 时，优先用带此后缀的重测文件
REDO_SUFFIX = "重"
