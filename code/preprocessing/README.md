# putt_preproc — 推杆 IMU 批量预处理

Noitom 全身 IMU 导出的 `take***<名字><杆次>_chr01.csv`（60 Hz）+ 杆面数据表 → 每杆特征表、时间归一化曲线、运动链分解、QC 报告。

## 安装
```
pip install numpy pandas scipy matplotlib openpyxl
```
需要 Python ≥ 3.9。

## 运行
1. 先做检查（`--data` 填这批 csv 所在的文件夹，`--face` 填杆面表）：
   ```
   python run.py --dry-run --data "数据文件夹" --face "杆面表.xlsx"
   ```
   这一步会打印 **文件标签 → 受试者** 的映射，并在输出目录生成 `name_map.csv`。自动匹配只做大小写不敏感的完全相等，比如 `p01 → P01`。匹配不上的标签显示 `???`。在 `name_map.csv` 的 subject 列里手动填好以后，再跑一次 dry-run，直到映射和杆次对齐都没有报错。
   - 同一个人既有 `name` 又有 `name重` 时，自动只用 `name重`。
2. 正式处理：把 `--dry-run` 去掉，其余不变。200 个文件大约 2–3 分钟。
3. 输出默认放在数据文件夹旁边的 `<数据文件夹名>_processed`，不会覆盖旧批次的结果；要换位置用 `--out`。
4. 不再每杆出图，只输出 4 张 `overview_*.png` 总览图。想单独看某几杆：`--plots-for P06:8,P08:10`；全部出图：`--plots`。
5. `--save-raw`：另存每杆清洗后的完整四元数和关节位置（`raw/*.npz`），供以后做其它分析，不需要可以不加。

## 处理流程
1. 只用 10 个真实传感器节点（Hips、Spine2、左右 UpLeg/Leg/Arm/ForeArm）计算关节角。其余节点全程 `Sensor-Lost=1`，是软件推算的；它们只在运动链分解里使用（见下文），不参与关节角。
2. 丢帧（Lost=1 时数值保持上一帧不变）用插值补：四元数用 SLERP，位置用 PCHIP。
3. 四元数做半球连续化，消除 q/−q 跳变。
4. 4 阶零相位 Butterworth 低通滤波，6 Hz。
5. 事件检测基于握把点，也就是双手腕的中点：起杆 → 上杆顶点 → 击球 → 送杆结束。击球定义为握把回到准备姿势时的位置（亚帧插值），并用速度峰做交叉校验。
6. 坐标对齐：以准备姿势下骨盆的朝向建立坐标系。X 指向目标，Y 朝上，Z 指向身体前方（朝球的方向）。左右侧统一命名为 lead（前侧）和 trail（后侧）。
7. 先在准备姿势窗口建立参考，再做一轮事件检测和全部角度计算。
8. 运动链分解（新增，见下文）。

## 运动链分解（`putt/chain.py`）
回答"推杆是谁在摆"：把握把的运动、前臂的滚转，精确拆成各关节的贡献。

**原理**：对刚体链 骨盆 → 脊柱各环节 → 肩带 → 上臂 → 前臂，腕关节速度恒等于
`v_腕 = v_骨盆 + Σ_k (ω_k − ω_父环节) × (p_腕 − p_k)`，前臂滚转速率恒等于 `Σ_k (ω_k − ω_父环节)·d`（d 为前臂长轴）。每一项就是一个关节的相对转动对末端的贡献。这是运动学恒等式，不是统计。握把 = 双腕中点，所以两条手臂分别分解后取平均。

**分组**：骨盆平移、骨盆转动、躯干（各脊柱环节相对骨盆）、肩带（锁骨环节，软件推算，通常很小）、前/后侧肩关节（上臂相对肩带）、前/后侧肘与前臂（前臂相对上臂，含旋前旋后）、残差。

**环节选择**：CSV 里同时有姿态和位置列的环节都会用上（含软件推算的 Spine、Spine1、Shoulder 等）；没有的自动跳过，其运动归入残差。日志会打印实际用到的链。

**质量检查**：`chain_resid_x_pct` = 目标轴方向残差的 RMS / 实测握把速度的 RMS（挥杆窗口）；超过 `config.CHAIN_RESID_WARN_PCT`（默认 15%）只在 QC 里标 WARN，不剔除。`chain_link_sd_mm` = 相邻关节间距在挥杆中的标准差，正常应接近 0。用合成数据验证：各项之和与实测握把位移误差 < 0.3 mm，前臂滚转各项之和与原 `imp_d_forearm_roll_*` 完全一致。

## 输出（OUT_DIR）
| 文件 | 内容 |
|---|---|
| `features.csv` | 每杆一行，包含杆面四个变量和约 70 个动作特征 |
| `curves_wide.csv` | 每杆 × 22 个变量各一行；每行是 3 个阶段（back/down/follow）× 101 点的时间归一化曲线，数值都是相对准备姿势的变化量 Δ |
| `chain_features.csv` | **新增**：每杆一行，运动链分解的各项贡献（命名见下）+ 杆面四个变量 |
| `chain_curves.csv` | **新增**：格式同 curves_wide；`cv_*` 握把朝目标速度的各项贡献（m/s），`cd_*` 起杆后累积位移贡献（mm），`cr_lead_* / cr_trail_*` 前臂累积扭转贡献（°） |
| `events.csv` | 各事件的帧号（击球为小数帧）、丢帧统计 |
| `qc_report.csv` | 每杆的状态：OK / WARN / CHECK，以及具体原因 |
| `raw/*.npz` | 加 `--save-raw` 时输出 |
| `qc_plots/*.png` | 每杆一张图，用来人工复核事件点是否正确 |
| `name_map.csv`, `file_list.csv`, `face_table_tidy.csv`, `run_log.txt` | 映射、文件清单、整理后的杆面表、运行日志 |

### 特征命名
- `addr_*`：准备姿势时的绝对值（受标定影响，适合比较不同受试者）
- `top_d_*` / `imp_d_*`：上杆顶点 / 击球时刻相对准备姿势的变化量
- `imp_rate_*`：击球时刻的角速度（deg/s）
- `rng_*`：起杆到送杆结束之间的运动范围（反映稳定性）
- 节奏：`t_backswing_s`, `t_downswing_s`, `t_follow_s`, `tempo_ratio`
- 幅度与速度：`backswing_len_mm`, `follow_len_mm`, `grip_speed_peak`, `grip_speed_impact`
- 握把方向：`grip_path_impact_deg`（正 = 由内向外），`grip_attack_impact_deg`（正 = 向上）
- 运动链（`chain_features.csv`，`{g}` 为分组名 pelvis_trans / pelvis_rot / trunk / girdle / shoulder_lead / shoulder_trail / elbow_lead / elbow_trail / resid）：
  - `bs_{g}_mm`：对上杆长度的贡献（准备 → 顶点），之和 = `bs_total_mm`（≈ `backswing_len_mm`）
  - `ds_{g}_mm`：对下杆位移的贡献（顶点 → 击球），之和 = `ds_total_mm`
  - `iv_{g}` / `ivz_{g}` / `ivy_{g}`：击球时握把朝目标 / 离身 / 竖直速度的贡献（m/s）
  - `roll_{lead|trail}_{g}_top_deg` / `_imp_deg`：准备 → 顶点 / 击球，前臂扭转的贡献（g 为 pelvis_rot / trunk / girdle / shoulder / elbow），之和 = `roll_*_total_*`
  - **按转动轴拆分**（v2 新增）：把每个关节的相对角速度投影到转动环节自身的正交轴上，各分量之和精确等于原贡献。骨盆、躯干、肩带拆成 `twist`（绕长轴扭转）、`flex`（绕左右轴屈伸）、`bend`（绕前后轴侧屈）；肩关节、肘拆成 `axial`（绕长轴旋转：上臂内外旋 / 前臂旋前旋后）、`swing`（其余摆动）。列名为 `bsx_{g}__{ax}_mm`、`dsx_{g}__{ax}_mm`、`ivx_{g}__{ax}`、`ivzx_{g}__{ax}`、`rollx_{lead|trail}_{g}__{ax}_top_deg / _imp_deg`；曲线 `cvx_{g}__{ax}`。注意：推杆时上身前倾，教学里说的"摇肩"（肩膀一上一下）在环节坐标系里是**绕脊柱轴的扭转**，不是侧屈。

### 正负号约定（和杆面表一致，方便直接做相关）
- `forearm_roll_*` / `pronsup_*` / `roll_*`：正 = 杆面**打开**的方向
- `shoulder_tilt_leadup`：正 = 前肩抬高
- `*_turn_closed`：正 = 往上杆方向转（闭合）
- 杆面表：face 正 = OP，path 正 = I-O，attack 正 = UP

## QC 状态
- **CHECK**（必须看图）：握把不在身体前方（姿势或标定异常）、击球点只能用速度峰兜底、挥杆窗口内连续丢帧超过 0.1 s、送杆一直到文件结尾、真实传感器集合与预期不同
- **WARN**：同一文件里有多次挥杆（取的是速度最大的那一杆）、两种击球估计相差超过 5 帧、运动链残差偏大
- 如果事件选错，在 OUT_DIR 新建 `events_override.csv`，包含两列 `file,impact_frame`，填真实击球帧的大概位置，重跑后会在该帧附近重新检测

## 已知限制
- 手、腕、头没有真实传感器，**测不到手腕角度**。前臂滚转是目前能拿到的、最接近杆面转动的量；运动链分解也只能算到双腕中点。
- 握把方向只是杆头路径和攻角的近似，不等于杆头本身。
