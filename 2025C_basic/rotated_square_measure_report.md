# 旋转正方形测量（发挥部分4）— 实现原理与技术报告

## 1. 题目背景

**2025 年全国大学生电子设计竞赛 C 题：基于单目视觉的目标物测量装置**

发挥部分（4）要求：取出最后一个发挥目标物（带正方形图案的平面板），摆在轴线上某一指定位置，水平转动目标物，使物面与轴线间成 **30°~60°** 的夹角 θ。一键启动装置，测量并显示物面上正方形的边长 **x**。

## 2. 核心几何原理

### 2.1 水平旋转的透视效应

当正方形目标物绕竖直轴水平旋转角度 θ 后，在相机画面中产生**水平方向的透视压缩**：

```
        相机视角（俯视）

         相机
          │
          │  光轴
          │
    ──────┼──────  A4 参考平面（垂直于光轴）
          │         ┌─────┐
          │         │     │  ← 旋转后的正方形
          │         │  ✓  │     正面宽度被压缩
          │         └─────┘
          │           ↑
          │      法线偏转 θ
```

在 A4 透视校正后的俯视图中：

- **垂直方向（高度）**：未被压缩，保持真实边长 **x**
- **水平方向（宽度）**：被压缩为 **x · cos(θ)**
- 因此正方形在校正图中呈现为**矩形**

### 2.2 数学恢复公式

由 `cv2.minAreaRect()` 获得旋转矩形的宽 `w` 和高 `h`：

```
真实边长:  x = max(w, h) / SCALE
旋转角度:  θ = arccos( min(w, h) / max(w, h) )
```

其中 `SCALE = 20 px/cm`，为透视变换的分辨率。

### 2.3 验证计算

| 真实 x | θ | cos(θ) | 压缩后宽度 | 能否恢复？ |
|--------|---|--------|-----------|-----------|
| 6 cm (120px) | 30° | 0.866 | 104px | ✓ x=120/20=6cm |
| 6 cm (120px) | 45° | 0.707 | 85px | ✓ |
| 6 cm (120px) | 60° | 0.500 | 60px | ✓ |
| 10 cm (200px) | 30° | 0.866 | 173px | ✓ |
| 10 cm (200px) | 60° | 0.500 | 100px | ✓ |

## 3. 处理流程

```
┌─────────────────────────────────────────────────────┐
│                   main() 一键启动                      │
├─────────────────────────────────────────────────────┤
│  1. Picamera2 初始化 (1280×720 RGB888)                │
│  2. 等待 2s 曝光/白平衡稳定                            │
│  3. 拍摄一帧 → process_frame()                        │
│  4. 串口输出: D={距离},x={边长},type=square            │
│  5. 保存结果图 + 调试图到 images/                       │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              process_frame(frame)                     │
├─────────────────────────────────────────────────────┤
│  ┌─ 灰度化 → GaussianBlur(5×5)                       │
│  └─ adaptiveThreshold(21, 5) → MORPH_CLOSE(3×3)      │
│                          │                            │
│  ┌─ A4 外框检测（三层回退）                            │
│  │   Tier 1: find_a4_outer_quad()   ratio∈[1.20,1.68]│
│  │   Tier 2: 放宽长宽比              ratio∈[1.10,2.20]│
│  │   Tier 3: 最大轮廓→凸包→强制四边形                   │
│  └─ 失败 → 返回错误原因                                │
│                          │                            │
│  ┌─ 距离估算: get_outer_frame_params()                 │
│  │           → estimate_distance()                    │
│  └─ 透视变换: order_points() → getPerspectiveTransform│
│              → warpPerspective (420×594 px, 20px/cm)  │
│                          │                            │
│  ┌─ 裁剪 2cm 边框 → Otsu 二值化                        │
│  └─ 保存中间调试图 (warped / gray / binary)             │
│                          │                            │
│  ┌─ detect_rotated_square(inner_binary)               │
│  │   → RETR_EXTERNAL contours → 面积过滤              │
│  │   → minAreaRect → 尺寸验证 → 计算 θ                 │
│  └─ 失败 → 返回详细原因                                │
│                          │                            │
│  ┌─ x = true_side_px / SCALE                          │
│  │   → correct_square_size() → get_final_square_size()│
│  └─ 绘制: 旋转矩形框 + 文字叠加到原图                    │
└─────────────────────────────────────────────────────┘
```

## 4. 关键设计决策

### 4.1 函数复用

以下函数全部从 `detect_squares_pi_single_frame_v7.py` 导入，保持与基础部分一致的标定参数：

| 函数 | 用途 |
|------|------|
| `find_a4_outer_quad()` | A4 外框四边形检测 |
| `order_points()` | 四点排序（左上→右上→右下→左下） |
| `estimate_distance()` | 像素高度 → 距离插值 |
| `get_outer_frame_params()` | 提取左右边 + 中轴像素长度 |
| `correct_square_size()` | 尺寸标定曲线插值 |
| `get_final_square_size()` | 支架遮挡补偿 |

### 4.2 为什么用 `minAreaRect` 而不是现有的 `detect_squares_robust`

- 发挥部分(4)场景中**只有一个目标**，不需要多正方形分离、重叠恢复等复杂逻辑
- `minAreaRect` 直接返回旋转矩形，天然适合旋转正方形
- 代码量约 100 行 vs `detect_squares_robust` 约 300 行
- `detect_squares_robust` 假设正方形边与图像轴对齐，不适用于旋转场景

### 4.3 A4 检测的三层回退机制

实测中发现某些光照/角度条件下，A4 黑边框在二值图中断裂，`find_a4_outer_quad` 的长宽比检查（`[1.20, 1.68]`）过于严格。因此设计三层回退：

| 层级 | 方法 | 长宽比范围 | 触发条件 |
|------|------|-----------|---------|
| Tier 1 | `find_a4_outer_quad`（原函数） | [1.20, 1.68] | 始终尝试 |
| Tier 2 | 放宽 ratio + recty | [1.10, 2.20], recty≥0.60 | Tier 1 失败 |
| Tier 3 | 最大 RETR_EXTERNAL 轮廓 → 凸包强制四边形 | 无限制 | Tier 2 失败 |

### 4.4 旋转正方形边长验证

原始的 `MIN_SIDE_PX`（110px）用于同时检查 `min(rw, rh)` 和 `max(rw, rh)`，但旋转后短边被压缩，如 6cm 正方形在 60° 时短边仅 60px，远小于 110px。

**修复：** 将验证分为两个独立条件：

```python
# 长边（真实尺寸）：必须在 [MIN_SIDE_PX, MAX_SIDE_PX]
true_side_px < MIN_SIDE_PX or true_side_px > MAX_SIDE_PX → REJECT

# 短边（压缩尺寸）：只需 ≥ MIN_SIDE_PX × cos(65°) ≈ 46.5px
compressed_px < min_compressed → REJECT
```

### 4.5 θ 验证范围

题目要求 30°~60°，代码使用 **[25°, 65°]** 的略宽范围允许测量容差。超出范围时仍输出结果，但在画面中标记橙色警告 `(WARN: out of 30-60deg)`。

## 5. 函数详解

### 5.1 `detect_rotated_square(inner_binary)`

**输入：** 裁剪 2cm 边框后的 Otsu 二值图（黑色=255, 白色=0）

**处理步骤：**

1. `MORPH_CLOSE(3×3)` → 连接细小断裂
2. `findContours(RETR_EXTERNAL)` → 取所有外部轮廓
3. 按面积排序，取最大的 ≥ `0.4 × MIN_SIDE_PX²` 的轮廓
4. `cv2.minAreaRect()` → 获取最小外接旋转矩形
5. 边长验证（长边 ∈ [110, 250]px，短边 ≥ 46.5px）
6. 计算 θ = arccos(短边/长边)

**返回：** 包含宽高、旋转角、角点坐标、验证状态的 dict，或 `None`

### 5.2 `draw_rotated_box_on_original()`

将 inner 坐标系中的 `minAreaRect` 角点经过逆透视变换，绘制回原始相机画面。关键步骤：

```python
pts += border_px                          # inner → warped 坐标
cv2.perspectiveTransform(pts, M_inv)       # warped → original 坐标
cv2.polylines(img, [orig], True, color)    # 绘制
```

### 5.3 `process_frame(frame)`

完整的单帧处理编排函数。返回值增加 `error_msg` 字段，使得 `main()` 能打印具体失败原因。

### 5.4 `main()`

一键测量入口：
- 初始化相机 → 等待 2 秒稳定 → 拍摄 → 处理 → 串口输出
- 成功：`D=105.3,x=8.05,type=square\r\n`
- 失败：`D=0.0,x=0.00,type=square\r\n` + 终端打印具体原因

## 6. 调试支持

### 6.1 终端诊断输出

每个关键步骤和每个 `return None` 路径都有详细打印：

```
Gray shape: (720, 1280), outer_binary foreground: 47154px
  A4 diag: 1820 contours, image=1280x720, area_threshold=[23040, 829440]
  A4 diag: 2 quads passed area+convex check: [...]
A4 candidate: area=245678.0, aspect=1.423
  detect_rotated_square: found 8 contours, area threshold=4840px^2
    contour[0]: area=12800px^2 (~sqrt=113px = 5.7cm)
    contour[1]: area=3200px^2 (~sqrt=57px = 2.8cm)
  detect_rotated_square: largest contour area=12800px^2,
    minAreaRect=(103.5, 120.0)px = (5.17, 6.00)cm
D=105.3 cm, x_raw=6.000cm, x=5.89cm, theta=30.4deg, valid=True
```

### 6.2 调试图片

| 文件名 | 内容 |
|--------|------|
| `rotated_00_original.jpg` | 原始相机画面 |
| `rotated_01_gray.jpg` | 灰度图 |
| `rotated_02_outer_binary.jpg` | A4 外框二值图 |
| `rotated_01_warped.jpg` | 透视校正后 A4 全图 |
| `rotated_02_inner_gray.jpg` | 裁剪 2cm 后的灰度图 |
| `rotated_03_inner_binary.jpg` | Otsu 二值化结果 |
| `rotated_result.jpg` | 最终标注结果图 |

### 6.3 失败原因提示

终端输出格式：

```
===== 测量失败 =====
原因: 旋转正方形未检测到，可能原因：
  1) 目标物未摆入A4纸面内
  2) 旋转角度超出范围(需30~60度)
  3) 正方形边长不在6~12cm范围
  4) 二值化阈值不理想(查看images/debug图片)
====================
```

## 7. 调试过程中的关键问题与修复

### 问题 1：旋转正方形的压缩边被错误过滤

**现象：** 处理时间仅 ~20ms，`detect_rotated_square` 返回 None。

**根因：** `min(rw, rh) < MIN_SIDE_PX`（110px）对压缩边检查过于严格。60° 旋转的 6cm 正方形压缩边仅 60px。

**修复：** 分离长边和短边验证条件（见 4.4 节）。

### 问题 2：A4 外框在二值图中断裂

**现象：** `find_a4_outer_quad` 找不到合格的 A4 四边形。终端显示找到 2 个四边形但长宽比 ~2.05，超出 `[1.20, 1.68]`。

**根因：** 某光照条件下 A4 黑边框在二值图中不连续，adaptive Threshold 的前景仅占 5%。

**修复：** 
1. 改用与 `calibrate_stable_pi.py` 一致的参数（`block_size=21, C=5`）
2. 添加 `MORPH_CLOSE(3×3)` 连接断边
3. 实现三层回退检测（见 4.3 节）

### 问题 3：缺少中间调试信息

**现象：** 失败时只能看到最终结果，无法定位具体失败环节。

**修复：** 在所有 `return None` 路径添加诊断打印，保存 7 张中间调试图片，`process_frame` 返回 `error_msg`。

## 8. 使用方法

### 8.1 独立运行

```bash
cd ~/ELEProject/2025C_basic
python3 rotated_square_measure_pi.py
```

### 8.2 集成到 central_control（可选）

在 `central_control.py` 的 `COMMANDS` 字典中添加：

```python
COMMANDS = {
    'A': 'basic_pi_pc_debug.py',
    'B': 'detect_squares_pi_single_frame_v7.py',
    'C': 'numbered_square_pi_v3.py',
    'D': 'rotated_square_measure_pi.py',   # ← 新增
}
```

### 8.3 操作步骤

1. 将 A4 参考纸放置在相机视野中（黑色边框完整可见）
2. 取出正方形目标物，置于轴线上指定位置
3. 水平旋转目标物，使物面与轴线成 30°~60° 夹角
4. 运行脚本（一键启动）
5. 终端显示 x 和 θ，结果图保存到 `images/rotated_result.jpg`
6. 串口输出格式：`D={距离},x={边长},type=square`

## 9. 文件清单

| 文件 | 说明 |
|------|------|
| `rotated_square_measure_pi.py` | 发挥部分(4) 主程序（本文件，约 415 行） |
| `detect_squares_pi_single_frame_v7.py` | 依赖模块（A4 检测、距离标定、尺寸标定） |
| `docs/superpowers/specs/2026-07-14-rotated-square-measure-design.md` | 设计文档 |
| `docs/superpowers/plans/2026-07-14-rotated-square-measure.md` | 实现计划 |
