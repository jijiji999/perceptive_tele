# G1 原装传感器 / Isaac Sim / nvblox 建图基线

当前 `live` / `sphere` 默认局部范围为 **1.2 m 半径**，使用固定在 `torso_link` 上的虚拟感知坐标系 F。腰部零位时，F 的原点位于原骨盆坐标原点下方 **20 cm**；不是从骨盆网格自动找出的点。相机和雷达相对于 F 的外参固定，F 会随躯干移动和倾斜。历史基线和对比图保留原来的坐标定义，以各文件的 `coordinate_frame` / 元数据为准。

可视化默认采用 `--inspection-angle overview`：从固定位置斜向下俯视整个环境，不跟随机器人平移或转向。窗口、截图和录屏使用同一个固定视角，默认开启场景真值叠加；可取消 `Overlay scene ground truth` 只看体素。`Restore overview` 恢复固定视角。原有近景仍可通过 `--inspection-angle front` / `oblique` 选择。

已在本机 `lab_v5.1` 的 **Isaac Sim 5.1** 中生成场景、采集 60 帧观测，并使用真正的 NVIDIA **nvblox TSDF** 完成相机、雷达、融合三组重建。默认体素边长 **5 cm**。

## 从 GitHub 获取代码后

仓库包含脚本、场景配置和文档；本地 Python/CUDA 依赖、`third_party/` 源码副本以及 `outputs/` 仿真数据不随仓库上传。下文列出的已有结果文件是在开发机器上生成的，新克隆的目录需要先生成结果。

运行前需要安装 Isaac Sim 5.1，并准备 Unitree G1 的 URDF 及其引用的完整网格目录。本机默认使用 `lab_v5.1` 和下文所列 URDF 路径；其他机器可指定：

```bash
export LAB_PYTHON=/path/to/isaacsim/python
bash scripts/setup_mapping.sh

# live 从空地图开始，不需要预先下载 outputs
./run.sh live --urdf /path/to/g1_description/g1_29dof_rev_1_0.urdf \
  --robot-pose tpose --inspection-angle overview --output outputs/live_overview

# 先生成离线基线，再打开离线查看器
./run.sh all --urdf /path/to/g1_description/g1_29dof_rev_1_0.urdf --frames 60
./run.sh view fused --urdf /path/to/g1_description/g1_29dof_rev_1_0.urdf
```

`setup_mapping.sh` 的平台和依赖限制见下文“运行环境与已修复的问题”。URDF 和网格需自行从 Unitree 的对应机器人描述包准备。

## 直接查看

```bash
cd /home/rx/project/sliouzhou04/perceptive_tele
./run.sh view fused
```

在 Isaac Sim 窗口中使用 `camera / lidar / fused` 按钮切换结果，勾选 `Overlay scene ground truth` 对照真实几何。体素默认隐藏地面，便于检查障碍物；**文件中仍保留地面**。空白处表示没有显示已观测的表面，不能据此当作可通行自由空间。

已经生成的文件在 `outputs/baseline/`：

| 文件 | 内容 |
|---|---|
| `scene.png` | Isaac Sim 场景截图 |
| `comparison.png` | 场景真值、相机、雷达、融合的四格对比图 |
| `sensors.png` | 首帧 RGB、有效深度、雷达距离图 |
| `fused_isaac_voxels.png` | 实际 TSDF 表面体素在 Isaac Sim 中的截图 |
| `capture.npz` | 逐帧深度、雷达球面距离、相机/雷达/机器人真值位姿、内参 |
| `{camera,lidar,fused}_voxels.npz` | 体素中心 `centers[N,3]`、`tsdf[N]`、`weight[N]`、体素边长 |
| `{camera,lidar,fused}_surface.ply` | 表面体素中心点云 |
| `{camera,lidar,fused}_mesh.ply` | nvblox 提取的三角网格 |
| `scene.usda` / `fused_voxel_scene.usda` | 可在 Isaac Sim 打开的场景 / 体素场景 |
| `metadata.json` | 场景参数、外参、近似模型说明 |
| `metrics.json` / `validation.json` | 几何偏差指标 / 验证结果 |

显示条件为 `weight > 0.1` 且 `abs(TSDF) <= 0.75 * voxel_size`。这是**已观测表面的体素带**，不是填实物体，也不是完整占据图或 ESDF。未观测体素不会因为 TSDF 初始为零而错误显示为障碍物。

## 场景与传感器

场景包含地面、三面墙、40 cm 矮箱、120 cm 高箱、10/20/30 cm 台阶、直径 10 cm 柱，以及桌面和桌腿。传感器随 G1 做刚性平移及一周转向；不运行行走控制器或 SONIC，先单独验证感知建图链路。

传感器相对躯干的变换读取本机 Unitree URDF：

`/home/rx/project/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`

| 传感器 link | 相对 torso 的 xyz / m | 绕 Y 的旋转 |
|---|---|---|
| `d435_link` | (0.0576235, 0.01753, 0.42987) | 0.8307767 rad，约向下 47.6° |
| `mid360_link` | (0.0002835, 0.00003, 0.41618) | 0.0401426 rad |

使用 URDF 的躯干链计算相对骨盆的完整变换，并明确转换相机机体坐标与 optical 坐标。骨盆高度假设为 0.79 m；所有关节固定为零位。**这是当前 URDF 的安装假设，实机前应核对机器人版本及实际标定。**

- D435i：RTX 理想针孔深度，640×360，水平视场 87°，保持方形像素，因此垂直视场约 56.19°；深度按光轴 Z，保留 0.3–3 m。RGB 来自同一理想视角，供检查，不模拟真实 RGB/深度相机间基线。
- MID360：对同一场景几何进行 GPU Warp raycast，水平 360°、垂直 −7° 到 +52°，每个名义 0.1 s 帧抽取 20,000 个角度网格单元，测距上限为本场景设置的 12 m。**使用随机稀疏角度采样近似，不是 Livox 的真实非重复扫描轨迹，也不是原生 RTX MID360 传感器模型。**
- 雷达的 nvblox Python 接口只提供对称视场，因此使用 ±52° 球面图并将 −7° 以下的行置为无效。每个距离必须与正确的角度单元对齐，不能将 MID360 当成固定线束旋转雷达。
- 机器人自身回波会过滤。URDF 头部 STL 是没有传感器窗口的封闭外壳，会包住 MID360 原点，故从遮挡过滤网格中排除头壳；手臂、躯干、腿等仍参与遮挡。相机观测也会与独立射线几何校验，过滤自身像素及大于 8 cm 的不一致。
- 使用仿真真值位姿，不含 SLAM、位姿漂移、扫描运动畸变、同步误差、相机缺测/噪声、反射率和透明材质影响。这里的“10 Hz”是离线观测序列的名义采样率，不是实测运行速度；为保证图像与位姿一致，每次采样在固定姿态下等待多次渲染。

在这个安装角度和运动路径下，相机主要覆盖地面及近距离低处，雷达覆盖较远墙面和较高区域。融合能补充观测，但不能自动消除近脚边、遮挡后方或未看见表面的盲区。应先观察箱体、台阶、桌腿的缺失位置，再决定采样运动和局部网络输入范围。

## 重新运行

```bash
# 一次完成采样、三组 nvblox 重建、验证、出图
./run.sh all --frames 60

# 各阶段也可单独执行
./run.sh capture --frames 90
./run.sh map
./run.sh validate
./run.sh plot

# 重用同一观测，对比体素边长（会覆盖该目录的重建结果）
./run.sh map --voxel-size 0.025
./run.sh validate
./run.sh plot

# 导出 Isaac Sim 体素截图及 USD，完成后自动退出
./run.sh view fused --headless

# 显示地面体素（默认仅隐藏 z < 0.08 m）
./run.sh view fused --voxel-min-z -0.1
```

场景、轨迹、传感器近似参数修改 `config/scene.json`。改变场景后需重新采样。`--output` 可指定另一组输出目录；单阶段运行时保持同一目录。

## 运行环境与已修复的问题

Isaac Sim 进程始终使用 `lab_v5.1` 的 Python/PyTorch 2.7；nvblox 的预编译包需要更新的 PyTorch ABI，因此建图阶段使用从该环境派生的项目内 `.mapping_env`，其中仅使用独立 PyTorch 2.9.1。两进程通过 NPZ 交换观测，不修改现有 Conda 环境。

依赖位于 `.mapping_env/`、`.deps/`、`.cuda_extra/`，未纳入版本管理。在本机首次安装或需要重建时：

```bash
bash scripts/setup_mapping.sh
```

安装脚本针对本机 Ubuntu 22.04、CUDA 12.8、`lab_v5.1` 现有的 NumPy/SciPy/Open3D/Matplotlib 环境；不是跨机器通用安装器。可用 `LAB_PYTHON=/path/to/python` 覆盖 Isaac Sim Python 路径。

固定使用 `nvblox_torch 0.0.10.dev1+cu12ubuntu22`。这个版本的 `PySensor::fromLidar` 将 `(vertical_fov, min_range)` 传给接收 `(min_range, vertical_fov)` 的 C++ 构造函数。`scripts/reconstruct.py` 做了版本限定的适配，并以墙面高度验证，防止雷达重建被压扁。升级包时必须重新检查该适配。

## 如何理解指标，以及后续 3D CNN 输入

`metrics.json` 测量的是**已选中表面体素中心到场景几何表面的距离**，受体素量化、表面带选择与场景对齐影响。本场景的轴对齐墙面甚至可能刚好经过体素中心，使该指标非常小。它**不是实际 D435i/MID360 传感器精度，不是完整率，也不能证明实机实时性能**。验证主要检查坐标变换、有效输入、已观测权重和雷达重建高度。

当前导出的表面体素适合直观检查，不应直接当作完整的策略输入。后续建议从 nvblox 的完整局部 TSDF/ESDF 层采样固定大小、以机器人为中心的网格，并至少保留 `observed/weight` 通道，避免把未知空间当作自由空间。实机阶段需接入深度内参/外参、MID360 点云去畸变与时间同步，以及可靠的里程计位姿。

参考：[nvblox 官方文档](https://nvidia-isaac.github.io/nvblox/)、[nvblox 源码](https://github.com/nvidia-isaac/nvblox)、[MID360 官方规格](https://www.livoxtech.com/mid-360/specs)、[D435i 官方规格](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html)。

## 多位置观测对比

新增五个观测位置，场景中的物体保持不变：

| 下拉菜单 | 位置 | 世界坐标 X、Y / m | 初始朝向 |
|---|---|---|---|
| `1. steps` | 台阶前方 | (1.10, −1.70) | 0° |
| `2. boxes` | 高箱侧前方 | (1.00, 1.90) | −35° |
| `3. pole` | 细柱前方 | (0.50, 0.25) | 0° |
| `4. side_wall` | 靠近侧墙 | (1.00, −2.90) | −90° |
| `5. table` | 桌子前方 | (2.25, 2.50) | 0° |

每处固定位置采集 18 帧，朝向在初始方向附近小范围转动 ±30°；每处分别从空地图开始重建相机、雷达和融合结果，**不会继承其他位置看见的表面**。体素边长仍为 5 cm。配置见 `config/positions.json`。

```bash
# 打开查看器，在 Observer position 下拉菜单切换五个位置
./run.sh view fused --output outputs/positions/steps

# 重新生成全部五个位置的观测、三组重建、校验及对比图
./run.sh positions

# 只重新生成一个位置
./run.sh positions --only table
```

新增查看器选项 `Show ground surface voxels` 可以显示地面体素，`Overlay scene ground truth` 用于检查缺失表面。切换位置会更新地图和 G1 位姿；camera/lidar/fused 三个按钮始终对应当前所选位置。原始基线仍可在下拉菜单中选择。

结果位于 `outputs/positions/`：`locations.png` 标注所有位置和朝向，`positions_comparison.png` 并列展示位置、相机首帧视角与融合体素；各子目录保存完整的观测、三种重建、截图和校验文件。总览图右列截取机器人前方的局部区域，坐标轴为前方 X、左方 Y、距地面高度 Z，仅为展示，不改变世界坐标下保存的地图。每处对应的 `comparison.png` 和 Isaac Sim 查看器显示整个重建范围。

## 机器人周围的球形区域

```bash
# 当前默认：细柱前方、虚拟点为球心、半径 1.2 m
./run.sh sphere

# 选择初始位置和半径
./run.sh sphere --output outputs/positions/table --sphere-radius 2.0

# 重新生成五个位置 × 三种半径的对比图与裁剪数据
/home/rx/miniconda3/envs/lab_v5.1/bin/python scripts/plot_spherical.py

# 离线导出 Isaac Sim 截图、USD 和裁剪后的 NPZ
./run.sh sphere --headless
```

查看器新增：

- `Crop to virtual-reference ball`：启用球形距离筛选；取消勾选可对照完整地图。
- `Sphere radius (m)`：拖动半径滑条，默认可调 0.5–3.0 m。
- `Show orange sphere boundary`：显示/隐藏橙色球形边界。它只是范围提示，不属于障碍物。
- `Focus on robot`：聚焦机器人附近；切换位置时球心、裁剪结果、机器人和视角一起更新。
- `Show ground surface voxels`：球形模式默认开启地面体素，方便检查脚前方覆盖。

当前查看器使用 `norm(voxel_center_world - virtual_origin_world) <= radius` 保留**整个球内**已观测表面，不设内半径，也不是仅保留球面。边界按体素中心判定，方块的角点可能略微越过线框。默认是全方向球形区域，没有再排除侧面或后方。

原始 nvblox 地图和传感器观测保留不变。此步骤是对已有重建的距离筛选与可视化，尚未接入实机流式更新，不会自动减少原始融合过程的计算量。蓝色仍表示已观测表面，球内的空白区域可能是未知区域。

结果在 `outputs/spherical/`：

| 文件 | 含义 |
|---|---|
| `radius_comparison.png` | 同一位置下 1.0 / 1.5 / 2.0 m 的范围对比 |
| `all_positions.png` | 五个位置 × 三个半径的完整对比 |
| `summary.json` / `validation.json` | 保留体素数量 / 几何筛选检查 |
| `位置/fused_sphere_r半径_voxels.npz` | 机体坐标体素中心、用于仿真对照的世界坐标、原 TSDF/权重、半径和骨盆位姿 |
| `view/` | `./run.sh sphere` 启动时导出的 Isaac Sim 截图、USD 和裁剪数据 |

当前 sphere 查看器导出的 `centers`、`centers_robot`、`centers_reference` 均为虚拟坐标，`coordinate_frame='sensor_rig'`；`T_W_F` / `T_P_F` / `T_anchor_F` 说明该坐标系，`centers_world` 仅保留作仿真对照。原始 `outputs/baseline/` 和 `outputs/positions/` 的完整重建文件仍使用世界坐标。早先 `plot_spherical.py` 生成的 1.0 / 1.5 / 2.0 m 对比图与各位置 NPZ 为历史骨盆参考系，`coordinate_frame='pelvis'`，不代表当前球心设置。当前零腰姿态的地面在 F 中约为 −0.59 m。显示地面开关只影响可视化。

## 刚体移动与在线更新

```bash
# 在 lab_v5.1 / Isaac Sim 5.1 中打开实时查看器，默认半径 1.2 m
./run.sh live

# 双手侧平举，固定上帝视角，斜向下俯视整个环境
./run.sh live --robot-pose tpose --inspection-angle overview --output outputs/live_overview

# 双手侧平举，虚拟球心下移 20 cm，正面观察 1.2 m 球；初始采样一次后暂停
./run.sh live --robot-pose tpose --inspection-angle front --start-paused \
  --reference-frame sensor_rig --origin-z-offset -0.2 --sphere-radius 1.2 \
  --output outputs/live_reference

# 录制一条完整的 18 秒名义路径，逐帧保存机体坐标数据
./run.sh live --headless --frames 181 --sensor-hz 10 --motion-duration 18 \
  --archive-every 1 --record --output outputs/live_demo

# 校验移动、转向后的坐标和重建几何
/home/rx/miniconda3/envs/lab_v5.1/bin/python scripts/validate_live.py --output outputs/live_demo
```

本模式不加载静态重建。从空 nvblox 地图开始，每次刚性移动 G1 和传感器，重新生成该姿态的深度及雷达观测，融合后取当前虚拟参考点周围 1.2 m 球内的表面体素。全身关节固定，不运行行走或残差控制器。默认路径经过细柱、台阶、侧墙、高箱和桌子附近；路径见 `config/motion.json`。GUI 默认沿路径运行一次后暂停。

`--robot-pose neutral` 为原 URDF 零位，`--robot-pose tpose` 为双手侧平举：左右肩 roll 分别为 +90° / −90°，双肘为 +90°（此 URDF 的零位前臂向前，需该肘角才能与侧举上臂伸直），其余关节保持零位。网格和相机/雷达自身遮挡使用同一关节姿态；传感器外参由同一关节链计算。`--inspection-angle overview` 为默认固定全场俯视；`front` / `oblique` 为可选机体近景。`--start-paused` 仅用于 live GUI，完成初始采样后保持静止，可用 Pause / resume 启动原自动路径。静态 sphere 查看器也支持姿态和视角参数，但不会重建其已存储的历史观测。

为了让实时显示的体素、球框和辅助坐标轴不进入传感器观测，本模式的相机深度由 **Warp GPU 对 Isaac 场景几何进行理想针孔射线计算**，沿用基线的 D435 视场、内参规则、URDF 外参、轴向深度、有效距离和自身遮挡过滤。它不是原离线采集的 RTX 深度图，也不模拟 D435 主动双目、噪声或缺测。MID360 仍使用既有随机稀疏角度近似；不是实际 Livox 扫描规律。用于传感器的几何只包含真实场景与自身遮挡网格，显示几何不参与采样。

地图内部使用任意局部里程计坐标 O，其原点是首次虚拟参考点位置；传感器位姿使用 `T_O_F @ T_F_sensor`，不依赖已知的绝对世界坐标。导出时通过 `p_F = R_O_F.T @ (p_O - t_O_F)` 转到当前虚拟坐标。仿真真值位姿仅用于模拟相对里程计与校验；实机仍需要可靠的相对运动和时间同步。固定传感器外参不能替代连续融合所需的里程计。

当前 URDF 中 `d435_link` 和 `mid360_link` 通过固定关节直接连接 `torso_link`；骨盆与 torso 之间有腰部 yaw / roll / pitch 关节。为使传感器外参固定，F 锚定 torso，在零腰姿态通过 `T_torso_F = inverse(T_P_torso_zero) @ translate(0,0,-0.2)` 标定位置。当前数值为 torso 中 `(0.0039635, 0, -0.244)` m，方向与零位骨盆系一致。腰部运动后 `T_P_F(q) = T_P_torso(q) @ T_torso_F` 会变化，F 不保证始终位于运动中的骨盆下方 20 cm；`T_F_C` 和 `T_F_L` 保持不变。这是软件定义的坐标系，不需要在模型或实机上增加实体点。

F 的轴随躯干转动，Z 是躯干上方向，不保证始终与重力对齐。如果需要抑制走路时的躯干晃动，应另外使用 IMU/里程计建立重力对齐的局部坐标系；传感器到该稳定系的完整变换会随姿态更新。`--reference-frame pelvis --origin-z-offset 0 --sphere-radius 1.5` 可以恢复原骨盆球心设置。`scripts/validate_reference.py` 校验 100 组腰部关节姿态与整机平移/旋转下的固定外参关系。

查看器控制：

- `Pause / resume`：暂停/恢复自动路径和采样；`Reset map + route`：清空地图、回到起点并重新运行。
- `Repeat automatic route`：重复路径；完整路径起终点相同，重复时继续融合，重新看建图过程可点击 Reset。
- 默认 `overview` 保持全场相机固定；`Restore overview` 可以恢复该视角。选择 `front` / `oblique` 时才显示 `Follow body frame`，用于切换机体近景跟随与固定全场视角。
- Forward / Back / Left / Right / Turn：按**当前机体坐标**平移 0.1 m 或转动 10°，暂停自动路径并采样一次。手动移动不带碰撞控制。
- `Update sensors at current pose`：当前位置新增一帧观测。
- `Overlay scene ground truth` 和 `Show ground surface voxels`：对照场景与地面覆盖。
- 红、绿、蓝坐标轴分别表示虚拟坐标的 X 前方、Y 左方、Z 躯干上方向；蓝色体素和橙色边界含义不变。

采样与融合通过独立常驻 nvblox 进程逐帧交换，序号和姿态必须一致，不会把一帧图像配到另一帧位姿。没有结果时继续渲染 GUI，收到结果后再推进刚体路径。`--sensor-hz` 是目标速率上限，界面显示实测更新 Hz；若算力不足，路径按名义观测时钟慢于墙钟运行。视频使用名义采样率回放，不表示严格的墙钟实时性能。

当前球形筛选限制输出范围；nvblox 内部仍保存这条有限路径沿途的历史地图，没有实现长期运行的滚动地图回收或动态障碍物清除。当前输出是机体坐标下的稀疏表面体素中心；机器人转向后，它们不再对齐机体固定轴的整齐网格，接入 3D CNN 时需要进一步重采样为固定体积的张量并保留未知状态。

输出默认在 `outputs/live/`，录制示例在 `outputs/live_demo/`：

| 文件 | 内容 |
|---|---|
| `latest_body_voxels.npz` | 最近一帧的机体坐标体素、TSDF、权重、局部里程计位姿与帧号 |
| `frames/000000.npz` 等 | 按 `--archive-every` 间隔归档的相同格式数据 |
| `history.jsonl` | 每帧观测时间、局部运动、体素数量、融合/提取耗时、实测更新率 |
| `metadata.json` | 坐标、外参和近似模型说明 |
| `scene_config.json` / `motion_config.json` / `ground_truth.ply` | 仅用于仿真复现与几何校验的场景及路径 |
| `frame_*.png` / `last_frame.png` | Isaac Sim 实际渲染的移动过程截图 |
| `motion.mp4` | 使用 `--record` 时保存的在线重建视频 |
| `metrics.json` / `validation.json` | 结束时的性能统计 / 移动坐标与场景表面对齐检查 |

在线 NPZ 的 `centers[N,3]`、`centers_robot[N,3]` 与 `centers_reference[N,3]` 都是**当前虚拟感知坐标 F**，单位米，没有 `centers_world`。当前默认 `coordinate_frame='sensor_rig'`、`reference_anchor='torso_link'`；`T_O_F[4,4]` 是 F 到局部里程计的变换，`T_O_P` / `T_P_F` 保留骨盆与 F 的关系，`T_anchor_F` / `T_F_C` / `T_F_L` 保留固定参考点和传感器外参。新虚拟参考系文件不再用含义不明确的 `T_O_B`；历史 pelvis 数据仍保留该旧键，校验脚本兼容两者。`frame_id`、`sequence_id` 和 `sensor_time_s` 用来匹配观测，其中 `sensor_time_s` 是本次运行的单调墙钟采样时间，`trajectory_time_s` 是自动路径的名义运动时间。`tsdf[N]`、`weight[N]`、`voxel_size`、`radius_m` 含义沿用前述定义。未知区域不能从缺少蓝色体素推断为空闲。
