# xArm 6 + Gemini 336 + Robotiq 2F-85 抓取

新电脑接手部署、硬件联调、标定和首次抓取验收请先阅读 [HANDOFF_GUIDE_ZH.md](HANDOFF_GUIDE_ZH.md)。

独立于已有 ROS 商超工程的 Python 入口。控制器 IP 已设为 `192.168.1.209`，相机安装在腕部，夹爪通过 xArm 末端 Modbus 工具接口控制。

当前实现：输入商品名称/别名 → 对应观察位姿 → RGB-D 对齐 → YOLO 识别 → 深度反投影 → 眼在手上坐标转换 → 保持观察位的 TCP 姿态 → 工具 Y 左右对齐 → 工具 Z 上下对齐 → 工具 X 深入 → 闭合并检查接触 → 沿工具 Z 轻抬 → 沿工具 X 原路退出。程序不估计或改变瓶子抓取姿态，不含自动放置或避障规划。

## 1. 安装

当前相机代码使用工作区 `../pyorbbecsdk` 对应的 Orbbec SDK v2 API。Git clone 得到的是源码和原生 SDK 文件，不能直接作为 Python 扩展导入；项目依赖中的 `pyorbbecsdk2` 安装后仍使用 `import pyorbbecsdk`。推荐使用项目虚拟环境：

```powershell
cd D:\Robotics\xarm-shop-demo\shop-workspace
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windows 首次使用 Gemini 336 时，需要以管理员权限执行仓库自带的官方 UVC metadata 注册脚本，随后拔插相机：

```powershell
.\.venv\Scripts\python.exe ..\pyorbbecsdk\scripts\env_setup\setup_env.py
```

先用独立诊断命令确认型号、彩色格式、软件 D2C 对齐、深度尺度和内参；该命令不连接机械臂：

```powershell
.\.venv\Scripts\python.exe -m xarm_grasp.check_camera
```

成功后检查 `runs/camera_check/color.jpg` 和 `camera.json`。标定与商品识别共用同一个 Gemini 336 相机封装，因此诊断通过后两者使用完全相同的彩色转换和对齐深度数据。本地 xArm SDK自动从相邻目录加载。


## 2. 配置商品和观察位姿

编辑 `config.json`。没有测量的参数用 `null`，不会使用猜测值运动。

- `observation`: 关节模式 `{"type":"joint","values":[J1,J2,J3,J4,J5,J6]}`，单位度；TCP 模式 `{"type":"tcp","values":[x,y,z,roll,pitch,yaw]}`，单位毫米/度，基座坐标系。数组中这里的字母表示需要替换的实测数值。
- `robot.tcp_offset`: 法兰到实际抓取 TCP 的 `[x,y,z,roll,pitch,yaw]`。先在 xArm Studio 完成工具 TCP/负载设置，再将 TCP 偏移填入；程序检查控制器设置一致，不自动覆盖。TCP 应为夹爪实际抓取参考点。
- `grasp_depth_offset_mm`: 从深度相机测到的瓶子正面表面，沿本次工具 X 深入方向继续走到瓶子夹持中心的距离，填正数，单位毫米。通常接近瓶身在抓取高度处的半径，必须按实物和 TCP 定义测量。
- `open_position` / `close_position`: Robotiq 原始位置 0..255，0 全开、255 全闭，**不是毫米**。当前夹爪速度/力均设为原始值 64。
- `tool_motion.lift_mm`: 闭合夹爪后的轻抬距离。程序根据工具 Z 与基座 +Z 的夹角自动选择 Z 正负方向。
- `tool_motion.axis_deadband_mm`: 左右/上下对齐误差小于此数值时跳过该轴动作。
- `tool_motion.max_alignment_mm` 和 `min/max_approach_mm`: 分轴移动的安全限值；超出时终止，不执行抓取。
- `tool_motion.vertical_axis_min_up_component`: 确认观察姿态的工具 Z 足够接近竖直方向。默认 0.7 约等于与竖直方向夹角不超过 45°。

手动将机械臂移动到观察位置并停稳后，可用只读命令获得配置片段：

```powershell
python -m xarm_grasp.teach --type joint
# 或
python -m xarm_grasp.teach --type tcp
```

该命令只连接并读取机械臂，不使能、不移动。把输出的 `observation` 整段复制到对应商品配置。同一观察区域中的商品可以共用一个观察位姿。观察姿态必须使工具 Y 对应左右、工具 Z 对应上下、工具 X 指向瓶子；执行前先在 xArm Studio 的工具坐标点动模式下分别验证三个轴的物理方向。轴角色由 `lateral_axis`、`vertical_axis`、`depth_axis` 配置，当前分别为 Y、Z、X。

## 3. 眼在手上标定

固定棋盘格到桌面/支架，在整个采集过程中保持不动。测量格子边长，`cols`/`rows` 是**内角点**数量。使用不对称行列数量的棋盘格，保持角点原点方向一致，避免棋盘格翻转导致编号反向。相机支架必须刚性固定。

在 xArm Studio 手动示教到不同位姿并停稳，每个位置运行一次（示例为 9×6 内角点、25 mm 格子，需要按实物修改）：

```powershell
python -m xarm_grasp.calibrate capture --cols 9 --rows 6 --square-mm 25
```

此命令只读取机械臂和相机，不使能、不运动。采集至少 12 组，建议 15–25 组，覆盖多个距离、位置以及绕不同轴的明显旋转，保持棋盘完整可见；仅平移或绕单轴旋转不足以标定。样本和角点检查图保存在 `calibration/`。

```powershell
python -m xarm_grasp.calibrate solve
```

结果是 `calibration/result.json` 中的 `T_flange_camera`：**彩色相机坐标 → 法兰坐标**的 4×4 齐次矩阵，平移单位 mm。求解有旋转多样性检查、棋盘位置/朝向残差检查，但这些是样本内部一致性检查，不代表真实抓取精度已验收。

把结果中的 `camera_serial`、`T_flange_camera` 复制到 `config.json` 的 `calibration` 对应字段。用未参与标定的位姿和已知固定点验证基座三维位置误差，确认满足抓取精度后才将 `calibration.validated` 设为 `true`。相机重新安装后重新标定。求解不会自动开启运动。

坐标链为：

```text
T_base_camera = T_base_tcp × inverse(T_flange_tcp) × T_flange_camera
P_base = T_base_camera × P_camera
```

深度先对齐到彩色图，使用彩色内参及支持的畸变模型反投影；深度原始值乘 SDK scale 转为 mm。无效深度、深度跳变、多目标实例会终止本次抓取。

## 4. 先视觉，后执行

安装依赖和提供模型后，默认模式只使用相机，不连接机械臂，不需要手眼标定：

```powershell
python -m xarm_grasp --item demo
```

输出包含目标框、相机坐标三维点；保存 `runs/latest/detection.jpg`、`depth_mm.npy`、`result.json`。该模式在相机当前视角识别，不移动到观察位姿。多次运行默认覆盖 latest，可用 `--output runs/自定义名称` 分开保存。

完成标定、TCP、商品抓取参数、实测工作范围和实际路径检查后，将 `motion_enabled` 设为 `true`；在 Studio 确认机器人无错误且已退出停止/暂停状态。现场执行：

```powershell
python -m xarm_grasp --item demo --execute
```

会发生真实机械臂和夹爪运动。机器人异常会停止运动，不自动清错、自动重试或松开已经夹持的物体。完成后夹持保持，当前不包含放置动作。

工作范围和逆解检查覆盖每个分轴动作的目标 TCP 点；**不检查动作之间整条路径、机械臂连杆、相机/夹爪外形和货架碰撞**。观察关节运动途中以及 Y/Z/X 各段路径需要现场点动验证。第一轮应在无障碍、低速条件下检查工具轴方向和实际轨迹。

## 5. 离线验证

```powershell
python -m unittest discover -s tests -v
```

测试使用合成数据及模拟机器人，不连接硬件。覆盖 TCP 偏移消除、坐标转换、深度失败、固定姿态分轴规划、纯工具平移命令、工作范围、禁止未标定运动、空抓检测、异常停止和手眼标定恢复。SDK/固件、RGB-D 实机对齐和真实抓取仍需要现场验证。





