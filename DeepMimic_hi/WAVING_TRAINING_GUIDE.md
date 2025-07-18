# 挥手动作训练优化指南

## 问题分析

您遇到的机器人站立不稳问题主要源于以下原因：

1. **缺少平衡奖励**：原始配置中多数稳定性奖励被注释掉了
2. **奖励权重不当**：跟踪奖励过强，稳定性奖励过弱
3. **训练参数不合适**：学习率过高，容忍度不足

## 优化方案

### 1. 奖励函数优化

已启用并优化了以下关键奖励：

```python
# 平衡奖励（新增）
keep_balance = RewTerm(func = keep_balance, weight=0.5)

# 动作跟踪奖励（调整容忍度）
tracking_dof_pos = RewTerm(func = tracking_dof_pos, weight=2.0, params = {"std": 0.8})
tracking_capture_points = RewTerm(func = tracking_capture_points, weight=1.0, params = {"std": 0.8})

# 足部约束（重新启用）
flat_feet_force = RewTerm(func=flat_feet_force, weight=-0.05)
feet_horizontal = RewTerm(func=mdp.feet_horizontal_l2, weight=-0.5)
feet_heading = RewTerm(func=mdp.feet_heading_l2, weight=-0.5)

# 关节限制（重新启用）
joint_pos_limit = RewTerm(func = mdp.joint_pos_limits, weight = -5.0)
joint_vel_limit = RewTerm(func = mdp.joint_vel_limits, weight = -2.0)
```

### 2. 训练参数优化

```python
# 学习率降低
learning_rate=5e-4  # 从1e-3降低到5e-4

# 增加训练稳定性
num_learning_epochs=8  # 增加到8
num_mini_batches=8     # 增加到8
gamma=0.995           # 提高折扣因子
lam=0.97             # 提高GAE参数

# 梯度裁剪
max_grad_norm=0.5    # 降低到0.5
```

### 3. 环境配置优化

```python
# 减少噪声
noise_scales.ang_vel = 0.1        # 从0.2降低到0.1
noise_scales.joint_pos = 0.01     # 从0.02降低到0.01
noise_scales.joint_vel = 0.8      # 从1.5降低到0.8

# 减少初始随机性
reset_robot_joints.position_range = (-0.05, 0.05)  # 从(-0.1, 0.1)减小

# 增加摩擦力
static_friction_range = [0.8, 1.2]   # 增加摩擦力范围
dynamic_friction_range = [0.6, 1.0]  # 增加动态摩擦力
```

## 使用方法

### 1. 训练命令

使用优化后的配置训练：

```bash
cd DeepMimic_hi
python mimic_real/scripts/train.py --task=hi_mimic --num_envs=4096 --headless --device=cuda:0
```

### 2. 监控训练

使用tensorboard监控训练进度：

```bash
tensorboard --logdir logs/
```

关键指标：
- `Episode/episodic_reward`: 应该逐渐增加
- `Episode/episode_length`: 应该逐渐增加
- `Policy/mean_reward`: 应该稳定在正值
- `Policy/learning_rate`: 应该逐渐降低

### 3. 评估和测试

训练完成后，使用play脚本测试：

```bash
python mimic_real/scripts/play.py --task=hi_mimic --num_envs=2 --device=cuda:0
```

### 4. 调试建议

如果仍然遇到问题，可以按以下步骤调试：

#### 4.1 检查动作文件

```bash
python mimic_real/scripts/vis_motion.py --task=hi_mimic
```

确保waving.json文件中的动作数据合理。

#### 4.2 调整目标高度

在`keep_balance`函数中，根据您的机器人调整目标高度：

```python
target_height = 0.45  # 根据机器人实际高度调整
```

#### 4.3 降低训练难度

如果训练仍然困难，可以进一步调整：

```python
# 进一步增加容忍度
tracking_dof_pos = RewTerm(func = tracking_dof_pos, weight=2.0, params = {"std": 1.0})
tracking_capture_points = RewTerm(func = tracking_capture_points, weight=1.0, params = {"std": 1.0})

# 增加平衡奖励权重
keep_balance = RewTerm(func = keep_balance, weight=1.0)
```

#### 4.4 检查终止条件

如果机器人频繁终止，可以调整：

```python
# 增加终止容忍度
self.terminate.capture_points_distance_threshold = 1.0  # 从0.8增加到1.0
```

## 预期效果

经过优化后，您应该看到：

1. **更稳定的站立**：机器人能够保持平衡而不倒下
2. **更收敛的曲线**：tensorboard中的奖励曲线应该更平滑
3. **更长的回合**：每个回合的长度应该增加
4. **更好的挥手动作**：在保持平衡的同时执行挥手动作

## 常见问题

### Q1: 训练速度很慢怎么办？
A1: 可以减少环境数量，例如使用2048个环境：
```bash
python mimic_real/scripts/train.py --task=hi_mimic --num_envs=2048 --headless --device=cuda:0
```

### Q2: 显存不足怎么办？
A2: 进一步减少环境数量或使用更小的网络：
```python
actor_hidden_dims=[256, 128, 64]
critic_hidden_dims=[256, 128, 64]
```

### Q3: 机器人仍然不稳定怎么办？
A3: 增加平衡奖励权重并减少动作跟踪权重：
```python
keep_balance = RewTerm(func = keep_balance, weight=1.0)
tracking_dof_pos = RewTerm(func = tracking_dof_pos, weight=1.0, params = {"std": 1.0})
```

## 总结

通过这些优化，您的机器人应该能够：
- 保持稳定的站立姿态
- 执行平滑的挥手动作
- 获得更好的训练收敛性

如果还有问题，请检查动作数据文件是否合理，并根据您的具体机器人型号调整参数。 