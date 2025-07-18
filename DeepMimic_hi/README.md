### 安装:
1. 获得一个装有isaaclab的环境（本项目使用的isaacsim版本为4.5, isaaclab版本为2.1.0）
2. pip install -e .

### 使用
#### 查看轨迹文件正确性
```
python mimic_real/scripts/vis_motion.py --task=hi_mimic
```
#### 训练
```
python mimic_real/scripts/train.py --task=hi_mimic --num_envs=4096 --headless --device=cuda:0
```
#### 评估
```
python mimic_real/scripts/play.py --task=hi_mimic --num_envs=2 --headless --device=cuda:0
```
