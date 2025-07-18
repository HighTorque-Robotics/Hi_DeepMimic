# 临时readme

视频流数据 --[gvhmr]--> SMPL运动数据 --[运动学重定向]--> 机器人运动数据 --[mimic训练]--> 运动策略

### 安装

1. GVHMR安装
https://github.com/zju3dv/GVHMR 
安装后将原项目demo.py替换为[text](demo.py)
即可在识别数据时生成对应格式SMPL运动文件

2. 本项目安装
```
conda create -n retargeting python=3.12
conda activate retargeting
pip install -r requirements.txt
pip install -e smpl_skeleton
pip install -e pyroki
```
3. mimic代码：
https://github.com/HighTorque-Locomotion/mimic_hi

### 本项目重定向流程
1.
GVHMR生成SMPL运动数据

2.
``` 
python scripts/display_amass.py # 输出SMPL运动文件
```
3. 
```
python scripts/retargeting.py # 进行运动学重定向
```
