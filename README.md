# Hi_DeepMimic

视频流数据 --[gvhmr]--> SMPL运动数据 --[运动学重定向]--> 机器人运动数据 --[mimic训练]--> 运动策略

#安装

# 1. GVHMR安装

（1）下载链接：https://github.com/zju3dv/GVHMR
安装后将原项目/path/yo/GVHMR/tools/demo路径下的demo.py的内容替换为/path/to/retarget_lab路径下的demo.py的内容
即可在识别数据时生成对应格式SMPL运动文件

（2）根据https://github.com/zju3dv/GVHMR 下载相关依赖，数据集训练和评估的数据集不需要下载，Install只用到下方截图这里
![341839ed-56eb-400a-8721-449c08494c46](https://github.com/user-attachments/assets/9f7692dd-59dd-414e-afb0-a2a90cf1cd7c)

（3）对单个视频进行演示：python tools/demo/demo.py --video=docs/example_video/name.mp4 -s
#### 注意：name.mp4可以是自己录制的视频流，需要放在GVHMR/docs/example_video路径下

# 2. 重定向：retarget_lab安装

##2.1 环境配置

创建名为retargeting的conda虚拟环境，指定Python版本为3.12：
‘’‘
conda create -n retargeting python=3.12 
‘’‘

激活retargeting环境，后续命令在该环境中执行：conda activate retargeting               

通过requirements.txt文件安装项目依赖包：pip install -r requirements.txt         

以可编辑模式安装smpl_skeleton包：pip install -e smpl_skeleton           

以可编辑模式安装pyroki包：pip install -e pyroki                  

##2.2 GVHMR生成SMPL运动数据 (生成pkl文件)

输出SMPL运动文件：python scripts/display_amass.py

##2.3 生成json文件
进行运动学重定向：python scripts/retargeting.py

# 3. DeepMimic训练代码：

### 3.1 安装mimic_hi
1. 获得一个装有isaaclab的环境（本项目使用的isaacsim版本为4.5, isaaclab版本为2.1.0）
2. pip install -e .

### 3.2 使用
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






