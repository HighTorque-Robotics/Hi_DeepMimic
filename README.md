# Hi_DeepMimic

![power](https://github.com/HighTorque-Robotics/Hi_DeepMimic/blob/main/%E9%A3%9E%E4%B9%A620250711-153009.gif)

## 流程：SMPL运动数据 --[运动学重定向]--> 机器人运动数据 --[mimic训练]--> 运动策略

## 1. 重定向：retarget_lab安装

### 1.1 环境配置

#### 克隆本仓库，在retarget_lab文件夹下打开终端

创建名为retargeting的conda虚拟环境，指定Python版本为3.12：
```
 conda create -n retargeting python=3.12 
```
激活retargeting环境，后续命令在该环境中执行：
```
conda activate retargeting
```             

通过requirements.txt文件安装项目依赖包：
```  
pip install -r requirements.txt         
```

以可编辑模式安装smpl_skeleton包：
```  
pip install -e smpl_skeleton           
```

以可编辑模式安装pyroki包：
```  
pip install -e pyroki                  
```

### 1.2 AMASS数据集生成SMPL运动数据 (生成pkl文件)

找到retarget_lab/data/from_video路径下的npz文件，在display_amass.py下找到这一行
```
amass_data = load_amass_data("data/from_video/crawl.npz")
```
修改npz文件名为对应的data/from_vide下的文件名，最后执行下方指令，输出SMPL运动文件（pkl格式）：
```
python scripts/display_amass.py
```  

### 1.3 PHC重定向生成SMPL运动数据 
按照提示输入刚才生成的pkl格式文件（刚才的pkl格式文件名，注意加上后缀），进行运动学重定向，输出json文件（文件名自己命名，注意加上后缀）：
```
python scripts/retargeting.py
```  

## 2. DeepMimic训练代码：

#### 克隆本仓库，在DeepMimic_hi文件夹下打开终端

### 2.1 安装DeepMimic_hi

#### 安装isaacsim和isaaclab环境（本项目使用的isaacsim版本为4.5, isaaclab版本为2.1.0）

安装完成后，进入DeepMimic_hi目录下
```
pip install -e .
```

#### 2.2 查看轨迹文件正确性
```
python mimic_real/scripts/vis_motion.py --task=hi_mimic
```
![vis_motion](https://github.com/HighTorque-Robotics/Hi_DeepMimic/blob/main/%E9%A3%9E%E4%B9%A620250710-101616.gif)

#### 2.3 训练
在DeepMimic_hi/mimic_real/envs/mimic路径下的hi_mimic_config.py中找到这一行
```
self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/crawl.json"
```
并且检查json文件已放入DeepMimic_hi/mimic_real/data/hi目录下,准备就绪后输入下方指令进行训练
```
python mimic_real/scripts/train.py --task=hi_mimic --num_envs=4096 --headless --device=cuda:0
```
在训练过程中建议
#### 2.4 评估
输入下方指令后，会得到onnx文件被放在logs下日期的文件夹里的exported文件夹下
```
python mimic_real/scripts/play.py --task=hi_mimic --num_envs=2 --headless --device=cuda:0
```






