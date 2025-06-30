
# Hi_DeepMimic

视频流数据 --[gvhmr]--> SMPL运动数据 --[运动学重定向]--> 机器人运动数据 --[mimic训练]--> 运动策略


# 1. GVHMR安装

（1）下载链接：https://github.com/zju3dv/GVHMR
安装后将原项目/path/yo/GVHMR/tools/demo路径下的demo.py的内容替换为/path/to/retarget_lab路径下的demo.py的内容
即可在识别数据时生成对应格式SMPL运动文件


（2）根据https://github.com/zju3dv/GVHMR 下载相关依赖，数据集训练和评估的数据集不需要下载


（3）对单个视频进行演示：python tools/demo/demo.py --video=docs/example_video/name.mp4 -s
#### 注意：name.mp4可以是自己录制的视频流，需要放在GVHMR/docs/example_video路径下,执行命令后会生成npz文件。

# 2. 重定向：retarget_lab安装

## 2.1 环境配置

#### 克隆本仓库，解压retarget_lab.zip文件，放入主目录下

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

##2.2 GVHMR生成SMPL运动数据 (生成pkl文件)

将生成的npz文件放在data/from_video，在display_amass.py下找到"amass_data = load_amass_data("data/from_video/mgg_walk.npz")"，修改npz文件名为对应的data/from_vide下的文件名，最后执行下方指令，输出SMPL运动文件（pkl文件）：
```
python scripts/display_amass.py
```  

## 2.3 生成json文件
按照提示输入pkl文件（刚才的pkl文件名，注意加上后缀），进行运动学重定向，输出json文件（文件名自己命名，注意加上后缀）：
```
python scripts/retargeting.py
```  

# 3. DeepMimic训练代码：

#### 克隆本仓库，解压DeepMimic_hi.zip文件，放入主目录下

### 3.1 安装DeepMimic_hi

#### 获得一个装有isaaclab的环境（本项目使用的isaacsim版本为4.5, isaaclab版本为2.1.0）

进入DeepMimic_hi目录下
```
pip install -e .
```

### 3.2 使用

#### 查看轨迹文件正确性
```
python mimic_real/scripts/vis_motion.py --task=hi_mimic
```
#### 训练
在DeepMimic_hi/mimic_real/envs/mimic/hi_mimic_config下将self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/crawl.json"这一行的"/hi/crawl.json"更换为自己的json文件名，并且将json文件放入DeepMimic_hi/mimic_real/data/hi目录下,准备就绪后输入下方指令进行训练
```
python mimic_real/scripts/train.py --task=hi_mimic --num_envs=4096 --headless --device=cuda:0
```
#### 评估
输入下方指令后，会得到onnx文件被放在logs下日期的文件夹里的exported文件夹下
```
python mimic_real/scripts/play.py --task=hi_mimic --num_envs=2 --headless --device=cuda:0
```






