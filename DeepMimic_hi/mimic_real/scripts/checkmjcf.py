import mujoco
import numpy as np

def print_joint_info(model_path):
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"成功加载模型: {model_path}")
    
    # 获取关节数量
    n_joints = model.njnt
    
    if n_joints == 0:
        print("模型中未找到关节。")
        return

    # 打印表头
    print("\n关节信息 (按模型定义顺序):")
    print(f"{'序号':<5} {'名称':<20} {'类型':<10} {'位置 (x, y, z)':<25} {'轴 (x, y, z)' if model.jnt_axis.size > 0 else ''}")
    print("-" * 80)
    
    # 遍历并打印每个关节
    for i in range(n_joints):
        joint_id = i
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(joint_name)
            

if __name__ == "__main__":
    model_path = '/home/sunteng/lab_ws/mimic_real'\
                            + '/mimic_real/assets/urdf/hi/mjcf/hi_23dof_250425.xml'
    print_joint_info(model_path)    