import numpy as np
import torch
import matplotlib.pyplot as plt
from motion_loader import MotionLoader
from mimic_real.data import WAVING_MOTION_DATA_DIR

def test_phase_transition():
    """测试相位过渡逻辑是否工作正常"""
    
    # 设置
    joint_names_mujoco = [
        'waist_joint', 'l_shoulder_pitch_joint', 'l_shoulder_roll_joint',
        'l_upper_arm_joint', 'l_elbow_joint', 'l_wrist_joint',
        'r_shoulder_pitch_joint', 'r_shoulder_roll_joint', 'r_upper_arm_joint',
        'r_elbow_joint', 'r_wrist_joint', 'l_hip_pitch_joint',
        'l_hip_roll_joint', 'l_thigh_joint', 'l_calf_joint',
        'l_ankle_pitch_joint', 'l_ankle_roll_joint', 'r_hip_pitch_joint',
        'r_hip_roll_joint', 'r_thigh_joint', 'r_calf_joint',
        'r_ankle_pitch_joint', 'r_ankle_roll_joint',
    ]
    
    # 加载动作数据
    motion_loader = MotionLoader(WAVING_MOTION_DATA_DIR, joint_names_mujoco, device="cpu")
    
    # 模拟相位变化
    dt = 0.001
    decimation = 20
    record_time = motion_loader.record_time
    
    # 记录数据用于绘图
    phases = []
    joint_positions = []
    smooth_joint_positions = []
    
    last_phase = 0.0
    last_target_q = None
    in_transition = False
    transition_steps = 10
    transition_counter = 0
    
    # 模拟2个完整的动作循环
    total_steps = int(2 * record_time / dt)
    
    for step in range(total_steps):
        # 计算原始相位
        phase_raw = (step * dt) / record_time
        phase_raw = phase_raw % 1.0
        
        # 检测相位跳跃
        if step > 0:
            phase_diff = phase_raw - last_phase
            if phase_diff < -0.5:  # 检测到跳跃
                in_transition = True
                transition_counter = 0
                print(f"Step {step}: Phase jump detected: {last_phase:.3f} -> {phase_raw:.3f}")
        
        # 获取关节位置
        if step % decimation == 0:  # 只在控制频率时计算
            # 原始方法（会有跳跃）
            dof_pos_original = motion_loader.get_dof_pos_batch(phase=torch.Tensor([phase_raw]))[0].cpu().numpy()
            
            # 改进方法（平滑过渡）
            if in_transition and transition_counter < transition_steps:
                blend_factor = transition_counter / transition_steps
                dof_pos_end = motion_loader.get_dof_pos_batch(phase=torch.Tensor([0.99]))[0].cpu().numpy()
                dof_pos_start = motion_loader.get_dof_pos_batch(phase=torch.Tensor([0.0]))[0].cpu().numpy()
                dof_pos_smooth = dof_pos_end * (1 - blend_factor) + dof_pos_start * blend_factor
                transition_counter += 1
                if transition_counter >= transition_steps:
                    in_transition = False
            else:
                dof_pos_smooth = motion_loader.get_dof_pos_batch(phase=torch.Tensor([phase_raw]))[0].cpu().numpy()
            
            # 添加目标位置平滑
            if last_target_q is not None:
                smooth_alpha = 0.1
                dof_pos_smooth = (1 - smooth_alpha) * dof_pos_smooth + smooth_alpha * last_target_q
            
            last_target_q = dof_pos_smooth.copy()
            
            # 记录数据
            phases.append(phase_raw)
            joint_positions.append(dof_pos_original)
            smooth_joint_positions.append(dof_pos_smooth)
        
        last_phase = phase_raw
    
    # 转换为numpy数组
    phases = np.array(phases)
    joint_positions = np.array(joint_positions)
    smooth_joint_positions = np.array(smooth_joint_positions)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 选择几个关节进行比较
    joint_indices = [0, 5, 10, 15]  # waist, l_wrist, r_wrist, l_ankle_pitch
    joint_names = ['waist', 'l_wrist', 'r_wrist', 'l_ankle_pitch']
    
    for i, (joint_idx, joint_name) in enumerate(zip(joint_indices, joint_names)):
        plt.subplot(2, 2, i+1)
        plt.plot(phases, joint_positions[:, joint_idx], 'r-', alpha=0.7, label='Original (with jumps)')
        plt.plot(phases, smooth_joint_positions[:, joint_idx], 'b-', label='Smoothed')
        plt.xlabel('Phase')
        plt.ylabel('Joint Position (rad)')
        plt.title(f'{joint_name} Joint Position')
        plt.legend()
        plt.grid(True)
        
        # 计算跳跃幅度
        original_diff = np.diff(joint_positions[:, joint_idx])
        smooth_diff = np.diff(smooth_joint_positions[:, joint_idx])
        max_original_jump = np.max(np.abs(original_diff))
        max_smooth_jump = np.max(np.abs(smooth_diff))
        
        print(f"{joint_name}: Max original jump = {max_original_jump:.4f}, Max smooth jump = {max_smooth_jump:.4f}")
    
    plt.tight_layout()
    plt.savefig('phase_transition_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("测试完成，结果已保存为 phase_transition_test.png")

if __name__ == "__main__":
    test_phase_transition() 