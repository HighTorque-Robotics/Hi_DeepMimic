import numpy as np
import mujoco, mujoco_viewer
import mujoco.viewer

from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from pynput import keyboard
import random
import math
import torch
import time
class env:
    num_single_obs = 81
    num_his = 10
    num_his_obs = num_single_obs * num_his
    num_actions = 23

    joint_names_lab = ['l_hip_pitch_joint', 
                'r_hip_pitch_joint', 
                'waist_joint', 
                'l_hip_roll_joint', 
                'r_hip_roll_joint', 
                'l_shoulder_pitch_joint', 
                'r_shoulder_pitch_joint', 
                'l_thigh_joint', 
                'r_thigh_joint', 
                'l_shoulder_roll_joint', 
                'r_shoulder_roll_joint', 
                'l_calf_joint', 
                'r_calf_joint', 
                'l_upper_arm_joint', 
                'r_upper_arm_joint', 
                'l_ankle_pitch_joint', 
                'r_ankle_pitch_joint', 
                'l_elbow_joint',
                'r_elbow_joint', 
                'l_ankle_roll_joint', 
                'r_ankle_roll_joint', 
                'l_wrist_joint', 
                'r_wrist_joint']
    
    joint_names_mujoco = [
        'waist_joint',
        'l_shoulder_pitch_joint',
        'l_shoulder_roll_joint',
        'l_upper_arm_joint',
        'l_elbow_joint',
        'l_wrist_joint',
        'r_shoulder_pitch_joint',
        'r_shoulder_roll_joint',
        'r_upper_arm_joint',
        'r_elbow_joint',
        'r_wrist_joint',
        'l_hip_pitch_joint',
        'l_hip_roll_joint',
        'l_thigh_joint',
        'l_calf_joint',
        'l_ankle_pitch_joint',
        'l_ankle_roll_joint',
        'r_hip_pitch_joint',
        'r_hip_roll_joint',
        'r_thigh_joint',
        'r_calf_joint',
        'r_ankle_pitch_joint',
        'r_ankle_roll_joint',
    ]

    device = "cpu"

from mimic_real.data import PUSHUP_MOTION_DATA_DIR, WAVING_MOTION_DATA_DIR,CRAWL_MOTION_DATA_DIR,SWING_MOTION_DATA_DIR

class Sim2simCfg:
    class sim_config:
        mujoco_model_path = '/home/youyou/IsaacLab/humanoid_amp_hi/xml/hi_guogan/mjcf/hi_23dof_250425.xml'
        # model_path = '/home/youyou/IsaacLab/mimic_hi-main/logs/hi_mimic/2025-06-25_11-47-47/exported/policy.pt'
        # model_path = '/home/youyou/IsaacLab/mimic_hi-main/logs/hi_mimic/2025-07-10_15-17-27/exported/policy.pt'
        model_path = '/home/youyou/IsaacLab/mimic_hi-main/logs/hi_mimic/2025-07-18_15-17-2/exported/policy11600.pt'


        # model_path = '/home/youyou/IsaacLab/DeepMimic_hi_zixiang/logs/hi_mimic//home/youyou/mimic_hi-main/logs/hi_mimic/2025-06-25_11-47-47/exported1/policy.pt'

        # model_path = '/home/youyou/IsaacLab/mimic_hi-main/logs/hi_mimic/2025-07-15_15-02-33/exported/policy.pt'

        # mujoco_model_path = '/home/sunteng/lab_ws/mimic_real'\
        #                     + '/mimic_real/assets/urdf/hi/mjcf/hi_new.xml'
        # model_path = '/home/sunteng/lab_ws/mimic_real'\
        #                     + '/logs/hi_mimic/2025-05-22_11-36-39/exported/policy.pt'
        # motion_path = CRAWL_MOTION_DATA_DIR
        motion_path = WAVING_MOTION_DATA_DIR
        print('motion_path',motion_path)

        sim_duration = 60.0
        dt = 0.001
        decimation = 20

    class robot_config:
        kps = np.ones(env.num_actions) * 80.0
        kds = np.ones(env.num_actions) * 1.0
        tau_limit = np.ones(env.num_actions) * 20.0
        use_filter = False


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quat_apply(quat: np.array, vec: np.array) -> np.array:
    # assert quat.shape == (1, 4) and vec.shape == (1, 3)
    xyz = quat[:3]
    w = quat[-1]
    t = np.cross(xyz, vec) * 2
    return (vec + w * t + np.cross(xyz, t)) # xyz.cross(t, dim=-1)


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (qpos, dq, quat, v, omega, gvec)


def low_pass_action_filter(actions, last_actions):
  alpha = 0.2
  actons_filtered = last_actions * alpha + actions * (1 - alpha)
  return actons_filtered


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd 

def get_trans_index(joint_names_lab, joint_names_mujoco):
    assert len(joint_names_lab) == len(joint_names_mujoco)
    joint_num = len(joint_names_mujoco)
    mjc2lab_list = []
    lab2mjc_list = []
    for i in range(joint_num):
        mjc2lab_list.append(joint_names_lab.index(joint_names_mujoco[i]))
        lab2mjc_list.append(joint_names_mujoco.index(joint_names_lab[i]))
    return mjc2lab_list, lab2mjc_list

def run_mujoco():
    from motion_loader import MotionLoader
    motion_loader = MotionLoader(Sim2simCfg.sim_config.motion_path, env.joint_names_mujoco, device=env.device, add_static_frame=False)

    st1 = time.time()

    model = mujoco.MjModel.from_xml_path(Sim2simCfg.sim_config.mujoco_model_path)
    st2 = time.time()
    print("st2-st1",st2-st1)
    model.opt.timestep = Sim2simCfg.sim_config.dt
    data = mujoco.MjData(model)
    
    # import ipdb; ipdb.set_trace();

    data.qpos[-env.num_actions :] = motion_loader.get_dof_pos_batch(phase=torch.Tensor([0]))[0].cpu().numpy()
    data.qpos[0:3] = motion_loader.get_root_trans_batch(phase=torch.Tensor([0]))[0].cpu().numpy()
    data.qpos[3:7] = motion_loader.get_root_rot_batch(phase=torch.Tensor([0]))[0].cpu().numpy()

    mujoco.mj_step(model, data)
    # viewer = mujoco_viewer.MujocoViewer(model, data)


    last_action_lab = np.zeros((env.num_actions), dtype=np.double)
    action = np.zeros((env.num_actions), dtype=np.double)
    hist_obs = np.zeros([1, env.num_his_obs])
    last_tau = np.zeros((env.num_actions), dtype=np.double)
    
    # 添加平滑处理变量
    last_target_q_mujoco = np.zeros((1, env.num_actions), dtype=np.double)
    motion_finished = False
    hold_final_pose = False

    count_lowlevel = 0 

    model_path = Sim2simCfg.sim_config.model_path
    policy_jit = torch.jit.load(model_path)

    mjc2lab_list, lab2mjc_list = get_trans_index(env.joint_names_lab, env.joint_names_mujoco)
    # from mimic_real.envs.motion_loader.motion_loader import MotionLoader
   
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:

            step_start = time.time()
            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)
            # import ipdb; ipdb.set_trace();
            q = q[-env.num_actions :]
            dq = dq[-env.num_actions :]

            q_lab = np.zeros_like(q)
            dq_lab = np.zeros_like(dq)
            for i in range(env.num_actions):
                q_lab[i] = q[lab2mjc_list[i]]
                dq_lab[i] = dq[lab2mjc_list[i]]        

            phase = (count_lowlevel * Sim2simCfg.sim_config.dt) / motion_loader.record_time 
            
            # 修复相位处理逻辑
            if phase >= 1.0:
                if not motion_finished:
                    motion_finished = True
                    hold_final_pose = True
                    print("Motion finished, holding final pose...")
                phase = 0.99  # 保持在最后一帧
            else:
                motion_finished = False
                hold_final_pose = False
                
            # 1000hz -> 50hz        
            if count_lowlevel % Sim2simCfg.sim_config.decimation == 0:
                obs = np.zeros([1, env.num_single_obs])
                _q = quat
                _v = np.array([0.0, 0.0, -1.0])
                projected_gravity = quat_rotate_inverse(_q, _v)

                # 如果动作结束，使用固定的相位值避免跳跃
                if hold_final_pose:
                    phase_for_obs = 0.99
                else:
                    phase_for_obs = phase
                    
                phase2pi = 2 * 3.1415926 * phase_for_obs
                obs[0, 0:3] = omega 
                obs[0, 3:6] = projected_gravity 
                obs[0, 6:29] = q_lab 
                obs[0, 29:52] = dq_lab 
                obs[0, 52:75] = last_action_lab 
                obs[0, 75] = math.sin(phase2pi)
                obs[0, 76] = math.cos(phase2pi)
                obs[0, 77] = math.sin(2 * phase2pi)
                obs[0, 78] = math.cos(2 * phase2pi)
                obs[0, 79] = math.sin(4 * phase2pi)
                obs[0, 80] = math.cos(4 * phase2pi)

                hist_obs = np.concatenate((hist_obs[:, env.num_single_obs:], obs[:, :env.num_single_obs]), axis=-1).astype(np.float32)
                
                if not hold_final_pose:
                    action_lab = policy_jit(torch.from_numpy(hist_obs.astype(np.float32))).detach().numpy()
                    last_action_lab = action_lab
                else:
                    # 动作结束时，逐渐减小动作幅度
                    action_lab = last_action_lab * 0.95  # 逐渐衰减到0
                    last_action_lab = action_lab
                
                action = np.clip(
                    action,
                    -100.0,
                    100.0,
                )
                
                target_q_delta_lab = action_lab * 0.25
                target_q_mujoco = np.zeros_like(target_q_delta_lab)
                target_q_delta_mjc = np.zeros_like(target_q_delta_lab)
                for i in range(env.num_actions):
                    target_q_delta_mjc[0][i] = target_q_delta_lab[0][mjc2lab_list[i]] 
                
                dof_pos_ref = motion_loader.get_dof_pos_batch(phase=torch.Tensor([phase]))[0].cpu().numpy()
                for i in range(env.num_actions):
                    target_q_mujoco[0][i] = dof_pos_ref[i] + target_q_delta_mjc[0][i] 
                
                # 添加目标位置的平滑过渡
                if count_lowlevel > Sim2simCfg.sim_config.decimation:  # 不是第一次
                    alpha = 0.1  # 平滑系数
                    target_q_mujoco = alpha * target_q_mujoco + (1 - alpha) * last_target_q_mujoco
                
                last_target_q_mujoco = target_q_mujoco.copy()

            # target_q_mujoco = np.zeros((env.num_actions), dtype=np.double)
            target_dq = np.zeros((env.num_actions), dtype=np.double)
            
            tau  = pd_control(
                target_q_mujoco, q, Sim2simCfg.robot_config.kps, target_dq, dq, Sim2simCfg.robot_config.kds
            )  # Calc torques
            tau = np.clip(
                tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit
            )[0]  # Clamp torques
            
            # 添加力矩平滑
            tau_smooth_alpha = 0.2
            tau = tau_smooth_alpha * tau + (1 - tau_smooth_alpha) * last_tau
            last_tau = tau
            
            data.ctrl = tau

            mujoco.mj_step(model, data)
            count_lowlevel+=1
            
            end_time = time.time()
            
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_mujoco()