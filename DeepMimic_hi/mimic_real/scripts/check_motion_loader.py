from mimic_real.envs.motion_loader.motion_loader import MotionLoader
import torch
data_file_path = "mimic_real/data/hi/walk.pkl"
motion_loader = MotionLoader(data_file_path)
# print(motion_loader.root_trans.shape)
phase = torch.ones((12)) * 0.99
print(motion_loader.get_root_trans_batch(phase))
# print(motion_loader.get_frame_ids_batch(phase))