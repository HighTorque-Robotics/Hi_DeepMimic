import numpy as np
import json
import joblib

loaded_data = joblib.load("mimic_real/data/hi/pushup.pkl")

root_trans = loaded_data["default"]["root_trans_offset"]
root_rot = loaded_data["default"]["root_rot"][:, [3, 0, 1, 2]]
capture_points = loaded_data["default"]["capture_points"]
dof_pos = loaded_data["default"]["dof"]
fps = loaded_data["default"]["fps"]
capture_points_link_names = loaded_data["marker_link_names"]
mjcf_joint_names = loaded_data["mjcf_joint_names"]
data = {
    "fps": fps,
    "target_link_names": capture_points_link_names,
    "data_joint_names": mjcf_joint_names,
    "root_trans": root_trans.tolist(),
    "root_wxyz": root_rot.tolist(),
    "target_link_pos": capture_points.tolist(),
    "dof_pos": dof_pos.tolist(),
}
with open('mimic_real/data/hi/pushup.json', 'w') as f:
    json.dump(data, f, indent=2)

# 创建一个包含 NumPy 数组的字典
# data = {
#     "integers": np.array([1, 2, 3, 4]),
#     "floats": np.array([1.5, 2.5, 3.5], dtype=np.float32),
#     "matrix": np.array([[1, 2], [3, 4]]),
# }

# # 转换为可序列化的字典
# serializable_data = {}
# for key, value in data.items():
#     if isinstance(value, np.ndarray):
#         serializable_data[key] = value.tolist()
#     else:
#         serializable_data[key] = value

# # 保存为 JSON
# with open('data.json', 'w') as f:
#     json.dump(serializable_data, f, indent=2)

# # 从 JSON 加载并恢复
with open('mimic_real/data/hi/pushup.json', 'r') as f:
    loaded_data = json.load(f)
import ipdb; ipdb.set_trace();

# restored_data = {}
# for key, value in loaded_data.items():
#     restored_data[key] = np.array(value)

# # 验证恢复结果
# print("原始数据类型:", {k: type(v) for k, v in data.items()})
# print("恢复数据类型:", {k: type(v) for k, v in restored_data.items()})
# print("恢复的矩阵:\n", restored_data["matrix"])