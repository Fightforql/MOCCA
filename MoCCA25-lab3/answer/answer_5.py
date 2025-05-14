# 姓名：梁书怡
# 学号：2300013147
import numpy as np
from scipy.spatial.transform import Rotation

from joints import Joint

from typing import List, Tuple

def PD_control(x: np.ndarray, R: np.ndarray, v: np.ndarray, w: np.ndarray,
               joints: List[Joint], kp: np.ndarray | float, kd: np.ndarray | float,
               target_joint_rotations: np.ndarray) -> np.ndarray:
    num_joints = len(joints)
    joint_torques = np.zeros((num_joints, 3))

    for i, joint in enumerate(joints):
        bodyA = joint.bodyA
        bodyB = joint.bodyB

        # 当前旋转：B 相对于 A 的旋转
        R_A = R[bodyA]
        R_B = R[bodyB]
        R_current = R_A.T @ R_B

        # 旋转误差：目标 * 当前^T
        R_target = target_joint_rotations[i]
        R_delta = R_target @ R_current.T

        # 转为旋转向量（轴角），表示目标-当前的差值
        delta_q = Rotation.from_matrix(R_delta).as_rotvec()
        delta_q = R[bodyA] @ delta_q
        
        joint_avel = w[bodyB] - w[bodyA]

        # PD 控制力矩（注意表示在世界坐标系下）
        tau = kp * delta_q - kd * joint_avel
        joint_torques[i] = tau

    return joint_torques