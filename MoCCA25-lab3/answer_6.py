# 姓名：梁书怡
# 学号：2300013147
import numpy as np
from scipy.spatial.transform import Rotation

from joints import Joint

from typing import List, Tuple

from answer_5 import PD_control
from answer_4 import gravity_compensation


def CCD(x:np.ndarray,joints:List[Joint],positions:np.ndarray,
        orientations:np.ndarray,target_position:np.ndarray,rotations:np.ndarray,offsets:np.ndarray,max_times=10):
    end_position=positions[-1]
    distance = np.sqrt(np.sum(np.square(end_position - target_position)))
    num_joints=len(joints)
    
    while distance > 0.01:
        for i in range(num_joints-1, 0, -1):
            bodyA=joints[i].bodyA
            bodyB=joints[i].bodyB
            #print(i,bodyA)
            #与空间相连的那个关节是固定的
            if bodyA == -1:
                continue
            
            #求出向量的旋转轴和旋转角
            a = target_position - positions[i]
            b = positions[-1] - positions[i]
            axis = np.cross(b, a)
            axis = axis / np.linalg.norm(axis)

            cos =np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
            cos = np.clip(cos, -1.0, 1.0)
            angle = np.arccos(cos)
            if angle < 0.0001:
                continue
            rotation = Rotation.from_rotvec(angle * axis).as_matrix()
           
            orientations[i] = rotation @ orientations[i]
            rotations[i] = np.linalg.inv(orientations[i - 1])@ orientations[i]
            for j in range(i + 1, num_joints):
                orientations[j] = orientations[j - 1] @ rotations[j]
                positions[j] = np.dot(orientations[j - 1], offsets[j]) + positions[j - 1]
            distance = np.sqrt(np.sum(np.square(positions[-1] - target_position)))
            #print(positions)
            print(distance)
    return positions,orientations,rotations




def end_effector_track_control1(
    m:np.ndarray, I:np.ndarray, g:np.ndarray, 
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    joints:List[Joint], 
    end_effector_body:int, end_effector_from_body:np.ndarray,
    target_end_effector_pos:np.ndarray, target_end_effector_vel:np.ndarray,
    ) -> np.ndarray:
    ''' 末端点位置控制器
    
    计算每个关节的控制力矩，使得机械臂的末端点位置跟踪目标位置。

    输入：
        m: (num_bodies)       刚体的质量
        I: (num_bodies, 3, 3) 刚体的惯性矩阵
        g: (3)                重力加速度
        x: (num_bodies, 3)    刚体（质心）的位置
        R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
        v: (num_bodies, 3)    刚体的线速度
        w: (num_bodies, 3)    刚体的角速度
        joints:               关节定义列表，包含 num_joints 个关节
        
        end_effector_body: int       末端点所在的刚体索引
        end_effector_from_body: (3)  末端点在刚体坐标系下的位置
        target_end_effector_pos: (3) 目标末端点位置
        target_end_effector_vel: (3) 目标末端点速度
    
    输出：
        joint_torques: (num_joints, 3) 每个关节的控制力矩
        
        注意，我们假设 bodyA 是父刚体，bodyB 是子刚体。joint_torques[j] 是施加在 bodyB 上的力矩，
        bodyA 获得的力矩为 -joint_torques[j]。
        
    提示：
        * 你可以综合使用 PD_control 和 gravity_compensation 来实现末端点位置控制器，也可以使用你认为合适的其他方法
        * 可以考虑的实现思路：
            1. 基于 Jacobian transpose 方法
                a. 计算重力补偿力矩 tau_g，使得机械臂保持静止
                b. 根据当前末端点位置和目标末端点位置，利用 PD 控制计算虚拟力 f_end，使得末端点位置跟踪目标位置
                c. 利用 Jacobian transpose 方法将 f_end 转换为关节力矩 tau
            2. 基于 PD 控制方法
                a. 根据目标末端点位置，利用 IK 计算目标关节旋转
                b. 计算关节 PD 控制力矩 tau，跟踪 IK 得到的目标关节旋转
            3. 基于 PD + 重力补偿方法
                a. 在 2 中的基础上，计算重力补偿力矩 tau_g，使得机械臂保持静止
            4. 其他方法
                a. 你可以使用其他方法来实现末端点位置控制器，比如基于模型预测控制的方法等
    '''
    def get_init_positions():
        positions=np.zeros((num_joints,3))
        for i in range(1,len(joints)):
            bodyA=joints[i].bodyA
            orientation_p=R[bodyA]
            positions[i]=Rotation.from_matrix(orientation_p).apply(joints[i].from_bodyA)+x[bodyA]
        return positions
    
    def get_rotations():
        rotations=np.zeros((num_joints,3,3))
        rotations[0]=R[joints[0].bodyB]
        for i in range(1,len(joints)):
            bodyA=joints[i].bodyA
            bodyB=joints[i].bodyB
            rotations[i]=R[bodyB]@np.linalg.inv(R[bodyA])
        return rotations
    
    def get_offsets():
        offsets=np.zeros((num_joints,3))
        for i in range(1,len(joints)):
            offsets[i]=Rotation.from_matrix(R[joints[i].bodyA]).apply(joints[i].from_bodyA)-Rotation.from_matrix(R[joints[i].bodyA]).apply(joints[i-1].from_bodyB)
        return offsets
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
        
    # 计算当前末端点位置和速度
    end_effector_from_body_global = R[end_effector_body] @ end_effector_from_body
        
    current_end_effector_pos = x[end_effector_body] + end_effector_from_body_global
    current_end_effector_vel = v[end_effector_body] + np.cross(w[end_effector_body], end_effector_from_body_global)
    
    # 需要施加的关节力矩
    joint_torques = np.zeros((num_joints, 3))
    
    #---------------------------------利用CCD IK先计算出目标关节的旋转---------------------------------
    positions=get_init_positions()
    offsets=get_offsets()
    rotations=get_rotations()
    orientations=R[:-1]
    npositions,norientations,nrotations=CCD(x,joints,positions,orientations,target_end_effector_pos,rotations,offsets)


    #---------------------------------------------------------------
    
    return joint_torques
kp=1000
kd=400




def end_effector_track_control(
    m:np.ndarray, I:np.ndarray, g:np.ndarray, 
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    joints:List[Joint], 
    end_effector_body:int, end_effector_from_body:np.ndarray,
    target_end_effector_pos:np.ndarray, target_end_effector_vel:np.ndarray,
    ) -> np.ndarray:
    ''' 末端点位置控制器
    
    计算每个关节的控制力矩，使得机械臂的末端点位置跟踪目标位置。

    输入：
        m: (num_bodies)       刚体的质量
        I: (num_bodies, 3, 3) 刚体的惯性矩阵
        g: (3)                重力加速度
        x: (num_bodies, 3)    刚体（质心）的位置
        R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
        v: (num_bodies, 3)    刚体的线速度
        w: (num_bodies, 3)    刚体的角速度
        joints:               关节定义列表，包含 num_joints 个关节
        
        end_effector_body: int       末端点所在的刚体索引
        end_effector_from_body: (3)  末端点在刚体坐标系下的位置
        target_end_effector_pos: (3) 目标末端点位置
        target_end_effector_vel: (3) 目标末端点速度
    
    输出：
        joint_torques: (num_joints, 3) 每个关节的控制力矩
        
        注意，我们假设 bodyA 是父刚体，bodyB 是子刚体。joint_torques[j] 是施加在 bodyB 上的力矩，
        bodyA 获得的力矩为 -joint_torques[j]。
        
    提示：
        * 你可以综合使用 PD_control 和 gravity_compensation 来实现末端点位置控制器，也可以使用你认为合适的其他方法
        * 可以考虑的实现思路：
            1. 基于 Jacobian transpose 方法
                a. 计算重力补偿力矩 tau_g，使得机械臂保持静止
                b. 根据当前末端点位置和目标末端点位置，利用 PD 控制计算虚拟力 f_end，使得末端点位置跟踪目标位置
                c. 利用 Jacobian transpose 方法将 f_end 转换为关节力矩 tau
            2. 基于 PD 控制方法
                a. 根据目标末端点位置，利用 IK 计算目标关节旋转
                b. 计算关节 PD 控制力矩 tau，跟踪 IK 得到的目标关节旋转
            3. 基于 PD + 重力补偿方法
                a. 在 2 中的基础上，计算重力补偿力矩 tau_g，使得机械臂保持静止
            4. 其他方法
                a. 你可以使用其他方法来实现末端点位置控制器，比如基于模型预测控制的方法等
    '''   
    def get_init_positions():
        positions=np.zeros((len(joints),3))
        for i in range(1,len(joints)):
            bodyA=joints[i].bodyA
            orientation_p=R[bodyA]
            positions[i]=Rotation.from_matrix(orientation_p).apply(joints[i].from_bodyA)+x[bodyA]
        return positions
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
        
    # 计算当前末端点位置和速度
    end_effector_from_body_global = R[end_effector_body] @ end_effector_from_body
        
    current_end_effector_pos = x[end_effector_body] + end_effector_from_body_global
    current_end_effector_vel = v[end_effector_body] + np.cross(w[end_effector_body], end_effector_from_body_global)
    
    # 需要施加的关节力矩
    joint_torques = np.zeros((num_joints, 3))
    axes=np.zeros((num_joints,3))
    for i in range(num_joints):
        axes[i]=R[joints[i].bodyB]@np.array([0., 0., 1.])
    #------------------------------------------------------------------
    tau_g = gravity_compensation(m, g, x, R, joints)

    #----------------------------------------------------------------
    pos_error = target_end_effector_pos - current_end_effector_pos
    vel_error = target_end_effector_vel - current_end_effector_vel

    f_end = kp * pos_error + kd * vel_error 
    tau=np.zeros_like(tau_g)
    positions=get_init_positions()
    J = np.zeros((3, num_joints))
    
        
    
    for i in range(num_joints):
        tau[i]=np.cross(current_end_effector_pos-positions[i],f_end)
        joint_torques[i]=tau[i]+tau_g[i]
    return joint_torques

