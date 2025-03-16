##############
# 姓名：
# 学号：
##############

import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    joint_name=[]
    depth=0
    st=[]
    for i in range(len(lines)):
        line=lines[i]
        if line.strip().startswith('ROOT') or line.strip().startswith('JOINT'):
            values = lines[i+2].split()[1:]
            offset = tuple(float(val) for val in values)
            st.append((line.split()[-1],depth,offset))
        
        if line.strip().startswith('End'):
            values = lines[i+2].split()[1:]
            offset = tuple(float(val) for val in values)
            node=st[-1]
            st.append((node[0]+"_end",depth,offset))
        if line.strip().startswith('{'):
            depth+=1
        if line.strip().startswith('}'):
            depth-=1
        if line.strip().startswith('MOTION'):
            break
    #由st的节点得到joint_name和joint_parent
    for i in range(len(st)):
        node=st[i]
        curdepth=node[1]
        name=node[0]
        offset=node[2]
        joint_name.append(name)
        joint_offset.append(offset)
        if curdepth==0:
            joint_parent.append(-1)
        else:
            for j in range(i-1,-1,-1):
                if st[j][1]==curdepth-1:
                    joint_parent.append(j)
                    break
    joint_offset=np.array(joint_offset)
    print(joint_name)
    print(joint_offset)
    print(joint_parent)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    data=motion_data[frame_id]
    position=[]
    orientations=[]
    position.append(np.array((data[0],data[1],data[2])))
    orientations.append((R.from_euler('XYZ', [data[3],  data[4], data[5]], degrees=True)).as_matrix())
    index=0
    for i in range(1,len(joint_name)):
        parent=joint_parent[i]
        offset=joint_offset[i]
        name=joint_name[i]
        if "_end" in name:
            global_position=position[parent]+np.array(offset)
            global_rotation=orientations[parent]
        else:
            index+=1
            Xrotation=data[3*index+3]
            Yrotation=data[3*index+4]
            Zrotation=data[3*index+5]
            euler_angle = R.from_euler('XYZ', [Xrotation,  Yrotation, Zrotation], degrees=True)
            global_position=position[parent]+R.from_matrix(orientations[parent]).apply(np.array(offset))
            global_rotation=np.dot(orientations[parent],euler_angle.as_matrix())
        position.append(global_position)
        orientations.append(global_rotation)
    for i in range(len(orientations)):
        rotation= R.from_matrix(orientations[i])
        joint_orientations.append(rotation.as_quat())
    joint_positions=np.array(position)
    joint_orientations=np.array(joint_orientations)
    #print(joint_orientations)
    return joint_positions, joint_orientations


def align_joint(joint_nameA,joint_parentA,joint_offsetA,joint_nameT,joint_parentT,joint_offsetT):
    #建立A的由name到index的索引
    dic={}
    index=0
    for name in joint_nameA:
        dic[name]=index
        index+=1
    names=[]
    parents=[]
    offsets=[]
    for name in joint_nameT:
        if name in joint_nameA:
            names.append(name)
            parents.append(joint_parentA[dic[name]])
            offsets.append(joint_offsetA[dic[name]])
    return names,parents,offsets

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    rotation_corrections = {
        "lShoulder": [0, 0, -45],
        "rShoulder": [0, 0, 45],
        "lHip": [0, 0, 45],
        "rHip": [0, 0, -45],
    }
    joint_nameA,joint_parentA,joint_offsetA=part1_calculate_T_pose(A_pose_bvh_path)
    joint_nameT,joint_parentT,joint_offsetT=part1_calculate_T_pose(T_pose_bvh_path)
    #调整joint name顺序一致
    joint_nameA,joint_parentA,joint_offsetT=align_joint(joint_nameA,joint_parentA,joint_offsetA,joint_nameT,joint_parentT,joint_offsetT)
    motion_dataA=load_motion_data(A_pose_bvh_path)
    I=np.eye(3)
    for i in range(motion_dataA.shape[0]): #处理每一帧
        orientationsB_A=[]
        index=0
        for j in range(0,len(joint_nameA)):
            parent=joint_parentA[j]
            name=joint_nameA[j]
            if '_end' in name:
                orientationsB_A.append(orientationsB_A[parent])
            else:
                Xrotation=motion_dataA[i][3*index+3]
                Yrotation=motion_dataA[i][3*index+4]
                Zrotation=motion_dataA[i][3*index+5]
                euler_angle1 = R.from_euler('XYZ', [Xrotation,  Yrotation, Zrotation], degrees=True)
                if name=='RootJoint':
                    orientationsB_A.append(I)
                else:
                    if joint_nameA[j] in rotation_corrections:
                        RB_A=R.from_euler('XYZ',rotation_corrections[joint_nameA[j]] , degrees=True).as_matrix()
                    else:
                        RB_A=I
                    matrix=np.dot(orientationsB_A[parent],RB_A)
                    orientationsB_A.append(matrix)
                res=np.dot(euler_angle1.as_matrix(),orientationsB_A[j])
                euler=R.from_matrix(res).as_euler('XYZ',degrees=True)
                motion_dataA[i][3*index+3]=euler[0]
                motion_dataA[i][3*index+4]=euler[1]
                motion_dataA[i][3*index+5]=euler[2]
                index+=1
    return motion_dataA
    