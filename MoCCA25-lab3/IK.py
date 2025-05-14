import numpy as np
from scipy.spatial.transform import Rotation as R


def get_joint_rotations(joint_name,joint_parent,joint_orientations):
    joint_rotations = np.empty(joint_orientations.shape)
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
        else:
            joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
    return joint_rotations

def get_joint_offsets(joint_name,joint_parent,joint_positions):
    joint_offsets = np.empty(joint_positions.shape)
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            joint_offsets[i] = np.array([0.,0.,0.])
        else:
            joint_offsets[i] = joint_positions[i] - joint_positions[joint_parent[i]]
    return joint_offsets
def CCD(positions,path,path2,orientations,rotations,offsets,target_pose,joint_parent,max_times=10):
    distance = np.sqrt(np.sum(np.square(positions[-1] - target_pose)))
    while distance > 0.01:
        for i in range(len(path) - 2, -1, -1):
            if joint_parent[path[i]] == -1:
                continue
            #这里跳过root joint的旋转 因为它的子关节不只是一个
            #不容易处理
            #求出向量的旋转轴和旋转角
            a = target_pose - positions[i]
            b = positions[-1] - positions[i]
            axis = np.cross(b, a)
            axis = axis / np.linalg.norm(axis)

            cos =np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
            cos = np.clip(cos, -1.0, 1.0)
            angle = np.arccos(cos)
            if angle < 0.0001:
                continue
            rotation = R.from_rotvec(angle * axis)
            # 更新当前关节的朝向和local rotation 更新子关节的朝向和position
            orientations[i] = rotation * orientations[i]
            rotations[i] = orientations[i - 1].inv() * orientations[i]
            for j in range(i + 1, len(path)):
                orientations[j] = orientations[j - 1] * rotations[j]
                positions[j] = np.dot(orientations[j - 1].as_matrix(), offsets[j]) + positions[j - 1]
            distance = np.sqrt(np.sum(np.square(positions[-1] - target_pose)))
            #print(positions)
            print(distance)
    return positions,orientations,rotations
def init(path,joint_offsets,joint_rotations,joint_orientations,joint_positions,path2):
    positions=[]
    rotations=[]
    orientations=[]
    offsets=[]
    offsets.append(np.array([0.,0.,0.])) #根节点是不动的
    positions.append(joint_positions[path[0]])
    if len(path2)>1:
        rotations.append(R.from_quat(joint_orientations[path2[1]]).inv())
    else:
        rotations.append(R.from_quat(joint_orientations[path[0]]))
    orientations.append(rotations[0])
    for i in range(1,len(path)):
        index=path[i]
        positions.append(joint_positions[index]) #positions直接放入即可
        if index in path2:
            offsets.append(-joint_offsets[path[i-1]]) 
            '''
            我们的chain的offset和上面计算的offset的方向是相反的 即chain：2->1 
            我们计算的offset[1]表示l1的长度和方向 1->2
            所以计算chain上的offset[2]= 原来的-offset[1] 长度相等 方向相反
            orientation：chain 6->4->1 原来 1->4->6 
            Q6=Q4*R相对 Q4=Q6*R逆
            显然rotation是原来的逆 orientation变成后一个的orientation

            '''
            orientations.append(R.from_quat(joint_orientations[path[i+1]]))
            rotations.append((R.from_quat(joint_rotations[index]).inv()))
        else:#这里的顺序都是正常的
            offsets.append(joint_offsets[index])
            orientations.append(R.from_quat(joint_orientations[index]))
            rotations.append(R.from_quat(joint_rotations[index]))
    
    
    return np.array(positions),np.array(rotations),np.array(orientations),np.array(offsets)


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """


    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    print(path,path1,path2)
    if len(path2) == 1 and path2[0] != 0:
        path2 = []

    joint_rotations = get_joint_rotations(joint_name,joint_parent,joint_orientations) #quat形式
    joint_offsets = get_joint_offsets(joint_name,joint_parent,joint_initial_position)
    positions,rotations,orientations,offsets=init(path,joint_offsets,joint_rotations,joint_orientations,joint_positions,path2)
    npositions,norientations,nrotations=CCD(positions,path,path2,orientations,rotations,offsets,target_pose,joint_parent)

    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = npositions[i]
        if index in path2:
            joint_rotations[index] = nrotations[i].inv().as_quat()
        else:
            joint_rotations[index] = nrotations[i].as_quat()

    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * norientations[0]).as_quat()

    # 更新rootjoint的位置，为了之后的更新
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = norientations[root_index].as_quat()
            joint_positions[0] = npositions[root_index]


    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])


    return joint_positions, joint_orientations