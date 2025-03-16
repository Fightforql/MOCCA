##############
# 姓名：梁书怡
# 学号：2300013147
##############
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import copy
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
def get_rotation_from_orientations(orientations,joint_parent):
    rotation = []
    rotation.append(R.from_quat(orientations[0]).as_matrix())
    for i in range(1, len(orientations)):
        matrix1 = R.from_quat(orientations[joint_parent[i]]).as_matrix()
        matrix2 = R.from_quat(orientations[i]).as_matrix()
        local_rotation = np.linalg.inv(matrix1) @ matrix2
        rotation.append(local_rotation)
    return np.array(rotation)


def get_offset_from_position(positions,joint_parent):
    offset=[]
    offset.append(np.array([0.,0.,0.]))
    for i in range(1,len(positions)):
        offset.append(positions[i]-positions[joint_parent[i]])
    return np.array(offset)

def matrix_to_quat(rotations):
    res=[]
    for i in range(len(rotations)):
        res.append( R.from_matrix(rotations[i]).as_quat())
    return np.array(res)

class IKmodel():
    def __init__(self,rotations,offsets,target_pose,path,root_index,lr=0.1,threshold=0.01):
        self.rotation_tensor=torch.tensor(rotations, requires_grad=True, dtype=torch.float32)
        self.offset_tensor=torch.tensor(offsets, requires_grad=False, dtype=torch.float32)
        self.target_position = torch.tensor(target_pose, requires_grad=False, dtype=torch.float32)
        self.path=path
        self.threshold=threshold
        self.lr=lr
        self.root_index=root_index
        self.joint_rotation=matrix_to_quat(rotations)
        self.optimizer = torch.optim.Adam([self.rotation_tensor], lr=self.lr)

    def forward(self):
        #利用当前rotation计算end_position 这里的rotation已经是矩阵形式
        cur_orientation=self.rotation_tensor[self.path[0]]
        cur_position=self.offset_tensor[self.path[0]]
        for i in range(1,len(self.path)):
            cur_orientation=cur_orientation@self.rotation_tensor[self.path[i]]
            cur_position=cur_position+cur_orientation@self.offset_tensor[self.path[i]]
                
        return cur_position
    
    def train(self):
        num=0
        while True:
            end_position=self.forward()
            dist=torch.norm(end_position-self.target_position)
            print(f"Distance: {dist.item()}")
            if dist.item()<self.threshold:
                break
            self.optimizer.zero_grad()
            dist.backward()
            self.optimizer.step()
        
        for i in range(len(self.path)):
            index = self.path[i]
            self.joint_rotation[index] = R.from_matrix(self.rotation_tensor[self.path[i]].detach().cpu().numpy()).as_quat()
            
        return self.joint_rotation
def rotation_matrix(a, b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    n = np.cross(a, b)
    # 旋转矩阵是正交矩阵，矩阵的每一行每一列的模，都为1；并且任意两个列向量或者任意两个行向量都是正交的。
    # n=n/np.linalg.norm(n)
    # 计算夹角
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(n)
    theta = np.arctan2(sin_theta, cos_theta)
    # 构造旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    rotation_matrix = np.array([[n[0]*n[0]*v+c, n[0]*n[1]*v-n[2]*s, n[0]*n[2]*v+n[1]*s],
                                 [n[0]*n[1]*v+n[2]*s, n[1]*n[1]*v+c, n[1]*n[2]*v-n[0]*s],
                                 [n[0]*n[2]*v-n[1]*s, n[1]*n[2]*v+n[0]*s, n[2]*n[2]*v+c]])
    return rotation_matrix

def inv_safe(data):
    # return R.from_quat(data).inv()
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return np.linalg.inv(R.from_quat(data).as_matrix())
    
def from_quat_safe(data):
    # return R.from_quat(data)
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return R.from_quat(data).as_matrix()

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
    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path), 3), dtype=float)
    offset_chain = np.empty((len(path), 3), dtype=float)

    # 对chain进行初始化
    if len(path2) > 1:
        rotation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv().as_euler('XYZ')
    else:
        rotation_chain[0] = R.from_quat(joint_orientations[path[0]]).as_euler('XYZ')

    # position_chain[0] = joint_positions[path[0]]
    start_position = torch.tensor(joint_positions[path[0]], requires_grad=False)
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        if index in path2:
            # essential
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv().as_euler('XYZ')
            offset_chain[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            rotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
            offset_chain[i] = joint_offsets[index]

    # pytorch autograde
    rotation_chain_tensor = torch.tensor(rotation_chain, requires_grad=True, dtype=torch.float32)
    offset_chain_tensor = torch.tensor(offset_chain, requires_grad=False, dtype=torch.float32)
    target_position = torch.tensor(target_pose, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_path = path.index(0)
    max_times = 50
    lr = 0.1
    while max_times > 0:
        # 向前计算end position
        max_times -= 1
        cur_position = start_position
        cur_orientation = rotation_chain_tensor[0]
        for i in range(1, len(path)):
            cur_position = euler_angles_to_matrix(cur_orientation, 'XYZ') @ offset_chain_tensor[i] + cur_position
            orientation_matrix = euler_angles_to_matrix(cur_orientation, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ')
            cur_orientation = matrix_to_euler_angles(orientation_matrix, 'XYZ')
            # joint_positions[path[i]] = cur_position.detach().numpy()
            # joint_orientations[path[i]] = R.from_euler('XYZ', cur_orientation.detach().numpy()).as_quat()
        dist = torch.norm(cur_position - target_position)
        if dist < 0.01 or max_times == 0:
            break

        # 反向传播
        dist.backward()
        rotation_chain_tensor.grad[rootjoint_index_in_path].zero_()
        rotation_chain_tensor.data.sub_(rotation_chain_tensor.grad * lr)
        rotation_chain_tensor.grad.zero_()

    # return joint_positions, joint_orientations

    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        if index in path2:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).inv().as_quat()
        else:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).as_quat()


    # 当IK链不过rootjoint时，IK起点的rotation需要特殊处理
    if path2 == [] and path[0] != 0:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() 
                                    * R.from_euler('XYZ', rotation_chain_tensor[0].detach().numpy())).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if 0 in path and rootjoint_index_in_path != 0:
        rootjoint_pos = start_position
        rootjoint_ori = rotation_chain_tensor[0]
        for i in range(1, rootjoint_index_in_path + 1):
            rootjoint_pos = euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ offset_chain_tensor[i] + rootjoint_pos
            rootjoint_ori = matrix_to_euler_angles(euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ'), 'XYZ')
        joint_orientations[0] = R.from_euler('XYZ', rootjoint_ori.detach().numpy()).as_quat()
        joint_positions[0] = rootjoint_pos.detach().numpy()

    
    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])

    return joint_positions, joint_orientations



        


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    joint_positions = None
    joint_orientations = None
   
    return joint_positions, joint_orientations
    
