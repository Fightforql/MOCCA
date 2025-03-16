##############
# 姓名：梁书怡
# 学号：2300013147
##############
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import copy

#计算path上joint(不包括end)的相对旋转 一律用欧拉角表示 输入的orientations是所有joint的
def get_rotation_from_orientations(orientations,joint_parent,path):
    rotation = []
    #path的根节点的相对旋转R=Q
    rotation.append(R.from_quat(orientations[path[0]]).as_euler('XYZ',degrees=True))
    for i in range(1, len(path)-1):
        matrix1 = R.from_quat(orientations[joint_parent[path[i]]]).as_matrix() #事实上 path[i]关节的parent就是path[i-1]
        matrix2 = R.from_quat(orientations[path[i]]).as_matrix()
        local_rotation = np.linalg.inv(matrix1) @ matrix2
        rotation.append(R.from_matrix(local_rotation).as_euler('XYZ',degrees=True))
    return np.array(rotation)

#计算每个joint的offset
def get_offset_from_position(joint_name,positions,joint_parent):
    offset=[]
    offset.append(positions[0])
    for i in range(1,len(joint_name)):
        offset.append(positions[i]-positions[joint_parent[i]])
    return np.array(offset)

#计算雅可比矩阵 不包括end Geometric Approach
def getJacobian(path,target_pose,joint_positions,joint_orientations,joint_angles):
    Jacobian = []
    for i in range(len(path) - 1):#不包括end joint
        r = target_pose - joint_positions[path[i]]
        if i == 0:
            last_joint_orientation = R.from_quat([0.,0.,0.,1.0])
        else:
            last_joint_orientation = R.from_quat(joint_orientations[path[i-1]])
        curr_joint_orientation = joint_angles[i]
        rotation_x = R.from_euler('XYZ',[curr_joint_orientation[0], 0., 0.], degrees=True)
        rotation_xy = R.from_euler('XYZ',[curr_joint_orientation[0], curr_joint_orientation[1], 0.], degrees=True)
        
        x = [1.,0.,0.]
        y = [0.,1.,0.]
        z = [0.,0.,1.]

        axis_x = last_joint_orientation.apply(x)
        axis_y = last_joint_orientation.apply(rotation_x.apply(y))
        axis_z = last_joint_orientation.apply(rotation_xy.apply(z))

        Jacobian.append((np.cross(axis_x, r),np.cross(axis_y, r),np.cross(axis_z, r)))

    Jacobian = np.array(Jacobian)
    return Jacobian

#每次梯度下降之后要更新chain上的joint的position和orientation
#这里容易错的是 需要先还原上一次朝向 因为我们梯度下降得到的永远是针对于最初的T-pose而言的旋转
def update(path,rotation_angle,joint_parent,last_orientations,last_positions,offsets):
    orientations=copy.deepcopy(last_orientations)
    positions=copy.deepcopy(last_positions)
    orientations[path[0]]=(R.from_euler('XYZ',rotation_angle[0]).as_quat())
    #更新朝向 这里也不需要还原
    for i in range(1,len(path)-1):
        orientation_parent=orientations[path[i-1]]
        cur_orientation=R.from_quat(orientation_parent).as_matrix()@R.from_euler('XYZ',rotation_angle[i]).as_matrix()
        orientations[path[i]]=(R.from_matrix(cur_orientation).as_quat())
    orientations[path[-1]]=orientations[path[len(path)-2]]
    #更新位置 不需要还原position 还原朝向即可 我们知道path的root不动 所以每次都可以重新开始计算
    positions[path[0]]=(last_positions[path[0]])
    for i in range(1,len(path)):
        orientation=R.from_quat(orientations[path[i-1]])*R.inv(R.from_quat(last_orientations[path[i-1]]))
        positions[path[i]]=(positions[path[i-1]]+orientation.apply(offsets[path[i]]))
    return orientations,positions
#梯度下降
def iterations(joint_positions,path,target_pose,joint_orientations,joint_parent,offsets,max_iterations=20,lr=0.5):
    end_index=path[-1]
    positions=joint_positions
    orientations=joint_orientations
    while max_iterations>0:
        max_iterations-=1
        distance=positions[end_index]-target_pose
        print(distance)
        if np.linalg.norm(distance)<0.01:
            break
        rotation_angle=get_rotation_from_orientations(orientations,joint_parent,path)
        #rotation_angle 是不包括end的欧拉角形式的局部旋转np
        jacobian=getJacobian(path,target_pose,positions,orientations,rotation_angle)
        #雅可比矩阵同样不包括end
        rotation_angle=rotation_angle-lr*jacobian@distance
        #进行一次梯度下降
        orientations,positions=update(path,rotation_angle,joint_parent,orientations,positions,offsets)
    return positions,orientations

'''
class IKmodel():
    def __init__(self,rotations,offsets,target_pose,path,root_index,lr=0.1,threshold=0.01):
        self.rotation_tensor=torch.tensor(rotations, requires_grad=True, dtype=torch.float32)
        self.offset_tensor=torch.tensor(offsets, requires_grad=False, dtype=torch.float32)
        self.target_position = torch.tensor(target_pose, requires_grad=False, dtype=torch.float32)
        self.path=path
        self.threshold=threshold
        self.lr=lr
        self.root_index=root_index
        #self.joint_rotation=matrix_to_quat(rotations)
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

'''
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
    joint_name=meta_data.joint_name
    joint_parent=meta_data.joint_parent
    root_joint=meta_data.root_joint
    end_joint=meta_data.end_joint 
    path,path_name,path1,path2=meta_data.get_path_from_root_to_end()
    offsets=get_offset_from_position(joint_name,joint_positions,joint_parent)
    joint_positions,joint_orientations=iterations(joint_positions,path,target_pose,joint_orientations,joint_parent,offsets)

    return joint_positions, joint_orientations



        


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    joint_positions = None
    joint_orientations = None
   
    return joint_positions, joint_orientations
    
