##############
# 姓名：梁书怡
# 学号：2300013147
##############
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


#计算path上joint(不包括end)的相对旋转 一律用欧拉角表示 输入的orientations是所有joint的
def get_rotation_from_orientations(orientations,joint_parent,path):
    rotation = []
    #path的根节点的相对旋转R=Q
    rotation.append(R.from_quat(orientations[path[0]]).as_euler('XYZ',degrees=True))
    for i in range(1, len(path)-1):
        matrix1 = R.from_quat(orientations[path[i-1]]).as_matrix() #事实上 path[i]关节的parent就是path[i-1]
        matrix2 = R.from_quat(orientations[path[i]]).as_matrix()
        local_rotation = np.linalg.inv(matrix1) @ matrix2
        rotation.append(R.from_matrix(local_rotation).as_euler('XYZ',degrees=True))
    rotation= np.reshape(rotation,(-1,3))
    return rotation

#计算每个joint的offset
def get_offset_from_position(joint_name,positions,joint_parent):
    offset=[]
    offset.append(np.array(positions[0]).reshape(1,-1))
    for i in range(1,len(joint_name)):
        offset.append(np.array(positions[i]-positions[joint_parent[i]]).reshape(1,-1))
    offsets = np.concatenate(offset, axis=0)
    return offsets

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

        Jacobian.append(np.cross(axis_x, r))
        Jacobian.append(np.cross(axis_y, r))
        Jacobian.append(np.cross(axis_z, r))

    Jacobian = np.reshape(Jacobian,(-1,3))
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
def iterations(joint_positions,path,target_pose,joint_orientations,joint_parent,offsets,max_iterations=20,lr=1):
    end_index=path[-1]
    positions=joint_positions
    orientations=joint_orientations
    while max_iterations>0:
        max_iterations-=1
        distance=np.array(positions[end_index] - target_pose).reshape(3,-1)
        print(np.linalg.norm(distance))
        if np.linalg.norm(distance)<0.01:
            break
        rotation_angle=get_rotation_from_orientations(orientations,joint_parent,path)
        theta = np.concatenate(rotation_angle, axis = None)
        theta = theta.reshape(-1,1)
        #rotation_angle 是不包括end的欧拉角形式的局部旋转np
        jacobian=getJacobian(path,target_pose,positions,orientations,rotation_angle)
        #雅可比矩阵同样不包括end
        theta = theta - lr * np.dot(jacobian, distance)
        theta = np.reshape(theta, (-1,3))
        #进行一次梯度下降
        orientations,positions=update(path,theta,joint_parent,orientations,positions,offsets)
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

'''
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

def CCD(positions,path,path2,orientations,rotations,offsets,target_pose,joint_parent,max_times=10):
    distance = np.sqrt(np.sum(np.square(positions[-1] - target_pose)))
    while distance > 0.001 and max_times>0:
        max_times -= 1
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
            print(positions)
            print(distance)
    return positions,orientations,rotations


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




        


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    #只需要改变lShoulder到lWrist的path
    target_pose = np.array([relative_x + joint_positions[0][0], target_height, relative_z + joint_positions[0][2]])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations
