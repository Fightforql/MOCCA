##############
# 姓名：梁书怡
# 学号：2300013147
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
np.set_printoptions(threshold=np.inf)
#对于每一帧，构建一个27维向量：0-2 左脚位置 3-5右脚位置 6-8左脚速度 9-11 右脚速度 12-14 hip速度 15-20 未来20，40，60帧的xz朝向 21-26 未来20，40，60帧的速度
def quat_mul_vec3(q,v):
    return R.from_quat(q).apply(v)

def normalize_features(feature_vectors,feature_weight):
        # compute mean and std
    features_offset = np.mean(feature_vectors, axis=0)
    features_scale = np.std(feature_vectors, axis=0)

    # weight
    features_scale = features_scale /feature_weight

    return features_offset, features_scale



#------------------------------------分界线---------------------------------------------------
class database():
    def __init__(self):
        self.motion0=BVHMotion("motion_material/walk.bvh")
        self.length=self.motion0.motion_length
        for i in range(5):
            motion = build_loop_motion(self.motion0,0.5, 0.2)
            pos = motion.joint_position[-1,0,[0,2]]
            rot = motion.joint_rotation[-1,0]
            facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            new_motion = motion.translation_and_rotation(0, pos, facing_axis)
            self.motion0.append(new_motion)
        #print(self.motion0.motion_length)
        self.motion1=BVHMotion("motion_material/turn_left.bvh")
        self.motion2=BVHMotion("motion_material/turn_right.bvh")
        self.motion3=BVHMotion("motion_material/spin_clockwise.bvh")
        self.motion4=BVHMotion("motion_material/spin_counter_clockwise.bvh")
        self.motions=self.get_motions()
        self.joint_name=self.motions.joint_name
        self.frame_num=self.motions.motion_length
        
        self.data = []
        self.data.append(np.zeros((self.motion0.motion_length, 27), dtype=np.float32))
        self.data.append(np.zeros((self.motion1.motion_length, 27), dtype=np.float32))
        self.data.append(np.zeros((self.motion2.motion_length, 27), dtype=np.float32))
        self.data.append(np.zeros((self.motion3.motion_length, 27), dtype=np.float32))
        self.data.append(np.zeros((self.motion4.motion_length, 27), dtype=np.float32))
        #self.data = np.zeros((5,self.motion1.motion_length, 27), dtype=np.float32)
        self.dt=1.0/60.0
        

    def get_motions(self):
        motions=self.motion0.raw_copy()
        motions.append(self.motion1)
        motions.append(self.motion2)
        motions.append(self.motion3)
        motions.append(self.motion4)
        return motions
    
    def get_motion_index(self,motion):
        if motion==self.motion0:
            return 0
        if motion==self.motion1:
            return 1
        if motion==self.motion2:
            return 2
        if motion==self.motion3:
            return 3
        if motion==self.motion4:
            return 4
        
    def compute_trajectory_position_feature(self,motion):
        motion_index=self.get_motion_index(motion)
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        for i in range(motion.motion_length-60):
            #这里还需要考虑一点，是否需要将每个bvh的range分开
            t0=i+20
            t1=i+40
            t2=i+60
            pos0=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),joint_translation[t0][0]-joint_translation[i][0])
            pos1=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),joint_translation[t1][0]-joint_translation[i][0])
            pos2=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),joint_translation[t2][0]-joint_translation[i][0])
            self.data[motion_index][i][0]=pos0[0]
            self.data[motion_index][i][1]=pos0[2]
            
            self.data[motion_index][i][2]=pos1[0]
            self.data[motion_index][i][3]=pos1[2]

            self.data[motion_index][i][4]=pos2[0]
            self.data[motion_index][i][5]=pos2[2]
        

    def compute_trajectory_direction_feature(self,motion):
        motion_index=self.get_motion_index(motion)
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        for i in range(motion.motion_length-60):
            t0=i+20
            t1=i+40
            t2=i+60
            rot0=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),quat_mul_vec3(joint_orientation[t0][0],np.array([0.,0.,1.])))
            rot1=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),quat_mul_vec3(joint_orientation[t1][0],np.array([0.,0.,1.])))
            rot2=quat_mul_vec3(R.from_quat(joint_orientation[i][0]).inv().as_quat(),quat_mul_vec3(joint_orientation[t2][0],np.array([0.,0.,1.])))
            self.data[motion_index][i][6]=rot0[0]
            self.data[motion_index][i][7]=rot0[2]

            self.data[motion_index][i][8]=rot1[0]
            self.data[motion_index][i][9]=rot1[2]
            
            self.data[motion_index][i][10]=rot2[0]
            self.data[motion_index][i][11]=rot2[2]
    
    def compute_other_feature(self,motion):
        motion_index=self.get_motion_index(motion)
        joint_translation, joint_orientation=motion.batch_forward_kinematics()
        for i in range(motion.motion_length):
            cur_orientation_inv = R.from_quat(joint_orientation[i][0]).inv()
            self.data[motion_index][i,12:15]=\
                cur_orientation_inv.apply(joint_translation[i, self.joint_name.index('rAnkle'), :] - joint_translation[i, 0, :])
            self.data[motion_index][i,15:18]=\
                cur_orientation_inv.apply(joint_translation[i, self.joint_name.index('lAnkle'), :] - joint_translation[i, 0, :])
            if i!=0:
                delta_time = self.dt
                self.data[motion_index][i, 18:21] = cur_orientation_inv.apply(\
                    (joint_translation[i, self.joint_name.index('RootJoint'), :] - joint_translation[i - 1, self.joint_name.index('RootJoint'), :]) / delta_time)
                hip_velocity = self.data[motion_index][i, 18:21]
                self.data[motion_index][i, 21:24] = cur_orientation_inv.apply(\
                    (joint_translation[i, self.joint_name.index('rAnkle'), :] - joint_translation[i - 1, self.joint_name.index('rAnkle'), :]) / delta_time) - hip_velocity
                self.data[motion_index][i, 24:27] = cur_orientation_inv.apply(\
                    (joint_translation[i, self.joint_name.index('lAnkle'), :] - joint_translation[i - 1, self.joint_name.index('lAnkle'), :]) / delta_time) - hip_velocity
    



    def bulid(self):
        self.compute_trajectory_position_feature(self.motion0)
        self.compute_trajectory_position_feature(self.motion1)
        self.compute_trajectory_position_feature(self.motion2)
        self.compute_trajectory_position_feature(self.motion3)
        self.compute_trajectory_position_feature(self.motion4)
        self.compute_trajectory_direction_feature(self.motion0)
        self.compute_trajectory_direction_feature(self.motion1)
        self.compute_trajectory_direction_feature(self.motion2)
        self.compute_trajectory_direction_feature(self.motion3)
        self.compute_trajectory_direction_feature(self.motion4)
        self.compute_other_feature(self.motion0)
        self.compute_other_feature(self.motion1)
        self.compute_other_feature(self.motion2)
        self.compute_other_feature(self.motion3)
        self.compute_other_feature(self.motion4)



class CharacterController():
    def __init__(self, controller) -> None:
        # 手柄/键盘控制器
        self.controller = controller
        # 读取graph结构
        self.graph = Graph('./nodes.npy')
        self.graph.load_from_file()
        # node name组成的List
        self.node_names = [nd.name for nd in self.graph.nodes]
        # edge name组成的List
        self.edge_names = []
        for nd in self.graph.nodes:
            for eg in nd.edges:
                self.edge_names.append(eg.label)

        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        # 当前角色的参考root位置
        self.cur_root_pos = None
        # 当前角色的参考root旋转
        self.cur_root_rot = None
        # 当前角色处于Graph的哪一个节点
        self.cur_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1

        # 初始化上述参数
        self.initialize()
        self.counter=3
        self.feature_weight = np.array([1.0]*6+[1.5]*6+[0.75]*6+[1.25]*9)
        self.db=database()
        self.db.bulid()
        #print(self.db.data[0].shape)
        self.feature_vectors=np.concatenate(self.db.data, axis=0)
        #print(self.feature_vectors.shape)
        self.features_offset,self.features_scale=normalize_features(self.feature_vectors,self.feature_weight)
        
        self.db.data = [
            (d - self.features_offset[None, :]) / self.features_scale[None, :]
            for d in self.db.data
        ]
        
        self.walk=True
        self.walk_num=50
        self.next_motion=None
        self.next_frame=0
        self.motion0=BVHMotion("motion_material/walk.bvh")
        self.walk_motion=self.motion0.raw_copy()
        self.length=self.motion0.motion_length
        for i in range(2):
            motion = build_loop_motion(self.motion0,0.5, 0.2)
            pos = motion.joint_position[-1,0,[0,2]]
            rot = motion.joint_rotation[-1,0]
            facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            new_motion = motion.translation_and_rotation(0, pos, facing_axis)
            self.motion0.append(new_motion)
        self.graph.nodes[0].motion=self.motion0
        
        self.inertialize_active = False
        self.half_life = 0.15  # 过渡时间参数
        self.bone_offset_positions = None
        self.bone_offset_vels = None
        self.bone_offset_rotations = None
        self.bone_offset_avels = None
        self.transition_src_position = np.zeros(3)
        self.transition_src_rotation = np.array([0,0,0,1.0])
        self.transition_dst_position = np.zeros(3)
        self.transition_dst_rotation = np.array([0,0,0,1.0])
        self.target_motion = None
        self.target_frame = 0
        self.dt=1.0/60.0
        self.walk_flag=False
        
    def initialize(self):
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        self.first_frame=True
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])



    def _is_input_stable(self):
        """检查最近5帧输入方向是否一致"""
        valid_directions = [d for d in self.last_input_directions if d is not None]
        if len(valid_directions) < self.stable_threshold:
            return False
        
        # 计算平均方向
        mean_dir = np.mean(np.array(valid_directions), axis=0)
        mean_dir /= np.linalg.norm(mean_dir)  # 归一化
        
        # 检查所有方向是否接近平均方向
        for d in valid_directions:
            if np.dot(d, mean_dir) < 0.99:  # 余弦相似度阈值
                return False
        return True
    


    def find_near_frame(self,q,index):
        # Normalize database
        # 对于查询向量 q，形状是 (27,)
        kd_tree0=KDTree(self.db.data[0][:self.db.length])
        kd_tree1=KDTree(self.db.data[1])
        kd_tree2=KDTree(self.db.data[2])
        kd_tree3=KDTree(self.db.data[3])
        kd_tree4=KDTree(self.db.data[4])
        distance0, index0 = kd_tree0.query(q)
        distance1, index1 = kd_tree1.query(q)
        distance2, index2 = kd_tree2.query(q)
        distance3, index3 = kd_tree3.query(q)
        distance4, index4 = kd_tree4.query(q)
        d_list=[distance0,distance1,distance2,distance3,distance4]
        index_list=[index0,index1,index2,index3,index4]
        d_min=min(d_list)
        #print(d_list)
        min_index=d_list.index(d_min)
        '''
        if abs(d_min-d_list[index])<3.0:
            min_index=index
        '''
        return (min_index,index_list[min_index])
        
    def switch_to_walk(self,motion1,motion2,mix_frame1,mix_frame2):
        res1=motion1.raw_copy()
        res2 = motion2.raw_copy()
        #下面这种直接拼肯定是不行的(
        #将二者拼接在一起
        res1=res1.sub_sequence(0,mix_frame1)
        res2=res2.sub_sequence(mix_frame2,res2.joint_position.shape[0])
        half_life=0.27 
        rotations1=res1.joint_rotation
        rotations2=res2.joint_rotation
        #首先计算开始帧与结束帧的差异
        avel1=quat_to_avel(rotations1, 1/60)
        avel2=quat_to_avel(rotations2,1/60)
        rot_diff = (R.from_quat(rotations1[-1]) * R.from_quat(rotations2[0].copy()).inv()).as_rotvec()
        avel_diff = (avel1[-1] - avel2[0])
        pos_diff = res1.joint_position[-1] - res2.joint_position[0]
        #pos_diff[:,[0,2]] = 0
        vel1 = res1.joint_position[-1] - res1.joint_position[-2]
        vel2 = res2.joint_position[1] -res2.joint_position[0]
        vel_diff = (vel1 - vel2)/60
        # 将旋转差均匀分布到每一帧
        for i in range(res2.motion_length):
            #计算diff的衰减offset
            offset= decay_spring_implicit_damping_rot(rot_diff, avel_diff, half_life, i/60)
            offset_rot = R.from_rotvec(offset[0])
            res2.joint_rotation[i] = -(offset_rot * R.from_quat(res2.joint_rotation[i])).as_quat() 

        for i in range(res2.motion_length):
            
            offset = decay_spring_implicit_damping_pos(pos_diff, vel_diff, half_life, i/60)
            res2.joint_position[i] += offset[0]
        
        return res2
   

    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

        '''
        # # 一个简单的例子，循环播放第0个动画，不会响应输入信号
            
        
        joint_name = self.cur_node.motion.joint_name
        
        joint_translation1, joint_orientation1 = self.cur_node.motion.batch_forward_kinematics()
        joint_translation = joint_translation1[self.cur_frame]
        joint_orientation = joint_orientation1[self.cur_frame]
        
        # 更新self.cur_root_pos, self.cur_root_rot
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        
        '''
        if self.first_frame:
            self.first_frame=False
            joint_name = self.cur_node.motion.joint_name
            joint_translation, joint_orientation = self.cur_node.motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            
            # 更新你的表示角色的信息
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame = (self.cur_frame + 1) % self.cur_node.motion.motion_length
            # 一直处于第0个动画所在的node
            self.cur_node = self.graph.nodes[0]
            # 不会切换，所以一直不会播放transition动画
            self.cur_edge = None
        #首先构建出当前frame对应的特征向量
        '''
        k=self.cur_node.identity
        l=self.cur_frame
        current_forward = R.from_quat(self.cur_root_rot).apply([0,0,1])[[0,2]]
        input_direction = desired_vel_list[0][[0,2]]
        if np.dot(input_direction/np.linalg.norm(input_direction), 
                current_forward/np.linalg.norm(current_forward)) > 0.95 and self.next_motion!=None and self.walk_flag:
            joint_translation1, joint_orientation1 = self.next_motion.batch_forward_kinematics()
            joint_translation = joint_translation1[self.next_frame]
            joint_orientation = joint_orientation1[self.next_frame]
            self.next_frame+=1
            return joint_name,joint_translation,joint_orientation
        '''
        if self.walk_num<50:
            if np.dot(input_direction/np.linalg.norm(input_direction), 
                current_forward/np.linalg.norm(current_forward)) > 0.95:
                self.walk_num+=1
            if self.walk_num==50:
                name=self.cur_node.name
                self.cur_node=self.graph.nodes[0]
                facing_axis = R.from_quat(self.graph.nodes[k].motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                self.cur_node.motion=self.cur_node.motion.translation_and_rotation(0, self.graph.nodes[k].motion.joint_position[self.cur_frame, 0, [0, 2]],facing_axis)
                self.cur_frame=0
                self.cur_edge=Edge(name+'->'+self.cur_node.name,self.cur_node)
        '''
        query=np.zeros([27], dtype=np.float32)
        query[0:2]=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[1]-self.cur_root_pos)[[0,2]]
        query[2:4]=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[2]-self.cur_root_pos)[[0,2]]
        query[4:6]=R.from_quat(self.cur_root_rot).inv().apply(desired_pos_list[3]-self.cur_root_pos)[[0,2]]
        query[6:8]=R.from_quat(self.cur_root_rot).inv().apply(R.from_quat(desired_rot_list[1]).apply(np.array([0.,0.,1.])))[[0,2]]
        query[8:10]=R.from_quat(self.cur_root_rot).inv().apply(R.from_quat(desired_rot_list[2]).apply(np.array([0.,0.,1.])))[[0,2]]
        query[10:12]=R.from_quat(self.cur_root_rot).inv().apply(R.from_quat(desired_rot_list[3]).apply(np.array([0.,0.,1.])))[[0,2]]
        query[12:27]=self.db.data[self.cur_node.identity][self.cur_frame][12:27]
        #从database中找到最接近的一个frame()
        query[:12] = (query[:12] - self.features_offset[:12]) / self.features_scale[:12]
        #定位这个frame所在的bvh
        #更新self.cur_node self.cur_edge self.cur_frame(在当前bvh的index 而非所有的)
        #print(self.cur_node.identity)
        a,b=self.find_near_frame(query,self.cur_node.identity)
        #print(a)
        
        
        
        if a!=self.cur_node.identity and np.linalg.norm(desired_vel_list[0])> 0.1:
            print("切换",self.cur_node.identity,a)
            self.counter-=1
            if self.counter==0:
                self.counter=3
                
                #self.graph.nodes[k].motion=self.motions[k]
                name=self.cur_node.name
                self.cur_node=self.graph.nodes[a]
                facing_axis = R.from_quat(self.graph.nodes[k].motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                self.cur_node.motion=self.cur_node.motion.translation_and_rotation(b, self.graph.nodes[k].motion.joint_position[self.cur_frame, 0, [0, 2]],facing_axis)
                self.cur_frame=b
                self.cur_edge=Edge(name+'->'+self.cur_node.name,self.cur_node)
                '''
                if a==0:
                    self.walk_flag=True
                    self.next_motion=self.switch_to_walk(self.graph.nodes[k].motion,self.cur_node.motion,l,self.cur_frame)
                else:
                    self.walk_flag=False
                    self.next_frame=0
                '''
                #self.cur_node.motion=concatenate_two_motions(self.graph.nodes[k].motion,self.cur_node.motion,l)
            else:
                self.next_motion=None
                self.cur_frame = (self.cur_frame + 1) % self.cur_node.motion.motion_length
                self.cur_edge=None
        else:
            print(self.cur_frame,self.cur_node.motion.motion_length)
            #self.next_motion=None
            self.cur_frame = (self.cur_frame + 1) % self.cur_node.motion.motion_length
            self.cur_edge=None
        if self.cur_node.identity!=0:
            self.walk_num=0
        return joint_name, joint_translation, joint_orientation
    


