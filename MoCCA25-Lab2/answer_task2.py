##############
# 姓名：毛思雨
# 学号：2300013216
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
from scipy.spatial import KDTree

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

        self.feats=None
        self.feats0=None
        self.feats1=None
        self.feats2=None
        self.feats3=None
        self.feats4=None
        
        self.feature0_KD_tree=None
        self.feature1_KD_tree=None
        self.feature2_KD_tree=None
        self.feature3_KD_tree=None
        self.feature4_KD_tree=None
        
        self.feature_offset=None
        self.feature_scale=None
        
        self.feature_weight =None
        self.counter=3
        # 初始化上述参数
        self.motion0=BVHMotion("motion_material/walk.bvh")
        for _ in range(2):
            motion = build_loop_motion(self.motion0,0.5, 0.2)
            pos = motion.joint_position[-1,0,[0,2]]
            rot = motion.joint_rotation[-1,0]
            facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            new_motion = motion.translation_and_rotation(0, pos, facing_axis)
            self.motion0.append(new_motion)
        self.graph.nodes[0].motion=self.motion0
        self.initialize()
    
    def extract_advanced_features(self,mocap):
        frames = mocap.motion_length
        dt = mocap.frame_time
        joint_translation, joint_orientation = mocap.batch_forward_kinematics()
        frame_feaure=np.zeros([mocap.motion_length, 27])
        for i in range(frames):
            quat = joint_orientation[i, 0, :]
            if np.linalg.norm(quat) < 1e-5:
                quat = np.array([0., 0., 0., 1.])
            cur_orientation_inv = R.from_quat(quat).inv()

            if i < mocap.motion_length - 60:
                # 未来关节位置变化（20、40、60帧后的位置变化，局部坐标系）
                frame_feaure[i,0:2] = cur_orientation_inv.apply(joint_translation[i + 20, 0, :] - joint_translation[i, 0, :])[[0, 2]]
                frame_feaure[i,2:4] = cur_orientation_inv.apply(joint_translation[i + 40, 0, :] - joint_translation[i, 0, :])[[0, 2]]
                frame_feaure[i,4:6] = cur_orientation_inv.apply(joint_translation[i + 60, 0, :] - joint_translation[i, 0, :])[[0, 2]]
                # 未来关节朝向变化（20、40、60帧后的旋转变化，局部坐标系）
                frame_feaure[i,6:8] = cur_orientation_inv.apply(R.from_quat(joint_orientation[i + 20, 0, :]).apply(np.array([0.,0.,1.])))[[0,2]]
                frame_feaure[i,8:10] = cur_orientation_inv.apply(R.from_quat(joint_orientation[i + 40, 0, :]).apply(np.array([0.,0.,1.])))[[0,2]]
                frame_feaure[i,10:12] = cur_orientation_inv.apply(R.from_quat(joint_orientation[i + 60, 0, :]).apply(np.array([0.,0.,1.])))[[0,2]]
            #脚的局部位置
            frame_feaure[i,12:15] = cur_orientation_inv.apply(joint_translation[i, mocap.joint_name.index('lAnkle'), :] - joint_translation[i, 0, :])
            frame_feaure[i,15:18] = cur_orientation_inv.apply(joint_translation[i, mocap.joint_name.index('rAnkle'), :] - joint_translation[i, 0, :])
            if i != 0:
                #跟节点速度
                frame_feaure[i,18:21] = cur_orientation_inv.apply(\
                    (joint_translation[i, mocap.joint_name.index('RootJoint'), :] - joint_translation[i - 1, mocap.joint_name.index('RootJoint'), :]) / dt)
                #脚步局部速度
                root_velocity = frame_feaure[i,18:21]
                frame_feaure[i,21:24] = cur_orientation_inv.apply(\
                        (joint_translation[i, mocap.joint_name.index('lAnkle'), :] - joint_translation[i - 1, mocap.joint_name.index('lAnkle'), :]) / dt) - root_velocity
                frame_feaure[i,24:27] = cur_orientation_inv.apply(\
                        (joint_translation[i, mocap.joint_name.index('rAnkle'), :] - joint_translation[i - 1, mocap.joint_name.index('rAnkle'), :]) / dt) - root_velocity
        return frame_feaure
        
    # 批量处理目录下所有bvh文件
    def batch_extract_bvh_features(self):
        features_list = []
        #filenames = []
        #file_path = os.path.join("..", "motion_material", "walk.bvh")

        for filepath in ["./motion_material/walk.bvh","./motion_material/turn_left.bvh","./motion_material/turn_right.bvh","./motion_material/spin_clockwise.bvh","./motion_material/spin_counter_clockwise.bvh"]:
            if filepath=="./motion_material/walk.bvh":
                mocap=self.motion0
            else:
                mocap = BVHMotion(filepath)
            feats = self.extract_advanced_features(mocap)#成功调用类内的函数
            features_list.append(feats)
        return features_list

    def normalize_features(self,feat):
        # compute mean and std
        #由于这些东西不等长
        features_offset = np.mean(feat, axis=0)
        features_scale = np.std(feat, axis=0)

        # weight
        features_scale = features_scale / self.feature_weight

        return features_offset, features_scale
    
    def initialize(self):
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
        #自己加进来的
        self.feats=self.batch_extract_bvh_features()
        self.feature_weight = np.array([1.0]*6+[1.5]*6+[0.75]*6+[1.25]*9)
        
        features=np.concatenate(self.feats)
        self.feature_offset, self.feature_scale = self.normalize_features(features)
        
        self.feats0=(self.batch_extract_bvh_features()[0]-self.feature_offset)/self.feature_scale#要进行归一化
        self.feats1=(self.batch_extract_bvh_features()[1]-self.feature_offset)/self.feature_scale
        self.feats2=(self.batch_extract_bvh_features()[2]-self.feature_offset)/self.feature_scale
        self.feats3=(self.batch_extract_bvh_features()[3]-self.feature_offset)/self.feature_scale
        self.feats4=(self.batch_extract_bvh_features()[4]-self.feature_offset)/self.feature_scale
        
        #print("self.feats.shape: ",len(self.feats[0]))
        #print("self.feats.shape: ",len(self.feats[1]))
        #建树
        self.feature0_kd_tree=KDTree(self.feats0)
        self.feature1_kd_tree=KDTree(self.feats1)
        self.feature2_kd_tree=KDTree(self.feats2)
        self.feature3_kd_tree=KDTree(self.feats3)
        self.feature4_kd_tree=KDTree(self.feats4)
        # 单个bvh文件的动作提取器
        '''
        代码提取了以下几类特征：

        未来关节位置变化（20、40、60帧后的位置变化，局部坐标系）

        未来关节朝向变化（20、40、60帧后的旋转变化，局部坐标系）

        当前脚部位置（右脚和左脚踝的位置）

        臀部局部速度（通过根关节的运动计算）

        脚部局部速度（右脚和左脚踝的速度）
        '''
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        #获得特征向量
        feature_list=self.feats
        #获得特征向量之后用KD树组织起来 方便之后进行查找
        #不把这个变量加到self中的坏处就在于每次要重新构建一遍
        
        #构建现在这个desired的feature 之后可以进行query
        #首先，当前pose是当前bvh文件中的cur_frame
        bvh_now=self.cur_node.motion
        joint_name=bvh_now.joint_name
        joint_translation_, joint_orientation_ =bvh_now.batch_forward_kinematics()
        joint_translation, joint_orientation=joint_translation_[self.cur_frame], joint_orientation_[self.cur_frame]
        self.cur_root_pos=joint_translation[0,:]
        self.cur_root_rot=joint_orientation[0,:]
        root_position=self.cur_root_pos
        root_orientation=self.cur_root_rot.flatten()
        '''
        print("root_position: ",root_position)
        print("root_orientation: ",root_orientation.flatten())
        print("R.from_quat(root_orientation): ",R.from_quat(root_orientation.flatten()))
        '''
        cur_feature_vector = np.zeros([27])
        root_orientation_inv=R.from_quat(root_orientation).inv()
        # print("root_orientation_inv:",root_orientation_inv)
        #期望的未来关节位置变化
        cur_feature_vector[0:2] = root_orientation_inv.apply(desired_pos_list[1] - root_position)[[0, 2]]
        cur_feature_vector[2:4] = root_orientation_inv.apply(desired_pos_list[2] - root_position)[[0, 2]]
        cur_feature_vector[4:6] = root_orientation_inv.apply(desired_pos_list[3] - root_position)[[0, 2]]
        #期望的未来关节朝向变化
        cur_feature_vector[6:  8] = root_orientation_inv.apply(R.from_quat(desired_rot_list[1]).apply(np.array([0.,0.,1.])))[[0,2]]
        cur_feature_vector[8: 10] = root_orientation_inv.apply(R.from_quat(desired_rot_list[2]).apply(np.array([0.,0.,1.])))[[0,2]]
        cur_feature_vector[10:12] = root_orientation_inv.apply(R.from_quat(desired_rot_list[3]).apply(np.array([0.,0.,1.])))[[0,2]]
        # 对当前做归一化
        #当前的offset和scale还没算出来
        #修改 对query归一化
        cur_feature_vector[:12]=(cur_feature_vector[:12]-self.feature_offset[:12])/self.feature_scale[:12]
            
        # 当前bvh文件中的feature
        feats=feature_list[self.cur_node.identity]
        # 当前帧的feature
        feats_now=feats[self.cur_frame]
        cur_feature_vector[12:27]=feats_now[12:27]
        '''
        print("feature_offset: ",self.feature_offset)#这是一个标量
        print("feature_scale: ",self.feature_scale)
        print("cur_feature_vector: ",cur_feature_vector)
        '''
        # 查找下一帧的最好帧(我可以在不同的bvh文件中分别查找 这样之后更新就会方便一些)
        best_cost0, best_frame0 = self.feature0_kd_tree.query(cur_feature_vector)  
        best_cost1, best_frame1 = self.feature1_kd_tree.query(cur_feature_vector)  
        best_cost2, best_frame2 = self.feature2_kd_tree.query(cur_feature_vector)  
        best_cost3, best_frame3 = self.feature3_kd_tree.query(cur_feature_vector)  
        best_cost4, best_frame4 = self.feature4_kd_tree.query(cur_feature_vector)  
        
        costs=[best_cost0,best_cost1,best_cost2,best_cost3,best_cost4]
        bvh_files=["./motion_material/walk.bvh","./motion_material/turn_left.bvh","./motion_material/turn_right.bvh","./motion_material/spin_clockwise.bvh","./motion_material/spin_counter_clockwise.bvh"]
        frames=[best_frame0,best_frame1,best_frame2,best_frame3,best_frame4]
        min_index = costs.index(min(costs))
        best_bvh = bvh_files[min_index]
        best_frame = frames[min_index]
        best_cost = costs[min_index]
        #print("best_bvh: ",best_bvh)
        #如果和当前帧不同 则变 否则不变
        if min_index!=self.cur_node.identity and np.linalg.norm(desired_vel_list[0])> 0.1:
            #要对下一帧做切片 和当前帧衔接上
            #获取当前帧的内容
            self.counter-=1
            if self.counter==0:
                self.counter=3
                name=self.cur_node.name
                id=self.cur_node.identity
                frame=self.cur_frame
                facing_axis = R.from_quat(self.graph.nodes[id].motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                self.cur_node=self.graph.nodes[min_index]
                #self.cur_end_frame = self.cur_node.motion.motion_length
                self.cur_node.motion=self.cur_node.motion.translation_and_rotation(best_frame, self.graph.nodes[id].motion.joint_position[frame, 0, [0, 2]],facing_axis)
                self.cur_frame=best_frame
                self.cur_edge=Edge(name+'->'+self.cur_node.name,self.cur_node)
            else:
                self.cur_frame=(self.cur_frame+1)%self.cur_node.motion.motion_length
                self.cur_edge=None
        else:
            #前进一帧即可
            self.cur_frame=(self.cur_frame+1)%self.cur_node.motion.motion_length
            self.cur_edge=None
            #self.cur_root_pos,self.cur_root_rot=self.cur_node.motion.joint_position[self.cur_frame, 0,:],self.cur_node.motion.joint_rotation[self.cur_frame, 0,:]
        return joint_name, joint_translation, joint_orientation