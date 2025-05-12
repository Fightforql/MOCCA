##############
# 姓名：梁书怡
# 学号：2300013147
##############
"""
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bvh_motion import BVHMotion
from smooth_utils import decay_spring_implicit_damping_pos,decay_spring_implicit_damping_rot,quat_to_avel,align_quat

# part1
def lerp(pos1,pos2,alpha):
    return (1-alpha)*pos1+alpha*pos2
def blend_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, v:float=None, input_alpha:np.ndarray=None) -> BVHMotion:
    '''
    输入: 两个将要blend的动作，类型为BVHMotion
          将要生成的BVH的速度v
          如果给出插值的系数alpha就不需要再计算了
          target_fps,将要生成BVH的fps
    输出: blend两个BVH动作后的动作，类型为BVHMotion
    假设两个动作的帧数分别为n1, n2
    首先需要制作blend 的权重适量 alpha
    插值系数alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    Tips:
        1. 计算给出两端动作的速度，两个BVH已经将Root Joint挪到(0.0, 0.0)的XOZ位置上了，为了便于你计算，我们假定提供的bvh都是沿着z轴正方向向前运动的
        2. 利用v计算插值系数alpha
        3. 线性插值以及Slerp
    '''
    walk_forward = BVHMotion('motion_material/walk_forward_resampled.bvh')
    run_forward = BVHMotion('motion_material/run_forward_resampled.bvh')
    run_forward.adjust_joint_name(walk_forward.joint_name)
    
    # 调整方向和位置, 对齐第一帧
    # 将Root Joint挪到(0.0, 0.0)的XOZ位置上
    walk_forward = walk_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    run_forward = run_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    #print(walk_forward.joint_position[-1,0,2])
    #初始z方向位置为0，用最后一帧计算的z位置/(帧数*frametime) 即为速度
    v1 = walk_forward.joint_position[-1,0,2] /(walk_forward.motion_length*walk_forward.frame_time)
    v2 = run_forward.joint_position[-1,0,2] / (run_forward.motion_length*run_forward.frame_time)
    blend_weight = (v-v1)/(v2-v1)
    #利用公式计算插值的帧数
    d= (1-blend_weight)*v1*walk_forward.motion_length + blend_weight*v2*run_forward.motion_length
    cycle_time = np.around(d/v).astype(np.int32)
    input_alpha = np.ones((cycle_time,)) * blend_weight
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(input_alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(input_alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0
    n1=bvh_motion1.joint_position.shape[0]
    n2=bvh_motion2.joint_position.shape[0]
    joint_num=len(bvh_motion1.joint_name)
    #计算第i帧的插值动作
    for i in range(len(input_alpha)):
        t=float(i/len(input_alpha))
        index1=t*(n1-1)
        index2=t*(n2-1)
        k1=np.floor(index1).astype(int)
        k2=np.ceil(index1).astype(int)
        m1=np.floor(index2).astype(int)
        m2=np.ceil(index2).astype(int)
        for j in range(joint_num):
            position1=lerp(bvh_motion1.joint_position[k1][j],bvh_motion1.joint_position[k2][j],index1-k1)
            position2=lerp(bvh_motion2.joint_position[m1][j],bvh_motion2.joint_position[m2][j],index2-m1)
            rot1=R.from_quat([bvh_motion1.joint_rotation[k1][j],bvh_motion1.joint_rotation[k2][j]])
            rot2=R.from_quat([bvh_motion2.joint_rotation[m1][j],bvh_motion2.joint_rotation[m2][j]])
            time=[0,1]
            slerp1=Slerp(time,rot1)
            slerp2=Slerp(time,rot2)
            rotation1=slerp1(index1-k1)
            rotation2=slerp2(index2-m1)
            res.joint_position[i][j]=lerp(position1,position2,input_alpha[i])
            rot=R.from_quat([rotation1.as_quat(),rotation2.as_quat()])
            slerp=Slerp(time,rot)
            res.joint_rotation[i][j]=slerp(input_alpha[i]).as_quat()
    return res

# part2
def build_loop_motion(bvh_motion:BVHMotion, ratio:float, half_life:float) -> BVHMotion:
    '''
    输入: 将要loop化的动作，类型为BVHMotion
          damping在前在后的比例ratio, ratio介于[0,1]
          弹簧振子damping效果的半衰期 half_life
          如果你使用的方法不含上面两个参数，就忽视就可以了，因接口统一保留
    输出: loop化后的动作，类型为BVHMotion
    
    Tips:
        1. 计算第一帧和最后一帧的旋转差、Root Joint位置差 (不用考虑X和Z的位置差)
        2. 如果使用"inertialization"，可以利用`smooth_utils.py`的
        `quat_to_avel`函数计算对应角速度的差距，对应速度的差距请自己填写
        3. 逐帧计算Rotations和Postions的变化
        4. 注意 BVH的fps需要考虑，因为需要算对应时间
        5. 可以参考`smooth_utils.py`的注释或者 https://theorangeduck.com/page/creating-looping-animations-motion-capture
    
    '''
    res = bvh_motion.raw_copy()
    rotations = res.joint_rotation
    #首先计算开始帧与结束帧的差异
    avel = quat_to_avel(rotations, 1/60)
    rot_diff = (R.from_quat(rotations[-1]) * R.from_quat(rotations[0].copy()).inv()).as_rotvec()
    avel_diff = (avel[-1] - avel[0])
    pos_diff = res.joint_position[-1] - res.joint_position[0]
    pos_diff[:,[0,2]] = 0
    vel1 = res.joint_position[-1] - res.joint_position[-2]
    vel2 = res.joint_position[1] -res.joint_position[0]
    vel_diff = (vel1 - vel2)/60
    # 将旋转差均匀分布到每一帧
    for i in range(res.motion_length):
        #计算diff的衰减offset
        offset1 = decay_spring_implicit_damping_rot(ratio*rot_diff, ratio*avel_diff, half_life, i/60)
        offset2 = decay_spring_implicit_damping_rot((1-ratio)*rot_diff, (1-ratio)*avel_diff, half_life, (res.motion_length-i-1)/60)
        offset_rot = R.from_rotvec(offset1[0] - offset2[0])
        #print(offset_rot.as_matrix())
        res.joint_rotation[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 

    for i in range(res.motion_length):
        offset1 = decay_spring_implicit_damping_pos(ratio*pos_diff, ratio*vel_diff, half_life, i/60)
        offset2 = decay_spring_implicit_damping_pos((1-ratio)*pos_diff, (1-ratio)*vel_diff, half_life, (res.motion_length-i-1)/60)
        offset_pos = offset1[0] - offset2[0]
        res.joint_position[i] += offset_pos
    
    return res

# part3
def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    def find_best_match_frame(weight_pos=0.6, weight_rot=0.4):
   
        target_pos = bvh_motion1.joint_position[mix_frame1]
        target_rot = bvh_motion1.joint_rotation[mix_frame1]
        min_cost = float('inf')
        best_frame = 0
        
        for i in range(bvh_motion2.motion_length):
            # 位置差异（归一化）
            pos_diff = np.mean(np.linalg.norm(bvh_motion2.joint_position[i] - target_pos, axis=1))
            
            # 旋转差异（四元数夹角）
            rot_diff = 0
            for j in range(len(bvh_motion2.joint_name)):
                q1 = R.from_quat(target_rot[j])
                q2 = R.from_quat(bvh_motion2.joint_rotation[i,j])
                rot_diff += (q1.inv() * q2).magnitude() # 旋转差值
            
            # 加权成本
            cost = weight_pos * pos_diff + weight_rot * rot_diff
            if cost < min_cost:
                min_cost = cost
                best_frame = i
        return best_frame
    
    mix_frame2=find_best_match_frame()
    mix_frame2=20
    print(mix_frame2)
    translation_xz = bvh_motion1.joint_position[mix_frame1, 0, [0,2]]
    Ry, _ = bvh_motion1.decompose_rotation_with_yaxis(bvh_motion1.joint_rotation[mix_frame1, 0, :])
    #print(R.from_quat(Ry).as_matrix())
    facing_direction_xz = R.from_quat(Ry).as_matrix()[0][2,[0,2]]
    facing_direction_xz = [0, 1.0]
    bvh_motion2 = bvh_motion2.translation_and_rotation(mix_frame2, translation_xz, facing_direction_xz)
    res1=bvh_motion1.raw_copy()
    res2 = bvh_motion2.raw_copy()
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
    
    res1.append(res2)
    
    #res1.append(res2)
    return res1
    
