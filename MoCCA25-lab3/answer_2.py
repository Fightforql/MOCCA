# 姓名：梁书怡
# 学号：2300013147
import scipy.linalg
import utils
from joints import Joint

import numpy as np
import scipy
import copy
from scipy.spatial.transform import Rotation as RR
from typing import List, Tuple
from answer_1 import forward_dynamics
from answer_3 import hinge_Jacobian

def cross_matrix(x):
    res=np.array([[0.,-x[2],x[1]],
                 [x[2],0.,-x[0]],
                 [-x[1],x[0],0.]
                 ])
    return res


def create_joint(x:np.ndarray, R:np.ndarray, joint_pos:np.ndarray, bodyA:int, bodyB:int) -> Joint:
    ''' 根据输入信息创建关节定义
    
        给出一组刚体的位置、朝向，以及一组关节的信息，创建关节实例（参见 joints.py 中 Joint 类的定义）
        
        输入：
            x: (num_bodies, 3)      刚体（质心）的位置
            R: (num_bodies, 3, 3)   刚体的朝向，表示为一组矩阵
            joint_pos: (3)          关节的位置
            bodyA: int              关节连接的刚体A的index
            bodyB: int              关节连接的刚体B的index
                    
        x,R,joint_pos 均为世界坐标系表示
        
        输出：
            一个 Joint 实例，包含关节的定义信息
            
        提示：
            * 你需要计算每个关节的 from_bodyA, from_bodyB，即关节在对应刚体局部坐标系下的位置
            * from_bodyA, from_bodyB 表示同一个关节在两个刚体局部坐标系下的位置
            * 通过 joint_def = Joint(bodyA, bodyB, from_bodyA, from_bodyB) 生成关节定义
    '''
    
    
    ####### 你的回答 #######
    # 提示：需要计算每个关节的 from_bodyA, from_bodyB，即关节在对应刚体局部坐标系下的位置
    # 通过 joint_def = Joint(bodyA, bodyB, from_bodyA, from_bodyB) 生成关节定义
    
    # 下面的例子可以删除
    from_bodyA = np.zeros(3)
    from_bodyB = np.zeros(3)
    from_bodyA=np.linalg.inv(R[bodyA]) @ (joint_pos - x[bodyA])
    from_bodyB=np.linalg.inv(R[bodyB]) @ (joint_pos - x[bodyB])
    joint = Joint(bodyA, bodyB, from_bodyA, from_bodyB)
    #print(joint)
    return joint
        

def ball_Jacobian(x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, joints:List[Joint]) -> Tuple[np.ndarray, np.ndarray]:
    ''' 计算关节约束的 Jacobian 矩阵
    
        给出一组刚体的位置、朝向，以及一组关节的定义（参见 joints.py 中 Joint 类的定义）
        
        计算每个关节的约束 Jacobian 矩阵
            
        输入：
            x: (num_bodies, 3)    刚体（质心）的位置
            R: (num_bodies, 3, 3) 刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)    刚体的（质心）线速度
            w: (num_bodies, 3)    刚体的角速度
            joints:               关节定义列表，包含 num_joints 个关节
            
        x,R,v,w 均为世界坐标系表示
                    
        输出：
            J: (num_joints, 3, 12)  所有关节约束的Jacobian矩阵，对应速度向量 [bodyA_v, bodyA_w, bodyB_v, bodyB_w]
            rhs: (num_joints, 3)     关节约束的右端项，表示为一个修正项，用于修正仿真误差导致的关节错位
                
        提示：
            * 你需要计算每个关节的 Jacobian 矩阵，参考 ppt 中相关内容
            * 关节约束的速度表示为 J v = rhs，其中 rhs=0 或者是一个修正项，用于修正仿真误差导致的关节错位，参考 ppt 中关于 ERP 的内容
            * 你可能不需要用到所有的输入信息
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints)
    bodyA = [jnt.bodyA for jnt in joints]
    bodyB = [jnt.bodyB for jnt in joints]
    
    from_bodyA = np.array([jnt.from_bodyA for jnt in joints])
    from_bodyB = np.array([jnt.from_bodyB for jnt in joints])
    
    x_bodyA, x_bodyB = x[bodyA,:], x[bodyB,:]    
    R_bodyA, R_bodyB = R[bodyA,:,:], R[bodyB,:,:]       
    J = np.zeros((num_joints, 3, 12))
    rhs = np.zeros((num_joints, 3))
    
    ####### 你的回答 #######
    #I_3=np.eye(3)
    for i in range(num_joints):
        jnt=joints[i]
        bodyA=joints[i].bodyA
        bodyB=joints[i].bodyB
        J[i, :, 0:3] = np.eye(3)  # -I for bodyA v
        J[i, :, 3:6] = -cross_matrix(R[bodyA] @ joints[i].from_bodyA)  # [rA]× for bodyA w
        J[i, :, 6:9] = -np.eye(3)    # I for bodyB v
        J[i, :, 9:12] = cross_matrix(R[bodyB] @ joints[i].from_bodyB)  # -[rB]× for bodyB w
        v_jnt_A = v[bodyA] + np.cross(w[bodyA], R[bodyA] @ jnt.from_bodyA)
        v_jnt_B = v[bodyB] + np.cross(w[bodyB], R[bodyB] @ jnt.from_bodyB)
        alpha = 0.25
        delta_v=v_jnt_B  - v_jnt_A
        #rhs[i] = alpha * delta_v
        #J[i]=np.hstack([I_3,-cross_matrix(RR.from_matrix(R_bodyA[i]).apply(from_bodyA[i])),-I_3,cross_matrix(RR.from_matrix(R_bodyB[i]).apply(from_bodyB[i]))])
        
    return J, rhs

def forward_dynamics_with_constraints(
    m:np.ndarray, I:np.ndarray, inv_m:np.ndarray, inv_I:np.ndarray,
    x:np.ndarray, R:np.ndarray, v:np.ndarray, w:np.ndarray, 
    f:np.ndarray, tau:np.ndarray, h:float,
    joints:List[Joint]
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 带约束的刚体前向运动学函数
    
        给出一组刚体的位置、朝向，以及当前速度、角速度，
        给出一组关节的定义，利用欧拉积分计算下一时刻刚体的位置和朝向
    
        输入：
            m: (num_bodies, )         刚体的质量
            I: (num_bodies, 3, 3)     刚体的转动惯量
            inv_m: (num_bodies, )     刚体的质量倒数
            inv_I: (num_bodies, 3, 3) 刚体的转动惯量倒数
            
            x: (num_bodies, 3)        刚体（质心）的位置
            R: (num_bodies, 3, 3)     刚体的朝向，表示为一组矩阵
            v: (num_bodies, 3)        刚体的（质心）线速度
            w: (num_bodies, 3)        刚体的角速度
            
            f: (num_bodies, 3)        施加在每个刚体质心上的力
            tau: (num_bodies, 3)      施加在每个刚体上的力矩
            h: ()                     时间步长
            
            joints:             关节定义列表，包含 num_joints 个关节
        
        所有量均为世界坐标系表示
        
        输出：
            v_next: (num_bodies, 3)        刚体下一时刻的速度
            w_next: (num_bodies, 3)        刚体下一时刻的角速度
            
        提示：
            * 使用带约束的 Newton-Euler 方程来求解下一时刻刚体的状态，参考ppt中相关内容
            * 输入包括了场景中所有的刚体，你可以使用 for 循环来处理每个刚体（不推荐），或者
              使用 numpy 的向量化操作来处理所有刚体
            * 你可能不需要用到所有的输入信息
            * 你需要使用 ball_Jacobian 函数来计算关节约束的 Jacobian 矩阵
    '''
    
    num_bodies = x.shape[0]
    num_joints = len(joints) 
    
    hinge_joints = [jnt for jnt in joints if jnt.hinge_axis is not None]
    num_hinge_joints = len(hinge_joints)
    #num_joints=num_joints-num_hinge_joints
    v_next, w_next = v.copy(), w.copy()
    
    
    
    ####### 你的回答 #######
    #------------------------------------------------------------------
    # 以下是一个示例框架，你可以在其基础上修改，也可以删除改为你自己的实现
    # 注意：这个框架的效率比较低，更高效的计算需要考虑质量矩阵、约束 Jacobian 矩阵的稀疏性
    
    # 运动方程的矩阵形式： 由牛顿欧拉方程推导而来，考虑加速度的近似： a = (v_next - v) / h
    #   M * [v_next, w_next] = M * [v, w] - h * [0, w x Iw] + h * [f, tau] + h * J^T * lambda
    # 可以得到
    #   [v_next, w_next] = ([v, w] + h * M^-1 * ([f, tau] - [0, w x Iw])) + h * M^-1 * J^T * lambda
    #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                    = >[v0, w0]< + h * M^-1 * J^T * lambda
    # 如果你完成了问题1中的 forward_dynamics函数，你会发现第一项 [v0, w0] 就是 forward_dynamics 函数的输出
    # 约束方程为：
    #   J * [v_next, w_next] = rhs
    # 带入上面的结果，并整理可以得到
    #   A lambda = b
    # 其中 A = h * J * M^-1 * J^T
    #      b = rhs - J * [v0, w0]
    # 为了数值稳定，我们可以对 A 进行修正： diag(A) += cfms
    # 进而求解得到 
    #   lambda = A^-1 b
    # 然后带入上面的结果，得到 [v_next, w_next] = [v0, w0] + h * M^-1 * J^T * lambda    
    #------------------------------------------------------------------
        
    #------------------------------------------------------------------
    # [1]. 构造所有刚体的质量矩阵，M = diag([m_1, I_1,... m_n, I_n])，及其逆矩阵
    #------------------------------------------------------------------、
    inv_I_world = np.zeros((num_bodies, 3, 3))
    I_world=np.zeros((num_bodies, 3, 3))
    for i in range(num_bodies-1):
        inv_I_world[i] = R[i] @ inv_I[i] @ R[i].T
        I_world[i] = R[i] @ I[i] @ R[i].T
   
    M = np.zeros((num_bodies*6, num_bodies*6))
    inv_M = np.zeros((num_bodies*6, num_bodies*6))
    # 在对角线上填充质量和转动惯量及其倒数
    # 注意转动惯量 I 及其逆 inv_I 需要随着旋转进行相应的变换
    # 你可以考虑使用 scipy.linalg.block_diag 来构造对角矩阵
    blocks1=[]
    blocks2=[]
    
    
    for i in range(num_bodies):
        
        blocks1.append(m[i]*np.eye(3))
        blocks2.append(inv_m[i]*np.eye(3))
        blocks1.append(I_world[i])
        blocks2.append(inv_I_world[i])
    M=scipy.linalg.block_diag(*blocks1)
    inv_M=scipy.linalg.block_diag(*blocks2)
    
    
    
    #------------------------------------------------------------------
    # [2]. 计算无约束下刚体的速度和角速度 [v0, w0]
    #------------------------------------------------------------------
    # 你可以使用 forward_dynamics 来帮你计算，也可以复制代码到这里，因为后面的计算可能需要用到一些中间变量
    vw0=[]
    ##### 你的代码 #####
    v_next0, w_next0 =np.zeros_like(v),np.zeros_like(w)
    #对转动惯量的更新
    
    
    for i in range(num_bodies):
        a=inv_m[i]*f[i]
        v_next0[i]=v[i]+h*a
        Iw = I_world[i] @ w[i]
        dI_dt_w=np.cross(w[i],Iw)
        w_alpha=inv_I_world[i]@(tau[i]-dI_dt_w)
        w_next0[i]=w[i]+h*w_alpha
        vw0.append(v_next0[i])
        vw0.append(w_next0[i])
    ################### 
    #vw0.append(np.zeros(3))   
    #vw0.append(np.zeros(3))
    vw=np.concatenate(vw0).reshape(-1, 1) 
    #------------------------------------------------------------------
    # [3]. 构造所有约束的 Jacobian 矩阵 J 和右端项 rhs
    #------------------------------------------------------------------
    
    constraint_dim = num_joints * 3
    constraint_dim += num_hinge_joints * 2 # hinge 关节的额外约束
    
    J = np.zeros((constraint_dim, num_bodies*6))
    rhs = np.zeros((constraint_dim, 1))
    
    # 计算每个约束 Jacobian 矩阵 J 和右端项 rhs
    J_ball, rhs_ball = ball_Jacobian(x, R, v, w, joints)
    if num_hinge_joints > 0:
        J_hinge, rhs_hinge = hinge_Jacobian(x, R, v, w, hinge_joints)
        rhs_hinge_flat = rhs_hinge.reshape(-1, 1)
   
    # 将 J_ball, rhs_ball, J_hinge, rhs_hinge 填充到 J 和 rhs 中
    # 注意 J_ball, J_hinge 的大小都是 (nj, dof, 12), 如果你的实现正确，每个约束矩阵的最后一维的前 6 列应该对应 bodyA，后 6 列应该对应 bodyB
    #   其中对于 J_ball， dof = 3；对于 J_hinge， dof = 2，你需要将对应的矩阵块填充到 J 中的合适位置
    # 注意 rhs_ball, rhs_hinge 的大小都是 (nj, dof)，你可能需要 reshape 调整它们的形状    
    
    ##### 你的代码 #####
    offset=0
    for i in range(num_joints):
        #print(i)
        body1=joints[i].bodyA
        body2=joints[i].bodyB
        
        if body1==-1:
            body1=num_bodies-1
        if body2 == -1:
            body2=num_bodies-1
        
        #print(J[3*i:3*i+3,body1*6:body1*6+6].shape)
        J[3*i:3*i+3,body1*6:body1*6+6]=J_ball[i][:,0:6]
        J[3*i:3*i+3,body2*6:body2*6+6]=J_ball[i][:,6:12]
        #offset=3*i+3
    
    for i in range(num_joints,num_joints+num_hinge_joints):
        body1=hinge_joints[i-num_joints].bodyA
        body2=hinge_joints[i-num_joints].bodyB
        
        if body1==-1:
            body1=num_bodies-1
        if body2 == -1:
            body2=num_bodies - 1
        row = 3*num_joints + 2*(i-num_joints)
        
        J[row:row+2,body1*6:body1*6+6]=J_hinge[i-num_joints][:,0:6]
        J[row:row+2,body2*6:body2*6+6]=J_hinge[i-num_joints][:,6:12]
    

    for i in range(num_joints):
        #print(i)
        body1=joints[i].bodyA
        body2=joints[i].bodyB
        
        if body1==-1:
            body1=num_bodies-1
        if body2 == -1:
            body2=num_bodies-1
        
        #print(J[3*i:3*i+3,body1*6:body1*6+6].shape)
        J[3*i:3*i+3,body1*6:body1*6+6]=J_ball[i][:,0:6]
        J[3*i:3*i+3,body2*6:body2*6+6]=J_ball[i][:,6:12]

    
    #print(num_hinge_joints)
    rhs_ball_flat = rhs_ball.reshape(-1, 1)
    
    if num_hinge_joints!=0:
        rhs =np.concatenate([rhs_ball_flat, rhs_hinge_flat], axis=0)
    else:
        rhs=rhs_ball_flat
    
    ###################
        
    #------------------------------------------------------------------
    # [4]. 构造约束方程的矩阵 A 和右端项 b
    #A = h * J * M^-1 * J^T
    #------------------------------------------------------------------
    A = np.zeros((constraint_dim, constraint_dim))
    b = np.zeros((constraint_dim, 1))
    
    ##### 你的代码 #####
    #print(vw.shape)
    A=h*(J@inv_M@np.transpose(J)) #A是9*9矩阵 J是9*24矩阵 vw是24*1
    
    b = rhs - J@vw
    ###################
    
    #------------------------------------------------------------------
    # [5]. 求解约束方程 A lambda = b，得到 lambda
    #------------------------------------------------------------------
    A += np.diag(np.ones(constraint_dim) * 1e-12) # 数值稳定性修正 cfm = 1e-8
    lambd = np.linalg.solve(A, b) 
    
    #------------------------------------------------------------------
    # [6]. 计算约束对速度和角速度的影响，得到 [v_next, w_next] = [v0, w0] + h * M^-1 * J^T * lambda
    #------------------------------------------------------------------
    
    delta_vw = h * (inv_M @ np.transpose(J) @ lambd)
    for i in range(num_bodies):
        v_next[i] = v_next0[i] + delta_vw[i*6:i*6+3].reshape(-1)
        w_next[i] = w_next0[i] + delta_vw[i*6+3:i*6+6].reshape(-1)
    
    
    #######################
    #print('当前angular momentum:', utils.angular_momentum(m, I, x, R, v, w))
    
    return v_next, w_next