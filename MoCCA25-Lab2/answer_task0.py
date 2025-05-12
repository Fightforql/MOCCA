##############
# 姓名：梁书怡
# 学号：2300013147
##############
import numpy as np

# part 0
def load_meta_data(bvh_file_path):
    """
    请把lab1-FK-part1的代码复制过来
    请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        channels: List[int]，整数列表，joint的自由度，根节点为6(三个平动三个转动)，其余节点为3(三个转动)
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量
    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    joint_name=[]
    channels=[]
    depth=0
    st=[]
    for i in range(len(lines)):
        line=lines[i]
        if line.strip().startswith('ROOT') or line.strip().startswith('JOINT'):
            values = lines[i+2].split()[1:]
            offset = tuple(float(val) for val in values)
            if line.strip().startswith('ROOT'):
                st.append((line.split()[-1],depth,offset,6))
            if line.strip().startswith('JOINT'):
                st.append((line.split()[-1],depth,offset,3))
        
        if line.strip().startswith('End'):
            values = lines[i+2].split()[1:]
            offset = tuple(float(val) for val in values)
            node=st[-1]
            st.append((node[0]+"_end",depth,offset,0))
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
        channel=node[3]
        joint_name.append(name)
        joint_offset.append(offset)
        channels.append(channel)
        if curdepth==0:
            joint_parent.append(-1)
        else:
            for j in range(i-1,-1,-1):
                if st[j][1]==curdepth-1:
                    joint_parent.append(j)
                    break
    joint_offset=np.array(joint_offset)
    return joint_name, joint_parent, channels,joint_offset