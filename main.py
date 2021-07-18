import copy
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from sortedcontainers import SortedList
import time

def remove_small_clusters(sPCD, debug = False):

    # 4.3 take only Foot part remove small clusters
    # 4.3.1 clustering , 파라메터 튜닝 필요!!!
    seglabels = sPCD.cluster_dbscan(eps = 10, min_points = 20, print_progress=False)
    #Density Based Spatial Clustering of Applications with Noise=?DBSCAN algorithmn
    #np.array(pcd.points)순으로 그 점의 인덱스에 대한 배열 반환

    labelset, label_indexes, label_counts = np.unique(seglabels, return_index = True, return_counts= True)

    if debug:
        print("seglabels:", labelset)#label number
        print("label_counts:",  label_counts)#label에 해당하는 점의 개수
        print("label_indexes:",  label_indexes)#label의 최초점의 인덱스
    max_label = len(labelset)

    pcd_colors_original = o3d.utility.Vector3dVector(np.asarray(sPCD.colors))  # back up for restore original color

    foot_label = labelset[0]
    if debug:
        print("noise: %d, max label: %d"%(0, foot_label))

    # 4.3.2 taking the biggest cluster (foot)
    '''    
    color_segmentation_foot(foot_pcd, np.asarray(seglabels), foot_label)
    o3d.visualization.draw_geometries([foot_pcd, coord_axis],
                                   width = 640, height = 576,
                                  zoom=1.,
                                  front=[0.6452, -0.3036, -0.7011],
                                  lookat= [0., 0., 0.], # [1.9892, 2.0208, 1.8945],
                                  up=[-0.2779, -0.9482, 0.1556],
                                   window_name = "foot_segmented")   
    '''
    # get only foot points and color
    # to numpy (why not using numpy Open3D?)
    points_before = np.asarray(sPCD.points)
    normals_before = np.asarray(sPCD.normals)
    if debug:
        colors_before = np.asarray(pcd_colors_original)
    else:
        colors_before = np.asarray(sPCD.colors)
    # select
    points = points_before[seglabels != foot_label, : ]
    colors = colors_before[seglabels != foot_label, : ]
    normals = normals_before[seglabels != foot_label, :]
    # reconstruct points and colors
    #sPCD.points =  o3d.utility.Vector3dVector(points)
    tPCD = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tPCD.colors =  o3d.utility.Vector3dVector(colors)
    tPCD.normals=o3d.utility.Vector3dVector(normals)

    return tPCD


class candidate:
    def __init__(self,index,marker,distance):
        self.index=index
        self.marker=marker
        self.distance=distance
    def __lt__(self,other):
        return self.distance<other.distance

def init(kd_tree,watershed_list,gray,m1,m2,S,remain,z,normal,n,ws_color):
    add_neighbor(kd_tree,m1,1,gray,S,z,n,normal,ws_color,watershed_list)
    watershed_list[m1]=1#red
    remain=remain-1
    add_neighbor(kd_tree,m2,2,gray,S,z,n,normal,ws_color,watershed_list)
    watershed_list[m2]=2#blue 발
    remain=remain-1
    return watershed_list,remain

def add_neighbor(kd_tree,index,marker,gray,S, z , n, normal, ws_color,watershed_list):

    [k, idx, _] = kd_tree.search_knn_vector_3d(pcd.points[index], n)
    for i in range(k):
        '''if ws_color==True:
            distance =(np.tanh(1000*(1-abs(normal[index][2]*normal[idx[i]][2])))+ abs(gray[index] - gray[idx[i]])).astype(int)
            #distance = (10*abs(z[index]-z[idx[i]]) + abs(gray[index] - gray[idx[i]])).astype(int)
        else:
            #dot = abs(np.dot(normal[index], normal[idx[i]]))
            dot=abs(normal[index][2]*normal[idx[i]][2])
            distance=np.tanh(1000*(1-dot))
        #distance =(1000000*(1-dot)*(1-dot)).astype(int)'''
        distance = (10 * abs(z[index] - z[idx[i]]) + abs(gray[index] - gray[idx[i]])).astype(int)
        if watershed_list[idx[i]]==0:
            S.add(candidate(idx[i],marker,distance))
#점 사이 거리도 고려해야, 활성함수?

def watershed(gray,kd_tree,watershed_list,S,remain,z,n,normal,ws_color):

    while remain>0:
        if len(S)<1:
            print('list is empty')
            break
        c=S.pop(0)
        index=c.index
        marker=c.marker
        if watershed_list[index]==0:
            watershed_list[index]=marker
            remain=remain-1
            add_neighbor(kd_tree,index,marker,gray,S,z,n,normal,ws_color,watershed_list)
    return watershed_list,remain


def calculate_3D_rigid_transform(A, B):

    """ 3개의 점 => 3개의 점 변환

       (R|T) A = B
       http://nghiaho.com/?page_id=671 참고
       https://github.com/nghiaho12/rigid_transform_3D 에서 가져온것 같은데.
        이건 오차가 있는 점들이 많을 때 사용하는 방법인것 같고, 정확히 3점이 같을 때는 이렇게 안해도 될 꺼 같지 않나?
    """

    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def get_plane_align_matrix(plane_model, debug=False,draw=False):#원래 평면의 3점을 xy위에 대응되는 3점으로 매핑

    """

        축을 Z =0 평명으로 정렬 (목적은 바닥과 다리통 제거를 위함)
        a, b, c가 0이 아니어야 함. 특히 c?

                   +  tpy
            | d
            |
            |
            |
            +------------*
            tpz   d3     tpx

    """

    # 이거 앞에서 했는데 왜 또 하지?
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    if debug :
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # plane 1을 plane 2 (z=0)로 변환
    # 1. 현재 평명의 3점
    pts_src = np.array([[-d/a,0,0],[0,-d/b,0],[0,0,-d/c]])

    # 2. 타겟 평면의 3점
    pts_tgt = np.zeros_like(pts_src)
    d1 = np.linalg.norm(pts_src[1] - pts_src[2])  #distance(pts_src[1],pts_src[2])
    d2 = np.linalg.norm(pts_src[0] - pts_src[1])
    d3 = np.linalg.norm(pts_src[0] - pts_src[2])

    '''
    h2 = d2**2 - ((d3**2+d2**2-d1**2)**2)/(4*(d3**2))
    h  = math.sqrt(h2)
    '''
    # Haron's formula
    S = (d1 + d2 + d3)/2.0
    A = math.sqrt(S * (S - d1) * (S - d2) * (S - d3))
    h = 2.0*A/d3   # height
    k  = math.sqrt(d1**2-h**2)
    pts_tgt[0] = [d3,0,0]
    pts_tgt[1] = [k, h, 0]
    pts_tgt[2] = [0,0,0]

    # 두점집합을 넣고 변환 (이헐게 복잡하게 할필요가 있을까? 필요한건 Z = 0일 뿐인데)
    rotate, translate = calculate_3D_rigid_transform(pts_src.T,pts_tgt.T)
    T  = np.eye(4)
    T[:3,:3] = rotate
    T[:3, 3] = translate[:].T
    return T


def normalization(arr):
    square=arr*arr
    sum=np.sum(square,axis=1)
    dis=np.sqrt(sum)
    a=np.zeros_like(arr)
    a[:,0]=dis
    a[:,1]=dis
    a[:,2]=dis
    normalized_arr=arr/a
    return normalized_arr

def polarization(pcd):
    point = np.array(pcd.points)
    normal = np.array(pcd.normals)
    vec=normalization(point)
    mul=vec*normal
    dot=np.sum(mul,axis=1)
    normal[dot>0]*=-1
    pcd.normals = o3d.utility.Vector3dVector(normal)
    return pcd

def execute(pcd,m1,m2,ws_color,n,debug):
    camera=np.array([0,0,0,1])

    if ws_color:
        print("normal vector와 color정보 모두 사용")
    else:
        print("normal vector만 사용")

    print("사용한 인접점의 개수는 %d 개" % (n))
    S = SortedList()
    if debug:o3d.visualization.draw_geometries([pcd], width=720, height=480, window_name="input")

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        pcd=polarization(pcd)
    if debug:o3d.visualization.draw_geometries([pcd], width=720, height=480, window_name="estimate normal",point_show_normal=True)
    T = get_plane_align_matrix(pcd)
    pcd = pcd.transform(T)
    camera[0]=np.sum(T[0]*camera)
    camera[1] = np.sum(T[1] * camera)
    camera[2] = np.sum(T[2] * camera)
    camera[3] = np.sum(T[3] * camera)
    #print(T)
    if abs(np.min(np.array(pcd.points)[:, 2])) > np.max(np.array(pcd.points)[:, 2]):
        R = np.eye(4)
        R[2][2], R[1][1] = -1, -1
        print(R)
        pcd = pcd.transform(R)
        camera[0] = np.sum(R[0] * camera)
        camera[1] = np.sum(R[1] * camera)
        camera[2] = np.sum(R[2] * camera)
        camera[3] = np.sum(R[3] * camera)
    T=np.eye(4)
    T[0][3]=-np.min(np.array(pcd.points)[:,0])
    camera[0]-=np.min(np.array(pcd.points)[:,0])
    T[1][3]=-np.min(np.array(pcd.points)[:,1])
    camera[1] -= np.min(np.array(pcd.points)[:, 1])
    pcd=pcd.transform(T)
    if debug:print(f"camera location: {camera}mm")
    ###################################################################################################################
    points = np.array(pcd.points)
    if int(points[0][2] * 100) / 100 - points[0][2] != 0:
        points = (1000 * points).astype(int)  # kinect는 mm단위 Iphone는 m단위
        pcd.points = o3d.utility.Vector3dVector(points)
    ###################################################################################################################
    coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coord_axis.scale(100, center=(0, 0, 0))  # how to make thin lines?
    coord_axis = coord_axis.translate((pcd.points[0][0], pcd.points[0][1], 0))
    if debug:o3d.visualization.draw_geometries([pcd, coord_axis], width=720, height=480, window_name="transform")
    foot_pcd = copy.deepcopy(pcd)
    floor_pcd = copy.deepcopy(pcd)
    normal = np.array(pcd.normals)
    z = points[:, 2]
    color = (255 * np.array(pcd.colors)).astype(int)
    gray = np.zeros(len(pcd.points))
    gray = ((color[:, 0] * 0.299 + color[:, 1] * 0.587 + color[:, 2] * 0.114)).astype(int)

    watershed_list = np.zeros(len(pcd.points))
    remain = len(pcd.points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    watershed_list,remain = init(kd_tree, watershed_list, gray,m1,m2,S,remain,z,normal,n,ws_color)
    start = time.time()
    watershed_list,remain = watershed(gray, kd_tree, watershed_list,S,remain,z,n,normal,ws_color)
    print("watershed time", round(time.time() - start, 2), "sec")
    print("marking 안 된 점의 개수는", remain, "개")
    ws_color = np.zeros_like(color)
    ws_color[watershed_list == 1, 0] = 255
    ws_color[watershed_list == 2, 2] = 255
    pcd.colors = o3d.utility.Vector3dVector(ws_color)

    if debug:o3d.visualization.draw_geometries([pcd], width=720, height=480, window_name="output")

    a = np.where(watershed_list == 1)
    sel = a[0]
    floor = floor_pcd.select_by_index(sel, invert=False)
    if debug:o3d.visualization.draw_geometries([floor], width=720, height=480, window_name="floor",point_show_normal=True)
    a = np.where(watershed_list == 2)
    sel = a[0]
    foot = foot_pcd.select_by_index(sel, invert=False)
    if debug:o3d.visualization.draw_geometries([foot], width=720, height=480, window_name="foot",point_show_normal=True)
    #mesh, pt_map = foot.hidden_point_removal(camera_location=camera[:3], radius=125.)
    #foot = foot.select_by_index(pt_map,invert=True)
    foot=remove_small_clusters(foot,False)
    if debug: o3d.visualization.draw_geometries([foot], width=720, height=480, window_name="remove noise", point_show_normal=True)
    return foot

if __name__=="__main__":
    n = 8
    ws_color = True
    debug=True
    ###################################################################################################################
    '''marker=[[36905,66712,65787,66103,64377,66011,64634],[58028,51766,46474,42375,43610,44874,30391]]#바닥,발

    ###################################################################################################################
    for i in range(len(marker[0])):
        pcd = o3d.io.read_point_cloud("D:\download\IP-PLY-SOCK-A4-20210712T035528Z-001\IP-PLY-SOCK-A4/0%d.ply"%(i))
        segment=execute(pcd,marker[0][i],marker[1][i],ws_color,n,debug)
        o3d.io.write_point_cloud("D:\download\IP-PLY-SOCK-A4-20210712T035528Z-001\IP-PLY-SOCK-A4/segment%d.ply"%(i),segment,write_ascii=True)'''

    marker = [[28510, 26450, 14698, 18533, 12152, 18783],# 바닥
              [20088, 19816, 23178, 24689, 20635, 24460]]  # 발

    ###################################################################################################################
    for i in range(2,len(marker[0])):
        pcd = o3d.io.read_point_cloud("D:\download/7-14/0%d.ply" % (i))
        segment = execute(pcd, marker[0][i], marker[1][i], ws_color, n, debug)
        print(segment.points)
        o3d.io.write_point_cloud("D:\download/7-14/segment%d.ply" % (i),segment, write_ascii=True)






