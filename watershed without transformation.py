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

def init(kd_tree,watershed_list,gray,m1,m2,S,remain,points,normal,n,ws_color,plane_vector):
    add_neighbor(kd_tree,m1,1,gray,S,points,n,normal,ws_color,watershed_list,plane_vector)
    watershed_list[m1]=1#red
    remain=remain-1
    add_neighbor(kd_tree,m2,2,gray,S,points,n,normal,ws_color,watershed_list,plane_vector)
    watershed_list[m2]=2#blue 발
    remain=remain-1
    return watershed_list,remain

def add_neighbor(kd_tree,index,marker,gray,S, points , n, normal, ws_color,watershed_list,plane_vector):

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
        z=abs(np.dot(points[index]-points[idx[i]],np.array(plane_vector)))
        distance = (10 * z + abs(gray[index] - gray[idx[i]])).astype(int)
        if watershed_list[idx[i]]==0:
            S.add(candidate(idx[i],marker,distance))
#점 사이 거리도 고려해야, 활성함수?

def watershed(gray,kd_tree,watershed_list,S,remain,points,n,normal,ws_color,plane_vector):

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
            add_neighbor(kd_tree,index,marker,gray,S,points,n,normal,ws_color,watershed_list,plane_vector)
    return watershed_list,remain


def execute(pcd,m1,m2,ws_color,n,debug):
    if ws_color:
        print("normal vector와 color정보 모두 사용")
    else:
        print("normal vector만 사용")

    print("사용한 인접점의 개수는 %d 개" % (n))
    S = SortedList()
    if debug:o3d.visualization.draw_geometries([pcd], width=720, height=480, window_name="input")

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))

    ###################################################################################################################
    points = np.array(pcd.points)
    if int(points[0][2] * 100) / 100 - points[0][2] != 0:
        points = (1000 * points).astype(int)  # kinect는 mm단위 Iphone는 m단위
        pcd.points = o3d.utility.Vector3dVector(points)
    ###################################################################################################################
    plane_model, idx = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"index len: {len(idx)}")
    print(f"normal vector:{[a,b,c]}")
    plane_vector=[a,b,c]
    ###################################################################################################################
    foot_pcd = copy.deepcopy(pcd)
    floor_pcd = copy.deepcopy(pcd)
    normal = np.array(pcd.normals)
    color = (255 * np.array(pcd.colors)).astype(int)
    gray = ((color[:, 0] * 0.299 + color[:, 1] * 0.587 + color[:, 2] * 0.114)).astype(int)
    watershed_list = np.zeros(len(pcd.points))
    remain = len(pcd.points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    ####################################################################################################################
    watershed_list,remain = init(kd_tree, watershed_list, gray,m1,m2,S,remain,points,normal,n,ws_color,plane_vector)
    start = time.time()
    watershed_list,remain = watershed(gray, kd_tree, watershed_list,S,remain,points,n,normal,ws_color,plane_vector)
    print("watershed time", round(time.time() - start, 2), "sec")
    print("marking 안 된 점의 개수는", remain, "개")
    ####################################################################################################################
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
    foot = remove_small_clusters(foot, False)

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



    '''marker = [[28510, 26450, 14698, 18533, 12152, 18783],# 바닥
              [20088, 19816, 23178, 24689, 20635, 24460]]  # 발

    ###################################################################################################################
    for i in range(len(marker[0])):
        pcd = o3d.io.read_point_cloud("D:\download/7-14/0%d.ply" % (i))
        #pcd = o3d.io.read_point_cloud("/media/bbangsik/새 볼륨/download/7-14/0%d.ply" % (i))
        segment = execute(pcd, marker[0][i], marker[1][i], ws_color, n, debug)
        print(segment.points)
        o3d.io.write_point_cloud("D:\download/7-14/segment%d.ply" % (i),segment, write_ascii=True)'''

    pcd = o3d.io.read_point_cloud("D:/bbangsik/reconstruct.ply")
    segment = execute(pcd, 39569, 151165, ws_color, n, debug)





