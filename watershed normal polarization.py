import numpy as np
import open3d as o3d
import time


def init(kd_tree):
    global S, watershed_list
    add_neighbor(kd_tree,61907)
    watershed_list[61907]=1#확실한 normal


def add_neighbor(kd_tree,index):
    global S, n, normal, watershed_list

    [k, idx, _] = kd_tree.search_knn_vector_3d(pcd.points[index], n)
    for i in range(k):
        if watershed_list[idx[i]]==0:
            #dot = np.dot(normal[index], normal[idx[i]])
            angle = np.arccos(np.clip(np.dot(normal[index], normal[idx[i]]), -1.0, 1.0)) / np.pi * 180
            if angle>130:
                watershed_list[idx[i]]=-1
                normal[idx[i]]*=-1
                #print("inverse")
            else: watershed_list[idx[i]]=1
            S.append(idx[i])

def watershed(kd_tree):
    global S, watershed_list
    while len(S)>0:
        index=S.pop(0)
        add_neighbor(kd_tree,index)

def search_neighbor(remark,kd_tree):
    global watershed_list
    for i in range(len(remark)):
        radius=0.1
        index=-1
        while 1:
            [k, idx, _] = kd_tree.search_radius_vector_3d(pcd.points[remark[i]], radius)
            for j in range(k):
               if watershed_list[idx[j]]!=0:
                    index=idx[j]
                    break
            if index!=-1:break
            #radius+=0.1
            radius+=10
        dot = np.dot(normal[index], normal[remark[i]])


        if dot < 0:
            watershed_list[remark[i]] = -1
            normal[remark[i]] *= -1
            #print("inverse")
        else:
            watershed_list[remark[i]] = 1


if __name__=="__main__":

    n=10
    print("사용한 인접점의 개수는 %d 개"%(n))
    S=[]
    pcd = o3d.io.read_point_cloud("./rough.ply")
    print("총 점의 개수는 %d개"%(len(pcd.points)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=1000))
    o3d.visualization.draw_geometries([pcd],width=720,height=480,point_show_normal=True,window_name="input")
    start = time.time()
    normal = np.array(pcd.normals)
    watershed_list=np.zeros(len(pcd.points))

    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    init(kd_tree)
    watershed(kd_tree)
    remark = np.where(watershed_list == 0)[0]
    print("검사 안 된 점의 개수 %d개"%(len(remark)))
    search_neighbor(remark,kd_tree)

    remark = np.where(watershed_list == 0)[0]
    print("검사 안 된 점의 개수 %d개" % (len(remark)))

    pcd.normals = o3d.utility.Vector3dVector(normal)
    print("소요시간 %.2f sec"%(round(time.time()-start,2)))
    o3d.visualization.draw_geometries([pcd],width=720,height=480,point_show_normal=True,window_name="output")
