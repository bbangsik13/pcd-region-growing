import numpy as np
import open3d as o3d
import time

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
    global a,b,c

    normal = np.array(pcd.normals)
    plane_normal = np.zeros_like(normal)
    plane_normal[:,0],plane_normal[:,1],plane_normal[:,2]=a,b,c
    mul=plane_normal*normal
    dot=np.sum(mul,axis=1)
    normal[dot>0]*=-1
    pcd.normals = o3d.utility.Vector3dVector(normal)
    return pcd


if __name__=="__main__":
    pcd = o3d.io.read_point_cloud("D:\window_old\KINECT\out/color_to_depth1.ply")

    plane_model, idx = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"index len: {len(idx)}")
    if np.dot(np.array(pcd.points)[idx[0]],np.array([a,b,c]))<0:
        a,b,c=-a,-b,-c
        print(f"normal vector:{[a,b,c]} 반전됨")
    else: print(f"normal vector:{[a,b,c]}")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=1000))
    o3d.visualization.draw_geometries([pcd],width=720,height=480,point_show_normal=True,window_name="input")
    start=time.time()
    pcd=polarization(pcd)
    print(time.time()-start)
    o3d.visualization.draw_geometries([pcd], width=720, height=480, point_show_normal=True, window_name="output")
    o3d.io.write_point_cloud("./rough.ply",pcd,write_ascii=True)

