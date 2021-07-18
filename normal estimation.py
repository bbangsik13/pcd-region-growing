import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from sortedcontainers import SortedList
import time


class candidate:
    def __init__(self, index, marker, distance):
        self.index = index
        self.marker = marker
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


def init(kd_tree, watershed_list, gray):
    global S, remain
    add_neighbor(kd_tree, 243, 1, gray)
    watershed_list[243] = 1  # red
    remain = remain - 1
    add_neighbor(kd_tree, 1425, 2, gray)
    watershed_list[1425] = 2  # blue 발
    remain = remain - 1
    return watershed_list


def add_neighbor(kd_tree, index, marker, gray):
    global S, z, n, normal, ws_color

    [k, idx, _] = kd_tree.search_knn_vector_3d(pcd.points[index], n)
    for i in range(k):
        dot = abs(np.dot(normal[index], normal[idx[i]]))
        # distance=abs(gray[index]-gray[idx[i]])+10*abs(z[index]-z[idx[i]])
        if ws_color == True:
            distance = 10000000 * (1 - dot) * (1 - dot) + abs(gray[index] - gray[idx[i]])
        else:
            distance = 10000000 * (1 - dot) * (1 - dot)

        S.add(candidate(idx[i], marker, distance))


def watershed(gray, kd_tree, watershed_list):
    global S, remain
    while remain > 0:
        if len(S) < 1:
            print('list is empty')
            break
        c = S.pop(0)
        index = c.index
        marker = c.marker
        if watershed_list[index] == 0:
            watershed_list[index] = marker
            remain = remain - 1
            add_neighbor(kd_tree, index, marker, gray)
    return watershed_list


def estimate_normals(pcd, radius, max_nn, thres=0, debug=False):

    if not pcd.has_normals():
        assert radius and max_nn
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    pcd_points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_normals = np.asarray(pcd.normals)

    # pcd 표면의 점을 사용하여 convex hull을 만듦
    hull, pcd_point_idx = pcd.compute_convex_hull()
    hull.compute_vertex_normals(normalized=True)

    # convex hull 정점의 법선 벡터를 중심 좌표를 사용하여 정렬
    hull_vertex_normals = np.asarray(hull.vertex_normals)
    inner = np.sum(hull_vertex_normals * (np.asarray(hull.vertices) - hull.get_center()), axis=1)
    hull_vertex_normals[inner < 0] *= -1

    if debug:
        kpcd = o3d.geometry.PointCloud(pcd.points)
        kpcd.colors = pcd.colors
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        tpcd = o3d.geometry.PointCloud(hull.vertices)
        tpcd.normals = hull.vertex_normals
        o3d.visualization.draw_geometries([kpcd, hull_ls, tpcd])
        hull.vertex_normals = o3d.utility.Vector3dVector(np.asarray(hull.vertices) - hull.get_center())
        hull.normalize_normals()
        tpcd.normals = hull.vertex_normals
        o3d.visualization.draw_geometries([kpcd, tpcd, hull])

    # convex hull을 이루는 정점 중 pcd의 모서리 점을 배제하기 위해 이웃 수가 가장 많은 점을 선택
    num_neighbor_list = [pcd_tree.search_radius_vector_3d(pcd_points[idx], radius)[0] for idx in pcd_point_idx]
    source_point_idx = pcd_point_idx[np.argmax(num_neighbor_list)]

    source_point_normal = pcd_normals[source_point_idx]
    inner = np.sum(source_point_normal * hull_vertex_normals[np.argmax(num_neighbor_list)])
    pcd_normals[source_point_idx] = source_point_normal if inner > 0 else -source_point_normal

    # 너비 우선 탐색
    UNKNOWN, SEARCHING, SEARCHED = 0, 1, 2
    info = np.zeros(len(pcd_normals), dtype=np.uint8)  # UNKNOWN
    info[source_point_idx] = SEARCHING

    while (info == SEARCHING).any():  # not (info == SEARCHED).all()
        isSearching = info == SEARCHING
        source_points = pcd_points[isSearching]
        source_point_normals = pcd_normals[isSearching]

        for point, normal in zip(source_points, source_point_normals):
            # k, idx, _ = pcd_tree.search_knn_vector_3d(query=point, knn=max_nn)
            k, idx, _ = pcd_tree.search_radius_vector_3d(query=point, radius=radius)
            idx = np.asarray(idx)
            idx = idx[info[idx] == UNKNOWN]
            neighbor_point_normals = pcd_normals[idx]
            inner = np.sum(neighbor_point_normals * normal.reshape(1, -1), axis=1)
            neighbor_point_normals[inner < -thres] *= -1
            pcd_normals[idx] = neighbor_point_normals

            info[idx[np.abs(inner) > thres]] = SEARCHING

        info[isSearching] = SEARCHED

        #print(np.sum(info == SEARCHED), "/", len(info))


if __name__ == "__main__":
    for i in range(2,6):
        foot = o3d.io.read_point_cloud("D:\download/7-14/segment%d.ply"%(i))
        o3d.visualization.draw_geometries([foot], width=720, height=480, window_name="input",point_show_normal=True)
        start = time.time()
        estimate_normals(foot, 5, 30, 0.707)  # 0.01
        print("time:", time.time() - start, "sec")
        o3d.visualization.draw_geometries([foot], width=720, height=480, window_name="output",point_show_normal=True)