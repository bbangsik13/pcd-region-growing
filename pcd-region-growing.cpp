#include <string>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <Eigen/Dense>
#include <chrono>
#include <open3d/Open3D.h>
using namespace open3d;

class candidate {
public:
	int index;
	int marker;
	double distance;

	candidate(int index, int marker, int distance){
		this->index = index;
		this->marker = marker;
		this->distance = distance;
	}		
};

bool operator<(candidate t, candidate u){
    return (t.distance > u.distance);
}


void add_neighbor(	geometry::PointCloud& pcd, std::priority_queue<candidate> &pq, int* z, const int& nn, Eigen::Vector3d* normal, const bool& ws_color,
					geometry::KDTreeFlann &kdtree, const int& index, const int& marker, unsigned char* gray, int* watershed_list){
						std::vector<int> idx;
                        std::vector<double> dist;
						double dot;
						double distance;
                        int k = kdtree.SearchKNN(pcd.points_[index], nn, idx, dist);
						for (int i=0; i<k; i++){
							dot = std::abs(normal[index].dot(normal[idx[i]]));
							distance = 10*std::abs(z[index] - z[idx[i]]) + std::abs(gray[index] - gray[idx[i]]);
							if(!watershed_list[index]){
								pq.push(candidate(idx[i], marker, distance));
							}
						}
					}

void init(	geometry::PointCloud& pcd, std::priority_queue<candidate> &pq, int* z, const int& nn, Eigen::Vector3d* normal, const bool& ws_color, size_t &remain, 
			geometry::KDTreeFlann &kdtree, int* watershed_list, unsigned char* gray){
				add_neighbor(pcd, pq, z, nn, normal, ws_color, kdtree, 229387, 1, gray, watershed_list);
				watershed_list[229387]=1;
				remain--;
				add_neighbor(pcd, pq, z, nn, normal, ws_color, kdtree, 214008, 2, gray, watershed_list);
				watershed_list[214008]=2;
				remain--;
			}
void watershed (	geometry::PointCloud& pcd, std::priority_queue<candidate> &pq, int* z, const int& nn, Eigen::Vector3d* normal, const bool& ws_color, size_t &remain, 
					geometry::KDTreeFlann &kdtree, int* watershed_list, unsigned char* gray){

						int marker;
						int index;
						while (remain > 0){
							if (pq.empty()){
								std::cout<<"queue is empty"<<std::endl;
								break;
							}
							marker = pq.top().marker;
							index = pq.top().index;
							pq.pop();
							
							if(!watershed_list[index]){
								watershed_list[index]=marker;
								remain--;
								
								add_neighbor(pcd, pq, z, nn, normal, ws_color, kdtree, index, marker, gray, watershed_list);
							}
						}
			}
int main(int argc, char *argv[]) {
	std::priority_queue<candidate> pq;
	std::string path;
	geometry::PointCloud pcd;

	bool ws_color=false;
	int nn = 8;


	if (argc >= 2) path = argv[1];
	else { std::cout<<"E: please enter the path"<<std::endl; return 0;}
	if (argc >= 3) nn = std::stoi(argv[2]);
        if (argc >= 4) ws_color = std::stoi(argv[3]);	
	if ( !(io::ReadPointCloud(path, pcd)) ){
		std::cout<<"E: can't read pointcloud!"<<std::endl;	return 0;
        }
	
	visualization::Visualizer visualizer;
	std::shared_ptr<geometry::PointCloud> pcd_ptr(new geometry::PointCloud);
	std::shared_ptr<geometry::PointCloud> foot_pcd_ptr(new geometry::PointCloud);
	std::shared_ptr<geometry::PointCloud> floor_pcd_ptr(new geometry::PointCloud);
    *pcd_ptr = pcd;
    *foot_pcd_ptr = *pcd_ptr; //deep copy
    *floor_pcd_ptr = *pcd_ptr; //deep copy

	visualization::DrawGeometries({pcd_ptr}, "origin");

	const size_t pcd_point_size = pcd.points_.size();
	int z[pcd_point_size];
	unsigned char gray[pcd_point_size];
	int watershed_list[pcd_point_size];
	size_t remain = pcd_point_size;
	Eigen::Vector3d normal[pcd_point_size];
	// Eigen::Vector3d color[pcd_point_size];
	Eigen::Vector3d black(0,0,0);
	Eigen::Vector3d red (255,0,0);
	Eigen::Vector3d blue (0,0,255);
	geometry::KDTreeFlann kdtree;


    if (pcd.HasNormals()) {
		for (size_t i = 0; i < pcd_point_size; i++) {
			z[i] = ((int)(pcd.points_[i](2)));
			gray[i] = ((unsigned char)(85*(pcd.colors_[i](0)+pcd.colors_[i](1)+pcd.colors_[i](2))));// 255/3 *(r+g+b)
			normal[i] = (pcd.normals_[i]);
			watershed_list[i] = 0;
		}
	}else {std::cout<<"E: ply file has no normal vector"<<std::endl; return 0;}
	
	
    kdtree.SetGeometry(pcd);
	init(pcd, pq, z, nn, normal, ws_color, remain, kdtree, watershed_list, gray);

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	watershed(pcd, pq, z, nn, normal, ws_color, remain, kdtree, watershed_list, gray);
	std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;

	std::cout<<"watershed time: "<< sec.count() << " seconds" << std::endl;
	std::cout<<"# of points not marked: "<< remain <<std::endl;
	

	for (size_t i = 0; i < pcd_point_size; i++) {
		if (watershed_list[i] == 1) pcd.colors_[i]=red;
		else if (watershed_list[i] == 2) pcd.colors_[i]=blue;
		else pcd.colors_[i]=black;
	}
	*pcd_ptr = pcd;
	visualization::DrawGeometries({pcd_ptr}, "watershed_seg");

	
	return 0;
}
