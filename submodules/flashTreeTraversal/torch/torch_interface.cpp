#include "torch_interface.h"
#include "../hierarchy_loader.h"
#include "../hierarchy_writer.h"
#include "../flash_tree_traversal.h"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LoadHierarchy(std::string filename){
	HierarchyLoader loader;
	
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	
	loader.load(filename.c_str(), pos, shs, alphas, scales, rot, nodes, boxes);
	
	int P = pos.size();
	
	torch::TensorOptions options 	= torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor pos_tensor 		= torch::from_blob(pos.data(), {P, 3}, options).clone();
	torch::Tensor shs_tensor 		= torch::from_blob(shs.data(), {P, 16, 3}, options).clone();
	torch::Tensor alpha_tensor 		= torch::from_blob(alphas.data(), {P, 1}, options).clone();
	torch::Tensor scale_tensor 		= torch::from_blob(scales.data(), {P, 3}, options).clone();
	torch::Tensor rot_tensor 		= torch::from_blob(rot.data(), {P, 4}, options).clone();
	
	int N = nodes.size();
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	
	torch::Tensor nodes_tensor = torch::from_blob(nodes.data(), {N, 7}, intoptions).clone();
	torch::Tensor box_tensor = torch::from_blob(boxes.data(), {N, 2, 4}, options).clone();
	
	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, scale_tensor, rot_tensor, nodes_tensor, box_tensor);
}
void WriteHierarchy(std::string filename, torch::Tensor& pos, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& log_scales, torch::Tensor& rotations, torch::Tensor& nodes, torch::Tensor& boxes){
	HierarchyWriter writer;
	
	int allP = pos.size(0);
	int allN = nodes.size(0);
	
	writer.write(
		filename.c_str(),
		allP,
		allN,
		(Eigen::Vector3f*)pos.cpu().contiguous().data_ptr<float>(),
		(SHs*)shs.cpu().contiguous().data_ptr<float>(),
		opacities.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector3f*)log_scales.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector4f*)rotations.cpu().contiguous().data_ptr<float>(),
		(Node*)nodes.cpu().contiguous().data_ptr<int>(),
		(Box*)boxes.cpu().contiguous().data_ptr<float>()
	);
}

// cloud functions 
int ReorderNodes(torch::Tensor &nodes, torch::Tensor &boxes, torch::Tensor& depth_count) {
    return reorder_nodes(
        nodes.size(0), 
		nodes.contiguous().data_ptr<int>(), 
		boxes.contiguous().data_ptr<float>(), 
        depth_count.contiguous().data_ptr<int>() 
    );
}
std::tuple<int, float> FlashTreeTraversal( 
	torch::Tensor& nodes, 
	torch::Tensor& boxes, 
	torch::Tensor& means3D,
	float target_size, 
	torch::Tensor& viewpoint, 
	torch::Tensor& view_transform,
	torch::Tensor& projection_matrix, 
	bool frustum_culling, 
	int window_size, 
	torch::Tensor& least_recently, 
	torch::Tensor& render_indices,
	torch::Tensor& node_indices,
	torch::Tensor& num_siblings,
	int mode
) { 
    return flashTreeTraversal(
        nodes.size(0), 
		nodes.contiguous().data_ptr<int>(),
		boxes.contiguous().data_ptr<float>(), 
        means3D.contiguous().data_ptr<float>(), 
		target_size, 
		viewpoint.contiguous().data_ptr<float>(), 
		view_transform.contiguous().data_ptr<float>(), 
		projection_matrix.contiguous().data_ptr<float>(), 
		frustum_culling, 
		window_size, 
		least_recently.contiguous().data_ptr<int>(),  
		render_indices.contiguous().data_ptr<int>(),
		node_indices.contiguous().data_ptr<int>(),
		num_siblings.contiguous().data_ptr<int>(),
		mode 
    );
} 

std::tuple<int, float>  TransimissionCompress( 
	// input parameters
	int N,
	torch::Tensor& render_indices, 
	torch::Tensor& node_indices, 
	torch::Tensor& num_siblings, 
	// initial data
	torch::Tensor& means3D, 
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	// output 
	torch::Tensor& data_to_pass,
	torch::Tensor& sizes,
	float opacity_min, float inv_range 
){
	return transimissionCompress(
		// input parameters 
		N, 
		render_indices.contiguous().data_ptr<int>(),
		node_indices.contiguous().data_ptr<int>(),
		num_siblings.contiguous().data_ptr<int>(),
		// initial data
		means3D.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		rotations.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		shs.contiguous().data_ptr<float>(),
		boxes.contiguous().data_ptr<float>(),
		// output 
		data_to_pass.contiguous().data_ptr<uint8_t>(),
		sizes.contiguous().data_ptr<uint8_t>(),
		opacity_min, inv_range 
	);
}

// client 
int SubGraphTreeInit(
	torch::Tensor& compressed_data,
	torch::Tensor& sizes,
	torch::Tensor& nodes,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& num_siblings, 
	float opacity_min, float range_255 
){
	return subGraphTreeInit(
		sizes.size(0),
		compressed_data.contiguous().data_ptr<uint8_t>(),
		sizes.contiguous().data_ptr<uint8_t>(),
		nodes.contiguous().data_ptr<int>(), 
		means3D.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		rotations.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		shs.contiguous().data_ptr<float>(),
		boxes.contiguous().data_ptr<float>(),
		num_siblings.contiguous().data_ptr<int>(),
		opacity_min, range_255 
	);
}
std::tuple<int, float> SubGraphTreeExpand(
	torch::Tensor& nodes,
	torch::Tensor& depth_count,
	torch::Tensor& means3D, 
	torch::Tensor& boxes, 
	float threshold, 
	torch::Tensor& viewpoint, 
	bool frustum_culling,
	torch::Tensor& view_transform,
	torch::Tensor& projection_matrix,
	torch::Tensor& least_recently,
	torch::Tensor& render_indices 
) { 
	return subGraphTreeExpand(  
		nodes.size(0), 
		nodes.contiguous().data_ptr<int>(), 
		depth_count.size(0), 
		depth_count.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		boxes.contiguous().data_ptr<float>(),
		threshold,
		viewpoint.contiguous().data_ptr<float>(),
		frustum_culling, 
		view_transform.contiguous().data_ptr<float>(),
		projection_matrix.contiguous().data_ptr<float>(), 
		least_recently.contiguous().data_ptr<int>(), 
		render_indices.contiguous().data_ptr<int>()
    );
}
std::tuple<int, float> SubGraphTreeUpdate( 
	torch::Tensor& compressed_data,
	torch::Tensor& sizes,
	torch::Tensor& nodes,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& num_siblings,
	torch::Tensor& least_recently, int window_size, 
	float opacity_min, float range_255, int featureMaxx 
){
	return subGraphTreeUpdate(
		sizes.size(0), 
		compressed_data.contiguous().data_ptr<uint8_t>(),
		sizes.contiguous().data_ptr<uint8_t>(),
		nodes.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		rotations.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		shs.contiguous().data_ptr<float>(),
		boxes.contiguous().data_ptr<float>(),
		num_siblings.contiguous().data_ptr<int>(),
		least_recently.contiguous().data_ptr<int>(), 
		window_size, opacity_min, range_255, featureMaxx 
	);
}
