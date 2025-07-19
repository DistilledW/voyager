#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math, os, json
import torch 
from utils.loss_utils import ssim
from gaussian_renderer import render_post
import sys 
from scene import Scene, GaussianModel 
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser 
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips
from gaussian_hierarchy._C import ReorderNodes, expand_to_size, get_interpolation_weights
from scipy.spatial.transform import Rotation as R, Slerp 
import ffmpeg 
import numpy as np 

class camera_pose:
    def __init__(self, world_view_transform, projection_matrix):    
        self.world_view_transform   = world_view_transform.cuda() 
        self.projection_matrix      = projection_matrix.cuda() 
        self.full_proj_transform    = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).cuda()  
        self.camera_center          = world_view_transform.inverse()[3, :3].cuda() 
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4)) 
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
def getProjectionMatrix(znear, zfar, fovX, fovY, primx, primy):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    #primy = 0.5
    #primx = 0.5
    top = tanHalfFovY * znear
    bottom = (1 - primy) * 2 * -top
    top = primy * 2 * top
    right = tanHalfFovX * znear
    left = (1-primx) * 2 * -right
    right = primx * 2 * right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
def rotate_camera_pitch(R, angle_deg):
    angle_rad = np.radians(angle_deg)
    # 绕 X 轴旋转矩阵（相机坐标系中的局部 right 轴）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad),  np.cos(angle_rad)],
    ]) 
    # 注意：这是在相机局部坐标系中旋转，所以是右乘
    return R @ Rx 
def rotate_camera_yaw(R, angle_deg):
    """
    左右旋转视角（偏航）——绕相机局部 Y 轴（下轴）旋转。
    输入:
        R: 当前相机的旋转矩阵 (3x3)
        angle_deg: 要旋转的角度（正数 = 向右转）
    返回:
        新的旋转矩阵 R'
    """
    angle_rad = np.radians(angle_deg)
    # 绕 Y 轴旋转矩阵（相机坐标系中的局部 down 轴）
    Ry = np.array([
        [np.cos(angle_rad),  0, np.sin(angle_rad)],
        [0,                  1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)],
    ])
    # 注意：在相机局部坐标系中旋转，右乘
    return R @ Ry
def rotate_camera_roll(R, angle_deg):
    """
    绕相机自身 Z 轴（正前方）旋转，即图像顺/逆时针旋转。
    输入:
        R: 当前相机的旋转矩阵 (3x3)
        angle_deg: 旋转角度，正数 = 顺时针看，图像逆时针转
    返回:
        新的旋转矩阵 R'
    """
    angle_rad = np.radians(angle_deg)
    # 绕 Z 轴的旋转矩阵（局部坐标系：forward 轴）
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                  1],
    ])
    # 局部旋转，右乘
    return R @ Rz

def from_cameras_to_paths(scene):
    # R=rotate_camera_pitch(R, -30) 
    # # # R=R@np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) 
    # T=np.array([12, 5, 5]) # [5, -5, 30]
    # tanfovx = math.tan(viewpoint.FoVx * 0.5) 
    # threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)
    cameras = scene.getTestCameras() 
    camera_data = {} 
    pre_viewpoint = None 
    view_cameras = {} 
    for viewpoint in tqdm(cameras):
        if pre_viewpoint == None:
            pre_viewpoint = viewpoint 
        camera_data[viewpoint.image_name] = { 
            "R": viewpoint.R.tolist(), 
            "T": viewpoint.T.tolist() 
        } 
    # R=np.array([[-0.747316309407019,-0.4512447849695189,0.4877463251856286],
    #             [-0.1337947302856015,-0.616815677487202,-0.7756528799365132],
    #             [0.6508588970292067,-0.6449159356410852,0.4005822413863451]])
    # T=np.array([-10, -12, 3]) 
    # R=rotate_camera_roll(R, -90)  
    # R=rotate_camera_yaw(R, -30)
    # R=rotate_camera_pitch(R, 45) 
    # Rs.append(R) 
    # Ts.append(T) 
    # for idx, (R_cur, T_cur) in enumerate(zip(Rs, Ts)):
    #     world_view_transform = torch.tensor(
    #         getWorld2View2(R_cur, T_cur, trans, scale) 
    #     ).transpose(0, 1).cuda() 
    #     projection_matrix = getProjectionMatrix(
    #         znear=znear, 
    #         zfar=zfar, 
    #         fovX=settings["FoVx"], 
    #         fovY=settings["FoVy"], 
    #         primx=settings["primx"], 
    #         primy=settings["primy"] 
    #     ).transpose(0,1).cuda() 
    #     cameras[idx] = camera_pose(world_view_transform, projection_matrix) 
        # view_cameras[f"{idx}.png"] = {
        #     "R": R_cur.tolist(), 
        #     "T": T_cur.tolist() 
        # } 
    # with open(f"/workspace/code/dataset/camera_paths/camera_poses.json", "w")as fout:
    #     json.dump(view_cameras, fout, indent=4) 
def read_generate(args): # n -> ((n-1)*n_frames + 1) # interpolation 
    with open(os.path.join(args.cameras_dir, "setting.json"), "r") as f:
        settings = json.load(f) 
    # settings["image_width"] = 1368 
    # settings["image_height"] = 912 
    
    with open(os.path.join(args.cameras_dir, "camera_poses.json"), "r") as f:
        camera_poses = json.load(f) 
    camera_poses = dict(sorted(camera_poses.items())) 
    R_pre = None 
    T_pre = None 
    Rs = [] 
    Ts = [] 
    view_index = 0 
    for idx, (image_name, pose) in enumerate(camera_poses.items()):
        R_cur = np.array(pose["R"]) 
        T_cur = np.array(pose["T"]) 
        if R_pre is None:
            R_pre = R_cur 
            T_pre = T_cur 
            Rs.append(R_cur) 
            Ts.append(T_cur) 
            view_index+=1 
            continue 
        key_times = [0, 1] 
        key_rots = R.from_matrix([R_pre, R_cur])
        slerp = Slerp(key_times, key_rots) 
        for i in range(1, args.n_frames): 
            t = i / args.n_frames 
            R_mid = slerp([t])[0].as_matrix() 
            T_mid = (1 - t) * T_pre + t * T_cur 
            Rs.append(R_mid) 
            Ts.append(T_mid) 
            view_index+=1
        Rs.append(R_cur) 
        Ts.append(T_cur) 
        view_index+=1
        R_pre = R_cur 
        T_pre = T_cur 
    zfar = 100.0 
    znear = 0.01 
    trans = np.array([0.0, 0.0, 0.0]) 
    scale = 1.0 
    cameras = {} 
    for idx, (R_cur, T_cur) in enumerate(zip(Rs, Ts)):
        world_view_transform = torch.tensor(
            getWorld2View2(R_cur, T_cur, trans, scale) 
        ).transpose(0, 1).cuda() 
        projection_matrix = getProjectionMatrix(
            znear=znear, 
            zfar=zfar, 
            fovX=settings["FoVx"], 
            fovY=settings["FoVy"], 
            primx=settings["primx"], 
            primy=settings["primy"] 
        ).transpose(0,1).cuda() 
        cameras[idx] = camera_pose(world_view_transform, projection_matrix) 
    return cameras, settings 

@torch.no_grad() 
def render_set_accuracy(args, scene, pipe, out_dir, tau, eval):
    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    cameras = scene.getTestCameras() if eval else scene.getTrainCameras() 
    ReorderNodes( 
        scene.gaussians.nodes, 
        scene.gaussians.boxes, 
        False 
    ) 
    psnr_test = 0.0 
    ssims = 0.0 
    lpipss = 0.0 
    with open(args.log_file, "w") as fout:
        pass 
    K = 5 
    cameras_for_iteration = []
    for viewpoint in tqdm(cameras):
        cameras_for_iteration.append(viewpoint)
    for idx in range(len(cameras_for_iteration)//8):
        current = []
        for k in range(K):
            current.append(cameras_for_iteration[idx * 8 + k])
        
        viewpoint=cameras_for_iteration[idx * 8] 
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()
        tanfovx = math.tan(viewpoint.FoVx * 0.5) 
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

        to_render, _ = expand_to_size(
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            scene.gaussians.get_xyz.cuda(),
            threshold,
            viewpoint.camera_center,
            viewpoint.world_view_transform, 
            viewpoint.projection_matrix, 
            torch.zeros((3)),
            render_indices,
            0 
        ) 
        
        indices = render_indices[:to_render].int().contiguous() 
        for k in range(K):
            viewpoint = current[k] 
            viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
            viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
            viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
            viewpoint.camera_center = viewpoint.camera_center.cuda()
            render_ret = render_post( 
                viewpoint, 
                scene.gaussians, 
                pipe, 
                torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
                render_indices=indices,
                interp_python=False,
                use_trained_exp=args.train_test_exp 
            ) 
            image = torch.clamp(render_ret["render"], 0.0, 1.0) 
            
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0) 
            alpha_mask = viewpoint.alpha_mask.cuda() 
            try:
                torchvision.utils.save_image(image, os.path.join(out_dir, viewpoint.image_name.split(".")[0] + ".png"))
            except:
                os.makedirs(os.path.dirname(os.path.join(out_dir, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
                torchvision.utils.save_image(image, os.path.join(out_dir, viewpoint.image_name.split(".")[0] + ".png"))
            image *= alpha_mask 
            gt_image *= alpha_mask 
            psn = psnr(image, gt_image).mean().double()
            ssi = ssim(image, gt_image).mean().double()
            lpi = lpips(image, gt_image, net_type='vgg').mean().double()
            psnr_test   += psn 
            ssims       += ssi 
            lpipss      += lpi 
            with open(args.log_file, "a+")as fout:
                fout.write(f"image_name = {viewpoint.image_name}, {psn:.5f}, {ssi:.5f}, {lpi:.5f}\n")
            torch.cuda.empty_cache() 
    
    psnr_test /= len(scene.getTestCameras())
    ssims /= len(scene.getTestCameras())
    lpipss /= len(scene.getTestCameras())
    print(f"tau: {tau}, PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
    with open(args.log_file, "a+")as fout:
        fout.write(f"Average = {psnr_test:.5f}, {ssims:.5f}, {lpipss:.5f}\n")
@torch.no_grad() 
def render_set_performance(args, scene, pipe, tau, eval):
    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()
    ReorderNodes( 
        scene.gaussians.nodes, 
        scene.gaussians.boxes, 
        False 
    ) 
    elapse = 0 
    with open(args.log_file, "w") as fout:
        pass 
    iterations = 2 
    double_iterations = 2 * iterations 
    for viewpoint in tqdm(cameras):
        viewpoint=viewpoint 
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()

        tanfovx = math.tan(viewpoint.FoVx * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)
        e_elapse = 0 
        for i in range(double_iterations):
            to_render, expand_elapse = expand_to_size( 
                scene.gaussians.nodes,
                scene.gaussians.boxes,
                scene.gaussians.get_xyz.cuda(),
                threshold,
                viewpoint.camera_center, 
                viewpoint.world_view_transform, 
                viewpoint.projection_matrix, 
                torch.zeros((3)),
                render_indices,
                0 
            ) 
            if i >= iterations:
                e_elapse += expand_elapse 
        expand_elapse = e_elapse / iterations 
        indices = render_indices[:to_render].int().contiguous() 

        r_t = 0 
        b_t = [] 
        for i in range(double_iterations): 
            render_ret = render_post( 
                viewpoint, 
                scene.gaussians, 
                pipe, 
                torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
                render_indices = indices, 
                interp_python = False, 
                use_trained_exp=args.train_test_exp 
            ) 
            if i == iterations:
                r_t += render_ret["elapse"] 
                for f in render_ret["elapse_breakdown"]:
                    b_t.append(f) 
            elif i > iterations:
                r_t += render_ret["elapse"] 
                for i in range(len(render_ret["elapse_breakdown"])):
                    b_t[i] += render_ret["elapse_breakdown"][i]
        render_elapse = r_t / iterations 
        for i in range(len(b_t)):
            b_t[i] /= iterations 
        _elapse = expand_elapse + render_elapse 
        elapse += _elapse 
        with open(args.log_file, "a+")as fout:
            fout.write(f"Image_name = {viewpoint.image_name}: \n{expand_elapse:.5f}\n{0.0:.5f}\n{render_elapse:.5f}\n---\n")
            for elapse_brk in b_t: 
                fout.write(f"{elapse_brk:.5f}\n") 
            fout.write(f"{_elapse:.5f}\n{to_render}\n") 
        torch.cuda.empty_cache() 
    elapse /= len(scene.getTestCameras()) 
    print(f"tau: {tau} Elapse: {elapse:.5f}") 
    with open(args.log_file, "a+")as fout:
        fout.write(f"Tau = {tau}, Elapse = {elapse:.5f}\n")
@torch.no_grad() 
def render_camera_paths(args, scene, pipe, tau, eval):
    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    camera_poses, settings = read_generate(args) 
    ReorderNodes( 
        scene.gaussians.nodes, 
        scene.gaussians.boxes, 
        False 
    ) 
    # iterations = 1 
    # double_iterations = 2 * iterations 
    # result={"elapse" : 0.0, "expand": 0.0, "interpolation": 0.0, "render":0.0, "Render_breakdown": [], "to_render_point" : 0} 
    for idx, viewpoint in tqdm(camera_poses.items()): 
        viewpoint=viewpoint 
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()
        tanfovx = math.tan(settings["FoVx"] * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * settings["image_width"])
        # e_elapse = 0 
        # for i in range(double_iterations): 
        to_render, expand_elapse = expand_to_size( 
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            scene.gaussians.get_xyz.cuda(),
            threshold,
            viewpoint.camera_center, 
            viewpoint.world_view_transform, 
            viewpoint.projection_matrix, 
            torch.zeros((3)),
            render_indices, 
            0 
        ) 
        #     if i >= iterations: 
        #         e_elapse += expand_elapse 
        # expand_elapse = e_elapse / iterations 
        indices = render_indices[:to_render].int().contiguous() 
        # r_t = 0 
        # b_t = [] 
        # for i in range(double_iterations): 
        render_ret = render_post( 
            None,
            scene.gaussians, 
            pipe, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            render_indices = indices, 
            interp_python = False, 
            use_trained_exp=args.train_test_exp,
            viewpoint=viewpoint,
            settings=settings 
        ) 
        image = torch.clamp(render_ret["render"], 0.0, 1.0) 
        # file_path = os.path.join(args.out_dir, f"{idx:04d}.png") 
        # try:
        #     torchvision.utils.save_image(image, file_path)
        # except:
        #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #     torchvision.utils.save_image(image, file_path) 
        #     if i == iterations:
        #         r_t += render_ret["elapse"] 
        #         for f in render_ret["elapse_breakdown"]:
        #             b_t.append(f) 
        #     elif i > iterations:
        #         r_t += render_ret["elapse"] 
        #         for i in range(len(render_ret["elapse_breakdown"])):
        #             b_t[i] += render_ret["elapse_breakdown"][i]
        # render_elapse = r_t / iterations 
        # for i in range(len(b_t)):
        #     b_t[i] /= iterations 
        # elapse = expand_elapse + render_elapse 
        # with open(args.log_file, "a+")as fout:
        #     fout.write(f"frame_index = {idx}:  \n{expand_elapse:.5f}\n{0.0:.5f}\n{render_elapse:.5f}\n---\n")
        #     for elapse_brk in b_t: 
        #         fout.write(f"{elapse_brk:.5f}\n") 
        #     fout.write(f"{elapse:.5f}\n{to_render}\n") 
        # result["elapse"] += elapse 
        # result["expand"] += expand_elapse 
        # result["interpolation"] += 0.0  
        # result["render"] += render_elapse 
    #     if idx == 0:
    #         result["Render_breakdown"] = render_ret["elapse_breakdown"] 
    #     else:
    #         for i in range(len(result["Render_breakdown"])):
    #             result["Render_breakdown"][i] += render_ret["elapse_breakdown"][i] 
    #     result["to_render_point"] += to_render
    #     # torch.cuda.empty_cache() 
    # result["elapse"] /= len(camera_poses) 
    # result["expand"] /= len(camera_poses) 
    # result["interpolation"] /= len(camera_poses) 
    # result["render"] /= len(camera_poses) 
    # result["to_render_point"] /= len(camera_poses) 
    # for i in range(len(result["Render_breakdown"])): 
    #     result["Render_breakdown"][i] /= len(camera_poses) 
    # with open(args.res_file, "w")as fout:
    #     fout.write(f"frame_number = {len(camera_poses)}\n")
    #     fout.write(f"Elapse = {result["elapse"]:.5f}\n")
    #     fout.write(f"Breakdown:: expand = {result["expand"]:.5f}, interpolation = {result["interpolation"]:.5f}, render = {result["render"]:.5f}\n") 
    #     fout.write(f"\tRender Breakdown: ")
    #     for i in range(len(result["Render_breakdown"])): 
    #         fout.write(f"{result["Render_breakdown"][i]:.5f}, ")
    #     fout.write(f"\nto_render_point: {result["to_render_point"]:.1f}\n")
    # print(f"frame_number = {len(camera_poses)}")
    # print(f"Elapse = {result["elapse"]:.5f}")
    # print(f"Breakdown:: expand = {result["expand"]:.5f}, interpolation = {result["interpolation"]:.5f}, render = {result["render"]:.5f}") 
    # print(f"Render Breakdown: ") 
    # for i in range(len(result["Render_breakdown"])): 
    #     print(f"\t{result["Render_breakdown"][i]:.5f}")
    # print(f"to_render_point: {result["to_render_point"]:.1f}")

if __name__ == "__main__":
    # Set up command line argument parser 
    parser = ArgumentParser(description="Rendering script parameters") 
    lp = ModelParams(parser) 
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser) 
    parser.add_argument('--out_dir', type=str, default="") 
    parser.add_argument("--taus", nargs="+", type=float, default=[]) 
    parser.add_argument("--log_file", type=str, default="") 
    parser.add_argument('--res_file', type=str, default="") 
    parser.add_argument("--per", action="store_true") 
    parser.add_argument('--cameras_dir', type=str, default="") 
    parser.add_argument('--path', action="store_true") 
    parser.add_argument('--test', type=str, default="test.txt") 
    parser.add_argument("--n_frames", type=int, default=1)
    args = parser.parse_args(sys.argv[1:]) 
    print("Rendering " + args.model_path) 
    dataset, pipe = lp.extract(args), pp.extract(args) 
    gaussians = GaussianModel(dataset.sh_degree) 
    gaussians.active_sh_degree = dataset.sh_degree 
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True, test=args.test) 
    for tau in args.taus:
        if args.path:
            render_camera_paths(args, scene, pipe, tau, args.eval) 
        elif args.per:
            render_set_performance(args, scene, pipe, tau, args.eval) 
        else:
            render_set_accuracy(args, scene, pipe, args.out_dir, tau, args.eval) 