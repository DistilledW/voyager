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

import math
import os
import torch
from utils.loss_utils import ssim
import sys 
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips

from gaussian_hierarchy._C import force_search
# server-client 
from multiprocessing.reduction import send_handle, recv_handle 
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer 
import torch.multiprocessing as mp 
from multiprocessing import Event, queues 
import numpy as np 
import socket
import time
import json
import pickle 
import ast
import zlib  
from PIL import Image
import struct

class client_Camera:
    def __init__(self, FoVx:float, FoVy:float, image_height:int, image_width:int, world_view_transform:torch.Tensor, 
                 projection_matrix:torch.Tensor, full_proj_transform:torch.Tensor, image_name:str, camera_center:torch.Tensor, timestamp):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = image_height
        self.image_width = image_width
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = full_proj_transform
        self.image_name = image_name
        self.camera_center = camera_center
        self.timestamp = timestamp
    def serialize(self):
        return json.dumps({
            "FoVx": self.FoVx,
            "FoVy": self.FoVy,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "world_view_transform": self.world_view_transform.tolist(),
            "projection_matrix": self.projection_matrix.tolist(),
            "full_proj_transform": self.full_proj_transform.tolist(),
            "camera_center": self.camera_center.tolist(),
            "image_name": self.image_name,
            "timestamp": self.timestamp
        })

class client:
    def __init__(self, manager, tau=6.0):    
        self.shared_data = manager.dict({
            "tau": tau
        }) 

    def send(self, end_signal, send_start_signal, camera_queue, child_con, viewpointFilePath, buffer_size=10240): 
        fd = recv_handle(child_con) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        objects = [] 
        
        with open(viewpointFilePath, 'r') as file: 
            lines = [line.strip() for line in file]  # 读取 
            i = 0 
            while i < len(lines):
                FoVx = float(lines[i]) 
                FoVy = float(lines[i + 1]) 
                image_height = int(lines[i + 2]) 
                image_width = int(lines[i + 3]) 
                world_view_transform = torch.tensor(ast.literal_eval(lines[i + 4])) 
                projection_matrix = torch.tensor(ast.literal_eval(lines[i + 5])) 
                full_proj_transform = torch.tensor(ast.literal_eval(lines[i + 6])) 
                camera_center = torch.tensor(ast.literal_eval(lines[i + 7])) 
                image_name = lines[i + 8] 
                timestamp = None  # 这里暂时不处理时间戳 

                # 创建对象并存入列表 
                camera_obj = client_Camera(FoVx, FoVy, image_height, image_width, world_view_transform, 
                                        projection_matrix, full_proj_transform, image_name, camera_center, timestamp)
                objects.append(camera_obj)
                i += 9  # 每组数据有 9 行 
        # 按 image_name 排序
        objects.sort(key=lambda x: x.image_name) 
        # print("viewpoints is read successully, need to wait until until rendering process start.") 
        send_start_signal.wait()
        print("===========================================================")
        print(f"Sending process start.")
        code = 1 # 发送tau 值
        connection.sendall(code.to_bytes(4, byteorder='big')) 
        connection.sendall(struct.pack("f", self.shared_data["tau"])) 
        code = 0 
        for idx, obj in enumerate(objects): 
            if end_signal.is_set():
                break 
            connection.sendall(code.to_bytes(4, byteorder='big')) 
            message = obj.serialize().encode('utf-8') 
            data_size = len(message) 
            connection.sendall(data_size.to_bytes(4, byteorder='big')) 
            for i in range(0, data_size, buffer_size): 
                try: 
                    chunk = message[i : i + buffer_size] 
                    if not end_signal.is_set(): 
                        connection.sendall(chunk) 
                    else:
                        break 
                except Exception as e:
                    print("Send exception: ", e) 
                    end_signal.set() 
                    break 
            camera_queue.put(obj) 
            print(f"Send viewpoint[{idx}] sucessfully") 
            time.sleep(1) 
        end_signal.wait()
    def receive(self, end_signal, queue, child_conn, buffer_size=10240): 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        frame_index = 1 # 第一帧数据跳过 
        parameter_number = 14 
        print("===========================================================")
        print("Receive process start.")
        while not end_signal.is_set(): 
            try: 
                frame_index += 1 
                # i = 0 
                # 性能测试 to fix 
                start_time = time.time() 
                for idx in range(parameter_number):
                    str = connection.recv(4) 
                    message_length = int.from_bytes(str, byteorder='big') 
                    if str == b'': 
                        end_signal.set() 
                        break 
                    message = b"" 
                    while len(message) < message_length and not end_signal.is_set():
                        chunk = connection.recv(min(buffer_size, message_length - len(message)))  # 每次最多接收1024字节
                        if not chunk:
                            end_signal.set() 
                            break 
                        message += chunk 
                    if end_signal.is_set():
                        break 
                    decompressed = zlib.decompress(message) 
                    tensor = pickle.loads(decompressed) 
                    queue.put(tensor) 
                # 性能测试 to fix 
                end_time = time.time() 
                execution_time = end_time - start_time 
                # print(f"Read time = {execution_time}s, frame index = {frame_index}") 
                time.sleep(1)
            except Exception as e: 
                print("Receive exception: ", e) 
                end_signal.set() 
                break 
        end_signal.wait()
    def check_size(self, s1:int, s2:int, s3:int, s4:int, s5:int, s6:int):
        if (s1 != s2 or s3 != s4 or s5!=s6):
            return False
        elif (s1 != s3 or s1!=s5):
            return False
        return True
    @torch.no_grad()
    def render(self, end_signal, send_start_signal, queue, camera_queue, dataset, pipe, args):
        # create output folder if it doesn't exit  
        if not os.path.exists(args.render_dir):
            os.makedirs(args.render_dir)
            print(f"Create folder {args.render_dir}.")
        # send_start_signal.set() 
        # return 
        torch.cuda.init() 
        torch.cuda.set_device(torch.cuda.current_device())
        # 第一帧数据作为起始数据 
        # child 
        tensors = [] 
        tensor_directory = f"../dataset/first_frame/without/{self.shared_data['tau']}" 
        if args.frustum_culling:
            tensor_directory = f"../dataset/first_frame/with/{self.shared_data['tau']}" 
        for i in range(14): 
            tensors.append(torch.load(os.path.join(tensor_directory, f"{i}.pt"), weights_only=True))        
        child_means3D = tensors[0].cuda()
        child_rotations = tensors[1].cuda()
        child_opacity = tensors[2].cuda()
        child_scales = tensors[3].cuda()
        child_shs = tensors[4].cuda()
        child_boxes = tensors[5].cuda()
        
        # parent 
        parent_means3D = tensors[6].cuda()
        parent_rotations = tensors[7].cuda()
        parent_opacity = tensors[8].cuda()
        parent_scales = tensors[9].cuda()
        parent_shs = tensors[10].cuda()
        parent_boxes = tensors[11].cuda() 
        leafs_tag = tensors[12].cuda() 
        num_siblings = tensors[13].cuda()
        # skybox:
        sky_box_means3d = torch.load(os.path.join("../dataset/skybox/means3d.pt"), weights_only=True).cuda()
        sky_box_opacity = torch.load(os.path.join("../dataset/skybox/opacity.pt"), weights_only=True).cuda()
        sky_box_shs = torch.load(os.path.join("../dataset/skybox/shs.pt"), weights_only=True).cuda()
        sky_box_scales = torch.load(os.path.join("../dataset/skybox/scales.pt"), weights_only=True).cuda()
        sky_box_rotations = torch.load(os.path.join("../dataset/skybox/rotations.pt"), weights_only=True).cuda()

        # 初始化中间变量： 
        frame_index = 0 
        render_indices = torch.zeros(child_means3D.size(0), dtype=torch.int).cuda() 
        parent_indices = torch.zeros(child_means3D.size(0), dtype=torch.int).cuda() 
        nodes_for_render_indices = torch.zeros(child_means3D.size(0), dtype=torch.int).cuda() 
        interpolation_weights = torch.zeros(child_means3D.size(0), dtype=torch.float32).cuda() 
        last_frame = torch.zeros(child_means3D.size(0), dtype=torch.int).cuda()

        # Evaluation 
        psnr_test = 0.0 
        ssims = 0.0 
        lpipss = 0.0 
        overTime = 0 
        window_size = 10 
        print("===========================================================") 
        print(f"Rendering process start:: {child_means3D.size()}")
        if args.frustum_culling:
            log_path = os.path.join(args.logs_dir, "with", f"{self.shared_data['tau']}.log")
        else:
            log_path = os.path.join(args.logs_dir, "without", f"{self.shared_data['tau']}.log")

        send_start_signal.set() 
        while not end_signal.is_set(): 
            if (not self.check_size(child_means3D.size(0), child_rotations.size(0), child_opacity.size(0), child_scales.size(0), child_shs.size(0), child_boxes.size(0)) or (not self.check_size(parent_means3D.size(0), parent_rotations.size(0), parent_opacity.size(0), parent_scales.size(0), parent_shs.size(0), parent_boxes.size(0))) or child_means3D.size(0) != parent_means3D.size(0) or child_means3D.size(0) != leafs_tag.size(0)):
                print("Size Error") 
                print(child_means3D.size(0), child_rotations.size(0), child_opacity.size(0), child_scales.size(0), child_shs.size(0), child_boxes.size(0), 
                      parent_means3D.size(0), parent_rotations.size(0), parent_opacity.size(0), parent_scales.size(0), parent_shs.size(0), parent_boxes.size(0), 
                      leafs_tag.size(0))
                break 
            # 1. 这里应该从viewpoint队列里面获取一个视点 
            try:
                viewpoint = camera_queue.get(timeout=5) 
            except queues.Empty: 
                # print("Timeout exception: No data received within 5 seconds.")
                overTime += 1 
                if (overTime >= 12):
                    print("Timeout exception: No data received within 1 minute.")
                    break 
                else:
                    continue 
            except Exception as e:
                print("Render exception[1]: ", e) 
                break 
            overTime = 0 
            viewpoint.world_view_transform = viewpoint.world_view_transform.cuda() 
            viewpoint.projection_matrix = viewpoint.projection_matrix.cuda() 
            viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda() 
            viewpoint.camera_center = viewpoint.camera_center.cuda() 
            frame_index += 1 
            # 2. precompute 
            tanfovx = math.tan(viewpoint.FoVx * 0.5) 
            tanfovy = math.tan(viewpoint.FoVy * 0.5) 
            threshold = (2 * (self.shared_data["tau"] + 0.5)) * tanfovx / (0.5 * viewpoint.image_width) 
            # print("force_search") 
            to_render = force_search( 
                child_boxes, 
                parent_boxes, 
                child_means3D, 
                threshold, 
                viewpoint.camera_center, 
                args.frustum_culling, 
                viewpoint.world_view_transform, 
                viewpoint.projection_matrix, 
                torch.zeros((3)), 
                leafs_tag, 
                # output 
                last_frame, 
                render_indices, 
                interpolation_weights) 
            print("out of force_search:: ", to_render) 
            # 3. render 
            # 计算插值 
            indices = render_indices[:to_render].int().contiguous() 
            num_node_kids = torch.cat([num_siblings, torch.ones(sky_box_means3d.size(0), dtype=torch.int).cuda()], dim = 0) 
            interps = interpolation_weights[:to_render].unsqueeze(1) 
            interps_inv = (1 - interpolation_weights[:to_render]).unsqueeze(1) 
            means3D_base = (interps * child_means3D[indices] + interps_inv * parent_means3D[indices]).contiguous() 
            scales_base = (interps * child_scales[indices] + interps_inv * parent_scales[indices]).contiguous() 
            shs_base = (interps.unsqueeze(2) * child_shs[indices] + interps_inv.unsqueeze(2) * parent_shs[indices]).contiguous() 
            opacity_base = (interps * child_opacity[indices] + interps_inv * parent_opacity[indices]).contiguous() 

            parents_rots = parent_rotations[indices] 
            child_rots = child_rotations[indices] 
            dots = torch.bmm(child_rots.unsqueeze(1), parents_rots.unsqueeze(2)).flatten() 
            parents_rots[dots < 0] *= -1 
            rotations_base = ((interps * child_rots) + interps_inv * parents_rots).contiguous() 

            means3d = torch.cat([means3D_base, sky_box_means3d], dim = 0).contiguous() 
            scales = torch.cat([scales_base, sky_box_scales], dim = 0).contiguous() 
            rotations = torch.cat([rotations_base, sky_box_rotations], dim = 0).contiguous() 
            shs = torch.cat([shs_base, sky_box_shs], dim = 0).contiguous() 
            opacity = torch.cat([opacity_base, sky_box_opacity], dim = 0).contiguous() 
            # 渲染 
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint.image_height),
                image_width=int(viewpoint.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
                scale_modifier=1.0,
                viewmatrix=viewpoint.world_view_transform,
                projmatrix=viewpoint.full_proj_transform,
                sh_degree=3,
                campos=viewpoint.camera_center,
                prefiltered=False,
                debug=pipe.debug,
                render_indices=torch.Tensor([]).int(),
                parent_indices=torch.Tensor([]).int(),
                interpolation_weights=interpolation_weights,
                num_node_kids = num_node_kids, 
                do_depth=False 
            ) 
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            image, _, _ = rasterizer( 
                means3D = means3d,
                means2D = None,
                opacities = opacity,
                shs = shs,
                scales = scales,
                rotations = rotations) 
            image = image.clamp(0.0, 1.0) 
            alpha_mask_path = os.path.join(args.alpha_masks, f"{viewpoint.image_name.rsplit('.', 1)[0]}.png")
            pil_alpha_mask = Image.open(alpha_mask_path) 
            alpha_mask = (torch.from_numpy(np.array(pil_alpha_mask)) / 255.0).cuda()
            alpha_mask = alpha_mask.unsqueeze(dim=-1).permute(2, 0, 1)
            
            pil_gt_image = Image.open(os.path.join(args.images, viewpoint.image_name))
            gt_image = (torch.from_numpy(np.array(pil_gt_image)) / 255.0).cuda()
            gt_image = gt_image.permute(2, 0, 1)
            
            # try:
            #     torchvision.utils.save_image(image, os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png"))
            # except:
            #     os.makedirs(os.path.dirname(os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
            #     torchvision.utils.save_image(image, os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png"))
            if args.eval:
                image *= alpha_mask 
                gt_image *= alpha_mask 
                psnr_test_ = psnr(image, gt_image).mean().double()
                ssims_ = ssim(image, gt_image).mean().double()
                lpipss_ = lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test += psnr_test_ 
                ssims += ssims_ 
                lpipss += lpipss_ 
                with open(log_path, "a+") as fout:
                    fout.write(f"{viewpoint.image_name}: {to_render}, {psnr_test_}, {ssims_}, {lpipss_}\n") 
                print(f"frame_index = {frame_index}, to_render={to_render}, psnr_avg = {psnr_test / frame_index}, ssim_avg = {ssims / frame_index}, lpips_avg = {lpipss / frame_index}")
            # 4. 每隔一段时间，删掉一部分数据集中很久没有用到的高斯点 
            if frame_index % 10 == 0:
                number_of_delete_points = child_means3D.size(0) 
                # last_frame 中大于阈值的所有数据都剔除 
                masks = last_frame < window_size 
                # child 136 * 4 = 544 Bytes * 20 k = 10.880 MB/s 没有压缩！ 
                child_means3D = child_means3D[masks].contiguous() # 3
                child_rotations = child_rotations[masks].contiguous() # 4 
                child_opacity = child_opacity[masks].contiguous() # 1 
                child_scales = child_scales[masks].contiguous() # 3
                child_shs = child_shs[masks].contiguous() # 16 * 3 
                child_boxes = child_boxes[masks].contiguous() # 4 * 2 
                # parent 
                parent_means3D = parent_means3D[masks].contiguous() 
                parent_rotations = parent_rotations[masks].contiguous() 
                parent_opacity = parent_opacity[masks].contiguous() 
                parent_scales = parent_scales[masks].contiguous() 
                parent_shs = parent_shs[masks].contiguous() 
                parent_boxes = parent_boxes[masks].contiguous() 
                # others 
                leafs_tag = leafs_tag[masks].contiguous() # 1 
                num_siblings = num_siblings[masks].contiguous() # 1 
                
                # 修改临时变量的长度: 
                render_indices = render_indices[:child_means3D.size(0)].contiguous() 
                parent_indices = parent_indices[:child_means3D.size(0)].contiguous() 
                nodes_for_render_indices = nodes_for_render_indices[:child_means3D.size(0)].contiguous() 
                interpolation_weights = interpolation_weights[:child_means3D.size(0)].contiguous() 
                last_frame = last_frame[masks].contiguous() 
                # for debug 
                number_of_delete_points = number_of_delete_points - child_means3D.size(0) 
                print("Delete points: ", number_of_delete_points) 
                # torch.cuda.empty_cache() 
                pass 
            # 5. 将新接收到的高斯点加入到数据集中 
            # child 
            child_means3D = torch.cat([child_means3D, queue.get().cuda()], dim = 0)
            child_rotations = torch.cat([child_rotations, queue.get().cuda()], dim = 0)
            child_opacity  = torch.cat([child_opacity, queue.get().cuda()], dim = 0)
            child_scales  = torch.cat([child_scales, queue.get().cuda()], dim = 0)
            child_shs  = torch.cat([child_shs, queue.get().cuda()], dim = 0) 
            child_boxes  = torch.cat([child_boxes, queue.get().cuda()], dim = 0)
            # parent 
            parent_means3D = torch.cat([parent_means3D, queue.get().cuda()], dim = 0)
            parent_rotations = torch.cat([parent_rotations, queue.get().cuda()], dim = 0)
            parent_opacity = torch.cat([parent_opacity, queue.get().cuda()], dim = 0)
            parent_scales = torch.cat([parent_scales, queue.get().cuda()], dim = 0)
            parent_shs = torch.cat([parent_shs, queue.get().cuda()], dim = 0)
            parent_boxes = torch.cat([parent_boxes, queue.get().cuda()], dim = 0)
            leafs_tag = torch.cat([leafs_tag, queue.get().cuda()], dim = 0)
            num_siblings = torch.cat([num_siblings, queue.get().cuda()], dim = 0)
            # 补充临时变量 
            addition = child_means3D.size(0) - render_indices.size(0) 
            render_indices = torch.cat([render_indices, torch.zeros(addition, dtype=torch.int).cuda()], dim = 0)
            parent_indices = torch.cat([parent_indices, torch.zeros(addition, dtype=torch.int).cuda()], dim = 0)
            nodes_for_render_indices = torch.cat([nodes_for_render_indices, torch.zeros(addition, dtype=torch.int).cuda()], dim = 0)
            interpolation_weights = torch.cat([interpolation_weights, torch.zeros(addition, dtype=torch.int).cuda()], dim = 0)
            last_frame = torch.cat([last_frame, torch.zeros(addition, dtype=torch.int).cuda()], dim = 0) 
        end_signal.wait() 
# single: 10.147.18.182 
# four:   47.116.170.43 
# mine 10.147.18.242 
if __name__ == "__main__":
    # Set up command line argument parser 
    parser = ArgumentParser(description="Rendering script parameters") 
    lp = ModelParams(parser) 
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser) 
    parser.add_argument("--tau", type=float, default=6.0) 
    parser.add_argument("--ip", type=str, default='10.147.18.182') 
    parser.add_argument('--port', type=int, default=50000) 
    parser.add_argument("--frustum_culling", action="store_true")
    parser.add_argument('--render_dir', type=str, default="") 
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--logs_dir", type=str, default="")
    parser.add_argument("--viewpointFilePath", type=str, default="")
    args = parser.parse_args(sys.argv[1:]) 
    dataset, pipe = lp.extract(args), pp.extract(args) 
    # client 
    mp.set_start_method("spawn", force=True) 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client_socket.connect((args.ip, args.port)) 
    manager = mp.Manager() 
    tensor_queue = mp.Queue() 
    camera_queue = mp.Queue() 
    c = client(manager, args.tau) 
    # 启动进程 
    parent_conn1, child_conn1 = mp.Pipe() # 通信管道 
    parent_conn2, child_conn2 = mp.Pipe() 
    end_signal = Event() 
    send_start_signal = Event() # 预处理结束，开始发送数据 
    c_send = mp.Process(target=c.send, args=(end_signal, send_start_signal, camera_queue, child_conn1, args.viewpointFilePath, )) 
    c_send.start() 
    c_receive = mp.Process(target=c.receive, args=(end_signal, tensor_queue, child_conn2, )) 
    c_receive.start() 
    c_render = mp.Process(target=c.render, args=(end_signal, send_start_signal, tensor_queue, camera_queue, dataset, pipe, args, )) 
    c_render.start() 
    send_handle(parent_conn1, client_socket.fileno(), c_send.pid) 
    send_handle(parent_conn2, client_socket.fileno(), c_receive.pid) 
    
    c_send.join() 
    c_receive.join() 
    c_render.join() 
    try:
        data = client_socket.recv(4, socket.MSG_PEEK)
        if data: 
            client_socket.close()
    except Exception as e: 
        print("Error", e)