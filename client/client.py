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

from subgraph_expand._C import subgraph_tree_init, subgraph_expand, subgraph_update 
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer 

# server-client 
from multiprocessing.reduction import send_handle, recv_handle 
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
from tqdm import tqdm

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
    def __init__(self, manager, tau=6.0, ws = 10): 
        self.shared_data = manager.dict({
            "tau": tau,
            "window_size" : ws,
            "interpolation_number" : 15
        }) 

    def send(self, end_signal, render_isOk, receive_isOk, camera_queue, child_con, viewpointFilePath, buffer_size=10240): 
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
        connection.sendall(struct.pack("f", self.shared_data["tau"])) 
        connection.sendall(struct.pack("i", self.shared_data["window_size"])) 
        code = 0 
        receive_isOk.wait() 
        print("===========================================================")
        print(f"Sending process start.")
        for idx, obj in enumerate(objects): 
            if end_signal.is_set():
                break 
            if idx >= 1: 
                render_isOk.wait()
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
    def receiveOne(self, connection, buffer_size = 10240):
        dat = connection.recv(4) 
        if dat == b'': 
            return torch.empty(), -1 
        message_length = int.from_bytes(dat, byteorder='big') 
        message = b"" 
        digit_count = 0
        if message_length > 1000000:
            digit_count = len(str(message_length)) - 6
            buffer_size = 1024 * pow(10, digit_count)
            # print(digit_count, buffer_size, message_length ) 
        progress_bar = tqdm(total=message_length, desc="Receiving", unit="B", unit_scale=True)
        
        while len(message) < message_length:
            chunk = connection.recv(min(buffer_size, message_length - len(message))) 
            if not chunk:
                progress_bar.close()
                return torch.empty(), -1 
            message += chunk 
            progress_bar.update(len(chunk)) 
        decompressed = zlib.decompress(message) 
        return pickle.loads(decompressed), 0 
    
    def receive(self, end_signal, receive_isOk, queue, child_conn, args, buffer_size=10240): 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        frame_index = 1 # 第一帧数据跳过 
        parameter_number = 2 
        print("===========================================================")
        print("Receive process start.")
        depth_count, _ = self.receiveOne(connection, buffer_size) 
        queue.put(depth_count) 
        data = connection.recv(4)
        if data: # pointNumber
            queue.put(struct.unpack('!i', data)[0])
        else:
            end_signal.set() 
        data = connection.recv(4)
        if data: # opacity min
            queue.put(struct.unpack('!f', data)[0])
        else:
            end_signal.set() 
        data = connection.recv(4)
        if data: # opacity max 
            queue.put(struct.unpack('!f', data)[0])
        else:
            end_signal.set() 
        if args.local:
            print("Receive skybox")
            for i in range(5): # means3D, opacity, rotations, scales, shs 
                tensor, _ = self.receiveOne(connection, buffer_size)
                if _ < 0:
                    end_signal.set()
                    break 
                queue.put(tensor) 
        data = connection.recv(1024).decode() 
        if data == "START":
            receive_isOk.set()
        else:
            end_signal.set() 
        while not end_signal.is_set(): 
            try: 
                frame_index += 1 
                # start_time = time.time() 
                for _ in range(parameter_number):
                    if end_signal.is_set():
                        break 
                    tensor, ret = self.receiveOne(connection, buffer_size) 
                    if ret < 0:
                        end_signal.set() 
                        break 
                    queue.put(tensor) 
                # end_time = time.time() 
                # execution_time = end_time - start_time 
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
    def render(self, end_signal, render_isOk, queue, camera_queue, dataset, pipe, args):
        torch.cuda.init() 
        torch.cuda.set_device(torch.cuda.current_device()) 
        # return 
        # initialize 
        depth_count     = queue.get() 
        pointNumber     = queue.get() 
        opacity_min     = queue.get() 
        opacity_max     = queue.get() 
        range           = (opacity_max - opacity_min) / 255.0 
        # skybox:
        if args.local:
            sky_box_means3d     = queue.get().cuda()
            sky_box_opacity     = queue.get().cuda()
            sky_box_rotations   = queue.get().cuda()
            sky_box_scales      = queue.get().cuda()
            sky_box_shs         = queue.get().cuda()
        else:
            sky_box_means3d     = torch.load(os.path.join("../dataset/skybox/means3d.pt"), weights_only=True).cuda() 
            sky_box_opacity     = torch.load(os.path.join("../dataset/skybox/opacity.pt"), weights_only=True).cuda() 
            sky_box_shs         = torch.load(os.path.join("../dataset/skybox/shs.pt"), weights_only=True).cuda() 
            sky_box_scales      = torch.load(os.path.join("../dataset/skybox/scales.pt"), weights_only=True).cuda() 
            sky_box_rotations   = torch.load(os.path.join("../dataset/skybox/rotations.pt"), weights_only=True).cuda()
        
        compressed_data = queue.get().cuda() 
        sizes           = queue.get().cuda() 
        # compressed_data = torch.load("../dataset/compressed_data.pt").cuda() 
        # sizes = torch.load("../dataset/sizes.pt").cuda() 
        length_size = sizes.size(0) * 2 
        print(f"Length = {sizes.size(0)}") 
        means3D         = torch.zeros(length_size, 3, dtype=torch.float).cuda() 
        opacities       = torch.zeros(length_size, 1, dtype=torch.float).cuda() 
        rotations       = torch.zeros(length_size, 4, dtype=torch.float).cuda() 
        scales          = torch.zeros(length_size, 3, dtype=torch.float).cuda() 
        shs             = torch.zeros(length_size, 16, 3, dtype=torch.float).cuda() 
        boxes           = torch.zeros(length_size, 2, 4, dtype=torch.float).cuda() 
        num_siblings    = torch.zeros(length_size, dtype=torch.int).cuda() 
        
        render_indices  = torch.zeros(length_size, dtype=torch.int).cuda() 
        least_recently  = torch.empty(length_size, dtype=torch.int).cuda()
        length_size_half = sizes.size(0) 
        least_recently[:length_size_half] = 0 
        least_recently[length_size_half:] = 100 
        # return 
        nodes = torch.full((pointNumber, 2), -1, dtype=torch.int).cuda() 
        
        print("Tree initialize") 
        featureMaxx = subgraph_tree_init(
            compressed_data, sizes, 
            nodes, means3D, opacities, rotations, scales, shs, 
            boxes, num_siblings, 
            opacity_min, range
        ) 
        print(featureMaxx, flush=True) 
        # Evaluation 
        psnr_test   = 0.0 
        ssims       = 0.0 
        lpipss      = 0.0 
        overTime    = 0 
        print("===========================================================") 
        print(f"Rendering process start:: ", flush=True) 
        if args.frustum_culling:
            log_path = os.path.join(args.logs_dir, "with", f"{self.shared_data['tau']}.log")
        else:
            log_path = os.path.join(args.logs_dir, "without", f"{self.shared_data['tau']}.log")
        render_isOk.set() 
        frame_index = 0 
        preview = camera_queue.get() 
        # create output folder if it doesn't exit  
        if not os.path.exists(args.render_dir):
            os.makedirs(args.render_dir)
            print(f"Create folder {args.render_dir}.")
        while not end_signal.is_set(): 
            viewpoint = None
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
            # viewpoints = generate(preview, currentview)
            # for viewpoint in viewpoints:
            # update subgraph of the BFS-tree 
            if viewpoint is None:
                continue
            overTime = 0 
            compressed_data = queue.get().cuda() 
            sizes = queue.get().cuda() 
            # print("Update tree start")
            featureMaxx, update_elapse = subgraph_update( 
                compressed_data, sizes, 
                nodes, means3D, opacities, rotations, scales, shs, boxes, num_siblings, 
                least_recently, self.shared_data["window_size"], opacity_min, range, featureMaxx
            ) 
            # print("Update tree end", featureMaxx, f"{update_elapse:.5f}") 
            
            viewpoint.world_view_transform = viewpoint.world_view_transform.cuda() 
            viewpoint.projection_matrix = viewpoint.projection_matrix.cuda() 
            viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda() 
            viewpoint.camera_center = viewpoint.camera_center.cuda() 
            frame_index += 1 
            # 2. precompute 
            tanfovx = math.tan(viewpoint.FoVx * 0.5) 
            tanfovy = math.tan(viewpoint.FoVy * 0.5) 
            threshold = (2 * (self.shared_data["tau"] + 0.5)) * tanfovx / (0.5 * viewpoint.image_width) 
            print("subgraph_expand") 
            to_render, expand_elapse = subgraph_expand( 
                nodes, 
                depth_count, 
                means3D, 
                boxes, 
                threshold, 
                viewpoint.camera_center, 
                args.frustum_culling, 
                viewpoint.world_view_transform, 
                viewpoint.projection_matrix, 
                least_recently, 
                render_indices 
            ) 
            print("out of subgraph_expand:: ", to_render, f"{expand_elapse:.5f}") 
            # 3. render 
            indices = render_indices[:to_render] 
            mea = torch.cat([means3D[indices], sky_box_means3d], dim = 0).contiguous() 
            sca = torch.cat([scales[indices], sky_box_scales], dim = 0).contiguous() 
            rot = torch.cat([rotations[indices], sky_box_rotations], dim = 0).contiguous() 
            sh = torch.cat([shs[indices], sky_box_shs], dim = 0).contiguous() 
            opa = torch.cat([opacities[indices], sky_box_opacity], dim = 0).contiguous() 
            num_node_kids = torch.cat([num_siblings[indices], torch.ones(sky_box_means3d.size(0), dtype=torch.int).cuda()], dim = 0).contiguous() 

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
                interpolation_weights=torch.Tensor([]).float(), 
                num_node_kids = num_node_kids, 
                do_depth=False 
            ) 
            rasterizer = GaussianRasterizer(raster_settings=raster_settings) 
            image, num_rendered, elapse, elapse_breakdown = rasterizer( 
                means3D = mea,
                means2D = None,
                opacities = opa,
                shs = sh,
                scales = sca,
                rotations = rot
            ) 
            image = torch.clamp(image, 0.0, 1.0) 
            
            pil_alpha_mask = Image.open(os.path.join(args.alpha_masks, f"{viewpoint.image_name.rsplit('.', 1)[0]}.png")) 
            alpha_mask = (torch.from_numpy(np.array(pil_alpha_mask)).to(torch.uint8) / 255.0).cuda()
            alpha_mask = alpha_mask.unsqueeze(dim=-1).permute(2, 0, 1)
            
            pil_gt_image = Image.open(os.path.join(args.images, viewpoint.image_name))
            gt_image = (torch.from_numpy(np.array(pil_gt_image)).to(torch.uint8) / 255.0).cuda()
            gt_image = gt_image.permute(2, 0, 1)
            
            pil_alpha_mask.close() 
            pil_gt_image.close()
            
            if args.train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]

            try:
                torchvision.utils.save_image(image, os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png"))
            except:
                os.makedirs(os.path.dirname(os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
                torchvision.utils.save_image(image, os.path.join(args.render_dir, viewpoint.image_name.split(".")[0] + ".png"))
            if args.eval:
                image *= alpha_mask 
                gt_image *= alpha_mask 
                psnr_test_ = psnr(image, gt_image).mean().double()
                ssims_ = ssim(image, gt_image).mean().double()
                lpipss_ = lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test += psnr_test_ 
                ssims += ssims_ 
                lpipss += lpipss_ 
                # with open(log_path, "a+") as fout:
                #     fout.write(f"{viewpoint.image_name}: {to_render}, {psnr_test_}, {ssims_}, {lpipss_}\n") 
                print(f"frame_index = {frame_index}, to_render={to_render}, psnr_avg = {psnr_test_}, ssim_avg = {ssims_}, lpips_avg = {lpipss_}", flush=True) 
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
    parser.add_argument("--tt_mode", type=int, default=0) 
    parser.add_argument("--local", action="store_true")
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
    render_isOk = Event() # 预处理结束，开始发送数据 
    receive_isOk = Event() 
    c_send = mp.Process(target=c.send, args=(end_signal, render_isOk, receive_isOk, camera_queue, child_conn1, args.viewpointFilePath, )) 
    c_send.start() 
    c_receive = mp.Process(target=c.receive, args=(end_signal, receive_isOk, tensor_queue, child_conn2, args, )) 
    c_receive.start() 
    c_render = mp.Process(target=c.render, args=(end_signal, render_isOk, tensor_queue, camera_queue, dataset, pipe, args, )) 
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