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
from random import randint
from utils.loss_utils import ssim
import sys
from scene import Scene, GaussianModel 
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
# import torchvision
from lpipsPyTorch import lpips 
from flash_tree_traversal._C import reorder_nodes, flash_tree_traversal, transimission_compress

# server-client 
from multiprocessing.reduction import send_handle, recv_handle
import torch.multiprocessing as mp 
from multiprocessing import Event 
import socket 
import json 
import pickle 
import zlib 
import struct 
# from vector_quantize_pytorch import ResidualVQ

class client_Camera:
    def __init__(self, FoVx:float, FoVy:float, image_height:int, image_width:int,world_view_transform:torch.Tensor, 
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
    @classmethod
    def from_json(cls, json_data):
        """ 从 JSON 数据还原 client_Camera 对象 """
        return cls(
            FoVx=json_data["FoVx"],
            FoVy=json_data["FoVy"],
            image_height=json_data["image_height"],
            image_width=json_data["image_width"],
            world_view_transform=torch.tensor(json_data["world_view_transform"]),
            projection_matrix=torch.tensor(json_data["projection_matrix"]),
            full_proj_transform=torch.tensor(json_data["full_proj_transform"]), 
            camera_center=torch.tensor(json_data["camera_center"]),
            image_name=json_data["image_name"],
            timestamp=json_data["timestamp"]
        )

class server:
    def __init__(self, manager, codebook_size=64, num_quantizers=6):
        self.shared_data = manager.dict({
            "tau": 6.0,
            "window_size" : int(10)
        })
        # self.vq_scale = ResidualVQ(dim = 3, codebook_size = codebook_size, num_quantizers = num_quantizers, 
        #                            commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
        #                            learnable_codebook=True, in_place_codebook_optimizer=lambda *args, 
        #                            **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        # self.vq_rotation = ResidualVQ(dim = 4, codebook_size = codebook_size, num_quantizers = num_quantizers, 
        #                          commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
        #                          learnable_codebook=True, in_place_codebook_optimizer=lambda *args, 
        #                          **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        # self.rvq_bit = math.log2(codebook_size)
        # self.num_quantizers = args.num_quantizers 
        
    def Compress(self, to_pass, render_indices, node_indices, num_siblings, 
                 # initial data 
                 means3D, opacities, rotations, scales, shs, boxes, 
                 # output 
                 compressed_data, mode):
        # _, scale_idx, _ = self.vq_scale()           # [batch, dim, num_quantizers] = [20000, 3, 6]
        # _, rotation_idx, _ = self.vq_rotation() 
        # scale_codebooks = self.vq_scale.cpu().state_dict()
        # rotation_codebooks = self.vq_rot.cpu().state_dict()
        
        pass 

    def receive(self, end_signal, cameras_queue, child_conn, buffer_size=10240):
        print("===========================================================")
        print("Receive process start!")
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        self.shared_data["tau"] = struct.unpack('f', connection.recv(4))[0] 
        self.shared_data["window_size"] = struct.unpack('i', connection.recv(4))[0] 
        while not end_signal.is_set(): 
            try:
                str = connection.recv(4)
                code = int.from_bytes(str, byteorder='big')
                if code == 0: # viewpoint 
                    str = connection.recv(4) 
                    data_size = int.from_bytes(str, byteorder='big')
                    if str == b'':
                        end_signal.set() 
                        break 
                    data = b"" 
                    while len(data) < data_size:
                        if not end_signal.is_set(): 
                            chunk = connection.recv(min(buffer_size, data_size - len(data)))  # 每次最多接收1024字节
                            if not chunk:
                                end_signal.set()
                                break
                            data += chunk
                    if end_signal.is_set():
                        break
                    json_data = json.loads(data.decode('utf-8'))
                    viewpoint = client_Camera.from_json(json_data)
                    cameras_queue.put(viewpoint)
                # elif code == 1:# tau 值 
                #     str = connection.recv(4)
                #     if not str:
                #         end_signal.set()
                #         break 
                #     self.shared_data["tau"] = struct.unpack('f', str)[0] 
                else:
                    print("Code error: ", code)
                    end_signal.set()
                    break 
            except Exception as e:
                print("Receive exception: ", e)
                end_signal.set()
                break 
        # cameras_queue.close() 
        # gc.collect()
        # torch.cuda.empty_cache()
        end_signal.wait()
        print("Quit receive.")
    
    def sendOne(self, connection, tensor=torch.empty(0), intData=0, floatData=0.0, strData = "", mode=0, buffer_size=10240):
        if mode == 0: # send tensor 
            data_decom = pickle.dumps(tensor)
            message = zlib.compress(data_decom) 
            size_com = len(message) 
            connection.sendall(size_com.to_bytes(4, byteorder='big')) 
            if size_com > 1000000: # 自动调整 buffer size, 在1000以内传输完成 
                digit_count = len(str(size_com)) - 6 
                buffer_size = 1024 * pow(10, digit_count) 
            print(size_com, size_com/buffer_size) 
            try:
                for i in tqdm(range(0, size_com, buffer_size), desc="Sending", unit="B", unit_scale=True):
                    chunk = message[i : i + buffer_size]
                    connection.sendall(chunk)
            except Exception as e:
                print("Send exception: ", e)
                return -1
        elif mode == 1: # send int
            try: 
                data = struct.pack('!i', intData)
                connection.sendall(data) 
            except Exception as e:
                print("Send exception: ", e) 
                return -1 
        elif mode == 2: # send float
            try: 
                data = struct.pack('!f', floatData)
                connection.sendall(data) 
            except Exception as e:
                print("Send exception: ", e) 
                return -1 
        elif mode == 3: # send start
            try:
                connection.sendall(strData.encode()) 
            except Exception as e:
                print("Send exception: ", e) 
                return -1 
        else:
            return -1 
        return 0 
    def send(self, end_signal, queue, child_conn, args, buffer_size=10240):   
        print("===========================================================")
        print("Send process start!") 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        parameter_number = 2 
        frame_index = 0 
        if (self.sendOne(connection, tensor=queue.get(), mode=0, buffer_size=buffer_size) < 0):
            end_signal.set() # depth_count 
        if (self.sendOne(connection, intData=queue.get(), mode=1, buffer_size=buffer_size) < 0):
            end_signal.set() # pointNumber 
        if (self.sendOne(connection, floatData=queue.get(), mode=2, buffer_size=buffer_size) < 0):
            end_signal.set() # opacity_min 
        if (self.sendOne(connection, floatData=queue.get(), mode=2, buffer_size=buffer_size) < 0):
            end_signal.set() # opacity_max 
        if args.local:
            for i in range(5):# means3D, opacity, rotations, scales, shs 
                if (self.sendOne(connection, tensor=queue.get(), mode=0, buffer_size=buffer_size) < 0):
                    end_signal.set() 
        if (self.sendOne(connection, strData="START", mode=3, buffer_size=buffer_size) < 0):
            end_signal.set() 
        print("Send initialization is ok!") 
        while not end_signal.is_set(): 
            try:
                viewpoint = None 
                try:
                    viewpoint = queue.get(timeout=5) 
                except:
                    viewpoint = None
                    pass 
                if viewpoint is not None:
                    frame_index += 1 
                    print(f"Send [{frame_index}] start.") 
                    for i in range(parameter_number): 
                        if (self.sendOne(connection, tensor=queue.get(), mode=0, buffer_size=buffer_size) < 0):
                            end_signal.set() 
                            break 
                    print(f"Send [{frame_index}] over.")
            except Exception as e:
                print("Send exception: ", e)
                end_signal.set()
                break
        # queue.close()
        # gc.collect()
        # torch.cuda.empty_cache()
        end_signal.wait()
        print("Quit send.")
    
    @torch.no_grad() 
    def tree_traversal(self, end_signal, dataset, pipe, cameras_queue, queue, args):
        torch.cuda.init() 
        torch.cuda.set_device(torch.cuda.current_device()) 
        gaussians = GaussianModel(dataset.sh_degree) 
        gaussians.active_sh_degree = dataset.sh_degree 
        scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True) 
        point_number = scene.gaussians._xyz.size(0) 
        
        means3D = scene.gaussians.get_xyz.cuda() 
        opacities = scene.gaussians.get_opacity.cuda() 
        rotations = scene.gaussians.get_rotation.cuda() 
        scales = scene.gaussians.get_scaling.cuda() 
        shs = scene.gaussians.get_features.cuda() 
        
        # tags 
        least_recently  = torch.full((point_number, ), 100, dtype = torch.int).cuda() # last visit 
        render_indices  = torch.zeros(point_number, dtype = torch.int).cuda() 
        node_indices     = torch.zeros(point_number, dtype = torch.int).cuda() 
        num_siblings    = torch.zeros(point_number, dtype = torch.int).cuda() 
        compressed_data = torch.zeros(point_number * 270, dtype=torch.uint8).cuda() 
        sizes           = torch.zeros(point_number, dtype=torch.uint8).cuda() 
        
        depth_count = torch.zeros(100, dtype=torch.int).cpu() # hope that 100 is enough 
        torch.cuda.synchronize() 
        tree_height = reorder_nodes( 
            scene.gaussians.nodes, 
            scene.gaussians.boxes, 
            depth_count 
        ) 
        # transmmit to "send" 
        depth_count = depth_count[:tree_height].contiguous() 
        opacity_min = torch.amin(opacities) 
        opacity_max = torch.amax(opacities) 
        queue.put(depth_count) 
        queue.put(scene.gaussians.nodes.size(0)) 
        queue.put(opacity_min) 
        queue.put(opacity_max) 
        inv_range = 255.0 / (opacity_max - opacity_min) 
        # skybox 
        if args.local:
            if scene.gaussians.skybox_points == 0:
                skybox_inds = torch.Tensor([]).long()
            else:
                skybox_inds = torch.arange(point_number - scene.gaussians.skybox_points, point_number - 1, device="cuda").long()
            queue.put(means3D[skybox_inds].cpu().contiguous()) 
            queue.put(opacities[skybox_inds].cpu().contiguous()) 
            queue.put(rotations[skybox_inds].cpu().contiguous()) 
            queue.put(scales[skybox_inds].cpu().contiguous()) 
            queue.put(shs[skybox_inds].cpu().contiguous()) 
        frame_index = 0 
        print("===========================================================", flush=True) 
        print("tree traversal process start!", flush=True) 
        # performance_file = os.path.join("../dataset/logs", "") 
        while not end_signal.is_set():
            viewpoint = None 
            try:
                viewpoint = cameras_queue.get(timeout=3) 
            except:
                viewpoint = None 
                continue 
            if viewpoint is not None: 
                frame_index += 1 
                tanfovx = math.tan(viewpoint.FoVx * 0.5) 
                target_size = (2 * (self.shared_data["tau"]  + 0.5)) * tanfovx / (0.5 * viewpoint.image_width) 
                viewpoint.camera_center         = viewpoint.camera_center.cuda() 
                viewpoint.world_view_transform  = viewpoint.world_view_transform.cuda() 
                viewpoint.projection_matrix     = viewpoint.projection_matrix.cuda() 
                viewpoint.full_proj_transform   = viewpoint.full_proj_transform.cuda() 
                print("flash_tree_traversal", flush=True) 
                to_pass, ftt_elapse = flash_tree_traversal( 
                    scene.gaussians.nodes, 
                    scene.gaussians.boxes, 
                    means3D, 
                    target_size, 
                    viewpoint.camera_center, 
                    viewpoint.world_view_transform, 
                    viewpoint.projection_matrix, 
                    args.frustum_culling, 
                    self.shared_data["window_size"], 
                    least_recently, 
                    render_indices, 
                    node_indices, 
                    num_siblings, 
                    args.tt_mode 
                ) 
                print(f"idx = {frame_index}, to pass = ", to_pass, ftt_elapse, flush=True) 
                compress_size, tc_elapse = transimission_compress( 
                    # input parameters 
                    to_pass, 
                    render_indices, 
                    node_indices, 
                    num_siblings, 
                    # initial data 
                    means3D, 
                    opacities, 
                    rotations, 
                    scales, 
                    shs, 
                    scene.gaussians.boxes, 
                    # output 
                    compressed_data,
                    sizes, 
                    opacity_min, inv_range 
                ) 
                print(f"Compress Rate Equals {compress_size*100.0/(to_pass * 270):.2f}%, {tc_elapse}", flush=True) 
                # if frame_index > 1:
                queue.put(viewpoint) 
                queue.put(compressed_data[:compress_size].cpu().contiguous()) 
                queue.put(sizes[:to_pass].cpu().contiguous()) 
                # elif True: # 如果在同一台服务器上测试，不用处理这部分 
                #     torch.save(compressed_data[:compress_size].cpu().contiguous(), "../dataset/compressed_data.pt")
                #     torch.save(sizes[:to_pass].cpu().contiguous(), "../dataset/sizes.pt")
                print(f"Tree Traversal[{frame_index}] end.") 
        # cameras_queue.close() 
        # gc.collect() 
        # torch.cuda.empty_cache()
        end_signal.wait()
        print("quit tree traversal.") 
if __name__ == "__main__":
    # Set up command line argument parser 
    parser = ArgumentParser(description="Rendering script parameters") 
    lp = ModelParams(parser) 
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser) 
    parser.add_argument('--ip', type=str, default="127.0.0.1") 
    parser.add_argument('--port', type=int, default=50000) 
    parser.add_argument('--client', type=int, default=1) 
    parser.add_argument('--frustum_culling', action="store_true") 
    parser.add_argument('--tt_mode', type=int, default=0) 
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args(sys.argv[1:]) 
    dataset, pipe = lp.extract(args), pp.extract(args) 
    mp.set_start_method("spawn", force=True) 
    
    # server 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server_socket.bind((args.ip, args.port)) 
    server_socket.listen(args.client) 
    manager = mp.Manager() 
    print(f"Server listening on {args.ip}:{args.port}, while the number of client equals {args.client}\n") 
    
    # while True: 
    print("Wait to connect...") 
    for i in range(1): 
        client_sock, client_address = server_socket.accept() 
        # 进程间通信: 
        parent_conn1, child_conn1 = mp.Pipe() # 通信管道 
        parent_conn2, child_conn2 = mp.Pipe() 
        cameras_queue = mp.Queue() 
        tensor_queue = mp.Queue() 
        end_signal = Event() 
        ser = server(manager) 
        print(f"connect with {client_address}") 
        # 启动进程: 
        s_receive = mp.Process(target=ser.receive, args=(end_signal, cameras_queue, child_conn1, )) 
        s_receive.start() 
        s_send = mp.Process(target=ser.send, args=(end_signal, tensor_queue, child_conn2, args, )) 
        s_send.start() 
        s_tt = mp.Process(target=ser.tree_traversal, args=(end_signal, dataset, pipe, cameras_queue, tensor_queue, args, ))
        s_tt.start() 
        send_handle(parent_conn1, client_sock.fileno(), s_receive.pid) 
        send_handle(parent_conn2, client_sock.fileno(), s_send.pid) 
        s_receive.join() 
        s_tt.join() 
        s_send.join() 
        try:
            data = client_sock.recv(4, socket.MSG_PEEK) 
            if data:
                client_sock.close() 
        except Exception as e:
            print("Quit Error: ", e)
    server_socket.close()
