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
from gaussian_hierarchy._C import expand_to_size

# server-client
from multiprocessing.reduction import send_handle, recv_handle
import torch.multiprocessing as mp 
from multiprocessing import Event 
import socket 
import json 
import pickle 
import zlib 
import struct 
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
    def __init__(self, manager):
        self.shared_data = manager.dict({
            "tau": 6.0
        })

    def receive(self, end_signal, cameras_future, child_conn, buffer_size=10240):
        print("===========================================================")
        print("Receive process start!")
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
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
                    cameras_future.put(viewpoint)
                elif code == 1:# tau 值 
                    str = connection.recv(4)
                    if not str:
                        end_signal.set()
                        break 
                    self.shared_data["tau"] = struct.unpack('f', str)[0] 
                else:
                    print("Code error: ", code)
                    end_signal.set()
                    break 
            except Exception as e:
                print("Receive exception: ", e)
                end_signal.set()
                break 
        # cameras_future.close()
        # gc.collect()
        # torch.cuda.empty_cache()
        end_signal.wait()
        print("Quit receive.")
    
    def send(self, end_signal, queue, child_conn, buffer_size=10240):   
        print("===========================================================")
        print("Send process start!") 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        parameter_number = 14 
        frame_index = 0 
        with open("../dataset/compress.txt", "w")as fout:
            pass
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
                    with open("../dataset/compress.txt", "a+")as fout:
                        fout.write(f"Compress {frame_index}\n")
                    for i in range(parameter_number): 
                        tensor = queue.get() 
                        data_decom = pickle.dumps(tensor)
                        # size_decom = len(data_decom)
                        message = zlib.compress(data_decom) 
                        size_com = len(message) 
                        # if i == 0:
                        #     with open("../dataset/compress.txt", "a+") as fout:
                        #         fout.write(f"\t\tsize = {tensor.size(0)}\n")
                        # with open("../dataset/compress.txt", "a+") as fout:
                        #     fout.write(f"\t\t{i}:\t{size_decom}, {size_com}, {(size_com/size_decom):.4f}\n")
                        connection.sendall(size_com.to_bytes(4, byteorder='big')) 
                        for i in range(0, size_com, buffer_size):
                            try: 
                                chunk = message[i : i + buffer_size] 
                                if not end_signal.is_set(): 
                                    connection.sendall(chunk) 
                                else:
                                    print("Send process ends because of end_signal.")
                                    break 
                            except Exception as e:
                                print("Send exception: ", e) 
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
    def tree_traversal(self, end_signal, dataset, pipe, cameras_future, cameras_past, queue):
        print("===========================================================")
        print("tree traversal process start!")
        torch.cuda.init()
        torch.cuda.set_device(torch.cuda.current_device())
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.active_sh_degree = dataset.sh_degree
        scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
        point_number = scene.gaussians._xyz.size(0) 
        box_number = scene.gaussians.boxes.size(0) 
        # tag 
        last_frame = torch.full((point_number, ), -999, dtype=torch.int).cuda() # last frame index 
        child_indices = torch.zeros(point_number, dtype=torch.int).cuda()       # child index 
        parent_indices = torch.zeros(point_number, dtype=torch.int).cuda()      # parent index 
        child_box_indices = torch.zeros(box_number, dtype=torch.int).cuda()     # child box index 
        parent_box_indices = torch.zeros(box_number, dtype=torch.int).cuda()    # parent box index 
        leafs_tag = torch.zeros(box_number, dtype=torch.bool).cuda()            # whether child is leaf 
        num_siblings = torch.zeros(box_number, dtype = torch.int).cuda()        # the number of nodes' siblings 
        
        search_means3D = scene.gaussians.get_xyz.cuda()
        means3D = scene.gaussians.get_xyz.cpu() # 高斯球的中心坐标 
        rotations = scene.gaussians.get_rotation.cpu() # 旋转角度 
        opacity = scene.gaussians.get_opacity.cpu() # opacity 
        scales = scene.gaussians.get_scaling.cpu() # 高斯球三个方向的半径 
        shs = scene.gaussians.get_features.cpu() # 球谐函数 
        # print(means3D.size(), rotations.size(), opacity.size(), scales.size(), shs.size()) 
        # Tree Traversal 
        torch.cuda.synchronize() 
        frame_index = 0 # 记录第几帧数据 
        window_size = 10 # 最近的帧数据 
        # with open("to_pass.txt", "w") as fout:
        #     pass
        while not end_signal.is_set():
            viewpoint = None 
            try:
                viewpoint = cameras_future.get(timeout=3) 
            except:
                viewpoint = None 
                continue 
            if viewpoint is not None: 
                frame_index += 1 
                print(f"Tree Traversal[{frame_index}] start:") 
                # cameras_past.put(viewpoint) # 用于预测下一帧视点 
                tanfovx = math.tan(viewpoint.FoVx * 0.5) 
                threshold = (2 * (self.shared_data["tau"]  + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)
                viewpoint.camera_center = viewpoint.camera_center.cuda() 
                viewpoint.world_view_transform = viewpoint.world_view_transform.cuda() 
                viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
                viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda() 
                # print("expand to size") 
                to_pass = expand_to_size( 
                    scene.gaussians.nodes,      # nodes 
                    scene.gaussians.boxes,      # boxes 
                    search_means3D,             # means 
                    threshold,                  # pixel size 
                    viewpoint.camera_center,    # viewpoint 
                    torch.zeros((3)),           # viewpoint dir 
                    frame_index,                # frame index 
                    window_size,                # window size 
                    viewpoint.world_view_transform, 
                    viewpoint.projection_matrix, 
                    # list for clients 
                    last_frame,                 # last frame index 
                    child_indices,              # child index 
                    parent_indices,             # parents index 
                    child_box_indices,          # child box index 
                    parent_box_indices,         # parent box index 
                    leafs_tag,                  # whether child is leaf 
                    num_siblings)               # the number of nodes' siblings 
                # print("to pass = ", to_pass, threshold, viewpoint.camera_center) 
                # with open("to_pass.txt", "a+") as fout:
                #     fout.write(f"frame_index = {frame_index}, {to_pass}\n") 
                c_indices = child_indices[:to_pass].cpu().contiguous()
                c_box_indices = child_box_indices[:to_pass].cpu().contiguous()
                p_indices = parent_indices[:to_pass].cpu().contiguous()
                p_box_indices = parent_box_indices[:to_pass].cpu().contiguous()
                # child 
                child_means3D = means3D[c_indices].contiguous() 
                child_rotations = rotations[c_indices].contiguous() 
                child_opacity = opacity[c_indices].contiguous() 
                child_scales = scales[c_indices].contiguous() 
                child_shs = shs[c_indices].contiguous() 
                child_boxes = scene.gaussians.boxes[c_box_indices].contiguous() 
                
                # parent 
                parent_means3D = means3D[p_indices].cpu().contiguous() 
                parent_rotations = rotations[p_indices].cpu().contiguous() 
                parent_opacity = opacity[p_indices].cpu().contiguous() 
                parent_scales = scales[p_indices].cpu().contiguous() 
                parent_shs = shs[p_indices].cpu().contiguous() 
                parent_boxes = scene.gaussians.boxes[p_box_indices].cpu().contiguous() 
                # child is leaf ?
                leafs = leafs_tag[:to_pass].cpu().contiguous()
                siblings_to_render = num_siblings[:to_pass].cpu().contiguous() 
                # child: 
                if frame_index > 1 and to_pass > 0:
                    # 通过queue传输给 send thread 
                    queue.put(viewpoint) 
                    # children 
                    queue.put(child_means3D) 
                    queue.put(child_rotations) 
                    queue.put(child_opacity) 
                    queue.put(child_scales) 
                    queue.put(child_shs) 
                    queue.put(child_boxes) 
                    # parents 
                    queue.put(parent_means3D) 
                    queue.put(parent_rotations) 
                    queue.put(parent_opacity) 
                    queue.put(parent_scales) 
                    queue.put(parent_shs) 
                    queue.put(parent_boxes) 
                    # others 
                    queue.put(leafs)
                    queue.put(siblings_to_render)
                elif frame_index == 1 and False: 
                    os.makedirs(f"data/{self.shared_data['tau']}", exist_ok=True) 
                    torch.save(child_means3D, os.path.join(f"data/{self.shared_data['tau']}/0.pt"))
                    torch.save(child_rotations, os.path.join(f"data/{self.shared_data['tau']}/1.pt"))
                    torch.save(child_opacity, os.path.join(f"data/{self.shared_data['tau']}/2.pt"))
                    torch.save(child_scales, os.path.join(f"data/{self.shared_data['tau']}/3.pt"))
                    torch.save(child_shs, os.path.join(f"data/{self.shared_data['tau']}/4.pt"))
                    torch.save(child_boxes, os.path.join(f"data/{self.shared_data['tau']}/5.pt"))
                    torch.save(parent_means3D, os.path.join(f"data/{self.shared_data['tau']}/6.pt"))
                    torch.save(parent_rotations, os.path.join(f"data/{self.shared_data['tau']}/7.pt"))
                    torch.save(parent_opacity, os.path.join(f"data/{self.shared_data['tau']}/8.pt"))
                    torch.save(parent_scales, os.path.join(f"data/{self.shared_data['tau']}/9.pt"))
                    torch.save(parent_shs, os.path.join(f"data/{self.shared_data['tau']}/10.pt"))
                    torch.save(parent_boxes, os.path.join(f"data/{self.shared_data['tau']}/11.pt"))
                    torch.save(leafs, os.path.join(f"data/{self.shared_data['tau']}/12.pt"))
                    torch.save(siblings_to_render, os.path.join(f"data/{self.shared_data['tau']}/13.pt"))
                    break 
                print(f"Tree Traversal[{frame_index}] end.")
        # cameras_future.close()
        # cameras_past.close()
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
    for i in range(1): 
        print("Wait to connect...") 
        client_sock, client_address = server_socket.accept() 
        # 进程间通信: 
        parent_conn1, child_conn1 = mp.Pipe() # 通信管道 
        parent_conn2, child_conn2 = mp.Pipe() 
        cameras_future = mp.Queue() 
        cameras_past = mp.Queue() 
        tensor_queue = mp.Queue() 
        end_signal = Event() 
        ser = server(manager) 
        print(f"connect with {client_address}") 
        # 启动进程: 
        s_receive = mp.Process(target=ser.receive, args=(end_signal, cameras_future, child_conn1, )) 
        s_receive.start() 
        s_send = mp.Process(target=ser.send, args=(end_signal, tensor_queue, child_conn2, )) 
        s_send.start() 
        s_tt = mp.Process(target=ser.tree_traversal, args=(end_signal, dataset, pipe, cameras_future, cameras_past, tensor_queue, ))
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
