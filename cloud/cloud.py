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
import socket, json, zlib, struct, io, sys, math 
from scene import Scene, GaussianModel 
from argparse import ArgumentParser 
from arguments import ModelParams 
from tqdm import tqdm 
from multiprocessing.reduction import send_handle, recv_handle
import torch 
import torch.multiprocessing as mp 
from multiprocessing import Event 
from vector_quantize_pytorch import ResidualVQ 
from flash_tree_traversal._C import reorder_nodes, flash_tree_traversal

class client_Camera:
    def __init__(self, world_view_transform:torch.Tensor, projection_matrix:torch.Tensor, image_name:str):
        self.world_view_transform   = world_view_transform
        self.projection_matrix      = projection_matrix 
        self.image_name             = image_name 
    def serialize(self):
        return json.dumps({
            "world_view_transform": self.world_view_transform.tolist(),
            "projection_matrix": self.projection_matrix.tolist(),
            "image_name": self.image_name 
        }) 
    @classmethod 
    def from_json(cls, json_data):
        return cls(
            world_view_transform=torch.tensor(json_data["world_view_transform"]),
            projection_matrix=torch.tensor(json_data["projection_matrix"]),
            image_name=json_data["image_name"] 
        )

class server:
    def __init__(self, manager):
        self.shared_data = manager.dict({
            "tau": 6.0,
            "window_size" : int(10),
            "FoVx" : 0.0,
            "image_width" : 0 
        }) 
        self.rvq_num = 6 
        self.parameter_number = 3 
    @torch.no_grad() 
    def receive(self, end_signal, cameras_queue, child_conn, buffer_size=10240):
        print("===========================================================")
        print("Receive process start!") 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        self.shared_data["tau"] = struct.unpack('f', connection.recv(4))[0] 
        self.shared_data["window_size"] = struct.unpack('i', connection.recv(4))[0] 
        self.shared_data["FoVx"] = struct.unpack('f', connection.recv(4))[0] 
        self.shared_data["image_width"] = struct.unpack('i', connection.recv(4))[0] 
        # print(self.shared_data["tau"], self.shared_data["window_size"], self.shared_data["FoVx"], self.shared_data["image_width"]) 
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
        end_signal.wait()
        print("Quit receive.")
    def sendOne(self, connection, compressed = "", intData=0, floatData=0.0, mode=0, buffer_size=10240):
        if mode == 0: # send str 
            size_com = len(compressed) 
            connection.sendall(size_com.to_bytes(4, byteorder='big'))
            if size_com > 1000000: # 自动调整 buffer size, 在1000以内传输完成 
                digit_count = len(str(size_com)) - 6 
                buffer_size = 1024 * pow(10, digit_count) 
            # print(size_com, size_com/buffer_size) 
            for i in tqdm(range(0, size_com, buffer_size), desc="Sending", unit="B", unit_scale=True):
                chunk = compressed[i : i + buffer_size]
                connection.sendall(chunk) 
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
        else:
            return -1 
        return 0 
    def send(self, end_signal, queue, child_conn, args, buffer_size=10240): 
        print("===========================================================")
        print("Send process start!") 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        frame_index = 0 
        if (self.sendOne(connection, compressed = queue.get(), mode=0, buffer_size=buffer_size) < 0):
            end_signal.set() 
        if (self.sendOne(connection, intData    = queue.get(), mode=1, buffer_size=buffer_size) < 0):
            end_signal.set() 
        if (self.sendOne(connection, compressed = queue.get(), mode=0, buffer_size=buffer_size) < 0):
            end_signal.set() 
        if (self.sendOne(connection, compressed = queue.get(), mode=0, buffer_size=buffer_size) < 0):
            end_signal.set() 
        if args.local:
            if (self.sendOne(connection, intData = queue.get(), mode=1, buffer_size=buffer_size) < 0):
                end_signal.set() # skybox number 
            for _ in range(5): # means3D, opacity, rotations, scales, shs 
                if (self.sendOne(connection, compressed=queue.get(), mode=0, buffer_size=buffer_size) < 0):
                    end_signal.set() 
        print("Send initialization is ok!") 
        while not end_signal.is_set(): 
            try:
                # viewpoint = None 
                code = 0 
                try:
                    code = queue.get(timeout=5) 
                except:
                    code = 0 
                    pass 
                if code == 1:
                    frame_index += 1 
                    print(f"Send [{frame_index}] start.") 
                    for i in range(self.parameter_number): 
                        if (self.sendOne(connection, compressed=queue.get(), mode=0, buffer_size=buffer_size) < 0):
                            end_signal.set() 
                            break 
                    print(f"Send [{frame_index}] over.")
                    pass 
            except Exception as e:
                print("Send exception: ", e)
                end_signal.set()
                break
        # queue.close()
        # gc.collect()
        # torch.cuda.empty_cache()
        end_signal.wait()
        print("Quit send.")
    def Compress(self, N, render_indices, node_indices, scene, shs_vq, queue):
        means3D     = scene.gaussians.get_xyz 
        opacities   = scene.gaussians.get_opacity 
        rotations   = scene.gaussians.get_rotation 
        scales      = scene.gaussians.get_scaling 
        boxes       = scene.gaussians.boxes 
        shs         = scene.gaussians.get_features 
        start       = torch.cuda.Event(enable_timing=True) 
        end         = torch.cuda.Event(enable_timing=True) 
        start.record() 
        render_indices  = render_indices[:N] 
        node_indices    = node_indices[:N] 
        features_bytes  = torch.cat((
            means3D[render_indices], 
            opacities[render_indices], 
            rotations[render_indices], 
            scales[render_indices], 
            boxes[node_indices].view(N, 8)[:, :7] 
        ), dim=1).cpu().numpy().tobytes()

        shs_32          = shs[render_indices].view(N, -1) 
        _, indices, _   = shs_vq(shs_32) 
        buffer = io.BytesIO() 
        torch.save(indices.cpu(), buffer) 
        buffer.seek(0) 
        shs_bytes = buffer.read() 

        indices_bytes   = zlib.compress(node_indices.cpu().numpy().tobytes() ) 
        features_bytes  = zlib.compress(features_bytes) 
        shs_bytes       = zlib.compress(shs_bytes) 
        
        end.record() 
        torch.cuda.synchronize() 
        elapsed_time_ms = start.elapsed_time(end) 

        queue.put(1) 
        queue.put(indices_bytes) 
        queue.put(features_bytes) 
        queue.put(shs_bytes) 
        base_size       = (3 * 4 + 3 + 1 + 4 + 3 + 4 * 2 + 7 + 16 * 3) * 4 
        compress_rate   = (len(indices_bytes) + len(features_bytes) + len(shs_bytes)) / (base_size * N) 
        return compress_rate, elapsed_time_ms 

    @torch.no_grad() 
    def tree_traversal(self, end_signal, dataset, cameras_queue, queue, args):
        torch.cuda.init() 
        torch.cuda.set_device(torch.cuda.current_device()) 
        gaussians = GaussianModel(dataset.sh_degree) 
        gaussians.active_sh_degree = dataset.sh_degree 
        scene = Scene(dataset, gaussians, create_from_hier=True) 
        
        means3D     = scene.gaussians.get_xyz 
        opacities   = scene.gaussians.get_opacity 
        rotations   = scene.gaussians.get_rotation 
        scales      = scene.gaussians.get_scaling 
        shs         = scene.gaussians.get_features 

        point_number    = means3D.size(0) 
        least_recently  = torch.full((point_number, ), 100, dtype = torch.int).cuda() # last visit 
        render_indices  = torch.zeros(point_number, dtype = torch.int).cuda() 
        node_indices    = torch.zeros(point_number, dtype = torch.int).cuda() 
        parent          = torch.full((scene.gaussians.nodes.size(0), ), -1, dtype = torch.int).cuda()

        depth_count = torch.zeros(100, dtype=torch.int).cpu() 
        shs_vq = ResidualVQ( 
            dim=48,
            codebook_size=256,
            num_quantizers=self.rvq_num,
            commitment_weight=0.0,
            kmeans_init=True,
            kmeans_iters=10,
            ema_update=False,
            learnable_codebook=True 
        ).cuda() 
        with torch.no_grad():
            _, _, _ = shs_vq(shs[::4].view(-1, 48)) # initialize shs_vq 
        state_dict = shs_vq.state_dict() 
        buf = io.BytesIO() 
        torch.save(state_dict, buf) 
        buf.seek(0) 

        torch.cuda.empty_cache() 
        tree_height = reorder_nodes( 
            scene.gaussians.nodes, 
            scene.gaussians.boxes, 
            depth_count,
            parent
        ) 
        # transmmit to "send" 
        depth_count = depth_count[:tree_height].contiguous() 
        queue.put(zlib.compress(depth_count.cpu().numpy().tobytes())) 
        queue.put(scene.gaussians.nodes.size(0)) 
        queue.put(zlib.compress(parent.cpu().numpy().tobytes()))
        queue.put(zlib.compress(buf.read())) 
        # skybox 
        if args.local:
            if scene.gaussians.skybox_points == 0:
                skybox_inds = torch.Tensor([]).long()
            else:
                skybox_inds = torch.arange(point_number - scene.gaussians.skybox_points, point_number, device="cuda").long()
            queue.put(scene.gaussians.skybox_points) 
            queue.put(zlib.compress(means3D[skybox_inds].cpu().numpy().tobytes())) 
            queue.put(zlib.compress(opacities[skybox_inds].cpu().numpy().tobytes())) 
            queue.put(zlib.compress(rotations[skybox_inds].cpu().numpy().tobytes())) 
            queue.put(zlib.compress(scales[skybox_inds].cpu().numpy().tobytes())) 
            queue.put(zlib.compress(shs[skybox_inds].cpu().numpy().tobytes())) 
        frame_index = 0 
        print("tree traversal process start!", flush=True) 
        with open(args.log_file, "w")as fout:
            pass 
        while not end_signal.is_set():
            viewpoint = None 
            try:
                viewpoint = cameras_queue.get(timeout=3) 
            except:
                viewpoint = None 
                continue 
            if viewpoint is not None: 
                frame_index += 1 
                viewpoint.world_view_transform  = viewpoint.world_view_transform.cuda() 
                viewpoint.projection_matrix     = viewpoint.projection_matrix.cuda() 
                
                viewpoint.full_proj_transform   = viewpoint.world_view_transform @ viewpoint.projection_matrix 
                viewpoint.camera_center         = viewpoint.world_view_transform.inverse()[3, :3].cuda() 
                tanfovx = math.tan(self.shared_data["FoVx"] * 0.5) 
                target_size = (2 * (self.shared_data["tau"]  + 0.5)) * tanfovx / (0.5 * self.shared_data["image_width"]) 
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
                    # degrees, 
                    args.tt_mode 
                ) 
                # print(f"idx = {frame_index}, to pass = ", to_pass, ftt_elapse, flush=True)
                compress_rate, tc_elapse = self.Compress( 
                    to_pass, render_indices, node_indices, scene, shs_vq, queue 
                ) 
                # print(compress_rate, tc_elapse) 
                # with open(args.log_file, "a+")as fout:
                #     compress_rate = compress_size * 100.0 / (to_pass * 270) 
                #     elapse = ftt_elapse + tc_elapse 
                #     fout.write(f"image_name = {viewpoint.image_name}:\n")
                #     fout.write(f"\tto_pass = {to_pass}\n")
                #     fout.write(f"\tcompress_rate = {compress_rate:.2f}\n")
                #     fout.write(f"\telapse = {elapse:.5f} = {ftt_elapse:.5f} + {tc_elapse:.5f}\n")
                print(f"Tree Traversal[{frame_index}] end.") 
        # cameras_queue.close() 
        # gc.collect() 
        # torch.cuda.empty_cache() 
        end_signal.wait() 
        print("quit tree traversal.") 
if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters") 
    lp = ModelParams(parser) 
    parser.add_argument('--ip', type=str, default="127.0.0.1") 
    parser.add_argument('--port', type=int, default=50000) 
    parser.add_argument('--client', type=int, default=1) 
    parser.add_argument('--frustum_culling', action="store_true") 
    parser.add_argument('--tt_mode', type=int, default=0) 
    parser.add_argument('--log_file', type=str, default="") 
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args(sys.argv[1:]) 
    dataset = lp.extract(args) 
    mp.set_start_method("spawn", force=True) 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server_socket.bind((args.ip, args.port)) 
    server_socket.listen(args.client) 
    manager = mp.Manager() 
    print(f"Server listening on {args.ip}:{args.port}, while the number of client equals {args.client}\n") 
    print("Wait to connect...") 
    for i in range(1): 
        client_sock, client_address = server_socket.accept() 
        parent_conn1, child_conn1 = mp.Pipe() 
        parent_conn2, child_conn2 = mp.Pipe() 
        cameras_queue = mp.Queue() 
        tensor_queue = mp.Queue() 
        end_signal = Event() 
        ser = server(manager) 
        print(f"connect with {client_address}") 
        s_receive = mp.Process(target=ser.receive, args=(end_signal, cameras_queue, child_conn1, )) 
        s_receive.start() 
        s_send = mp.Process(target=ser.send, args=(end_signal, tensor_queue, child_conn2, args, )) 
        s_send.start() 
        s_tt = mp.Process(target=ser.tree_traversal, args=(end_signal, dataset, cameras_queue, tensor_queue, args, ))
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
