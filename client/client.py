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

from argparse import ArgumentParser 
import math, os, sys, socket, time, json, zlib, struct, io 
import torchvision, torch 
from multiprocessing.reduction import send_handle, recv_handle 
import torch.multiprocessing as mp 
from multiprocessing import Event, queues 
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from flash_tree_traversal._C import subgraph_tree_init, subgraph_expand, subgraph_update 
from fast_hier import GaussianRasterizationSettings, GaussianRasterizer 
import numpy as np 
from PIL import Image
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ 
from einops import reduce
from utils.general_utils import PILtoTorch
from scipy.spatial.transform import Rotation as R, Slerp 

class client_Camera:
    def __init__(self, world_view_transform, projection_matrix, image_name):
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
def read_image(image_name, images_dir = "", masks_dir = "", resolution_scale = -1, train_test_exp = False):
    image = Image.open(os.path.join(images_dir, image_name)) 
    orig_w, orig_h = image.size
    if resolution_scale in [1, 2, 4, 8]:
        resolution = round(orig_w / resolution_scale), round(orig_h / resolution_scale)
    else:  # should be a type that converts to float 
        if resolution_scale == -1:
            if orig_w > 1600:
                global WARNED 
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) 
        resolution = (int(orig_w / scale), int(orig_h / scale))
    resized_image_rgb = PILtoTorch(image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    if masks_dir != "":
        try:
            pre_name, _ = os.path.splitext(image_name)  # 去除原始后缀
            mask_name = pre_name + '.png'
            alpha_mask = Image.open(os.path.join(masks_dir, mask_name))
        except FileNotFoundError:
            print(f"Error: The mask file at path '{masks_dir}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{masks_dir}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        alpha_mask = PILtoTorch(alpha_mask, resolution).cuda() 
    elif resized_image_rgb.shape[0] == 4:
        alpha_mask = resized_image_rgb[3:4, ...].cuda() 
    else:
        alpha_mask = None 
    original_image = gt_image.clamp(0.0, 1.0).cuda() 
    gt_image = gt_image.clamp(0.0, 1.0).cuda() 
    if masks_dir != "":
        original_image *= alpha_mask
    return gt_image, alpha_mask 
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
class client:
    def __init__(self, manager, tau=6.0, ws = 10): 
        self.shared_data = manager.dict({
            "tau": tau,
            "window_size" : ws,
            "FoVx" : 0.0,
            "FoVy" : 0.0,
            "image_height" : 0, 
            "image_width" : 0, 
            "interpolation_number" : 15 
        }) 
        self.rvq_num = 6 
        self.parameter_number = 3
    def send(self, end_signal, render_isOk, receive_isOk, camera_queue, child_con, viewpointFilePath, buffer_size=10240): 
        fd = recv_handle(child_con) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        with open("/workspace/code/dataset/bandwidth/camera_poses.json", "r")as fin:
            camera_poses = json.load(fin)  
        camera_poses = dict(sorted(camera_poses.items())) 
        with open("/workspace/code/dataset/bandwidth/setting.json", "r") as f:
            settings = json.load(f) 
        R_pre = None 
        T_pre = None 
        Rs = [] 
        Ts = [] 
        for idx, (image_name, pose) in enumerate(camera_poses.items()):
            R_cur = np.array(pose["R"]) 
            T_cur = np.array(pose["T"]) 
            if R_pre is None:
                R_pre = R_cur 
                T_pre = T_cur 
                Rs.append(R_cur) 
                Ts.append(T_cur) 
                continue 
            key_times = [0, 1] 
            key_rots = R.from_matrix([R_pre, R_cur])
            slerp = Slerp(key_times, key_rots) 
            # n_f = set_frames[idx-1] if n_frames == -1 else 
            n_f = 15 
            for i in range(1, n_f):
                t = i / n_f 
                R_mid = slerp([t])[0].as_matrix() 
                T_mid = (1 - t) * T_pre + t * T_cur 
                Rs.append(R_mid) 
                Ts.append(T_mid) 
            Rs.append(R_cur) 
            Ts.append(T_cur) 
            R_pre = R_cur 
            T_pre = T_cur 
        zfar = 100.0 
        znear = 0.01 
        trans = np.array([0.0, 0.0, 0.0]) 
        scale = 1.0 
        cameras = {} 
        projection_matrix = getProjectionMatrix(
            znear=znear, 
            zfar=zfar, 
            fovX=settings["FoVx"], 
            fovY=settings["FoVy"], 
            primx=settings["primx"], 
            primy=settings["primy"] 
        ).transpose(0,1).cuda() 
        for idx, (R_cur, T_cur) in enumerate(zip(Rs, Ts)):
            world_view_transform = torch.tensor(
                getWorld2View2(R_cur, T_cur, trans, scale) 
            ).transpose(0, 1).cuda() 
            cameras[idx] = client_Camera(world_view_transform, projection_matrix, f"{idx}.png")  
        self.shared_data["FoVx"] = settings["FoVx"] 
        self.shared_data["FoVy"] = settings["FoVy"] 
        self.shared_data["image_height"] = settings["image_height"]  
        self.shared_data["image_width"] = settings["image_width"] 
        connection.sendall(struct.pack("f", self.shared_data["tau"])) 
        connection.sendall(struct.pack("i", self.shared_data["window_size"]))  
        connection.sendall(struct.pack("f", self.shared_data["FoVx"])) 
        connection.sendall(struct.pack("i", self.shared_data["image_width"])) 
        code = 0 
        receive_isOk.wait() 
        print("===========================================================") 
        print(f"Sending process start.") 
        for idx, (image_name, pose) in enumerate(cameras.items()): 
            if idx % self.shared_data["window_size"] == 0:
                if end_signal.is_set():
                    break 
                if idx >= 1: 
                    render_isOk.wait() 
                connection.sendall(code.to_bytes(4, byteorder='big')) 
                message = pose.serialize().encode('utf-8') 
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
            camera_queue.put(pose) 
            # print(f"Send viewpoint[{idx}] sucessfully") 
            # time.sleep(1) 
        end_signal.wait() 
    def receiveOne(self, connection, buffer_size = 10240):
        dat = connection.recv(4) 
        if dat == b'': 
            return "", -1 
        message_length = int.from_bytes(dat, byteorder='big') 
        if message_length > 1000000:
            digit_count = len(str(message_length)) - 6
            buffer_size = 1024 * pow(10, digit_count)
        message = b"" 
        progress_bar = tqdm(total=message_length, desc="Receiving", unit="B", unit_scale=True)
        while len(message) < message_length:
            chunk = connection.recv(min(buffer_size, message_length - len(message))) 
            if not chunk:
                progress_bar.close()
                return "", -1 
            message += chunk 
            progress_bar.update(len(chunk)) 
        return message, 0 
    def receive(self, end_signal, receive_isOk, queue, child_conn, args, buffer_size=10240): 
        fd = recv_handle(child_conn) 
        connection = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) 
        frame_index = 0 
        print("Receive process start.") 
        depth_count, _ = self.receiveOne(connection, buffer_size) 
        queue.put(depth_count) 
        data = connection.recv(4)
        if data: # pointNumber 
            queue.put(struct.unpack('!i', data)[0])
        else:
            end_signal.set() 
        parents, _ = self.receiveOne(connection, buffer_size) 
        queue.put(parents)
        shs_vq, _ = self.receiveOne(connection, buffer_size) 
        queue.put(shs_vq) 
        print("Receive skybox") 
        data = connection.recv(4)
        if data: # skybox number  
            queue.put(struct.unpack('!i', data)[0])
            for i in range(5): # means3D, opacity, rotations, scales, shs 
                str, _ = self.receiveOne(connection, buffer_size)
                if _ < 0:
                    end_signal.set()
                    break 
                queue.put(str) 
        else:
            end_signal.set() 
        receive_isOk.set() 
        while not end_signal.is_set(): 
            try: 
                frame_index += 1 
                for _ in range(self.parameter_number):
                    if end_signal.is_set():
                        break 
                    str, ret = self.receiveOne(connection, buffer_size) 
                    if ret < 0:
                        end_signal.set() 
                        break 
                    queue.put(str) 
                time.sleep(1) 
            except Exception as e: 
                print("Receive exception: ", e) 
                end_signal.set() 
                break 
        end_signal.wait() 
    def Decompress(self, queue, shs_vq, first_trail=False): 
        start       = torch.cuda.Event(enable_timing=True) 
        end         = torch.cuda.Event(enable_timing=True) 
        indices_bytes = queue.get()
        features_bytes = queue.get() 
        shs_bytes = queue.get() 
        start.record() 
        indices_bytes   = zlib.decompress(indices_bytes)
        features_bytes  = zlib.decompress(features_bytes)
        shs_bytes       = zlib.decompress(shs_bytes)
        N = len(indices_bytes) // 4 
        indices  = torch.from_numpy((np.frombuffer(indices_bytes, dtype=np.int32)).copy()).cuda()
        features = torch.from_numpy((np.frombuffer(features_bytes, dtype=np.float16)).reshape((N, 18)).copy()).cuda().to(torch.float32)
        buffer   = io.BytesIO(shs_bytes) 
        shs_indices = torch.load(buffer).cuda() 
        if first_trail:
            indices_1_restored = shs_indices[:N//2] 
            indices_2_restored = shs_indices[N//2:] 
            codes   = shs_vq.get_codes_from_indices(indices_1_restored.reshape(-1, 1, self.rvq_num)) 
            shs_1   = shs_vq.project_out(reduce(codes, 'q ... -> ...', 'sum')).squeeze(1).reshape(-1, 16, 3) 
            codes   = shs_vq.get_codes_from_indices(indices_2_restored.reshape(-1, 1, self.rvq_num)) 
            shs_2   = shs_vq.project_out(reduce(codes, 'q ... -> ...', 'sum')).squeeze(1).reshape(-1, 16, 3) 
            shs     = torch.cat([shs_1, shs_2], dim=0) 
        else:
            codes    = shs_vq.get_codes_from_indices(shs_indices.reshape(-1, 1, self.rvq_num)) 
            shs      = shs_vq.project_out(reduce(codes, 'q ... -> ...', 'sum')).squeeze(1).reshape(-1, 16, 3)
        end.record() 
        torch.cuda.synchronize() 
        elapsed_time_ms = start.elapsed_time(end) 
        return indices, features, shs, N, elapsed_time_ms 
    @torch.no_grad() 
    def render(self, end_signal, render_isOk, queue, camera_queue, args):
        render_isOk.set() 
        torch.cuda.init() 
        torch.cuda.set_device(torch.cuda.current_device()) 
        # initialize 
        shs_vq = ResidualVQ( 
            dim=48,
            codebook_size=64,
            num_quantizers=self.rvq_num,
            commitment_weight=0.0,
            kmeans_init=False,
            kmeans_iters=0,
            ema_update=False,
            learnable_codebook=False 
        ).cuda() 
        depth_count = torch.from_numpy(np.frombuffer(zlib.decompress(queue.get()), dtype=np.int32).copy()) 
        pointNumber = queue.get() 
        parents     = torch.from_numpy(np.frombuffer(zlib.decompress(queue.get()), dtype=np.int32).copy()).cuda() 
        shs_vq.load_state_dict(torch.load(io.BytesIO(zlib.decompress(queue.get())))) 
        # skybox:
        skybox_number = queue.get() 
        sky_box_means3d     = torch.from_numpy((np.frombuffer(zlib.decompress(queue.get()), dtype=np.float32)).reshape((skybox_number, 3)).copy()).cuda()
        sky_box_opacity     = torch.from_numpy((np.frombuffer(zlib.decompress(queue.get()), dtype=np.float32)).reshape((skybox_number, 1)).copy()).cuda()
        sky_box_rotations   = torch.from_numpy((np.frombuffer(zlib.decompress(queue.get()), dtype=np.float32)).reshape((skybox_number, 4)).copy()).cuda()
        sky_box_scales      = torch.from_numpy((np.frombuffer(zlib.decompress(queue.get()), dtype=np.float32)).reshape((skybox_number, 3)).copy()).cuda()
        sky_box_shs         = torch.from_numpy((np.frombuffer(zlib.decompress(queue.get()), dtype=np.float32)).reshape((skybox_number, 16, 3)).copy()).cuda()
        indices_cur, features_cur, shs_cur, length, _ = self.Decompress(queue, shs_vq, first_trail=True) 
        print(indices_cur.shape, features_cur.shape, shs_cur.shape, length) 
        length_size = length * 2 
        means3D         = torch.zeros(length_size, 3, dtype=torch.float).cuda() 
        opacities       = torch.zeros(length_size, 1, dtype=torch.float).cuda() 
        rotations       = torch.zeros(length_size, 4, dtype=torch.float).cuda() 
        scales          = torch.zeros(length_size, 3, dtype=torch.float).cuda() 
        shs             = torch.zeros(length_size, 16, 3, dtype=torch.float).cuda() 
        boxes           = torch.zeros(length_size, 2, 4, dtype=torch.float).cuda() 
        back_pointer    = torch.zeros(length_size, dtype=torch.int).cuda()
        
        render_indices  = torch.zeros(length_size, dtype=torch.int).cuda() 
        least_recently  = torch.zeros(length_size, dtype=torch.int).cuda()
        starts          = torch.full((pointNumber, ), -1, dtype=torch.int).cuda() 
        featureMaxx = subgraph_tree_init( 
            length, indices_cur, features_cur, shs_cur, 
            starts, means3D, opacities, rotations, scales, shs, 
            boxes, back_pointer 
        ) 
        # Evaluation 
        psnr_test   = 0.0 
        ssims       = 0.0 
        lpipss      = 0.0 
        overTime    = 0 
        # print(f"Rendering process start:: ", flush=True) 
        frame_index = 0 
        # render_isOk.set() 
        preview = camera_queue.get() 
        # create output folder if it doesn't exit 
        # if not os.path.exists(args.out_dir):
        #     os.makedirs(args.out_dir)
        #     print(f"Create folder {args.out_dir}.") 
        # with open(args.log_file, "w")as fout:
        #     pass 
        # if args.train_test_exp and os.path.exists(os.path.join("/workspace/data/h_3dgs/exposure.json")): 
        #     with open("/workspace/data/h_3dgs/exposure.json", "r") as f:
        #         exposures = json.load(f) 
        #     pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        # else:
        #     pretrained_exposures = None 
        with open(f"{args.log_file}/elapse.txt", "w")as fout:
            pass 
        while not end_signal.is_set(): 
            viewpoint = None 
            try:
                viewpoint = camera_queue.get(timeout=5) 
            except queues.Empty: 
                # print("Timeout exception: No data received within 5 seconds.")
                overTime += 1 
                if (overTime >= 12):
                    print("Timeout exception: No data received within 1 minute.")
                    end_signal.set() 
                    break 
                else:
                    continue 
            except Exception as e:
                print("Render exception[1]: ", e) 
                break 
            # viewpoints = generate(preview, currentview)
            # for viewpoint in viewpoints:
            if viewpoint is None:
                continue 
            frame_index += 1 
            overTime = 0 
            update_elapse = 0 
            if frame_index % self.shared_data["window_size"] == 1:
                indices_cur, features_cur, shs_cur, length, decom_elapse = self.Decompress(queue, shs_vq) 
                print(indices_cur.shape, features_cur.shape, shs_cur.shape, length) 
                print("Update tree start") 
                featureMaxx, ud_elapse = subgraph_update( 
                    length, indices_cur, features_cur, shs_cur, 
                    starts, means3D, opacities, rotations, scales, shs, boxes, back_pointer, 
                    least_recently, self.shared_data["window_size"], featureMaxx
                ) 
                update_elapse = decom_elapse + ud_elapse 
            print("Update tree end", featureMaxx, f"{update_elapse:.5f}") 
            if featureMaxx > length_size:
                print("error") 
                exit(-1) 
            viewpoint.world_view_transform  = viewpoint.world_view_transform.cuda() 
            viewpoint.projection_matrix     = viewpoint.projection_matrix.cuda() 
            full_proj_transform   = viewpoint.world_view_transform @ viewpoint.projection_matrix 
            camera_center         = viewpoint.world_view_transform.inverse()[3, :3].cuda() 
            # 2. precompute 
            tanfovx = math.tan(self.shared_data["FoVx"] * 0.5) 
            tanfovy = math.tan(self.shared_data["FoVy"] * 0.5) 
            threshold = (2 * (self.shared_data["tau"] + 0.5)) * tanfovx / (0.5 * self.shared_data["image_width"]) 
            # print("subgraph_expand") 
            to_render, expand_elapse = subgraph_expand( 
                starts, parents, 
                depth_count, 
                means3D, 
                boxes, 
                threshold, 
                camera_center, 
                True, 
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
            sh  = torch.cat([shs[indices], sky_box_shs], dim = 0).contiguous() 
            opa = torch.cat([opacities[indices], sky_box_opacity], dim = 0).contiguous() 
            raster_settings = GaussianRasterizationSettings( 
                image_height=int(self.shared_data["image_height"]),
                image_width=int(self.shared_data["image_width"]),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
                scale_modifier=1.0,
                viewmatrix=viewpoint.world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=3,
                campos=camera_center, 
                prefiltered=False,
                debug=False,
                render_indices=torch.Tensor([]).int(),
                parent_indices=torch.Tensor([]).int(),
                interpolation_weights=torch.Tensor([]).float(), 
                num_node_kids=torch.Tensor([]).int(),
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
            with open(f"{args.log_file}/elapse.txt", "a+")as fout:
                fout.write(f"{update_elapse:.5f}, {expand_elapse:.5f}, {elapse:.5f}\n") 
            # if args.train_test_exp and pretrained_exposures is not None:
            #     try:
            #         exposure = pretrained_exposures[viewpoint.image_name] 
            #         image = torch.matmul(image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
            #     except Exception as e:
            #         # print(f"Exposures should be optimized in single. Missing exposure for image {viewpoint.image_name}")
            #         pass 
            # image = image.clamp(0, 1) 
            # image_name = os.path.join(args.out_dir, viewpoint.image_name)
            # try:
            #     # torchvision.utils.save_image(image, os.path.join(args.out_dir, viewpoint.image_name.split(".")[0] + ".png"))
            #     torchvision.utils.save_image(image, os.path.join(args.out_dir, image_name))
            # except:
            #     os.makedirs(os.path.dirname(os.path.join(args.out_dir, image_name)), exist_ok=True)
            #     torchvision.utils.save_image(image, os.path.join(args.out_dir, image_name)) 
            # if args.eval:
            #     gt_image, alpha_mask = read_image(viewpoint.image_name, images_dir=args.images, masks_dir=args.alpha_masks, resolution_scale=args.resolution) 
            #     if alpha_mask != None:
            #         gt_image    *= alpha_mask
            #         image       *= alpha_mask 
            #     psnr_test_ = psnr(image, gt_image).mean().double() 
            #     ssims_ = ssim(image, gt_image).mean().double() 
            #     lpipss_ = lpips(image, gt_image, net_type='vgg').mean().double()
            #     psnr_test += psnr_test_ 
            #     ssims += ssims_ 
            #     lpipss += lpipss_ 
            # with open(args.log_file, "a+") as fout:
                # _elapse = update_elapse + expand_elapse + elapse 
                # fout.write(f"Image_name = {viewpoint.image_name}: \n{psnr_test_:.5f}\n{ssims_:.5f}\n") #, {lpipss_:.5f}\n") 
                # fout.write(f"{update_elapse:.5f}\n{expand_elapse:.5f}\n{elapse:.5f}\n---\n")
                # for elapse_brk in elapse_breakdown: 
                #     fout.write(f"{elapse_brk:.5f}\n") 
            #     fout.write(f"{_elapse:.5f}\n") 
            #     fout.write(f"{to_render}, {num_rendered}\n") 
            # with open(args.log_file, "a+") as fout:
            #     # fout.write(f"frame_index = {frame_index}: psnr = {psnr_test_:.5f}, ssim = {ssims_:.5f}, lpi = {lpipss_:.5f}\n") 
            #     fout.write(f"image_name = {viewpoint.image_name}: psnr = {psnr_test_:.5f}, ssim = {ssims_:.5f}, lpi = {lpipss_:.5f}\n") 
            # print(f"image_name = {viewpoint.image_name}: psnr = {psnr_test_}, ssim = {ssims_}, lpips = {lpipss_}\n") 
            # # print(f"frame_index = {frame_index}: psnr_avg = {psnr_test_}, ssim_avg = {ssims_}, lpips_avg = {lpipss_}\n") 
            # print(f"to_render={to_render}, num_render={num_rendered}", flush=True) 
        # with open(args.log_file, "a+") as fout:
            # fout.write(f"psnr = {psnr_test/frame_index:.5f}, ssim = {ssims/frame_index:.5f}, lpi = {lpipss/frame_index:.5f}\n") 
        end_signal.set() 

if __name__ == "__main__":
    # Set up command line argument parser 
    parser = ArgumentParser(description="Rendering script parameters") 
    parser.add_argument("--ip", type=str, default='10.147.18.182') 
    parser.add_argument('--port', type=int, default=50000) 
    parser.add_argument("--viewpointFilePath", type=str, default="") 
    parser.add_argument("--images", type=str, default="images") 
    parser.add_argument("--alpha_masks", type=str, default="") 
    parser.add_argument('--out_dir', type=str, default="") 
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--tt_mode", type=int, default=0) 
    parser.add_argument("--tau", type=float, default=6.0) 
    parser.add_argument("--resolution", type=int, default=-1) 
    parser.add_argument("--eval", action="store_true") 
    parser.add_argument("--train_test_exp", action="store_true") 
    parser.add_argument("--window_size", type=int, default=16)
    args = parser.parse_args(sys.argv[1:]) 
    mp.set_start_method("spawn", force=True) 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client_socket.connect((args.ip, args.port)) 
    manager = mp.Manager() 
    tensor_queue = mp.Queue() 
    camera_queue = mp.Queue() 
    c = client(manager, args.tau, ws=args.window_size) 
    parent_conn1, child_conn1 = mp.Pipe() 
    parent_conn2, child_conn2 = mp.Pipe() 
    end_signal = Event() 
    render_isOk = Event()  
    receive_isOk = Event() 
    c_send = mp.Process(target=c.send, args=(end_signal, render_isOk, receive_isOk, camera_queue, child_conn1, args.viewpointFilePath, )) 
    c_send.start() 
    c_receive = mp.Process(target=c.receive, args=(end_signal, receive_isOk, tensor_queue, child_conn2, args, )) 
    c_receive.start() 
    c_render = mp.Process(target=c.render, args=(end_signal, render_isOk, tensor_queue, camera_queue, args, )) 
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