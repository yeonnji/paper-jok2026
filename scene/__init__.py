#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import cv2
from tqdm import tqdm
from utils.image_utils import load_mask_tensor

def generate_glow_points(core_points, core_colors, num_points_per_core=50, spread_factor=0.1):
    if len(core_points) == 0:
        return None, None
    num_core_points = len(core_points)
    num_total_glow = num_core_points * num_points_per_core
    repeated_core_points = np.repeat(core_points, num_points_per_core, axis=0)
    repeated_core_colors = np.repeat(core_colors, num_points_per_core, axis=0)
    if num_core_points > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(core_points)
        distances, _ = tree.query(core_points, k=2)
        avg_dist = np.mean(distances[:, 1])
        spread = avg_dist * spread_factor
    else:
        spread = spread_factor
    offsets = np.random.normal(0, spread, (num_total_glow, 3))
    glow_points = repeated_core_points + offsets
    glow_colors = repeated_core_colors
    return glow_points, glow_colors

class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        elif "C3VD" in args.source_path:
            scene_info=sceneLoadTypeCallbacks["colon"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        print("Loading masks for training...")
        for resolution_scale in resolution_scales:
            for cam in self.train_cameras[resolution_scale]:
                if hasattr(cam, 'soft_mask_path') and cam.soft_mask_path and os.path.exists(cam.soft_mask_path):
                    cam.soft_mask = load_mask_tensor(cam.soft_mask_path, cam.image_height, cam.image_width)
                else:
                    cam.soft_mask = None
            for cam in self.test_cameras[resolution_scale]:
                if hasattr(cam, 'soft_mask_path') and cam.soft_mask_path and os.path.exists(cam.soft_mask_path):
                    cam.soft_mask = load_mask_tensor(cam.soft_mask_path, cam.image_height, cam.image_width)
                else:
                    cam.soft_mask = None
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            ### MODIFIED BLOCK START ###
            print("Starting Physics-Based Initialization with Class Tagging...")
            pcd = scene_info.point_cloud
            
            # --- 1. Tagging Core Light Source Points ---
            # Initialize all points as Surface (class 0)
            initial_class_tags = np.zeros((len(pcd.points),), dtype=np.int8)

            print("Tagging core light source points (class 1) based on masks...")
            points3D = pcd.points
            points3D_hom = np.hstack((points3D, np.ones((len(points3D), 1))))

            for cam_info in tqdm(scene_info.train_cameras, desc="Projecting points to masks"):
                if not cam_info.soft_mask_path or not os.path.exists(cam_info.soft_mask_path):
                    continue
                
                mask_img = cv2.imread(cam_info.soft_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue

                R = cam_info.R; T = cam_info.T
                w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = -R.T @ T
                FovX = cam_info.FovX; FovY = cam_info.FovY
                znear, zfar = 0.01, 100.0
                tanHalfFovY = np.tan(FovY / 2); tanHalfFovX = np.tan(FovX / 2)
                proj = np.zeros((4,4)); proj[0, 0] = 1 / tanHalfFovX; proj[1, 1] = 1 / tanHalfFovY; proj[2, 2] = zfar / (zfar - znear); proj[3, 2] = -(zfar * znear) / (zfar - znear); proj[2, 3] = 1.
                full_proj_transform = proj @ w2c
                points_cam = (full_proj_transform @ points3D_hom.T).T
                points_ndc = points_cam[:, :3] / points_cam[:, 3, None]
                points_pixel_x = (points_ndc[:, 0] * 0.5 + 0.5) * cam_info.width
                points_pixel_y = (-points_ndc[:, 1] * 0.5 + 0.5) * cam_info.height
                
                in_frustum_mask = (points_cam[:, 3] > 0) & (points_pixel_x >= 0) & (points_pixel_x < cam_info.width) & (points_pixel_y >= 0) & (points_pixel_y < cam_info.height)
                px = points_pixel_x[in_frustum_mask].astype(int); py = points_pixel_y[in_frustum_mask].astype(int)
                mask_values = mask_img[py, px]
                is_in_mask_region = mask_values > 0
                original_indices = np.where(in_frustum_mask)[0]
                light_indices = original_indices[is_in_mask_region]
                initial_class_tags[light_indices] = 1 # Tag as Core Light Source

            num_core_lights = np.sum(initial_class_tags == 1)
            print(f"Tagged {num_core_lights} points as core light sources (class 1).")

            # --- 2. Pre-generating Glow Points (class 2) ---
            if num_core_lights > 0:
                print("Pre-generating glow points (class 2) around core light sources...")
                core_points = pcd.points[initial_class_tags == 1]
                core_colors = pcd.colors[initial_class_tags == 1]
                
                num_points_per_core = 50; spread_factor = 0.05
                
                glow_points, glow_colors = generate_glow_points(core_points, core_colors, num_points_per_core, spread_factor)
                
                if glow_points is not None:
                    augmented_points = np.vstack((pcd.points, glow_points))
                    augmented_colors = np.vstack((pcd.colors, glow_colors))
                    augmented_normals = np.zeros_like(augmented_points)
                    augmented_pcd = BasicPointCloud(points=augmented_points, colors=augmented_colors, normals=augmented_normals)
                    
                    # Create the final class tags array
                    glow_tags = np.full((len(glow_points),), 2, dtype=np.int8) # Tag new points as Glow
                    final_class_tags = np.concatenate((initial_class_tags, glow_tags))
                    
                    print(f"Added {len(glow_points)} new glow points. Total points: {len(augmented_points)}.")
                    self.gaussians.create_from_pcd(augmented_pcd, scene_info.train_cameras, self.cameras_extent, initial_class_tags=final_class_tags)
                else:
                    self.gaussians.create_from_pcd(pcd, scene_info.train_cameras, self.cameras_extent, initial_class_tags=initial_class_tags)
            else:
                self.gaussians.create_from_pcd(pcd, scene_info.train_cameras, self.cameras_extent, initial_class_tags=initial_class_tags)
            ### MODIFIED BLOCK END ###
            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]