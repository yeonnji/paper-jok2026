import torch
import os
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
import cv2
import numpy as np

"""
이 스크립트는 지정된 체크포인트(.pth)를 로드하여, 
'light_feature' 값이 설정된 임계값을 넘는 가우시안(고정된 가우시안)과
그렇지 않은 가우시안(활성 가우시안)을 분리하여 별도의 이미지로 렌더링합니다.
"""

def visualize(model_path, iteration, light_threshold, camera_index):
    # --- 1. 학습 설정 로드 및 보완 ---
    print("Reading configuration from training...")
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"Error: cfg_args file not found at {cfg_path}.")
        sys.exit(1)

    # 인자 파서 그룹들을 생성
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # [수정] 누락된 인자를 채우기 위한 로직 추가
    # 1. 먼저, 모든 파라미터의 기본값을 가진 Namespace를 생성합니다.
    full_default_args = parser.parse_args([])

    # 2. cfg_args 파일을 로드합니다.
    with open(cfg_path, 'r') as f:
        cfg_str = f.read()
    cfg_args_loaded = eval(cfg_str)

    # 3. 기본값 위에 로드한 값을 덮어써서 완전한 설정 객체를 만듭니다.
    #    이렇게 하면 cfg_args에 값이 없더라도 기본값이 사용됩니다.
    merged_args = full_default_args
    for key, value in vars(cfg_args_loaded).items():
        if hasattr(merged_args, key):
            setattr(merged_args, key, value)
    
    # 완성된 merged_args를 사용하여 파라미터 객체들 생성
    dataset = lp.extract(merged_args)
    dataset.eval = True
    opt = op.extract(merged_args)
    pipe = pp.extract(merged_args)

    # --- 2. 모델 및 Scene 로드 ---
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    checkpoint_path = os.path.join(model_path, "chkpnt" + str(iteration) + ".pth")
    print(f"Loading checkpoint: {checkpoint_path}")

    (model_params, _) = torch.load(checkpoint_path)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.restore(model_params, opt) # 이제 opt 객체에 모든 값이 있으므로 오류가 발생하지 않습니다.
    
    print("Loading scene...")
    scene = Scene(dataset, gaussians, shuffle=False)
    
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("Error: No test cameras found.")
        return
    if camera_index >= len(test_cameras):
        print(f"Error: Camera index {camera_index} is out of bounds.")
        return
    viewpoint_cam = test_cameras[camera_index]
    print(f"Rendering view from camera: {viewpoint_cam.image_name}")

    # --- 3. 시각화 로직 (이하 동일) ---
    original_opacity = gaussians._opacity.clone().detach()

    print("\n--- Rendering Frozen Gaussians ---")
    light_scores = gaussians.get_light_feature.detach()
    frozen_mask = (light_scores > light_threshold).squeeze()
    
    num_frozen = frozen_mask.sum().item()
    print(f"{num_frozen} Gaussians are identified as 'frozen'.")

    with torch.no_grad():
        gaussians._opacity.data[~frozen_mask] = -1e10

    frozen_img = render(viewpoint_cam, gaussians, pipe, background)["render"]
    frozen_img_np = (frozen_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    output_filename_frozen = f"frozen_gaussians_iter{iteration}_thr{light_threshold}.png"
    cv2.imwrite(output_filename_frozen, cv2.cvtColor(frozen_img_np, cv2.COLOR_RGB2BGR))
    print(f"Saved frozen render to: {output_filename_frozen}")

    print("\n--- Rendering Active Gaussians ---")
    with torch.no_grad():
        gaussians._opacity.data = original_opacity
        gaussians._opacity.data[frozen_mask] = -1e10

    active_img = render(viewpoint_cam, gaussians, pipe, background)["render"]
    active_img_np = (active_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    output_filename_active = f"active_gaussians_iter{iteration}_thr{light_threshold}.png"
    cv2.imwrite(output_filename_active, cv2.cvtColor(active_img_np, cv2.COLOR_RGB2BGR))
    print(f"Saved active render to: {output_filename_active}")
    print("\nVisualization complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualization script for frozen Gaussians")
    parser.add_argument('-m', '--model_path', type=str, required=True, help="Path to the model folder (e.g., ./output/experiment_name)")
    parser.add_argument('--iteration', type=int, default=15000, help="Iteration number of the checkpoint to load.")
    parser.add_argument('--threshold', type=float, default=0.85, help="Lightness threshold to identify frozen Gaussians.")
    parser.add_argument('--camera_index', type=int, default=0, help="Index of the test camera to render.")
    args = parser.parse_args()

    visualize(args.model_path, args.iteration, args.threshold, args.camera_index)