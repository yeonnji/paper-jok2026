import torch
import os
import sys
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from torchvision.utils import save_image

def render_light_maps(model_path, source_path):
    # ModelParams와 PipelineParams를 초기화하기 위한 파서 생성
    parser = ArgumentParser(description="Render script parser")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    ## MODIFIED ##
    # 렌더링에 필요한 인자를 딕셔너리로 정의
    arg_dict = {
        "source_path": source_path,
        "model_path": model_path,
        "images": "images",
        "eval": True,
        "depths": "",
        "train_test_exp": False,
        "resolution": -1,
        "data_device": "cuda" # 누락되었던 인자 추가
    }
    ##############

    # 딕셔너리를 Namespace 객체로 변환
    args_ns = Namespace(**arg_dict)

    # Namespace 객체를 extract 함수에 전달
    model_args = model.extract(args_ns)
    pipe_args = pipeline.extract(args_ns)
    
    # 모델 경로에서 파라미터 로드
    print(f"Loading model from: {model_path}")
    gaussians = GaussianModel(3)
    scene = Scene(model_args, gaussians, load_iteration=-1, shuffle=False)
    
    bg_color = [0, 0, 0] # 배경은 검은색
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 렌더링 결과 저장할 폴더 생성
    output_dir = os.path.join(model_path, "light_maps")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Light maps will be saved in: {output_dir}")

    # 테스트 카메라 중 일부 또는 전체를 렌더링
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("No test cameras found. Exiting.")
        return

    for idx, viewpoint_cam in enumerate(test_cameras):
        print(f"Rendering light map for: {viewpoint_cam.image_name}...")
        
        render_pkg = render(viewpoint_cam, gaussians, pipe_args, background, render_light_map=True)
        
        light_map_tensor = render_pkg["light_map"]

        if light_map_tensor is not None:
            output_filename = os.path.join(output_dir, f"{viewpoint_cam.image_name.split('.')[0]}_lightmap.png")
            save_image(light_map_tensor, output_filename)
            print(f"  -> Saved to {output_filename}")
        
        # if idx >= 4:
        #     break

if __name__ == "__main__":
    parser = ArgumentParser(description="Render light maps from a trained model.")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model directory (e.g., ./output/xxxxxxxxxx)")
    parser.add_argument("-s", "--source_path", required=True, help="Path to the original dataset directory (e.g., ./data/drjohnson)")
    args = parser.parse_args()

    safe_state(silent=True)
    render_light_maps(args.model_path, args.source_path)
    
    print("\n✅ Rendering complete.")