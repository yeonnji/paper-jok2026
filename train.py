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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def get_reg_scale(iteration, reg_start_iter, reg_ramp_iters):
    """Returns a [0, 1] scalar for ramping up regularization weight.

    - Before reg_start_iter: 0 (no regularization)
    - Between reg_start_iter and reg_start_iter+reg_ramp_iters: linear ramp
    - After ramp: 1 (full weight)
    """
    if iteration <= reg_start_iter:
        return 0.0
    if reg_ramp_iters <= 0:
        return 1.0
    ramp_progress = (iteration - reg_start_iter) / float(reg_ramp_iters)
    return min(ramp_progress, 1.0)


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, "
                 "please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    debug_log_path = os.path.join(scene.model_path, "class_debug_log.txt")
    debug_log_file = open(debug_log_path, 'w')

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init,
                                        opt.depth_l1_weight_final,
                                        max_steps=opt.iterations)

    # -- Thesis method flags --
    use_weight_map = getattr(opt, "use_weight_map", False)
    use_light_reg  = getattr(opt, "use_light_reg", False)
    beta           = getattr(opt, "beta_weight_map", 0.8)
    lambda_ssim    = getattr(opt, "lambda_ssim", 0.2)
    lambda_o       = getattr(opt, "lambda_opacity_reg", 0.05)
    lambda_s       = getattr(opt, "lambda_scale_reg", 0.05)
    reg_start_iter = getattr(opt, "reg_start_iter", 15_000)
    reg_ramp_iters = getattr(opt, "reg_ramp_iters", 0)
    use_legacy     = getattr(opt, "use_legacy_light_losses", False)
    lambda_geo_reg = getattr(opt, "lambda_geo_reg", 2.0)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0

    log_file_path = os.path.join(scene.model_path, "results.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    with open(log_file_path, 'w') as f:
        f.write(f"results for {scene.model_path}\n")
        f.write(f"use_weight_map={use_weight_map}  use_light_reg={use_light_reg}  "
                f"beta={beta}  lambda_ssim={lambda_ssim}  lambda_o={lambda_o}  "
                f"lambda_s={lambda_s}  reg_start_iter={reg_start_iter}  "
                f"reg_ramp_iters={reg_ramp_iters}  use_legacy={use_legacy}\n")

    best_model = {'iter': 0, 'test_avg_psnr': 0}
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress")

    for iteration in range(first_iter + 1, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            pass

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        try:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        except IndexError:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3,), device="cuda") if opt.random_background else background

        # Need alpha_light when weight map or legacy geo loss is active
        need_alpha = use_weight_map or use_light_reg or use_legacy
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            compute_light_alpha=need_alpha)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        alpha_light = render_pkg["alpha_light"]

        gt_image = viewpoint_cam.original_image.cuda()

        # ----------------------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------------------
        # G_light is defined as all Gaussians with gauss_class > 0 (Core + Glow).
        # See scene/gaussian_model.py: class 0 = Surface, 1 = Core, 2 = Glow.

        has_mask = (use_weight_map
                    and hasattr(viewpoint_cam, 'soft_mask')
                    and viewpoint_cam.soft_mask is not None)

        if has_mask:
            # Thesis Module 2 -- pixel-wise weighted L1
            # w(i,j) = 1 + beta * M(i,j)
            # L_weight = mean( w(i,j) * |I_render(i,j) - I_gt(i,j)| )
            M = viewpoint_cam.soft_mask.to("cuda")          # [1, H, W], values in [0,1]
            w = 1.0 + beta * M                               # [1, H, W]
            Ll1 = (w * torch.abs(image - gt_image)).mean()
        else:
            # Standard L1 (baseline 3DGS path)
            Ll1 = l1_loss(image, gt_image)

        # SSIM term -- L_SSIM = 1 - SSIM(I_render, I_gt)
        L_ssim = 1.0 - ssim(image, gt_image)

        # Thesis total reconstruction loss:
        #   L_total = L_weight + lambda_ssim * L_SSIM  (+ regularization terms below)
        # When use_weight_map=False this reduces to standard L1 + lambda_ssim*DSSIM,
        # which is identical to the original 3DGS loss when lambda_ssim == lambda_dssim.
        loss = Ll1 + lambda_ssim * L_ssim

        # ----------------------------------------------------------------
        # Thesis Module 3 -- Light attribute regularization
        # Applied only after reg_start_iter, with optional linear ramp.
        # G_light = {k : gauss_class_k > 0}  (Core + Glow)
        # L_o = mean_{k in G_light}( alpha_k * (1 - alpha_k) )
        # L_s = mean_{k in G_light}( ||s_k||_1 )
        # ----------------------------------------------------------------
        L_o_val = 0.0
        L_s_val = 0.0

        if use_light_reg:
            reg_scale = get_reg_scale(iteration, reg_start_iter, reg_ramp_iters)
            if reg_scale > 0.0:
                gauss_classes = gaussians.get_gauss_class
                light_mask = (gauss_classes > 0)  # G_light: Core(1) + Glow(2)

                if light_mask.sum() > 0:
                    light_opacities = gaussians.get_opacity[light_mask]  # sigmoid-activated
                    L_o_tensor = (light_opacities * (1.0 - light_opacities)).mean()
                    loss += reg_scale * lambda_o * L_o_tensor
                    L_o_val = L_o_tensor.item()

                    light_scales = gaussians.get_scaling[light_mask]     # exp-activated
                    L_s_tensor = light_scales.abs().mean()
                    loss += reg_scale * lambda_s * L_s_tensor
                    L_s_val = L_s_tensor.item()

        # ----------------------------------------------------------------
        # Legacy losses (NOT in paper; disabled by default)
        # Enable with --use_legacy_light_losses for backward-compat runs.
        # ----------------------------------------------------------------
        if use_legacy:
            gauss_classes = gaussians.get_gauss_class
            lambda_geo_reg_val = getattr(opt, "lambda_geo_reg", 2.0)
            lambda_core_op    = getattr(opt, "lambda_core_op_sustain", 0.02)
            lambda_glow_op    = getattr(opt, "lambda_glow_op_reg", 0.05)

            # Legacy geo loss: penalise light Gaussians outside mask region
            if hasattr(viewpoint_cam, 'soft_mask') and viewpoint_cam.soft_mask is not None and alpha_light is not None:
                gt_mask = viewpoint_cam.soft_mask.to("cuda")
                inverted_mask = 1.0 - gt_mask
                loss += lambda_geo_reg_val * (alpha_light * inverted_mask).mean()

            # Legacy core opacity sustain (encourages core toward opacity=1)
            core_mask = (gauss_classes == 1)
            if core_mask.sum() > 0:
                core_opacities = gaussians.get_opacity[core_mask]
                loss += lambda_core_op * (1.0 - core_opacities).mean()

            # Legacy glow opacity target (MSE toward 0.3)
            glow_mask = (gauss_classes == 2)
            if glow_mask.sum() > 0:
                glow_opacities = gaussians.get_opacity[glow_mask]
                loss += lambda_glow_op * ((glow_opacities - 0.3) ** 2).mean()

        # ----------------------------------------------------------------
        # Depth supervision (unchanged from original)
        # ----------------------------------------------------------------
        Ll1depth = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_tensor = depth_l1_weight(iteration) * torch.abs(
                (invDepth - mono_invdepth) * depth_mask).mean()
            loss += Ll1depth_tensor
            Ll1depth = Ll1depth_tensor.item()

        loss.backward()

        # Gradient clipping for training stability (not a substitute for L_s)
        with torch.no_grad():
            if gaussians._scaling.grad is not None:
                gaussians._scaling.grad.clamp_(-0.1, 0.1)
            if gaussians._rotation.grad is not None:
                gaussians._rotation.grad.clamp_(-0.1, 0.1)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # Debug log every 500 iterations
            if iteration % 500 == 1:
                gauss_classes = gaussians.get_gauss_class
                n_surface = (gauss_classes == 0).sum().item()
                n_core    = (gauss_classes == 1).sum().item()
                n_glow    = (gauss_classes == 2).sum().item()
                debug_log_file.write(f"\n--- Debug [Iter {iteration}] ---\n")
                debug_log_file.write(f"Surface/Core/Glow: {n_surface}/{n_core}/{n_glow}\n")
                debug_log_file.write(f"L_o={L_o_val:.6f}  L_s={L_s_val:.6f}\n")
                debug_log_file.write(f"has_mask={has_mask}  "
                                     f"reg_scale={get_reg_scale(iteration, reg_start_iter, reg_ramp_iters):.4f}\n")
                debug_log_file.flush()

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "L_o": f"{L_o_val:.5f}",
                    "L_s": f"{L_s_val:.5f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None),
                            dataset.train_test_exp)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0 or
                        (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    debug_log_file.close()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test   = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                   compute_light_alpha=False,
                                   use_trained_exp=train_test_exp)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image    = image[...,    image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + f"_view_{viewpoint.image_name}/render",
                            image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + f"_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None], global_step=iteration)
                    l1_test   += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                      f"L1 {l1_test:.6f}  PSNR {psnr_test:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations",       nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations",       nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,  default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
