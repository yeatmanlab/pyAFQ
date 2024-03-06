import sys
import tempfile
import os.path as op
import argparse
import logging
from AFQ.data.fetch import download_hypvinn

current_script_dir = op.dirname(op.abspath(__file__))
fastsurfer_path = op.join(current_script_dir, "FastSurfer")
if fastsurfer_path not in sys.path:
    sys.path.append(fastsurfer_path)

try:
    from HypVINN.run_prediction import (
        get_prediction, load_volumes, set_up_cfgs)
    from HypVINN.inference import Inference
    from HypVINN.config.hypvinn_global_var import HYPVINN_CLASS_NAMES
except TypeError:
    raise ValueError("FastSurfer requires python 3.10 or higher")


logger = logging.getLogger('AFQ')


def run_hypvinn(t1, device="cpu"):
    ckpt_paths, cfg_paths = download_hypvinn()

    working_dir = op.dirname(t1)
    temp_dir = tempfile.gettempdir()

    args = argparse.Namespace(
        in_dir=working_dir,
        out_dir=temp_dir,
        sid='subject', log_name=temp_dir,
        orig_name=t1,
        t2=None, registration=True, qc_snapshots=False,
        reg_type='coreg', device=device, viewagg_device='auto',
        threads=8, batch_size=1, async_io=False, allow_root=False,
        ckpt_cor=ckpt_paths["coronal"],
        ckpt_ax=ckpt_paths["axial"],
        ckpt_sag=ckpt_paths["sagittal"],
        cfg_cor=cfg_paths["coronal"],
        cfg_ax=cfg_paths["axial"],
        cfg_sag=cfg_paths["sagittal"],
        t1=t1, mode='t1')

    view_ops = {}
    cfg_ax = set_up_cfgs(args.cfg_ax, args)
    view_ops["axial"] = {"cfg": cfg_ax, "ckpt": args.ckpt_ax}
    cfg_sag = set_up_cfgs(args.cfg_sag, args)
    view_ops["sagittal"] = {"cfg": cfg_sag, "ckpt": args.ckpt_sag}
    cfg_cor = set_up_cfgs(args.cfg_cor, args)
    view_ops["coronal"] = {"cfg": cfg_cor, "ckpt": args.ckpt_cor}

    modalities, affine, _, orig_zoom, orig_size = load_volumes(
        mode=args.mode, t1_path=args.t1, t2_path=args.t2)

    model = Inference(cfg=cfg_cor, args=args)

    pred_classes = get_prediction(
        args.sid, modalities, orig_zoom, model, gt_shape=orig_size,
        view_opts=view_ops, out_scale=None, mode=args.mode,
        logger=logger)

    return pred_classes, HYPVINN_CLASS_NAMES, affine
