import os
import json
import cv2
import numpy as np
import torch
from typing import List, Tuple
from utils import subdivide_route


def _to_list(data):
    if hasattr(data, 'cpu'):
        return data.cpu().detach().numpy().tolist()
    return data

# === visualizer helpers ===
from visualizer import (
    draw_lane,
    draw_map_bev,
    draw_vehicle_bev,
    draw_cam_view,
    combine_images,
    combine_cam_view_veh_bev
)

############################################################
# Helper utilities
############################################################

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [ensure_list(i) for i in x]
    return x

############################################################
# Demo generator (visual style identical to visualize_info)
############################################################

def run_demo(case_id: str,
             img_root: str = "/data/NuScene/v1.0-trainval/",
             output_root: str = "/artifact/demo",
             width: int = 800,
             height: int = 600,
             pixel_per_meter: float = 5.0,
             fps: int = 30,
             show: bool = False):
    """Render visualization frames for <case_id> and save as a 30fps video."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # TODO: Remove postfix of the filename since they are already separated by folders
    base_split = os.path.join(img_root, "sweeps/CAM_FRONT/split", case_id)
    src_json   = os.path.join(base_split, f"{case_id}_culane.json")
    olra_json  = os.path.join("/artifact/val/olra", case_id, "predicted_routes.json")
    olra_osm_json  = os.path.join("/artifact/val/olra_osm", case_id, "predicted_routes.json")
    olra_wo_dir_json  = os.path.join("/artifact/val/olra_wo_dir", case_id, "predicted_routes.json")
    raw_json   = os.path.join("/artifact/val/raw_gps", case_id, "predicted_routes.json")
    sensor_json  = os.path.join("/artifact/val/sensor", case_id, "predicted_routes.json")
    perfect_json  = os.path.join("/artifact/val/perfect_gps", case_id, "predicted_routes.json")
    openpilot_json   = os.path.join("/artifact/val/openpilot", case_id, "predicted_routes.json")

    # ------------------------------------------------------------------
    # Load JSONs
    # ------------------------------------------------------------------
    with open(src_json, "r") as f:
        src = json.load(f)
    with open(olra_json, "r") as f:
        pred_olra = json.load(f)
    with open(olra_osm_json, "r") as f:
        pred_olra_osm = json.load(f)
    with open(olra_wo_dir_json, "r") as f:
        pred_olra_wo_dir = json.load(f)
    with open(raw_json, "r") as f:
        pred_raw  = json.load(f)
    with open(sensor_json, "r") as f:
        pred_sensor = json.load(f)
    with open(perfect_json, "r") as f:
        pred_perfect = json.load(f)
    with open(openpilot_json, "r") as f:
        pred_openpilot  = json.load(f)

    if not (len(pred_olra) == len(pred_raw) == len(src["frame_data"])):
        raise ValueError("Frame count mismatch among JSON files")

    # static camera / route data
    bev_to_img_rot_t   = torch.tensor(src["ego2cam_rot"], dtype=torch.float32)
    bev_to_img_trans_t = torch.tensor(src["ego2cam_trans"], dtype=torch.float32)
    calib_mat_t        = torch.tensor(src["calib_mat"], dtype=torch.float32)
    en_route           = subdivide_route(torch.tensor(src["route_osm"]))

    # prepare output video
    out_dir = os.path.join(output_root)
    ensure_dir(out_dir)
    video_fp = os.path.join(out_dir, f"{case_id}.mp4")
    writer = None

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------
    for idx, frame in enumerate(src["frame_data"]):
        img_path = os.path.join(img_root, frame.get("filename", ""))
        if not os.path.isfile(img_path):
            print(f"[WARN] skip frame {idx}: missing image {img_path}")
            continue
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"[WARN] skip frame {idx}: cannot read {img_path}")
            continue

        # draw lanes
        cam_view_base = draw_lane(orig_img, frame["predicted_lanes"] )

        # prepare OLRA / RAW data
        veh_route_olra_np = np.asarray(pred_olra[idx]["veh_route"], dtype=np.float64)
        veh_route_olra_osm_np = np.asarray(pred_olra_osm[idx]["veh_route"], dtype=np.float64)
        veh_route_olra_wo_dir_np = np.asarray(pred_olra_wo_dir[idx]["veh_route"], dtype=np.float64)
        veh_route_raw_np  = np.asarray(pred_raw[idx]["veh_route"], dtype=np.float64)
        veh_route_sensor_np = np.asarray(pred_sensor[idx]["veh_route"], dtype=np.float64)
        veh_route_perfect_np  = np.asarray(pred_perfect[idx]["veh_route"], dtype=np.float64)
        veh_route_openpilot_np  = np.asarray(pred_openpilot[idx]["veh_route"], dtype=np.float64)
        if len(veh_route_olra_np) == 0:
            continue

        pose_perfect = ensure_list(pred_perfect[idx]["pose"])
        if len(pose_perfect) == 2:
            pose_perfect += [0.0]
        pose_olra = ensure_list(pred_olra[idx]["pose"])
        if len(pose_olra) == 2:
            pose_olra += [0.0]
        pose_olra_osm = ensure_list(pred_olra_osm[idx]["pose"])
        if len(pose_olra_osm) == 2:
            pose_olra_osm += [0.0]
        pose_olra_wo_dir = ensure_list(pred_olra_wo_dir[idx]["pose"])
        if len(pose_olra_wo_dir) == 2:
            pose_olra_wo_dir += [0.0]
        pose_sensor = ensure_list(pred_sensor[idx]["pose"])
        if len(pose_sensor) == 2:
            pose_sensor += [0.0]

        pose_raw  = ensure_list(pred_raw[idx]["pose"])
        if len(pose_raw) == 2:
            pose_raw += [0.0]

        veh_route_perfect_t = torch.tensor(veh_route_perfect_np, dtype=torch.float32)
        veh_route_olra_t = torch.tensor(veh_route_olra_np, dtype=torch.float32)
        veh_route_olra_osm_t = torch.tensor(veh_route_olra_osm_np, dtype=torch.float32)
        veh_route_olra_wo_dir_t = torch.tensor(veh_route_olra_wo_dir_np, dtype=torch.float32)
        veh_route_sensor_t = torch.tensor(veh_route_sensor_np, dtype=torch.float32)
        veh_route_raw_t  = torch.tensor(veh_route_raw_np, dtype=torch.float32)
        veh_route_openpilot_t = torch.tensor(veh_route_openpilot_np, dtype=torch.float32)

        # draw BEV / map
        map_px = 600
        map_w = map_px
        map_h = map_px
        map_range_m = 60.0
        ppm_map = map_w / (2 * map_range_m)

        map_bev = draw_map_bev({
            'poses': [
                {'pose': pose_perfect, 'color': [0,0,255], 'radius':5, 'heading_is_valid':1, 'name':'GT'},
                #{'pose': pose_olra, 'color': [255,255,0], 'radius':5, 'heading_is_valid':1, 'name':'OLRA with NuScene Route'},
                {'pose': pose_olra_osm,  'color': [0,255,0], 'radius':5, 'heading_is_valid':1, 'name': 'OLRA'},
                #{'pose': pose_olra_wo_dir,  'color': [128, 255, 128], 'radius':5, 'heading_is_valid':1, 'name': 'OLRA wo Direction Weights'},
                {'pose': pose_sensor,  'color': [0,255,255], 'radius':5, 'heading_is_valid':1, 'name': 'Sensor Only'},
                {'pose': pose_raw,  'color': [128, 128, 128], 'radius':5, 'heading_is_valid':0, 'name': 'Raw GPS'},
            ],        
            'route': {
            'points': _to_list(en_route),
            'color': [128, 128, 128],
            'radius': 5,
            'thickness': 3
            },
            'width': map_w,
            'height': map_h,
            'pixel_per_meter': ppm_map,
            'center_pose': [
                en_route[(len(en_route)//2)+1][0].item() if hasattr(en_route[(len(en_route)//2)+1][0], 'item') else en_route[(len(en_route)//2)+1][0],
                en_route[(len(en_route)//2)+1][1].item() if hasattr(en_route[(len(en_route)//2)+1][1], 'item') else en_route[(len(en_route)//2)+1][1]
            ],
        })

        veh_w = 600
        veh_h = 800
        forward_m = 60.0
        backward_m = 20.0
        veh_bev = draw_vehicle_bev({
            'routes':[
                {'points':veh_route_perfect_np,  'color':[0,0,255], 'thickness':2, 'radius':0.3, 'name': 'GT'},
                #{'points':veh_route_olra_np, 'color':[255,255,0], 'thickness':2, 'radius':0.3, 'name': 'OLRA with NuScene Route'},
                {'points':veh_route_olra_osm_np, 'color':[0,255,0], 'thickness':2, 'radius':0.3, 'name': 'OLRA'},
                #{'points':veh_route_olra_wo_dir_np, 'color':[128, 255, 128], 'thickness':2, 'radius':0.3, 'name': 'OLRA wo Direction Weights'},
                {'points':veh_route_sensor_np, 'color':[0, 255, 255], 'thickness':2, 'radius':0.3, 'name': 'Sensor Only'},
                {'points':veh_route_raw_np,  'color':[128, 128, 128], 'thickness':2, 'radius':0.3, 'name': 'Raw GPS'},
                #{'points':veh_route_openpilot_np,  'color':[255,0, 0], 'thickness':2, 'radius':0.3, 'name': 'OP-DeepDive'}
            ],
            'width': veh_w,
            'height': veh_h,
            'backward_meters': backward_m,
            'forward_meters': forward_m,
            'clip_distance': 5.0,
        })

        cam_view = draw_cam_view(
            cam_view_base, bev_to_img_rot_t, bev_to_img_trans_t, calib_mat_t,
            {'routes': [
                {'route': veh_route_perfect_t,  'color':[0,0,255],'width':2,'radius':3},
                #{'route': veh_route_olra_t, 'color':[255,255,0],'width':2,'radius':3},
                {'route': veh_route_olra_osm_t, 'color':[0, 255, 0],'width':2,'radius':3},
                #{'route': veh_route_olra_wo_dir_t, 'color':[128, 255, 128],'width':2,'radius':3},
                {'route': veh_route_sensor_t, 'color':[0, 255, 255],'width':2,'radius':3},
                {'route': veh_route_raw_t,  'color':[128, 128, 128],'width':2,'radius':3},
                #{'route': veh_route_openpilot_t,  'color':[255, 0, 0],'width':2,'radius':3}
            ]}
        )

        # combine images
        combined = combine_images(cam_view, veh_bev, map_bev)
        # If only openpilot and olra are compared, ignore the map bev.
        #combined = combine_cam_view_veh_bev(cam_view, veh_bev)

        # initialize writer on first frame
        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_fp, fourcc, fps, (w, h))

        # write frame to video
        writer.write(combined)
        print(f"Frame {idx:04d} written to video")

    # finalize
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Video saved to {video_fp}")

############################################################
# CLI entry
############################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizer demo (OLRA vs RAW) output as 30fps video")
    parser.add_argument("case_id", help="scene id, e.g. 0001")
    parser.add_argument("--img_root", default="/data/NuScene/v1.0-trainval/", help="NuScenes base path")
    parser.add_argument("--output",   default="/artifact/demo/olra-vs-sensor-only:better", help="output directory root")
    parser.add_argument("--fps", type=int, default=15, help="video framerate")
    parser.add_argument("--show", action="store_true", help="display frames interactively")
    args = parser.parse_args()

    # V.S. pilot
    # case_ids = ['0015', '0103', '0272', '0562', '0906', '0914', '0928', '0962', '0966', '1064']

    # Ablation study
    case_ids = ['0014', '0016', '0017']

    for case_id in case_ids:
        run_demo(case_id, args.img_root, args.output, show=args.show, fps=args.fps)
    #run_demo(args.case_id, args.img_root, args.output, show=args.show, fps=args.fps)

