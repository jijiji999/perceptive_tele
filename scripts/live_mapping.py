"""Persistent nvblox worker. IPC and output use local odometry and virtual frames.

The Isaac runtime and nvblox wheel require different torch ABIs. One atomic
request/response pair in a private temporary directory keeps frames matched,
provides back pressure, and avoids importing both runtimes into one process.
"""
import argparse
import json
import os
from pathlib import Path
import time
from importlib.metadata import version
import sys
import traceback

import numpy as np
from common import ROOT

sys.path.insert(0,str(ROOT/'.deps'))
import torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.sensor import Sensor
from nvblox_torch.indexing import get_local_voxel_center_grid


def atomic_npz(path, **arrays):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle,**arrays)
    temporary.replace(path)


def local_surface(mapper, voxel_size, pose, radius):
    """Bulk GPU extraction; sphere selection is in the current body frame."""
    blocks,indices=mapper.tsdf_layer_view().get_all_blocks()
    if not blocks:
        return np.empty((0,3),np.float32),np.empty((0,2),np.float32)
    values=torch.stack(blocks)
    origins=torch.stack(indices).to(device='cuda',dtype=torch.float32)*8*voxel_size
    centers=origins[:,None,None,None,:]+get_local_voxel_center_grid(voxel_size)
    rotation=torch.as_tensor(pose[:3,:3],device='cuda',dtype=torch.float32)
    translation=torch.as_tensor(pose[:3,3],device='cuda',dtype=torch.float32)
    body=(centers-translation)@rotation
    keep=(values[...,1]>.1)&(values[...,0].abs()<=.75*voxel_size)&((body*body).sum(-1)<=radius*radius)
    return body[keep].cpu().numpy(),values[keep][:,:2].cpu().numpy()


def main():
    parent_pid=os.getppid()
    parser=argparse.ArgumentParser()
    parser.add_argument('--ipc',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((args.ipc/'config.json').read_text())
    c,l=cfg['camera'],cfg['lidar']
    k=np.asarray(cfg['K'])
    camera=Sensor.from_camera(float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),c['width'],c['height'])
    if not version('nvblox_torch').startswith('0.0.10'):
        raise RuntimeError('Recheck pinned nvblox fromLidar argument order before upgrading')
    # See reconstruct.py: wheel 0.0.10 forwards these two arguments in reverse.
    lidar=Sensor.from_lidar(l['width'],l['height'],vertical_fov_rad=l['min_range_m'],
                            min_valid_range_m=np.deg2rad(2*l['max_elevation_deg']))
    params=MapperParams()
    integrator=params.get_projective_integrator_params()
    integrator.projective_integrator_max_integration_distance_m=c['max_depth_m']
    # A 3 m sensor-centric support range covers the default local sphere plus
    # sensor extrinsics and a TSDF band. The output radius is applied separately.
    integrator.lidar_projective_integrator_max_integration_distance_m=min(l['max_range_m'],cfg['integration_range_m'])
    integrator.projective_integrator_truncation_distance_vox=4.0
    params.set_projective_integrator_params(integrator)
    voxel=float(cfg['voxel_size_m'])
    mapper=Mapper(voxel_sizes_m=voxel,mapper_parameters=params)
    atomic_npz(args.ipc/'ready.npz',ready=True)
    request=args.ipc/'request.npz'
    # Isaac's fast shutdown can exit before Python finally blocks run. Do not
    # leave a CUDA mapper alive after the owning simulator has disappeared.
    while os.getppid()==parent_pid and not (args.ipc/'stop').exists():
        if not request.exists():
            time.sleep(.003)
            continue
        with np.load(request) as archive:
            data={key:archive[key].copy() for key in archive.files}
        request.unlink()
        if bool(data['reset']):
            mapper.clear()
        torch.cuda.synchronize()
        start=time.perf_counter()
        for image_key,pose_key,sensor in [('depth','T_O_C',camera),('lidar','T_O_L',lidar)]:
            image=torch.as_tensor(data[image_key],device='cuda',dtype=torch.float32).contiguous()
            pose=torch.from_numpy(data[pose_key]).float().contiguous()
            mapper.add_depth_frame(image,pose,sensor)
        torch.cuda.synchronize()
        integration_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        centers,values=local_surface(mapper,voxel,data['T_O_F'],float(data['radius_m']))
        torch.cuda.synchronize()
        extraction_ms=(time.perf_counter()-start)*1000
        atomic_npz(args.ipc/'response.npz',centers=centers,centers_robot=centers,centers_reference=centers,
            tsdf=values[:,0],weight=values[:,1],voxel_size=np.float32(voxel),
            surface_band_m=np.float32(.75*voxel),coordinate_frame=cfg['coordinate_frame'],
            reference_anchor=cfg['reference_anchor'],T_anchor_F=np.asarray(cfg['T_anchor_F']),
            T_F_C=np.asarray(cfg['T_F_C']),T_F_L=np.asarray(cfg['T_F_L']),
            origin_z_offset_m=cfg['origin_z_offset_m'],
            T_O_F=data['T_O_F'],T_O_P=data['T_O_P'],T_P_F=data['T_P_F'],
            radius_m=data['radius_m'],frame_id=data['frame_id'],
            sequence_id=data['sequence_id'],sensor_time_s=data['sensor_time_s'],
            trajectory_time_s=data['trajectory_time_s'],
            integration_ms=integration_ms,extraction_ms=extraction_ms,
            allocated_blocks=mapper.tsdf_layer_view().num_blocks())


if __name__=='__main__':
    try:
        main()
    except Exception:
        # Parent also checks exit status. Keep complete diagnostics in its log.
        traceback.print_exc()
        raise
