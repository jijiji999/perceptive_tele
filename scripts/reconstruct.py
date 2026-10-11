"""Actual NVIDIA nvblox TSDF integration, independent of Isaac's torch runtime."""
import argparse
import json
from pathlib import Path
import sys
import time
from importlib.metadata import version
import numpy as np
from common import ROOT

sys.path.insert(0,str(ROOT/'.deps'))
import torch
import open3d as o3d
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.sensor import Sensor
from nvblox_torch.indexing import get_voxel_center_grids

def extract(mapper, voxel_size):
    blocks,indices=mapper.tsdf_layer_view().get_all_blocks()
    grids=get_voxel_center_grids(indices,voxel_size)
    centers=[];values=[]
    for b,g in zip(blocks,grids):
        # Unobserved TSDF defaults to 0: weight check is essential.
        keep=(b[...,1]>0.1)&(torch.abs(b[...,0])<=voxel_size*0.75)
        if keep.any():
            centers.append(g[keep].cpu().numpy())
            values.append(b[keep][:,:2].cpu().numpy())
    if not centers:
        raise RuntimeError('No observed TSDF surface voxels were reconstructed')
    return np.concatenate(centers),np.concatenate(values)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline')
    p.add_argument('--mode',choices=['all','camera','lidar','fused'],default='all')
    p.add_argument('--voxel-size',type=float,help='Override metadata voxel size, metres')
    args=p.parse_args()
    if args.voxel_size is not None and not 0.01<=args.voxel_size<=0.2:
        p.error('--voxel-size must be in [0.01, 0.2] m')
    data=np.load(args.output/'capture.npz')
    meta=json.loads((args.output/'metadata.json').read_text())
    cfg=meta['config'];c=cfg['camera'];l=cfg['lidar']
    voxel=args.voxel_size or cfg['voxel_size_m']
    k=data['K']
    camera=Sensor.from_camera(float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),c['width'],c['height'])
    # v0.0.10's PySensor::fromLidar forwards (fov, min_range) to a C++ Lidar
    # constructor taking (min_range, fov). Without this adapter, a 104-degree
    # scan is incorrectly fused as a 0.1-radian scan and walls collapse in Z.
    # Pin the affected wheel rather than silently carrying this into upgrades.
    nv_version=version('nvblox_torch')
    if not nv_version.startswith('0.0.10'):
        raise RuntimeError(f'Recheck fromLidar argument order for nvblox {nv_version}')
    lidar=Sensor.from_lidar(l['width'],l['height'],
        vertical_fov_rad=l['min_range_m'],
        min_valid_range_m=np.deg2rad(2*l['max_elevation_deg']))
    params=MapperParams()
    ip=params.get_projective_integrator_params()
    ip.projective_integrator_max_integration_distance_m=c['max_depth_m']
    ip.lidar_projective_integrator_max_integration_distance_m=l['max_range_m']
    ip.projective_integrator_truncation_distance_vox=4.0
    params.set_projective_integrator_params(ip)
    gt=o3d.io.read_triangle_mesh(str(args.output/'ground_truth.ply'))
    gt_rays=o3d.t.geometry.RaycastingScene()
    gt_rays.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
    metrics={}
    for mode in ['camera','lidar','fused'] if args.mode=='all' else [args.mode]:
        mapper=Mapper(voxel_sizes_m=float(voxel),mapper_parameters=params)
        torch.cuda.synchronize();start=time.perf_counter()
        for i in range(len(data['depth'])):
            if mode in ['camera','fused']:
                d=torch.as_tensor(data['depth'][i],dtype=torch.float32,device='cuda').contiguous()
                pose=torch.from_numpy(data['T_W_C'][i].copy()).float().contiguous()
                mapper.add_depth_frame(d,pose,camera)
            if mode in ['lidar','fused']:
                d=torch.as_tensor(data['lidar'][i],dtype=torch.float32,device='cuda').contiguous()
                pose=torch.from_numpy(data['T_W_L'][i].copy()).float().contiguous()
                mapper.add_depth_frame(d,pose,lidar)
            if i%20==0:
                print(f'{mode}: integrated {i+1}/{len(data["depth"])}',flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        centers,values=extract(mapper,voxel)
        np.savez_compressed(args.output/f'{mode}_voxels.npz',centers=centers,tsdf=values[:,0],
            weight=values[:,1],voxel_size=np.float32(voxel),surface_band_m=np.float32(voxel*.75))
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
        cloud.paint_uniform_color([0.16,0.68,0.83])
        o3d.io.write_point_cloud(str(args.output/f'{mode}_surface.ply'),cloud)
        mapper.update_color_mesh()
        mesh=mapper.get_color_mesh().to_open3d()
        if not mesh.is_empty():
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(args.output/f'{mode}_mesh.ply'),mesh)
        # Distance of observed surface voxel centres to true surfaces. This is
        # a quantized accuracy metric, not sensor accuracy or map completeness.
        distance=gt_rays.compute_distance(o3d.core.Tensor(centers.astype(np.float32))).numpy()
        metrics[mode]={'surface_voxels':len(centers),'voxel_size_m':voxel,
            'surface_band_m':voxel*.75,'voxel_center_to_gt_mean_m':float(distance.mean()),
            'voxel_center_to_gt_p95_m':float(np.quantile(distance,.95)),
            'fraction_within_1_voxel':float(np.mean(distance<=voxel)),
            'integration_wall_s':elapsed,'frames':len(data['depth']),
            'allocated_tsdf_blocks':mapper.tsdf_layer_view().num_blocks()}
        print(mode,json.dumps(metrics[mode]),flush=True)
        del mapper
        torch.cuda.empty_cache()
    previous=json.loads((args.output/'metrics.json').read_text()) if (args.output/'metrics.json').exists() else {}
    previous.update(metrics)
    (args.output/'metrics.json').write_text(json.dumps(previous,indent=2))
    print('NVBLOX_COMPLETE',flush=True)

if __name__=='__main__':
    main()
