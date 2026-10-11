"""Validate moving body-frame exports against the fixed simulation geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from common import ROOT,transform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/live_demo')
    args=parser.parse_args()
    directory=args.output
    history=[json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
    assert np.all(np.diff([item['sensor_time_s'] for item in history])>0),'Capture timestamps must increase'
    cfg=json.loads((directory/'scene_config.json').read_text())
    metadata=json.loads((directory/'metadata.json').read_text())
    frame_name=metadata.get('coordinate_frame','pelvis')
    t_p_f=np.asarray(metadata.get('T_P_F',np.eye(4)))
    waypoints=json.loads((directory/'motion_config.json').read_text())['waypoints']
    _,x,y,yaw=waypoints[0]
    initial=transform([x,y,cfg['pelvis_height_m']],[0,0,np.deg2rad(yaw)])@t_p_f
    scene=o3d.t.geometry.RaycastingScene()
    mesh=o3d.io.read_triangle_mesh(str(directory/'ground_truth.ply'))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances=[];lattice_error=[];counts=[];poses=[];checks=0
    for item in history:
        if item['archive'] is None:continue
        with np.load(directory/item['archive']) as data:
            pose_key='T_O_F' if 'T_O_F' in data.files else 'T_O_B'
            points=data['centers'];pose=data[pose_key];radius=float(data['radius_m']);voxel=float(data['voxel_size'])
            assert str(data['coordinate_frame'])==frame_name
            assert 'centers_world' not in data.files
            assert np.array_equal(points,data['centers_robot'])
            assert points.shape==(len(data['tsdf']),3)
            assert np.isfinite(points).all()
            assert np.all(np.linalg.norm(points,axis=1)<=radius+1e-5)
            assert np.all(data['weight']>.1)
            assert np.all(np.abs(data['tsdf'])<=.75*voxel+1e-6)
            assert int(data['sequence_id'])==item['sequence_id']
            assert np.max(np.abs(pose-np.asarray(item[pose_key])))<1e-8
            if pose_key=='T_O_F':
                assert np.allclose(pose,data['T_O_P']@data['T_P_F'])
                assert np.allclose(data['T_P_F'],t_p_f)
                assert np.allclose(data['T_F_C'],metadata['T_F_C'])
                assert np.allclose(data['T_F_L'],metadata['T_F_L'])
                assert np.array_equal(data['centers_reference'],points)
            assert np.max(np.abs(pose[:3,:3].T@pose[:3,:3]-np.eye(3)))<1e-6
            # Transform back to fixed local odometry. Centres must fall on the
            # SAME nvblox lattice after translation AND yaw, rather than being
            # erroneously carried along with the body.
            odometry=points@pose[:3,:3].T+pose[:3,3]
            grid=odometry/voxel-.5
            if len(points):lattice_error.append(float(np.max(np.abs(grid-np.rint(grid)))))
            # Absolute coordinates appear only in this ground-truth validation.
            # The mapper and policy-facing exports use relative poses / points.
            absolute=odometry@initial[:3,:3].T+initial[:3,3]
            if len(points):
                distance=scene.compute_distance(o3d.core.Tensor(absolute.astype(np.float32))).numpy()
                distances.extend(distance.tolist())
            counts.append(len(points));poses.append(pose);checks+=1
    assert checks>=2,'Need at least two archived observations'
    assert max(lattice_error)<2e-4,'Body / odometry transform or voxel lattice mismatch'
    translation=np.asarray(poses)[:,:3,3]
    assert np.linalg.norm(np.ptp(translation,axis=0))>.1,'Robot did not translate'
    yaw=np.unwrap(np.arctan2(np.asarray(poses)[:,1,0],np.asarray(poses)[:,0,0]))
    assert np.ptp(yaw)>.1,'Robot did not turn'
    distance=np.asarray(distances)
    assert np.quantile(distance,.95)<.075,'Surface ghosts or misaligned moving frames'
    assert np.mean(distance<=.075)>.97,'Too many reconstructed points away from real surfaces'
    result={'passed':True,'checked_snapshots':checks,'checked_surface_points':len(distance),
        'coordinate_frame':frame_name,'radius_m':metadata['radius_m'],
        'origin_z_offset_m':metadata.get('origin_z_offset_m',0),
        'max_fixed_grid_error_voxels':max(lattice_error),
        'voxel_center_to_gt_mean_m':float(distance.mean()),
        'voxel_center_to_gt_p95_m':float(np.quantile(distance,.95)),
        'fraction_within_7_5cm':float(np.mean(distance<=.075)),
        'odometry_translation_span_m':np.ptp(translation,axis=0).tolist(),
        'yaw_span_deg':float(np.rad2deg(np.ptp(yaw))),
        'minimum_surface_voxels':min(counts),'maximum_surface_voxels':max(counts)}
    (directory/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
