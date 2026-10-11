"""Isaac Sim 5.1 scene, RTX depth capture and Warp raycast lidar baseline."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from common import ROOT, DEFAULT_URDF, config, transform, urdf_frames, robot_pose_angles, perception_reference, trajectory, lidar_rays, sphere_mask, sphere_wire_curves, world_to_robot

parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true')
parser.add_argument('--frames', type=int, default=90)
parser.add_argument('--output', type=Path, default=ROOT/'outputs/baseline')
parser.add_argument('--config', type=Path)
parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
parser.add_argument('--view', choices=['camera','lidar','fused'], help='Display an existing nvblox reconstruction')
parser.add_argument('--voxel-min-z', type=float, default=0.08, help='Hide floor voxels in viewer only')
parser.add_argument('--positions-root', type=Path, default=ROOT/'outputs/positions', help='Directory containing position manifest')
parser.add_argument('--sphere-radius', type=float, help='Keep observed voxel centres within this radius of the virtual reference, metres')
parser.add_argument('--export-dir', type=Path, help='Separate directory for viewer screenshots, USD and sphere crop')
parser.add_argument('--live', action='store_true', help='Online rigid-body motion, sensor sampling and nvblox fusion')
parser.add_argument('--sensor-hz', type=float, default=10.0, help='Maximum live observation rate; measured rate is shown separately')
parser.add_argument('--motion-duration', type=float, default=45.0, help='Seconds for the automatic rigid-body route')
parser.add_argument('--motion-config', type=Path, default=ROOT/'config/motion.json')
parser.add_argument('--record', action='store_true', help='Record the live overview as MP4')
parser.add_argument('--archive-every', type=int, default=10, help='Save every Nth live body-frame surface snapshot')
parser.add_argument('--robot-pose',choices=['neutral','tpose'],default='neutral',help='Fixed whole-body joint configuration')
parser.add_argument('--inspection-angle',choices=['overview','oblique','front'],default='overview',help='Fixed environment overview (default), or body-relative close-up')
parser.add_argument('--start-paused',action='store_true',help='Live GUI: sample the initial pose once, then wait for interaction')
parser.add_argument('--reference-frame',choices=['sensor_rig','pelvis'],default='sensor_rig',help='Anchor virtual coordinates to the rigid sensor torso, or pelvis')
parser.add_argument('--origin-z-offset',type=float,default=-.2,help='Virtual origin Z relative to zero-waist pelvis, metres; negative is down')
args = parser.parse_args()
if args.frames < 2:
    parser.error('--frames must be >= 2')
if args.sphere_radius is not None and (not np.isfinite(args.sphere_radius) or args.sphere_radius<=0):
    parser.error('--sphere-radius must be finite and positive')
if args.live and (not np.isfinite(args.sensor_hz) or not 0<args.sensor_hz<=60 or
                  not np.isfinite(args.motion_duration) or args.motion_duration<=0 or args.archive_every<1):
    parser.error('Live mode requires 0 < sensor-hz <= 60, positive motion-duration and archive-every >= 1')
if args.live and args.view:
    parser.error('--live and --view are separate modes')
if args.start_paused and (args.headless or not args.live):
    parser.error('--start-paused is only supported by the live GUI')
if not np.isfinite(args.origin_z_offset):
    parser.error('--origin-z-offset must be finite')
from isaacsim import SimulationApp
app = SimulationApp({'headless':args.headless, 'width':1280, 'height':900, 'anti_aliasing':0})

import omni.usd
from pxr import UsdGeom, UsdLux, Gf, Vt
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from scipy.spatial.transform import Rotation
from common import environment_mesh, robot_meshes, combined_mesh
import warp as wp
import open3d as o3d

@wp.kernel
def scan_rays(env_mesh: wp.uint64, body_mesh: wp.uint64,
              directions: wp.array(dtype=wp.vec3), sensor_pos: wp.vec3,
              sensor_rot: wp.mat33, pelvis_pos: wp.vec3, pelvis_inv: wp.mat33,
              max_range: float, output: wp.array(dtype=float)):
    i = wp.tid()
    d = sensor_rot * directions[i]
    hit = wp.mesh_query_ray(env_mesh, sensor_pos, d, max_range)
    body_hit = wp.mesh_query_ray(body_mesh, pelvis_inv*(sensor_pos-pelvis_pos), pelvis_inv*d, max_range)
    value = float(0.0)
    if hit.result:
        value = hit.t
        if body_hit.result and body_hit.t < hit.t:
            value = 0.0
    output[i] = value

def usd_mesh(stage, path, mesh):
    p = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    p.CreateFaceVertexCountsAttr([3]*len(triangles))
    p.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    p.CreateSubdivisionSchemeAttr('none')
    colors = np.asarray(mesh.vertex_colors)
    p.CreateDisplayColorAttr([Gf.Vec3f(*colors[0])])
    return p

def warp_mesh(mesh):
    return wp.Mesh(points=wp.array(np.asarray(mesh.vertices,dtype=np.float32),dtype=wp.vec3,device='cuda:0'),
                   indices=wp.array(np.asarray(mesh.triangles,dtype=np.int32).ravel(),dtype=wp.int32,device='cuda:0'))

def set_rig(rig, t):
    xf = UsdGeom.Xformable(rig)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))

def voxels_usd(stage, filename, min_z, center=None, radius=None):
    v = np.load(filename)
    centers = v['centers']
    keep = centers[:,2]>=min_z
    if radius is not None:
        keep &= sphere_mask(centers,center,radius)
    centers = centers[keep]
    vs = float(v['voxel_size'])
    inst = UsdGeom.PointInstancer.Define(stage,'/World/Reconstruction')
    cube = UsdGeom.Cube.Define(stage,'/World/Reconstruction/Cube')
    cube.CreateSizeAttr(vs*0.9)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.16,0.68,0.83)])
    inst.CreatePrototypesRel().SetTargets([cube.GetPath()])
    inst.CreateProtoIndicesAttr([0]*len(centers))
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(centers.astype(np.float32)))
    print(f'Viewing {len(centers)} actual nvblox TSDF surface voxels ({vs*100:.0f} cm)',flush=True)
    return len(centers),len(v['centers'])

def sphere_boundary_usd(stage,center,radius,visible=True):
    if radius is None:
        stage.RemovePrim('/World/SphereBoundary')
        return
    curves=sphere_wire_curves(center,radius)
    wire=UsdGeom.BasisCurves.Define(stage,'/World/SphereBoundary')
    wire.CreateTypeAttr('linear')
    wire.CreateWrapAttr('nonperiodic')
    wire.CreateCurveVertexCountsAttr([len(c) for c in curves])
    wire.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(curves).astype(np.float32)))
    wire.CreateWidthsAttr([0.006])
    wire.SetWidthsInterpolation('constant')
    wire.CreateDisplayColorAttr([Gf.Vec3f(0.95,0.52,0.12)])
    wire.MakeVisible() if visible else wire.MakeInvisible()

def inspection_view(center,radius,yaw=0,front=False):
    rot=Rotation.from_euler('z',yaw).as_matrix()
    if front:
        return np.asarray(center)+rot@np.array([3.1*max(radius,1.0),0,.15]),np.asarray(center)
    eye=np.asarray(center)+rot@np.array([1.7,-2.3,1.6])*max(radius,1.0)
    target=np.asarray(center)+rot@np.array([0.25,0,0])
    return eye,target

def point_camera(camera,pos,target):
    forward=np.asarray(target)-pos
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    q=Rotation.from_matrix(np.column_stack([right,down,forward])).as_quat()
    camera.set_world_pose(pos,q[[3,0,1,2]],camera_axes='ros')

def overview_view(cfg):
    """One fixed oblique view fitted to the environment, independent of G1."""
    lower=[];upper=[]
    for obj in cfg['objects']:
        center=np.asarray(obj['center'],dtype=float)
        extent=np.asarray(obj['size'],dtype=float)/2 if obj['kind']=='box' else np.array([obj['radius'],obj['radius'],obj['height']/2])
        lower.append(center-extent);upper.append(center+extent)
    low=np.min(lower,axis=0);high=np.max(upper,axis=0)
    target=(low+high)/2
    span=max(float(np.max(high[:2]-low[:2])),4.)
    eye=target+1.12*span*np.array([.8,-1.,1.1])
    return eye,target

def overview_camera(cfg=None):
    camera=Camera('/World/Overview',resolution=(1280,900))
    camera.set_focal_length(2.0)
    eye,target=overview_view(cfg or config())
    point_camera(camera,eye,target)
    return camera

def main():
    meta_file=args.output/'metadata.json'
    cfg=json.loads(meta_file.read_text())['config'] if args.view and args.config is None and meta_file.exists() else config(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    world=World(stage_units_in_meters=1.0,physics_dt=1/60,rendering_dt=1/60)
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    light=UsdLux.DomeLight.Define(stage,'/World/Light')
    light.CreateIntensityAttr(800)
    env=environment_mesh(cfg)
    joint_angles=robot_pose_angles(args.robot_pose)
    robot=robot_meshes(args.urdf,joint_angles)
    UsdGeom.Xform.Define(stage,'/World/Environment')
    for name,mesh in env:
        usd_mesh(stage,'/World/Environment/'+name,mesh)
    rig=UsdGeom.Xform.Define(stage,'/World/G1Rig').GetPrim()
    for name,mesh in robot:
        usd_mesh(stage,'/World/G1Rig/'+name,mesh)
    frames=urdf_frames(args.urdf,joint_angles)
    # URDF body frame +X forward +Y left +Z up -> optical +X right +Y down +Z forward.
    body_to_optical=np.eye(4)
    body_to_optical[:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    t_p_c=frames['d435_link']@body_to_optical
    t_p_l=frames['mid360_link']
    reference=perception_reference(args.urdf,args.reference_frame,args.origin_z_offset,joint_angles)
    t_p_f=reference['T_P_F']
    first=trajectory(cfg,args.frames)[0]
    initial=transform([first[0],first[1],cfg['pelvis_height_m']],[0,0,np.deg2rad(first[2])])
    set_rig(rig,initial)
    overview_eye,overview_target=overview_view(cfg)
    set_camera_view(eye=overview_eye.tolist(),target=overview_target.tolist())
    o3d.io.write_triangle_mesh(str(args.output/'ground_truth.ply'),combined_mesh(env))

    if args.live:
        from types import SimpleNamespace
        from live_sim import run_live
        api=SimpleNamespace(set_rig=set_rig,warp_mesh=warp_mesh,scan_rays=scan_rays,
            overview_camera=overview_camera,overview_view=overview_view,point_camera=point_camera,
            inspection_view=inspection_view,sphere_boundary_usd=sphere_boundary_usd)
        run_live(args,app,world,stage,cfg,rig,env,robot,t_p_c,t_p_l,api,reference)
        return

    if args.view:
        sphere_enabled=[args.sphere_radius is not None]
        sphere_radius=[args.sphere_radius or 1.2]
        current_pose=[initial]
        initial_reference=initial@t_p_f
        center=initial_reference[:3,3]
        min_z=-0.1 if sphere_enabled[0] else args.voxel_min_z
        counts=voxels_usd(stage,args.output/f'{args.view}_voxels.npz',min_z,center,args.sphere_radius)
        sphere_boundary_usd(stage,center,args.sphere_radius)
        scene_overlay=args.inspection_angle=='overview'
        environment_display=UsdGeom.Imageable(stage.GetPrimAtPath('/World/Environment'))
        environment_display.MakeVisible() if scene_overlay else environment_display.MakeInvisible()
        overview=overview_camera(cfg)
        if sphere_enabled[0] and args.inspection_angle!='overview':
            eye,target=inspection_view(center,sphere_radius[0],np.deg2rad(first[2]),args.inspection_angle=='front')
            point_camera(overview,eye,target)
            set_camera_view(eye=eye.tolist(),target=target.tolist())
        world.reset()
        overview.initialize()
        for _ in range(20):world.step(render=True)
        import imageio.v3 as imageio
        export_dir=args.export_dir or args.output
        export_dir.mkdir(parents=True,exist_ok=True)
        suffix=f'_sphere_r{args.sphere_radius:g}' if sphere_enabled[0] else ''
        imageio.imwrite(export_dir/f'{args.view}{suffix}_isaac_voxels.png',np.asarray(overview.get_rgba())[...,:3])
        stage.Export(str(export_dir/f'{args.view}{suffix}_voxel_scene.usda'))
        if sphere_enabled[0]:
            source=np.load(args.output/f'{args.view}_voxels.npz')
            mask=sphere_mask(source['centers'],center,sphere_radius[0])
            local=world_to_robot(source['centers'][mask],initial_reference)
            np.savez_compressed(export_dir/f'{args.view}{suffix}_voxels.npz',
                centers=local,centers_robot=local,centers_reference=local,centers_world=source['centers'][mask],
                coordinate_frame=reference['coordinate_frame'],reference_anchor=reference['anchor_link'],
                tsdf=source['tsdf'][mask],weight=source['weight'][mask],voxel_size=source['voxel_size'],
                radius_m=np.float64(sphere_radius[0]),center_world=center,T_W_P=initial,T_W_F=initial_reference,
                T_P_F=t_p_f,T_anchor_F=reference['T_anchor_F'],origin_z_offset_m=args.origin_z_offset)
        if args.headless:
            return
        import omni.ui as ui
        panel=ui.Window('G1 / nvblox inspection',width=460,height=520)
        current_directory=[args.output]
        current_mode=[args.view]
        display_floor=[sphere_enabled[0]]
        display_boundary=[True]
        status_label=[None]
        def update_display():
            stage.RemovePrim('/World/Reconstruction')
            radius=sphere_radius[0] if sphere_enabled[0] else None
            center=(current_pose[0]@t_p_f)[:3,3]
            count,total=voxels_usd(stage,current_directory[0]/f'{current_mode[0]}_voxels.npz',
                -0.1 if display_floor[0] else args.voxel_min_z,center,radius)
            sphere_boundary_usd(stage,center,radius,display_boundary[0])
            if status_label[0] is not None:
                text=f'R = {radius:.2f} m | {count:,} / {total:,} surface voxels' if radius is not None else f'Full map | {count:,} / {total:,} surface voxels'
                status_label[0].text=text
        def focus_robot():
            if args.inspection_angle=='overview':
                point_camera(overview,overview_eye,overview_target)
                set_camera_view(eye=overview_eye.tolist(),target=overview_target.tolist())
                return
            pose=current_pose[0]@t_p_f
            yaw=np.arctan2(pose[1,0],pose[0,0])
            eye,target=inspection_view(pose[:3,3],sphere_radius[0],yaw,args.inspection_angle=='front')
            set_camera_view(eye=eye.tolist(),target=target.tolist())
        def select_mode(mode):
            current_mode[0]=mode
            update_display()
        def show_ground(model):
            imageable=UsdGeom.Imageable(stage.GetPrimAtPath('/World/Environment'))
            imageable.MakeVisible() if model.as_bool else imageable.MakeInvisible()
        def show_floor(model):
            display_floor[0]=model.as_bool
            select_mode(current_mode[0])
        def enable_sphere(model):
            sphere_enabled[0]=model.as_bool
            update_display()
            if sphere_enabled[0] and args.inspection_angle!='overview':focus_robot()
        def change_radius(model):
            sphere_radius[0]=model.as_float
            update_display()
        def show_boundary(model):
            display_boundary[0]=model.as_bool
            update_display()
        presets=[('Baseline trajectory',ROOT/'outputs/baseline')]
        manifest=args.positions_root/'manifest.json'
        if manifest.exists():
            for n,position in enumerate(json.loads(manifest.read_text())['positions'],1):
                directory=args.positions_root/position['directory']
                if all((directory/f'{mode}_voxels.npz').exists() for mode in ['camera','lidar','fused']):
                    x,y,yaw=position['xy_yaw_deg']
                    presets.append((f'{n}. {position["id"]} ({x:.2f}, {y:.2f}, {yaw:.0f} deg)',directory))
        selected=next((n for n,(_,directory) in enumerate(presets) if directory.resolve()==args.output.resolve()),0)
        def select_position(model,item):
            index=model.get_item_value_model().as_int
            directory=presets[index][1]
            metadata=json.loads((directory/'metadata.json').read_text())
            position_cfg=metadata['config']
            x,y,yaw=position_cfg['trajectory_xy_yaw_deg'][0]
            pose=transform([x,y,position_cfg['pelvis_height_m']],[0,0,np.deg2rad(yaw)])
            set_rig(rig,pose)
            current_pose[0]=pose
            current_directory[0]=directory
            select_mode(current_mode[0])
            if sphere_enabled[0]:focus_robot()
            print(f'POSITION_SELECTED {directory.name} x={x} y={y} yaw={yaw}',flush=True)
        with panel.frame:
            with ui.VStack(spacing=8):
                ui.Label('Observed TSDF surface voxels')
                ui.Label(f'Reference: {reference["coordinate_frame"]}, Z offset {args.origin_z_offset:+.2f} m')
                ui.Label('Observer position (independent maps)')
                position_combo=ui.ComboBox(selected,*[label for label,_ in presets])
                position_combo.model.add_item_changed_fn(select_position)
                with ui.HStack():
                    for mode in ['camera','lidar','fused']:
                        ui.Button(mode,clicked_fn=lambda m=mode:select_mode(m))
                with ui.HStack():
                    ui.Label('Crop to virtual-reference ball')
                    sphere_checkbox=ui.CheckBox()
                    sphere_checkbox.model.set_value(sphere_enabled[0])
                    sphere_checkbox.model.add_value_changed_fn(enable_sphere)
                ui.Label('Sphere radius (m)')
                radius_slider=ui.FloatSlider(min=min(0.5,sphere_radius[0]),max=max(3.0,sphere_radius[0]),step=0.05)
                radius_slider.model.set_value(sphere_radius[0])
                radius_slider.model.add_value_changed_fn(change_radius)
                with ui.HStack():
                    ui.Label('Show orange sphere boundary')
                    boundary_checkbox=ui.CheckBox()
                    boundary_checkbox.model.set_value(True)
                    boundary_checkbox.model.add_value_changed_fn(show_boundary)
                ui.Button('Restore overview' if args.inspection_angle=='overview' else 'Focus on robot',clicked_fn=focus_robot)
                status_label[0]=ui.Label('')
                with ui.HStack():
                    ui.Label('Overlay scene ground truth')
                    checkbox=ui.CheckBox()
                    checkbox.model.set_value(scene_overlay)
                    checkbox.model.add_value_changed_fn(show_ground)
                with ui.HStack():
                    ui.Label('Show ground surface voxels')
                    floor_checkbox=ui.CheckBox()
                    floor_checkbox.model.set_value(display_floor[0])
                    floor_checkbox.model.add_value_changed_fn(show_floor)
                ui.Label('Blue = observed surface; orange = crop boundary.')
                ui.Label('Blank regions may be unknown, not necessarily free.')
                ui.Label('Ideal depth / approximate MID360 / exact poses')
        update_display()
        while app.is_running():
            world.step(render=True)
        return

    c=cfg['camera']
    camera=Camera('/World/D435i',resolution=(c['width'],c['height']),frequency=60)
    # Isaac's accessor uses USD tenths of a stage unit. A tiny focalLength
    # gets clamped by RTX; use 10 mm USD focalLength with the desired ratio.
    camera.set_focal_length(1.0)
    camera.set_horizontal_aperture(2*np.tan(np.deg2rad(c['horizontal_fov_deg'])/2))
    # Preserve square pixels: actual vertical FOV derives from 16:9 resolution.
    camera.set_clipping_range(0.01,20)
    overview=overview_camera(cfg)
    world.reset()
    camera.initialize();camera.add_distance_to_image_plane_to_frame()
    overview.initialize()
    k=np.asarray(camera.get_intrinsics_matrix(),dtype=np.float32)
    rays,valid_angle=lidar_rays(cfg)
    # Stock visual head is a closed shell with no sensor apertures; it encloses
    # the MID360 origin. Exclude that shell from self masking, keep all limbs.
    self_mesh=[(name,mesh) for name,mesh in robot if not name.startswith('head_link_')]
    env_wp=warp_mesh(combined_mesh(env));body_wp=warp_mesh(combined_mesh(self_mesh))
    ray_wp=wp.array(rays.reshape(-1,3),dtype=wp.vec3,device='cuda:0')
    ranges_wp=wp.zeros(rays.shape[0]*rays.shape[1],dtype=float,device='cuda:0')
    u,v=np.meshgrid(np.arange(c['width'])+0.5,np.arange(c['height'])+0.5)
    cam_rays=np.stack([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones_like(u)],axis=-1)
    cam_norm=np.linalg.norm(cam_rays,axis=-1)
    cam_ray_wp=wp.array((cam_rays/cam_norm[...,None]).reshape(-1,3).astype(np.float32),dtype=wp.vec3,device='cuda:0')
    cam_range_wp=wp.zeros(c['height']*c['width'],dtype=float,device='cuda:0')
    rng=np.random.default_rng(42)
    possible=np.flatnonzero(valid_angle)
    depth_frames=[];lidar_frames=[];cam_poses=[];lidar_poses=[];rig_poses=[];rgb_frames=[]
    actual_fov=2*np.rad2deg(np.arctan(c['height']/(2*k[1,1])))
    print(f'Capture: D435i {k.tolist()}, actual vertical FOV {actual_fov:.2f} deg',flush=True)
    for i,pose in enumerate(trajectory(cfg,args.frames)):
        t_w_p=transform([pose[0],pose[1],cfg['pelvis_height_m']],[0,0,np.deg2rad(pose[2])])
        set_rig(rig,t_w_p)
        t_w_c=t_w_p@t_p_c;t_w_l=t_w_p@t_p_l
        q=Rotation.from_matrix(t_w_c[:3,:3]).as_quat()
        camera.set_world_pose(t_w_c[:3,3],q[[3,0,1,2]],camera_axes='ros')
        # Settle render pipeline at fixed pose; every archived pose matches its image.
        for _ in range(5 if i else 20):
            world.step(render=True)
        depth=camera.get_depth()
        if depth is None or depth.size != c['height']*c['width']:
            raise RuntimeError('Isaac depth annotator produced no image')
        depth=np.asarray(depth,dtype=np.float32).copy()
        depth[~np.isfinite(depth)|(depth<c['min_depth_m'])|(depth>c['max_depth_m'])]=0
        wp.launch(scan_rays,dim=len(cam_ray_wp),inputs=[env_wp.id,body_wp.id,cam_ray_wp,
            wp.vec3(*t_w_c[:3,3]),wp.mat33(t_w_c[:3,:3].astype(np.float32)),
            wp.vec3(*t_w_p[:3,3]),wp.mat33(t_w_p[:3,:3].T.astype(np.float32)),
            20.0,cam_range_wp],device='cuda:0')
        expected_z=cam_range_wp.numpy().reshape(depth.shape)/cam_norm
        if i==0:
            valid_compare=(expected_z>0)&(depth>0)
            agreement=float(np.mean(np.abs(depth[valid_compare]-expected_z[valid_compare])<0.08))
            print(f'Camera RTX/raycast geometry agreement: {agreement:.1%}',flush=True)
            if agreement<0.85:
                raise RuntimeError('Camera intrinsics or pose mismatch: RTX depth disagrees with geometric raycast')
        # Reject own-body pixels, and catch stale images or frame-convention errors.
        depth[(expected_z<=0)|(np.abs(depth-expected_z)>0.08)]=0
        wp.launch(scan_rays,dim=len(ray_wp),inputs=[env_wp.id,body_wp.id,ray_wp,
            wp.vec3(*t_w_l[:3,3]),wp.mat33(t_w_l[:3,:3].astype(np.float32)),
            wp.vec3(*t_w_p[:3,3]),wp.mat33(t_w_p[:3,:3].T.astype(np.float32)),
            cfg['lidar']['max_range_m'],ranges_wp],device='cuda:0')
        ranges=ranges_wp.numpy().reshape(valid_angle.shape)
        sampled=np.zeros(ranges.size,dtype=bool)
        sampled[rng.choice(possible,min(len(possible),cfg['lidar']['samples_per_frame']),replace=False)]=True
        ranges[~sampled.reshape(ranges.shape)|(ranges<cfg['lidar']['min_range_m'])]=0
        if not np.any(ranges):
            raise RuntimeError('Lidar has zero valid returns; check pose and robot self occlusion')
        depth_frames.append(depth);lidar_frames.append(ranges.copy())
        cam_poses.append(t_w_c);lidar_poses.append(t_w_l);rig_poses.append(t_w_p)
        if i in [0,args.frames//2,args.frames-1]:
            rgb_frames.append((i,np.asarray(camera.get_rgba())[...,:3].copy()))
        if i==0:
            import imageio.v3 as imageio
            imageio.imwrite(args.output/'scene.png',np.asarray(overview.get_rgba())[...,:3])
        if i%10==0 or i==args.frames-1:
            print(f'Frame {i+1}/{args.frames}, camera valid {np.mean(depth>0):.1%}, lidar hits {np.count_nonzero(ranges)}',flush=True)
    np.savez_compressed(args.output/'capture.npz',depth=np.asarray(depth_frames),lidar=np.asarray(lidar_frames),
        T_W_C=np.asarray(cam_poses,dtype=np.float32),T_W_L=np.asarray(lidar_poses,dtype=np.float32),
        T_W_P=np.asarray(rig_poses,dtype=np.float32),K=k,lidar_directions=rays,frame_period_s=np.float32(0.1))
    for i,rgb in rgb_frames:
        imageio.imwrite(args.output/f'camera_{i:03d}.png',rgb)
    metadata={'config':cfg,'urdf':str(args.urdf),'frames':args.frames,
        'robot_pose':args.robot_pose,'joint_angles_rad':joint_angles,
        'T_P_C':t_p_c.tolist(),'T_P_L':t_p_l.tolist(),'camera_vertical_fov_deg':actual_fov,
        'assumptions':['RTX ideal pinhole axial depth, no active stereo/noise/dropout model',
        'MID360 FOV/rate approximation: seeded random angular cells, not true Livox scan law',
        'robot self returns masked with static URDF visuals except closed head shell (no sensor aperture)',
        'camera self mask validated against raycast geometry; >8cm axial disagreement rejected',
        f'fixed-joint kinematic G1 ({args.robot_pose}), no locomotion dynamics; exact sensor poses',
        'offline 10 Hz nominal sample sequence; render settle steps are not physical sensor time',
        'no SLAM, drift, motion distortion or timestamp offset']}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    stage.Export(str(args.output/'scene.usda'))
    print(f'CAPTURE_COMPLETE {args.output}',flush=True)

try:
    main()
finally:
    app.close()
