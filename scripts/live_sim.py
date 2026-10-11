"""Online rigid G1 experiment inside Isaac Sim; no prebuilt map is loaded."""
import collections
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT,transform,lidar_rays,combined_mesh,robot_pose_angles


def atomic_npz(path, **arrays):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle,**arrays)
    temporary.replace(path)


def run_live(args,app,world,stage,cfg,rig,env,robot,t_p_c,t_p_l,api,reference):
    import warp as wp
    import omni.ui as ui
    from pxr import UsdGeom,Gf,Vt
    from isaacsim.core.utils.viewports import set_camera_view
    import imageio.v3 as imageio

    output=args.output
    (output/'frames').mkdir(exist_ok=True)
    radius=float(args.sphere_radius or 1.2)
    t_p_f=reference['T_P_F']
    t_f_c=np.linalg.inv(t_p_f)@t_p_c
    t_f_l=np.linalg.inv(t_p_f)@t_p_l
    frame_name=reference['coordinate_frame']
    waypoint=np.asarray(json.loads(args.motion_config.read_text())['waypoints'],dtype=float)
    if waypoint.ndim!=2 or waypoint.shape[1]!=4 or len(waypoint)<2 or not np.isfinite(waypoint).all() or np.any(np.diff(waypoint[:,0])<=0):
        raise ValueError('Motion waypoints must be finite [time,x,y,yaw_deg] with increasing time')
    waypoint[:,0]=(waypoint[:,0]-waypoint[0,0])/(waypoint[-1,0]-waypoint[0,0])*args.motion_duration
    waypoint[:,3]=np.rad2deg(np.unwrap(np.deg2rad(waypoint[:,3])))

    def route_pose(t):
        x,y,yaw=[np.interp(t,waypoint[:,0],waypoint[:,i]) for i in range(1,4)]
        return transform([x,y,cfg['pelvis_height_m']],[0,0,np.deg2rad(yaw)])

    first_pose=route_pose(0)
    t_o_w=np.linalg.inv(first_pose@t_p_f)  # Arbitrary initial virtual frame; no global registration.
    pose=first_pose.copy()
    api.set_rig(rig,pose)

    # Visualization geometry is never used as a sensor target: both ideal
    # sensors raycast only the environment mesh, with separate body masking.
    env_wp=api.warp_mesh(combined_mesh(env))
    body_wp=api.warp_mesh(combined_mesh([(n,m) for n,m in robot if not n.startswith('head_link_')]))
    c,l=cfg['camera'],cfg['lidar']
    focal=c['width']/(2*np.tan(np.deg2rad(c['horizontal_fov_deg'])/2))
    k=np.array([[focal,0,c['width']/2],[0,focal,c['height']/2],[0,0,1]],np.float32)
    u,v=np.meshgrid(np.arange(c['width'])+.5,np.arange(c['height'])+.5)
    camera_rays=np.stack([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones_like(u)],-1)
    camera_norm=np.linalg.norm(camera_rays,axis=-1)
    camera_wp=wp.array((camera_rays/camera_norm[...,None]).reshape(-1,3).astype(np.float32),dtype=wp.vec3,device='cuda:0')
    camera_ranges=wp.zeros(c['height']*c['width'],dtype=float,device='cuda:0')
    rays,valid=lidar_rays(cfg)
    lidar_wp=wp.array(rays.reshape(-1,3),dtype=wp.vec3,device='cuda:0')
    lidar_ranges=wp.zeros(rays.shape[0]*rays.shape[1],dtype=float,device='cuda:0')
    rng=np.random.default_rng(42)
    possible=np.flatnonzero(valid)

    reference_root=UsdGeom.Xform.Define(stage,'/World/G1Rig/PerceptionFrame').GetPrim()
    api.set_rig(reference_root,t_p_f)
    inst=UsdGeom.PointInstancer.Define(stage,'/World/G1Rig/PerceptionFrame/LocalVoxels')
    cube=UsdGeom.Cube.Define(stage,'/World/G1Rig/PerceptionFrame/LocalVoxels/Cube')
    cube.CreateSizeAttr(cfg['voxel_size_m']*.9)
    cube.CreateDisplayColorAttr([Gf.Vec3f(.16,.68,.83)])
    inst.CreatePrototypesRel().SetTargets([cube.GetPath()])
    inst.CreateProtoIndicesAttr([])
    inst.CreatePositionsAttr([])
    api.sphere_boundary_usd(stage,np.zeros(3),radius)
    sphere=stage.GetPrimAtPath('/World/SphereBoundary')
    api.set_rig(sphere,pose@t_p_f)
    axes=UsdGeom.Xform.Define(stage,'/World/BodyAxes').GetPrim()
    for name,color,endpoint in [('X_forward',(1,.1,.1),(.65,0,0)),('Y_left',(.1,1,.1),(0,.65,0)),('Z_up',(.1,.35,1),(0,0,.65))]:
        line=UsdGeom.BasisCurves.Define(stage,'/World/BodyAxes/'+name)
        line.CreateTypeAttr('linear');line.CreateCurveVertexCountsAttr([2])
        line.CreatePointsAttr([Gf.Vec3f(0,0,0),Gf.Vec3f(*endpoint)])
        line.CreateWidthsAttr([.015]);line.SetWidthsInterpolation('constant')
        line.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    api.set_rig(axes,pose@t_p_f)
    environment=UsdGeom.Imageable(stage.GetPrimAtPath('/World/Environment'))
    fixed_overview=args.inspection_angle=='overview'
    environment.MakeVisible() if fixed_overview else environment.MakeInvisible()
    overview_eye,overview_target=api.overview_view(cfg)
    overview=api.overview_camera(cfg)
    world.reset()
    overview.initialize()

    state={'running':not args.start_paused,'auto':True,'loop':False,'follow':not fixed_overview,'floor':True,
           'reset':True,'single':False,'route_time':0.,'sequence':0,'frame':0,'manual':collections.deque()}
    labels={}

    def focus():
        if fixed_overview:
            api.point_camera(overview,overview_eye,overview_target)
            set_camera_view(eye=overview_eye.tolist(),target=overview_target.tolist())
            return
        virtual_pose=pose@t_p_f
        yaw=np.arctan2(virtual_pose[1,0],virtual_pose[0,0])
        eye,target=api.inspection_view(virtual_pose[:3,3],radius,yaw,args.inspection_angle=='front')
        # Close-up camera has a wider field of view than the sensor camera.
        api.point_camera(overview,eye,target)
        set_camera_view(eye=eye.tolist(),target=target.tolist())

    def display_mask(points):
        if state['floor']:return np.ones(len(points),bool)
        virtual_pose=pose@t_p_f
        # Height filtering uses the simulation floor, not the virtual origin.
        world_z=points@virtual_pose[2,:3]+virtual_pose[2,3]
        return world_z>.08

    def reset():
        state['reset']=True
        state['running']=True
        state['auto']=True

    def step_body(x=0,y=0,yaw=0):
        state['auto']=False
        state['running']=False
        state['manual'].append((x,y,yaw))
        state['single']=True

    def toggle_running():
        state['running']=not state['running']

    def show_environment(model):
        environment.MakeVisible() if model.as_bool else environment.MakeInvisible()

    def follow_view(model):
        state['follow']=model.as_bool
        if state['follow']:focus()
        else:
            api.point_camera(overview,overview_eye,overview_target)
            set_camera_view(eye=overview_eye.tolist(),target=overview_target.tolist())

    panel=ui.Window('G1 / LIVE local perception',width=470,height=590)
    with panel.frame:
        with ui.VStack(spacing=7):
            ui.Label('LIVE sensor sampling -> nvblox -> virtual frame')
            ui.Label(f'Ball radius {radius:.2f} m | voxel {cfg["voxel_size_m"]*100:.0f} cm')
            ui.Label(f'Anchor: {reference["anchor_link"]} | origin Z {args.origin_z_offset:+.2f} m')
            ui.Label('X forward, Y left, Z body-up | '+('fixed sensor extrinsics' if args.reference_frame=='sensor_rig' else 'waist-dependent extrinsics'))
            ui.Label(f'Robot pose: {args.robot_pose} | view: {args.inspection_angle}')
            with ui.HStack():
                ui.Button('Pause / resume',clicked_fn=toggle_running)
                ui.Button('Reset map + route',clicked_fn=reset)
            with ui.HStack():
                ui.Label('Repeat automatic route')
                checkbox=ui.CheckBox()
                checkbox.model.add_value_changed_fn(lambda m:state.update(loop=m.as_bool))
            if fixed_overview:
                ui.Label('Fixed overview camera | no robot tracking')
            else:
                with ui.HStack():
                    ui.Label('Follow body frame')
                    checkbox=ui.CheckBox();checkbox.model.set_value(True)
                    checkbox.model.add_value_changed_fn(follow_view)
            ui.Label('Manual rigid-body steps (pauses automatic route)')
            with ui.HStack():
                ui.Button('Forward +0.1 m',clicked_fn=lambda:step_body(x=.1))
                ui.Button('Back -0.1 m',clicked_fn=lambda:step_body(x=-.1))
            with ui.HStack():
                ui.Button('Left +0.1 m',clicked_fn=lambda:step_body(y=.1))
                ui.Button('Right -0.1 m',clicked_fn=lambda:step_body(y=-.1))
            with ui.HStack():
                ui.Button('Turn left +10 deg',clicked_fn=lambda:step_body(yaw=10))
                ui.Button('Turn right -10 deg',clicked_fn=lambda:step_body(yaw=-10))
            ui.Button('Update sensors at current pose',clicked_fn=lambda:state.update(single=True))
            with ui.HStack():
                ui.Label('Overlay scene ground truth')
                checkbox=ui.CheckBox();checkbox.model.set_value(fixed_overview)
                checkbox.model.add_value_changed_fn(show_environment)
            with ui.HStack():
                ui.Label('Show ground surface voxels')
                checkbox=ui.CheckBox();checkbox.model.set_value(True)
                checkbox.model.add_value_changed_fn(lambda m:state.update(floor=m.as_bool))
            ui.Button('Restore overview' if fixed_overview else 'Focus on robot',clicked_fn=focus)
            for key in ['frame','rate','timing','motion']:
                labels[key]=ui.Label('Starting mapping worker...' if key=='frame' else '')
            ui.Label('Blue: observed surface | orange: radius boundary')
            ui.Label('Red X / green Y / blue Z: virtual reference axes')
            ui.Label('Ideal raycast depth / approximate MID360 / exact odometry')
            ui.Label('Blank can mean unobserved. Static environment only.')
    focus()

    metadata={'coordinate_frame':frame_name,'axes':{'x':'forward','y':'left','z':'body-up'},
        'inspection_angle':args.inspection_angle,'inspection_camera_fixed':fixed_overview,
        'inspection_camera_eye_world':overview_eye.tolist(),'inspection_camera_target_world':overview_target.tolist(),
        'reference_anchor':reference['anchor_link'],'reference_definition':reference['definition'],
        'origin_z_offset_m':args.origin_z_offset,'T_anchor_F':reference['T_anchor_F'].tolist(),
        'T_P_F':t_p_f.tolist(),'T_F_C':t_f_c.tolist(),'T_F_L':t_f_l.tolist(),
        'extrinsics_are_fixed':args.reference_frame=='sensor_rig',
        'robot_pose':args.robot_pose,'joint_angles_rad':robot_pose_angles(args.robot_pose),
        'radius_m':radius,'voxel_size_m':cfg['voxel_size_m'],'sensor_hz_limit':args.sensor_hz,
        'motion_duration_s':args.motion_duration,'T_P_C':t_p_c.tolist(),'T_P_L':t_p_l.tolist(),
        'odometry_origin':'initial virtual-reference pose; arbitrary local origin, no absolute world position in output',
        'camera':'Ideal D435 pinhole axial depth via GPU Warp raycast of Isaac scene geometry, NOT RTX stereo simulation',
        'lidar':'MID360 FOV with seeded sparse angular sampling, NOT actual Livox scan pattern',
        'mapping':'Persistent NVIDIA nvblox TSDF, starting empty; static environment',
        'timing':'Capture and fusion matched by sequence id; motion advances by nominal sensor period, may run slower than wall time',
        'limitations':['exact simulated relative odometry; real deployment needs odometry/SLAM and synchronization',
            'no noise, depth dropout, scan distortion, articulated motion or collision controller',
            f'{radius:g}m crop limits exported data; internal map retains integrated observations along this bounded route',
            'reference is rigidly torso-attached, not gravity stabilized; it moves and tilts with the torso',
            f'{args.origin_z_offset:+.3f}m shift is defined at zero waist; distance to moving pelvis is not kept constant',
            'sparse rotated voxel centers in reference frame; not yet a resampled dense body-aligned CNN tensor']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (output/'scene_config.json').write_text(json.dumps(cfg,indent=2))
    (output/'motion_config.json').write_text(args.motion_config.read_text())
    record=None
    if args.record:
        import imageio.v2 as imageio_v2
        record=imageio_v2.get_writer(output/'motion.mp4',fps=args.sensor_hz,codec='libx264',quality=8,macro_block_size=2)

    runtime_env=os.environ.copy()
    lib_dirs=[ROOT/'.cuda_extra/nvidia'/name/'lib' for name in ['cusparselt','nvshmem','nccl']]
    runtime_env['LD_LIBRARY_PATH']=':'.join(map(str,lib_dirs))+':'+runtime_env.get('LD_LIBRARY_PATH','')
    runtime_env['LD_PRELOAD']=str(lib_dirs[-1]/'libnccl.so.2')+(':'+runtime_env['LD_PRELOAD'] if runtime_env.get('LD_PRELOAD') else '')
    worker=None
    completed=0
    timestamps=collections.deque(maxlen=31)
    samples=[]
    inflight=None
    last_start=0.
    last_centers=np.empty((0,3),np.float32)
    last_data=None
    last_floor=True
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='g1_live_') as temporary, (output/'mapping.log').open('w') as log, (output/'history.jsonl').open('w') as history:
        ipc=Path(temporary)
        worker_cfg=dict(cfg,K=k.tolist(),integration_range_m=max(3.,radius+1.),
            coordinate_frame=frame_name,reference_anchor=reference['anchor_link'],
            T_anchor_F=reference['T_anchor_F'].tolist(),T_F_C=t_f_c.tolist(),T_F_L=t_f_l.tolist(),
            origin_z_offset_m=args.origin_z_offset)
        (ipc/'config.json').write_text(json.dumps(worker_cfg))
        worker=subprocess.Popen([str(ROOT/'.mapping_env/bin/python'),str(ROOT/'scripts/live_mapping.py'),'--ipc',str(ipc)],
            cwd=ROOT,env=runtime_env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while app.is_running():
                world.step(render=True)
                now=time.monotonic()
                if worker.poll() is not None:
                    raise RuntimeError(f'nvblox worker exited {worker.returncode}; see {output/"mapping.log"}')
                if not (ipc/'ready.npz').exists():
                    if now-started>120:raise RuntimeError('Mapping worker startup timeout')
                    continue
                if inflight is not None:
                    if not (ipc/'response.npz').exists():
                        if now-inflight['start']>60:raise RuntimeError('Mapping response timeout')
                        continue
                    with np.load(ipc/'response.npz') as archive:
                        data={key:archive[key].copy() for key in archive.files}
                    (ipc/'response.npz').unlink()
                    if int(data['sequence_id'])!=inflight['sequence']:
                        raise RuntimeError('Sensor / map sequence mismatch')
                    # No pose change while a frame is in flight. Reference-local
                    # points are displayed under the matching T_W_P @ T_P_F.
                    if np.max(np.abs(data['T_O_F']-t_o_w@pose@t_p_f))>1e-5:
                        raise RuntimeError('Sensor / current virtual-reference pose mismatch')
                    centers=data['centers']
                    if not np.isfinite(centers).all() or np.any(np.linalg.norm(centers,axis=1)>radius+1e-5):
                        raise RuntimeError('Invalid body-frame sphere crop')
                    last_centers=centers
                    last_data=data
                    keep=display_mask(centers)
                    inst.GetProtoIndicesAttr().Set([0]*int(keep.sum()))
                    inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(centers[keep].astype(np.float32)))
                    atomic_npz(output/'latest_body_voxels.npz',**data)
                    archive_name=None
                    if completed%args.archive_every==0 or (args.headless and completed==args.frames-1):
                        archive_name=f'frames/{completed:06d}.npz'
                        atomic_npz(output/archive_name,**data)
                    timestamps.append(time.monotonic())
                    rate=(len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps)>1 else 0.
                    latency_ms=(time.monotonic()-inflight['start'])*1000
                    sample={'sequence_id':int(data['sequence_id']),'frame_id':int(data['frame_id']),
                        'sensor_time_s':float(data['sensor_time_s']),
                        'trajectory_time_s':float(data['trajectory_time_s']),'surface_voxels':len(centers),
                        'integration_ms':float(data['integration_ms']),'extraction_ms':float(data['extraction_ms']),
                        'capture_ms':inflight['capture_ms'],'pipeline_ms':latency_ms,
                        'update_hz':rate,'allocated_blocks':int(data['allocated_blocks']),
                        'coordinate_frame':frame_name,'T_O_F':data['T_O_F'].tolist(),
                        'T_O_P':data['T_O_P'].tolist(),'archive':archive_name}
                    samples.append(sample)
                    history.write(json.dumps(sample)+'\n');history.flush()
                    labels['frame'].text=f'Frame {int(data["frame_id"])} | {len(centers):,} reference-frame voxels'
                    labels['rate'].text=f'Measured updates {rate:.1f} Hz | requested <= {args.sensor_hz:g} Hz'
                    labels['timing'].text=f'Fusion {float(data["integration_ms"]):.1f} ms | crop {float(data["extraction_ms"]):.1f} ms'
                    rel=data['T_O_F'];yaw=np.rad2deg(np.arctan2(rel[1,0],rel[0,0]))
                    labels['motion'].text=f'Local odometry: {rel[0,3]:+.2f}, {rel[1,3]:+.2f} m | yaw {yaw:+.0f} deg'
                    if state['follow']:focus()
                    # The first populated instancer needs extra Hydra warm-up;
                    # otherwise its first screenshot can still show an empty map.
                    for _ in range(12 if completed==0 else 3):world.step(render=True)
                    if completed%30==0 or completed==args.frames-1:
                        imageio.imwrite(output/f'frame_{completed:04d}.png',np.asarray(overview.get_rgba())[...,:3])
                        print(f'LIVE frame={completed} body_voxels={len(centers)} rate={rate:.2f}Hz fusion={float(data["integration_ms"]):.1f}ms crop={float(data["extraction_ms"]):.1f}ms',flush=True)
                    if record is not None:record.append_data(np.asarray(overview.get_rgba())[...,:3])
                    completed+=1
                    state['frame']+=1
                    inflight=None
                    if state['auto'] and state['running'] and not state['reset']:
                        if state['route_time']>=args.motion_duration:
                            if state['loop']:state['route_time']=0.
                            else:state['running']=False
                        else:state['route_time']=min(args.motion_duration,state['route_time']+1/args.sensor_hz)
                    if args.headless and completed>=args.frames:break
                    continue
                if last_floor!=state['floor']:
                    keep=display_mask(last_centers)
                    inst.GetProtoIndicesAttr().Set([0]*int(keep.sum()))
                    inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(last_centers[keep].astype(np.float32)))
                    last_floor=state['floor']
                if not (state['running'] or state['single'] or state['reset']):continue
                if now-last_start<1/args.sensor_hz:continue
                do_reset=state['reset']
                if do_reset:
                    state['route_time']=0.;state['frame']=0;state['manual'].clear()
                    inst.GetPositionsAttr().Set([]);inst.GetProtoIndicesAttr().Set([])
                    pose=first_pose.copy()
                    state['reset']=False
                elif state['manual']:
                    x,y,yaw=state['manual'].popleft()
                    pose=pose@t_p_f@transform([x,y,0],[0,0,np.deg2rad(yaw)])@np.linalg.inv(t_p_f)
                elif state['auto'] and state['running']:
                    pose=route_pose(state['route_time'])
                state['single']=bool(state['manual'])
                state['sequence']+=1
                api.set_rig(rig,pose);api.set_rig(sphere,pose@t_p_f);api.set_rig(axes,pose@t_p_f)
                # Don't drag the previous local surface with the robot while
                # waiting for the new map. Keep its physical geometry fixed.
                if last_data is not None and not do_reset:
                    old_to_new=np.linalg.inv(t_o_w@pose@t_p_f)@last_data['T_O_F']
                    transformed=last_data['centers']@old_to_new[:3,:3].T+old_to_new[:3,3]
                    keep=np.linalg.norm(transformed,axis=1)<=radius
                    keep &= display_mask(transformed)
                    inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(transformed[keep].astype(np.float32)))
                    inst.GetProtoIndicesAttr().Set([0]*int(keep.sum()))
                if state['follow']:focus()
                capture_start=time.perf_counter()
                t_w_c=pose@t_p_c;t_w_l=pose@t_p_l
                for directions,ranges,sensor_pose,max_range in [
                    (camera_wp,camera_ranges,t_w_c,20.),(lidar_wp,lidar_ranges,t_w_l,l['max_range_m'])]:
                    wp.launch(api.scan_rays,dim=len(directions),inputs=[env_wp.id,body_wp.id,directions,
                        wp.vec3(*sensor_pose[:3,3]),wp.mat33(sensor_pose[:3,:3].astype(np.float32)),
                        wp.vec3(*pose[:3,3]),wp.mat33(pose[:3,:3].T.astype(np.float32)),max_range,ranges],device='cuda:0')
                depth=(camera_ranges.numpy().reshape(c['height'],c['width'])/camera_norm).astype(np.float32)
                depth[(depth<c['min_depth_m'])|(depth>c['max_depth_m'])|~np.isfinite(depth)]=0
                lidar=lidar_ranges.numpy().reshape(valid.shape)
                selected=np.zeros(lidar.size,bool)
                selected[rng.choice(possible,min(len(possible),l['samples_per_frame']),replace=False)]=True
                lidar[~selected.reshape(lidar.shape)|(lidar<l['min_range_m'])]=0
                relative=t_o_w@pose
                relative_f=relative@t_p_f
                # Check the fixed-extrinsic path against the URDF kinematic path.
                if not (np.allclose(relative_f@t_f_c,relative@t_p_c) and
                        np.allclose(relative_f@t_f_l,relative@t_p_l)):
                    raise RuntimeError('Virtual frame / sensor extrinsics mismatch')
                capture_ms=(time.perf_counter()-capture_start)*1000
                last_start=time.monotonic()
                atomic_npz(ipc/'request.npz',depth=depth,lidar=lidar,
                    T_O_C=relative_f@t_f_c,T_O_L=relative_f@t_f_l,
                    T_O_F=relative_f,T_O_P=relative,T_P_F=t_p_f,
                    radius_m=radius,reset=do_reset,frame_id=state['frame'],sequence_id=state['sequence'],
                    sensor_time_s=last_start-started,trajectory_time_s=state['route_time'])
                inflight={'sequence':state['sequence'],'start':last_start,'capture_ms':capture_ms}
        finally:
            (ipc/'stop').touch()
            try:worker.wait(timeout=8)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
            if record is not None:record.close()
            if samples:
                metrics={'completed_frames':completed,'duration_including_startup_s':time.monotonic()-started,
                    'integration_ms_median':float(np.median([s['integration_ms'] for s in samples[1:] or samples])),
                    'extraction_ms_median':float(np.median([s['extraction_ms'] for s in samples[1:] or samples])),
                    'measured_update_hz_last_window':samples[-1]['update_hz'],
                    'sphere_radius_m':radius,'primary_coordinates':frame_name,
                    'reference_anchor':reference['anchor_link'],'origin_z_offset_m':args.origin_z_offset,
                    'first_surface_voxels':samples[0]['surface_voxels'],'last_surface_voxels':samples[-1]['surface_voxels']}
                (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
                imageio.imwrite(output/'last_frame.png',np.asarray(overview.get_rgba())[...,:3])
                print('LIVE_COMPLETE '+json.dumps(metrics),flush=True)
