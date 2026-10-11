"""Shared geometry and coordinate conventions. All lengths are metres."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = Path('/home/rx/project/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf')

def config(path=None):
    return json.loads(Path(path or ROOT / 'config/scene.json').read_text())

def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    t[:3, 3] = xyz
    return t

def sphere_mask(centers, center, radius):
    """Retain voxel centres inside a ball (not a shell), in any rigid frame."""
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('Sphere radius must be finite and positive')
    center=np.asarray(center,dtype=np.float64)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError('Sphere centre must be a finite xyz vector')
    delta=np.asarray(centers,dtype=np.float64)-center
    return np.sum(delta*delta,axis=-1)<=float(radius)**2

def world_to_robot(points, pose):
    """Row-vector points: fixed frame -> pelvis, including all three rotations."""
    return (np.asarray(points)-pose[:3,3])@pose[:3,:3]

def sphere_wire_curves(center,radius,samples=97):
    """Latitude/longitude curves in world xyz. Visual boundary, not map data."""
    t=np.linspace(0,2*np.pi,samples)
    center=np.asarray(center)
    curves=[]
    for lat in np.deg2rad([-60,-30,0,30,60]):
        curves.append(center+radius*np.stack([np.cos(lat)*np.cos(t),np.cos(lat)*np.sin(t),np.full_like(t,np.sin(lat))],axis=-1))
    for lon in np.linspace(0,np.pi,6,endpoint=False):
        curves.append(center+radius*np.stack([np.cos(t)*np.cos(lon),np.cos(t)*np.sin(lon),np.sin(t)],axis=-1))
    return curves

def robot_pose_angles(name='neutral'):
    """URDF zero pose or straight, horizontal arms extended sideways."""
    if name=='neutral':
        return {}
    if name=='tpose':
        # In this G1 URDF the zero forearm points forward. Elbow +90 deg
        # aligns it with the downward upper arm before shoulder abduction.
        return {'left_shoulder_roll_joint':np.pi/2,
                'right_shoulder_roll_joint':-np.pi/2,
                'left_elbow_joint':np.pi/2,'right_elbow_joint':np.pi/2}
    raise ValueError(f'Unknown robot pose: {name}')

def urdf_frames(path=DEFAULT_URDF,joint_angles=None):
    root = ET.parse(path).getroot()
    joint_angles=joint_angles or {}
    frames = {'pelvis': np.eye(4)}
    joints = list(root.findall('joint'))
    while joints:
        progress = False
        for j in joints[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in frames:
                continue
            o = j.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            motion=np.eye(4)
            angle=float(joint_angles.get(j.get('name'),0.))
            kind=j.get('type')
            if kind in ['revolute','continuous','prismatic']:
                a=j.find('axis')
                axis=np.fromstring(a.get('xyz'),sep=' ') if a is not None else np.array([1.,0,0])
                axis=axis/np.linalg.norm(axis)
                limit=j.find('limit')
                if limit is not None and kind!='continuous' and not float(limit.get('lower','-inf'))-1e-8<=angle<=float(limit.get('upper','inf'))+1e-8:
                    raise ValueError(f'Joint {j.get("name")} exceeds URDF limits')
                if kind=='prismatic':motion[:3,3]=axis*angle
                else:motion[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
            frames[child] = frames[parent] @ transform(xyz, rpy) @ motion
            joints.remove(j)
            progress = True
        if not progress:
            raise ValueError('URDF has disconnected joints')
    return frames

def perception_reference(path=DEFAULT_URDF,kind='sensor_rig',z_offset=-.2,joint_angles=None):
    """Virtual frame F. The zero-waist calibration places F at P+(0,0,dz).

    sensor_rig anchors F to torso_link, the rigid parent of both sensors.
    Moving waist joints changes T_P_F, but never T_torso_F or T_F_sensor.
    It is NOT a gravity-stabilized or world-stationary coordinate system.
    """
    if not np.isfinite(z_offset):raise ValueError('Reference Z offset must be finite')
    if kind not in ['sensor_rig','pelvis']:raise ValueError(f'Unknown reference frame: {kind}')
    zero=urdf_frames(path)
    current=urdf_frames(path,joint_angles)
    anchor='torso_link' if kind=='sensor_rig' else 'pelvis'
    t_anchor_f=np.linalg.inv(zero[anchor])@transform([0,0,z_offset])
    t_p_f=current[anchor]@t_anchor_f
    return {'coordinate_frame':'sensor_rig' if kind=='sensor_rig' else ('pelvis' if z_offset==0 else 'pelvis_offset'),
        'anchor_link':anchor,'z_offset_m':float(z_offset),
        'T_anchor_F':t_anchor_f,'T_P_F':t_p_f,
        'definition':'Fixed torso frame, calibrated below zero-waist pelvis; follows torso orientation'
            if kind=='sensor_rig' else 'Fixed pelvis frame offset; follows pelvis orientation'}

def environment_mesh(cfg):
    import open3d as o3d
    meshes = []
    for obj in cfg['objects']:
        if obj['kind'] == 'box':
            mesh = o3d.geometry.TriangleMesh.create_box(*obj['size'])
            mesh.translate(-np.asarray(obj['size']) / 2)
        else:
            mesh = o3d.geometry.TriangleMesh.create_cylinder(obj['radius'], obj['height'], resolution=48)
        mesh.translate(obj['center'])
        mesh.paint_uniform_color(obj['color'])
        mesh.compute_vertex_normals()
        meshes.append((obj['name'], mesh))
    return meshes

def robot_meshes(path=DEFAULT_URDF,joint_angles=None):
    """Kinematic G1 visual geometry, also used for sensor self masking."""
    import open3d as o3d
    path = Path(path)
    frames = urdf_frames(path,joint_angles)
    root = ET.parse(path).getroot()
    result = []
    for link in root.findall('link'):
        for i, v in enumerate(link.findall('visual')):
            mesh_tag = v.find('geometry/mesh')
            if mesh_tag is None:
                continue
            filename = mesh_tag.get('filename')
            if filename.startswith('package://'):
                raise ValueError('Use a URDF with local mesh paths')
            mesh = o3d.io.read_triangle_mesh(str(path.parent / filename))
            if mesh.is_empty():
                raise ValueError(f'Missing mesh {filename}')
            scale = np.fromstring(mesh_tag.get('scale', '1 1 1'), sep=' ')
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
            o = v.find('origin')
            local = np.eye(4) if o is None else transform(np.fromstring(o.get('xyz','0 0 0'),sep=' '), np.fromstring(o.get('rpy','0 0 0'),sep=' '))
            mesh.transform(frames[link.get('name')] @ local)
            mesh.paint_uniform_color([0.48,0.52,0.56])
            mesh.compute_vertex_normals()
            result.append((f"{link.get('name')}_{i}", mesh))
    return result

def combined_mesh(meshes):
    import open3d as o3d
    result = o3d.geometry.TriangleMesh()
    for _, mesh in meshes:
        result += mesh
    return result

def trajectory(cfg, count):
    points = np.asarray(cfg['trajectory_xy_yaw_deg'])
    u = np.linspace(0, len(points) - 1, count)
    return np.stack([np.interp(u, np.arange(len(points)), points[:,i]) for i in range(3)], axis=1)

def lidar_rays(cfg):
    """Match nvblox pixel-centre convention. Pad to symmetric +/-52 degrees.

    Python binding lacks asymmetric FOV; rows below MID360's -7 deg are invalid.
    This is an angular grid with randomized sparse samples, not Livox's scan law.
    """
    c = cfg['lidar']
    az = -np.pi + np.arange(c['width']) * 2*np.pi/c['width']
    el = np.linspace(np.deg2rad(c['max_elevation_deg']), -np.deg2rad(c['max_elevation_deg']), c['height'])
    a, e = np.meshgrid(az, el)
    directions = np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=-1).astype(np.float32)
    valid = e >= np.deg2rad(c['min_elevation_deg'])
    return directions, valid
