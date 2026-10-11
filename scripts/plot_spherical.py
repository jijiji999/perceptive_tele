"""Radius comparisons from existing nvblox maps; no rescan or synthetic filling."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from common import ROOT,DEFAULT_URDF,sphere_mask,sphere_wire_curves,world_to_robot,transform,urdf_frames
from plot_results import cubes

FONT=FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')

def draw(ax,local,radius,voxel_size,bones,limit):
    cubes(ax,local,voxel_size,uniform_color=(0.16,0.68,0.83,1))
    for curve in sphere_wire_curves([0,0,0],radius):
        ax.plot(curve[:,0],curve[:,1],curve[:,2],color='#d88726',alpha=.48,linewidth=.6)
    for p,c in bones:
        xyz=np.stack([p,c]);ax.plot(*xyz.T,color='#26333c',linewidth=2)
    ax.scatter([0],[0],[0],c='#d88726',s=22,depthshade=False)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),
        xlabel='Forward X (m)',ylabel='Left Y (m)',zlabel='Up Z (m)')
    ax.set_box_aspect([1,1,1]);ax.view_init(elev=28,azim=-62);ax.grid(False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--positions-root',type=Path,default=ROOT/'outputs/positions')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/spherical')
    p.add_argument('--radii',type=float,nargs='+',default=[1.0,1.5,2.0])
    p.add_argument('--position',default='pole')
    args=p.parse_args()
    if any(not np.isfinite(r) or r<=0 for r in args.radii):p.error('Radii must be finite and positive')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.positions_root/'manifest.json').read_text())
    frames=urdf_frames()
    bones=[]
    for joint in ET.parse(DEFAULT_URDF).getroot().findall('joint'):
        parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
        bones.append((frames[parent][:3,3],frames[child][:3,3]))
    records=[]
    summary=[]
    for position in manifest['positions']:
        directory=args.positions_root/position['directory']
        metadata=json.loads((directory/'metadata.json').read_text());cfg=metadata['config']
        x,y,yaw=cfg['trajectory_xy_yaw_deg'][0]
        pose=transform([x,y,cfg['pelvis_height_m']],[0,0,np.deg2rad(yaw)])
        v=np.load(directory/'fused_voxels.npz');centers=v['centers'];row=[]
        crop_dir=args.output/position['id'];crop_dir.mkdir(parents=True,exist_ok=True)
        previous=None
        for radius in sorted(args.radii):
            mask=sphere_mask(centers,pose[:3,3],radius)
            if previous is not None:assert not np.any(previous&~mask)
            previous=mask
            local=world_to_robot(centers[mask],pose)
            assert np.all(np.linalg.norm(local,axis=1)<=radius+1e-6)
            assert np.all(v['weight'][mask]>.1)
            np.savez_compressed(crop_dir/f'fused_sphere_r{radius:g}_voxels.npz',
                centers=local,centers_robot=local,centers_world=centers[mask],coordinate_frame='pelvis',
                tsdf=v['tsdf'][mask],weight=v['weight'][mask],
                voxel_size=v['voxel_size'],radius_m=np.float64(radius),center_world=pose[:3,3],T_W_P=pose)
            row.append((radius,local,float(v['voxel_size'])))
            summary.append({'position':position['id'],'label':position['label'],'radius_m':radius,
                'center_world':pose[:3,3].tolist(),'retained_surface_voxels':int(mask.sum()),
                'original_surface_voxels':len(centers),'ground_surface_voxels':int((centers[mask,2]<.08).sum())})
        records.append((position,row))
    limit=max(args.radii)*1.04
    fig=plt.figure(figsize=(len(args.radii)*5.2,len(records)*4.9),layout='constrained')
    for row,(position,variants) in enumerate(records):
        for col,(radius,local,voxel_size) in enumerate(variants):
            ax=fig.add_subplot(len(records),len(args.radii),row*len(args.radii)+col+1,projection='3d')
            draw(ax,local,radius,voxel_size,bones,limit)
            ax.set_title(f'{position["label"]} | R = {radius:.1f} m | {len(local):,} 个表面体素',fontproperties=FONT,fontsize=12)
    fig.suptitle('以骨盆为中心的球形区域：保留球内已观测表面，包含地面\n蓝色：原 nvblox 表面体素；橙色：裁剪边界；黑色：机器人骨架；空白可能是未知区域',fontproperties=FONT,fontsize=16)
    fig.savefig(args.output/'all_positions.png',dpi=140,bbox_inches='tight',pad_inches=.3);plt.close(fig)
    selected=next((record for record in records if record[0]['id']==args.position),None)
    if selected is None:p.error(f'Unknown position: {args.position}')
    position,variants=selected
    fig=plt.figure(figsize=(len(args.radii)*5.4,5.5),layout='constrained')
    for col,(radius,local,voxel_size) in enumerate(variants):
        ax=fig.add_subplot(1,len(args.radii),col+1,projection='3d')
        draw(ax,local,radius,voxel_size,bones,limit)
        ax.set_title(f'R = {radius:.1f} m | {len(local):,} 个表面体素',fontproperties=FONT,fontsize=14)
    fig.suptitle(f'{position["label"]}：球形裁剪半径对比（同一尺度，包含地面）\n球心为骨盆；蓝色为已观测表面，橙色仅为边界；坐标相对于机器人',fontproperties=FONT,fontsize=17)
    fig.savefig(args.output/'radius_comparison.png',dpi=160,bbox_inches='tight',pad_inches=.3);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (args.output/'validation.json').write_text(json.dumps({'passed':True,'checks':['all cropped centres lie within their sphere',
        'world and robot frame distances agree','larger radii contain smaller-radius crops','all retained voxels are observed'],
        'cases':len(summary)},indent=2))
    print(f'SPHERE_COMPARISON_COMPLETE: {len(summary)} validated crops',flush=True)

if __name__=='__main__':main()
