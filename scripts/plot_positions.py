"""Compare observer positions, ideal RGB views, and independent nvblox maps."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.font_manager import FontProperties
from common import ROOT
from plot_results import cubes

FONT=FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')

def room_plan(ax,cfg,positions,active):
    for obj in cfg['objects']:
        if obj['name']=='floor':continue
        x,y,_=obj['center']
        if obj['kind']=='box':
            sx,sy,_=obj['size']
            ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,facecolor=obj['color'],edgecolor='#59636e',linewidth=.6))
        else:
            ax.add_patch(Circle((x,y),obj['radius'],facecolor=obj['color'],edgecolor='#59636e'))
    for n,p in enumerate(positions,1):
        x,y,yaw=p['xy_yaw_deg'];r=np.deg2rad(yaw)
        color='#cf443c' if p['id']==active else '#7e8791'
        ax.scatter(x,y,c=color,s=40,zorder=5)
        ax.arrow(x,y,.55*np.cos(r),.55*np.sin(r),width=.035,head_width=.18,color=color,zorder=6,length_includes_head=True)
        ax.annotate(str(n),(x,y),xytext=(-12,8),textcoords='offset points',color=color,weight='bold')
    ax.set(xlim=(-2,5.3),ylim=(-4.3,4.3),xlabel='World X (m)',ylabel='World Y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.13)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/positions');args=p.parse_args()
    manifest=json.loads((args.output/'manifest.json').read_text());positions=manifest['positions']
    cfg=json.loads((args.output/positions[0]['directory']/'metadata.json').read_text())['config']
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
    room_plan(ax,cfg,positions,None)
    for n,position in enumerate(positions,1):
        x,y,_=position['xy_yaw_deg']
        ax.annotate(f'{n}. {position["label"]}',(x,y),xytext=(12,-16),textcoords='offset points',fontproperties=FONT,fontsize=12)
    ax.set_title('同一场景中的五个观测位置（箭头表示初始朝向）',fontproperties=FONT,fontsize=16)
    fig.savefig(args.output/'locations.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(17,len(positions)*3.7),layout='constrained')
    results=[]
    for row,position in enumerate(positions):
        directory=args.output/position['directory'];x,y,yaw=position['xy_yaw_deg']
        ax=fig.add_subplot(len(positions),3,row*3+1)
        room_plan(ax,cfg,positions,position['id'])
        ax.set_title(f'{row+1}. {position["label"]} | ({x:.2f}, {y:.2f}), {yaw:.0f}°',fontproperties=FONT,fontsize=12)
        ax=fig.add_subplot(len(positions),3,row*3+2)
        ax.imshow(plt.imread(directory/'camera_000.png'));ax.axis('off')
        ax.set_title('相机首帧视角（理想 RGB）',fontproperties=FONT,fontsize=12)
        v=np.load(directory/'fused_voxels.npz');centers=v['centers'];r=np.deg2rad(yaw)
        rot=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
        local=(centers-np.array([x,y,0]))@rot
        keep=(local[:,0]>=-.5)&(local[:,0]<=3)&(np.abs(local[:,1])<=2)&(local[:,2]>.08)&(local[:,2]<=2.6)
        ax=fig.add_subplot(len(positions),3,row*3+3,projection='3d')
        cubes(ax,local[keep],float(v['voxel_size']))
        ax.set(xlim=(-.5,3),ylim=(-2,2),zlim=(0,2.6),xlabel='Forward X (m)',ylabel='Left Y (m)',zlabel='Height (m)')
        ax.set_box_aspect([3.5,4,2.6]);ax.view_init(elev=35,azim=-65);ax.grid(False)
        ax.set_title(f'融合体素：前方局部区域，隐藏地面 | {int(keep.sum()):,} 格',fontproperties=FONT,fontsize=12)
        metric=json.loads((directory/'metrics.json').read_text())
        results.append({**position,'fused_surface_voxels':metric['fused']['surface_voxels'],'local_display_voxels':int(keep.sum())})
    fig.suptitle(f'不同位置独立重建：每处 {manifest["frames_per_position"]} 帧、小范围转向 ±30°、5 cm 体素\n理想深度 + 近似 MID360 扫描；每个位置从空地图开始',fontproperties=FONT,fontsize=17)
    fig.savefig(args.output/'positions_comparison.png',dpi=140);plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(args.output/'positions_comparison.png',flush=True)

if __name__=='__main__':main()
