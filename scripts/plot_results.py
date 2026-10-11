"""Render actual observed TSDF voxels as cubes; unknown space is never filled."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from common import ROOT, environment_mesh

def cubes(ax,centers,size,uniform_color=None):
    # Deterministic display cap only. Exported NPZ retains every surface voxel.
    if len(centers)>18000:
        centers=centers[np.linspace(0,len(centers)-1,18000,dtype=int)]
    corners=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*size*.45
    idx=np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]])
    faces=(centers[:,None,None,:]+corners[idx][None]).reshape(-1,4,3)
    color=plt.get_cmap('viridis')(np.clip(centers[:,2]/2.5,0,1)) if uniform_color is None else np.tile(uniform_color,(len(centers),1))
    poly=Poly3DCollection(faces,facecolors=np.repeat(color,6,axis=0),edgecolors=(.1,.15,.2,.12),linewidths=.08)
    ax.add_collection3d(poly)

def setup(ax,title):
    ax.set(xlim=(-3,5.2),ylim=(-4.2,4.2),zlim=(0,2.7),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=title)
    ax.set_box_aspect([8,8,2.7]);ax.view_init(elev=45,azim=-65)
    ax.set_facecolor('#f6f8fa');ax.grid(False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline');args=p.parse_args()
    meta=json.loads((args.output/'metadata.json').read_text());cfg=meta['config']
    fig=plt.figure(figsize=(18,12),facecolor='white',layout='constrained')
    ax=fig.add_subplot(2,2,1,projection='3d');setup(ax,'Isaac Sim scene geometry (ground truth)')
    for name,mesh in environment_mesh(cfg):
        if name in ['floor','right_wall']:
            continue
        triangles=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        ax.add_collection3d(Poly3DCollection(triangles,facecolors=[np.asarray(mesh.vertex_colors)[0]],alpha=.4,edgecolors=(0,0,0,.15),linewidths=.2))
    metrics=json.loads((args.output/'metrics.json').read_text())
    for j,mode in enumerate(['camera','lidar','fused'],2):
        v=np.load(args.output/f'{mode}_voxels.npz');centers=v['centers']
        ax=fig.add_subplot(2,2,j,projection='3d')
        setup(ax,f'{mode.upper()} | {len(centers):,} observed surface voxels | {float(v["voxel_size"])*100:.0f} cm')
        # Hide the ground and near wall in visualization only, to reveal objects.
        keep=(centers[:,2]>.08)&(centers[:,1]>-3.8)
        cubes(ax,centers[keep],float(v['voxel_size']))
    fig.suptitle('NVIDIA nvblox TSDF reconstruction from Isaac Sim 5.1\nIdeal depth + approximate MID360 sparse rays; ground-truth poses; floor hidden for inspection',fontsize=17)
    fig.savefig(args.output/'comparison.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    depth=np.load(args.output/'capture.npz')['depth'][0]
    axes[0].imshow(plt.imread(args.output/'camera_000.png'));axes[0].set_title('D435i ideal RGB (first pose)')
    im=axes[1].imshow(np.ma.masked_equal(depth,0),vmin=.3,vmax=3,cmap='viridis');axes[1].set_title('Filtered axial depth (m)');fig.colorbar(im,ax=axes[1])
    d=np.load(args.output/'capture.npz')['lidar'][0]
    im=axes[2].imshow(np.ma.masked_equal(d,0),vmin=.1,vmax=12,cmap='viridis',aspect='auto');axes[2].set_title('MID360 approximation: sparse ranges (m)');fig.colorbar(im,ax=axes[2])
    for ax in axes:ax.set_xlabel('u (pixel / angular bin)');ax.set_ylabel('v (pixel / angular bin)')
    fig.savefig(args.output/'sensors.png',dpi=150);plt.close(fig)
    print(args.output/'comparison.png')

if __name__=='__main__':main()
