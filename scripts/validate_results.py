"""Geometry checks catch camera-frame errors and flattened lidar reconstructions."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline');args=p.parse_args()
d=np.load(args.output/'capture.npz');meta=json.loads((args.output/'metadata.json').read_text())
assert np.isfinite(d['depth']).all() and np.isfinite(d['lidar']).all()
assert (d['depth']>0).mean()>.5,'Camera coverage is unexpectedly low'
assert np.all((d['lidar']>0).sum(axis=(1,2))>1000),'Lidar has missing frames'
for key in ['T_W_C','T_W_L']:
    r=d[key][:,:3,:3]
    assert np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-5)
    assert np.allclose(np.linalg.det(r),1,atol=1e-5)
stats={}
for mode in ['camera','lidar','fused']:
    x=np.load(args.output/f'{mode}_voxels.npz');c=x['centers'];vs=float(x['voxel_size'])
    assert len(c)>500 and np.isfinite(c).all()
    assert np.all(x['weight']>.1),'Unknown voxels must not be visualized as surface'
    assert np.all(np.abs(x['tsdf'])<=vs*.751)
    if mode in ['lidar','fused']:
        assert c[:,2].max()>2.3,'Lidar wall collapsed: check fromLidar argument order/FOV'
    stats[mode]={'surface_voxels':len(c),'min_xyz':c.min(0).tolist(),'max_xyz':c.max(0).tolist()}
(args.output/'validation.json').write_text(json.dumps({'passed':True,'checks':stats},indent=2))
print('PASS: finite observations, rigid poses, valid frames, observed-only TSDF, lidar wall heights')
