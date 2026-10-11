"""Reconstruct each observer location independently in the unchanged scene."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/positions')
    parser.add_argument('--frames',type=int)
    parser.add_argument('--only',choices=['steps','boxes','pole','side_wall','table'])
    args=parser.parse_args()
    args.output=args.output.resolve()
    spec=json.loads((ROOT/'config/positions.json').read_text())
    frames=args.frames or spec['frames_per_position']
    if frames<2:parser.error('--frames must be at least 2')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'description':'Independent maps per position; no map shared across positions',
        'frames_per_position':frames,'positions':[]}
    for position in spec['positions']:
        directory=args.output/position['id']
        directory.mkdir(parents=True,exist_ok=True)
        cfg=config()
        x,y,yaw=position['xy_yaw_deg']
        cfg['trajectory_xy_yaw_deg']=[[x,y,yaw+offset] for offset in spec['scan_offsets_deg']]
        cfg['observation_position']=position
        cfg_file=directory/'scene_config.json'
        cfg_file.write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
        manifest['positions'].append({**position,'directory':position['id']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    for position in manifest['positions']:
        if args.only and position['id']!=args.only:continue
        directory=args.output/position['directory']
        commands=[('capture',['capture','--frames',str(frames),'--config',str(directory/'scene_config.json'),'--output',str(directory)]),
            ('map',['map','--output',str(directory)]),
            ('validate',['validate','--output',str(directory)]),
            ('plot',['plot','--output',str(directory)]),
            ('view',['view','fused','--headless','--config',str(directory/'scene_config.json'),'--output',str(directory)])]
        for stage,command in commands:
            print(f'{position["label"]} / {stage}',flush=True)
            with (directory/f'{stage}.log').open('w') as log:
                result=subprocess.run([str(ROOT/'run.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                tail=(directory/f'{stage}.log').read_text(errors='replace')[-5000:]
                raise RuntimeError(f'{position["id"]}: {stage} failed\n{tail}')
        print(f'COMPLETE {position["id"]}',flush=True)
    if not args.only:
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_positions.py'),'--output',str(args.output)],check=True)
    print('POSITIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
