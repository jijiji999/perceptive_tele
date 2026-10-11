"""Check torso-anchored extrinsics under waist articulation and whole-body motion."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from common import ROOT,DEFAULT_URDF,perception_reference,urdf_frames,transform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/reference_validation.json')
    args=parser.parse_args()
    root=ET.parse(DEFAULT_URDF).getroot()
    for child in ['d435_link','mid360_link']:
        joint=next(j for j in root.findall('joint') if j.find('child').get('link')==child)
        assert joint.get('type')=='fixed' and joint.find('parent').get('link')=='torso_link'
    neutral=perception_reference()
    assert np.allclose(neutral['T_P_F'],transform([0,0,-.2]))
    base=urdf_frames()
    expected={sensor:np.linalg.inv(neutral['T_P_F'])@base[sensor] for sensor in ['d435_link','mid360_link']}
    rng=np.random.default_rng(18)
    max_error=0.;max_pelvis_change=0.
    for _ in range(100):
        angles={'waist_yaw_joint':rng.uniform(-.8,.8),
                'waist_roll_joint':rng.uniform(-.4,.4),'waist_pitch_joint':rng.uniform(-.4,.4)}
        current=urdf_frames(joint_angles=angles)
        reference=perception_reference(joint_angles=angles)
        assert np.allclose(reference['T_anchor_F'],neutral['T_anchor_F'])
        # Global movement of the whole robot must cancel from sensor -> F too.
        t_w_p=transform(rng.uniform(-5,5,3),rng.uniform(-.5,.5,3))
        t_w_f=t_w_p@reference['T_P_F']
        for sensor in expected:
            actual=np.linalg.inv(t_w_f)@t_w_p@current[sensor]
            max_error=max(max_error,float(np.max(np.abs(actual-expected[sensor]))))
            assert np.allclose(actual,expected[sensor],atol=1e-12)
        max_pelvis_change=max(max_pelvis_change,float(np.linalg.norm(reference['T_P_F'][:3,3]-neutral['T_P_F'][:3,3])))
    assert max_pelvis_change>.05,'Reference should follow torso rather than stay locked to pelvis'
    result={'passed':True,'waist_and_whole_body_cases':100,'anchor_link':'torso_link',
        'zero_waist_origin_in_pelvis_m':neutral['T_P_F'][:3,3].tolist(),
        'T_torso_F':neutral['T_anchor_F'].tolist(),
        'max_fixed_sensor_extrinsic_error':max_error,
        'max_reference_motion_relative_to_pelvis_m':max_pelvis_change,
        'gravity_stabilized':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
