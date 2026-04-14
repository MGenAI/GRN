import os
import os.path as osp
import json
import glob
import sys

import numpy as np

data_root = sys.argv[1]
min_fid, best_dict, best_cfg = 999, None, None
for metric_file in sorted(glob.glob(osp.join(data_root, '*/metrics.json'))):
    with open(metric_file, 'r') as f:
        metrics = json.load(f)
    cur_fid = metrics['fid'] if 'fid' in metrics else metrics['frechet_inception_distance']
    if cur_fid < min_fid:
        min_fid = cur_fid
        best_dict = metrics
        best_cfg = [metric_file, metrics]
    print(metric_file, metrics)
print(f'\n {min_fid=} {best_dict=} {best_cfg=}')
