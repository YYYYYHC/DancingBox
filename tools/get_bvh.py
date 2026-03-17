import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from visualization.joints2bvh import Joint2BVHConvertor
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert results.npy joint positions to BVH files')
    parser.add_argument('source_file', help='Path to results.npy')
    parser.add_argument('-o', '--output', default=None, help='Output directory (default: ik_animations/ next to source file)')
    parser.add_argument('--iterations', type=int, default=100, help='IK solver iterations (default: 100)')
    parser.add_argument('--no-foot-ik', action='store_true', help='Disable foot IK')
    args = parser.parse_args()

    source_file = os.path.abspath(args.source_file)
    target_path = args.output or os.path.join(os.path.dirname(source_file), 'ik_animations')
    os.makedirs(target_path, exist_ok=True)

    convertor = Joint2BVHConvertor()
    source_datas = np.load(source_file, allow_pickle=True).item()['motion']
    source_datas = np.transpose(source_datas, (0, 3, 1, 2))
    for i in range(source_datas.shape[0]):
        print(f'Processing {i:03d} / {source_datas.shape[0]}')
        source_data = source_datas[i]
        convertor.convert(source_data, os.path.join(target_path, f'{i:03d}.bvh'),
                          iterations=args.iterations, foot_ik=not args.no_foot_ik)
