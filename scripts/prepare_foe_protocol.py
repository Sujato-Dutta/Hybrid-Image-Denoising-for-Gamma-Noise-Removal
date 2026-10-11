from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camera_ready_protocol import prepare_protocol, read_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze the FoE split and validation/test Gamma inputs.')
    parser.add_argument('--config', default='configs/uformer_gamma.json')
    args = parser.parse_args()
    manifest = prepare_protocol(read_config(args.config))
    print('Protocol:', manifest['protocol_hash'])
    print('Split:', {key: len(value) for key, value in manifest['images'].items()})
