"""Check data integrity and exact epoch-boundary training resumption."""
import json
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from run_uformer_gamma import train
from src.camera_ready_protocol import REPO_ROOT, prepare_protocol, read_config
from profile_complexity import ArithmeticCounter


class CameraReadyTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO_ROOT / 'tmp' / f'protocol_test_{uuid.uuid4().hex}'
        self.root.mkdir(parents=True)
        data = self.root / 'data'; data.mkdir()
        for index in range(20):
            image = np.random.default_rng(index).integers(10, 245, (24, 24), dtype=np.uint8)
            cv2.imwrite(str(data / f'{index:03d}.png'), image)
        self.config = read_config('configs/uformer_gamma.json')
        self.config.update(data_root=str(data), protocol_dir=str(self.root / 'protocol'),
                           output_dir=str(self.root / 'runs'), test_replicates=1,
                           epochs=2, cpu_threads=2)

    def tearDown(self):
        if not self.root.resolve().is_relative_to((REPO_ROOT / 'tmp').resolve()):
            raise RuntimeError('Test cleanup escaped the intended temporary directory.')
        shutil.rmtree(self.root)

    def test_cache_is_immutable_and_detects_changed_data(self):
        first = prepare_protocol(self.config)
        second = prepare_protocol(self.config)
        self.assertEqual(first['protocol_hash'], second['protocol_hash'])
        groups = [{p['path'] for p in first['images'][s]} for s in ('train', 'val', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        source = Path(self.config['data_root']) / '000.png'
        cv2.imwrite(str(source), np.zeros((24, 24), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'differs'):
            prepare_protocol(self.config)

    def test_resumed_run_matches_uninterrupted_training(self):
        # Exercise real augmentation, Gamma sampling, shuffled loading, dropout,
        # optimizer/scheduler and RNG restoration with a small test-only model.
        def toy_model(_config):
            return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Conv2d(4, 1, 3, padding=1))
        with patch('run_uformer_gamma.build_model', side_effect=toy_model):
            train(self.config, 1, torch.device('cpu'), max_epochs=1)
            train(self.config, 1, torch.device('cpu'), max_epochs=2)
            uninterrupted = {**self.config, 'output_dir': str(self.root / 'uninterrupted')}
            train(uninterrupted, 1, torch.device('cpu'), max_epochs=2)
        resumed = torch.load(Path(self.config['output_dir']) / 'L1/last.pt', weights_only=False)
        full = torch.load(Path(uninterrupted['output_dir']) / 'L1/last.pt', weights_only=False)
        for name, weights in resumed['model_state_dict'].items():
            torch.testing.assert_close(weights, full['model_state_dict'][name], rtol=0, atol=0)
        self.assertEqual(resumed['updates'], full['updates'])
        self.assertEqual(resumed['best_val'], full['best_val'])
        self.assertEqual(resumed['best_epoch'], full['best_epoch'])
        self.assertEqual(resumed['scheduler'], full['scheduler'])
        self.assertEqual([h['val_mse'] for h in resumed['history']], [h['val_mse'] for h in full['history']])

    def test_counter_captures_functional_convolution_and_matmul(self):
        counter = ArithmeticCounter()
        with torch.inference_mode(), counter:
            torch.nn.functional.conv2d(torch.ones(1, 1, 8, 8), torch.ones(2, 1, 3, 3))
            torch.ones(3, 4) @ torch.ones(4, 2)
        self.assertEqual(sum(counter.macs.values()), 648 + 24)


if __name__ == '__main__':
    unittest.main()
