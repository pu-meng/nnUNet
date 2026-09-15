"""Check post-training dispatch without launching training or validation."""
import unittest
from unittest.mock import Mock, patch

import torch
from nnunetv2.run import run_training as entry
from pumengyu.task02.stage2_mixin import Task02Stage2Mixin


class TrainingEntryTest(unittest.TestCase):
    def test_post_training_dispatch(self):
        for gpus in (1, 2):
            for automatic in (True, False):
                with self.subTest(gpus=gpus, automatic=automatic):
                    trainer = Mock(output_folder='/unused', AUTO_VALIDATE_AFTER_TRAINING=automatic)
                    with patch.object(entry, 'get_trainer_from_args', return_value=trainer), \
                         patch.object(entry, 'maybe_load_checkpoint') as load, \
                         patch.object(entry.mp, 'spawn') as spawn, \
                         patch.object(entry.torch.cuda, 'is_available', return_value=False), \
                         patch.object(entry, 'find_free_network_port', return_value=29591), \
                         patch.dict(entry.os.environ, {}, clear=False):
                        entry.run_training('3', '3d_fullres', 0, plans_identifier='test',
                                           num_gpus=gpus, device=torch.device('cuda'))
                    self.assertEqual(trainer.perform_actual_validation.call_count, int(automatic))
                    self.assertEqual(spawn.call_count, int(gpus == 2))
                    if gpus == 2 and not automatic:
                        load.assert_not_called()

    def test_explicit_validation_still_rejected(self):
        trainer = Mock(AUTO_VALIDATE_AFTER_TRAINING=False)
        trainer.perform_actual_validation.side_effect = lambda _: Task02Stage2Mixin.perform_actual_validation(trainer)
        with patch.object(entry, 'get_trainer_from_args', return_value=trainer), \
             patch.object(entry, 'maybe_load_checkpoint'), \
             patch.object(entry.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Task02 does not permit'):
                entry.run_training('3', '3d_fullres', 0, plans_identifier='test',
                                   only_run_validation=True)
        trainer.run_training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
