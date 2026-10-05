"""Pipeline invariants to implement alongside training/evaluation code."""
import unittest


class PipelineContractTests(unittest.TestCase):
    @unittest.skip('Pending model and checkpoint implementation')
    def test_checkpoint_inference_matches_saved_predictions(self):
        pass
