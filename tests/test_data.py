"""Data invariants to implement alongside the loader and frozen manifest."""
import unittest


class DataContractTests(unittest.TestCase):
    @unittest.skip('Pending audited manifest and loader implementation')
    def test_related_images_never_cross_splits(self):
        pass
