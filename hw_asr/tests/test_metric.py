import unittest
from typing import List, Tuple
import numpy as np


from hw_asr.metric.utils import calc_wer, calc_cer


class TestMetric(unittest.TestCase):
    samples: List[Tuple[str, str, float, float]] = [
            ("if you can not measure it you can not improve it",
             "if you can nt measure t yo can not i",
             0.454, 0.25),
            ("if you cant describe what you are doing as a process you dont know what youre doing",
             "if you cant describe what you are doing as a process you dont know what youre doing",
             0.0, 0.0),
            ("one measurement is worth a thousand expert opinions",
             "one  is worth thousand opinions",
             0.375, 0.392),
            ("", "sm", 1, 1),
            ("", "", 0, 0)
        ]

    def test_wer(self):
        for target, pred, expected_wer, _ in self.samples:
            wer = calc_wer(target, pred)
            self.assertTrue(np.isclose(wer, expected_wer, atol=1e-3))

    def test_cer(self):
        for target, pred, _, expected_cer in self.samples:
            cer = calc_cer(target, pred)
            self.assertTrue(np.isclose(cer, expected_cer, atol=1e-3))
