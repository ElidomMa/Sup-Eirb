import time
import unittest

import cv2
import numpy as np

from pipeline import Frame, LatestRequest, Request, process_pair


class RecordingProcessor:
    device = "cpu"
    scale = 2

    def __init__(self):
        self.inputs = []

    def upscale(self, image):
        self.inputs.append(image)
        return cv2.resize(image, None, fx=self.scale, fy=self.scale), 1.0


class PipelineTests(unittest.TestCase):
    def request(self, sequence, size=(32, 18), generation=0, fixed=False):
        image = np.random.default_rng(sequence).integers(0, 256, (91, 123, 3), dtype=np.uint8)
        return Request(Frame(image, sequence, time.perf_counter()), size, generation, fixed)

    def test_latest_frame_replaces_pending_without_repeating_active_frame(self):
        queue = LatestRequest()
        first = self.request(1)
        queue.offer(first)
        self.assertIs(queue.take(), first)
        queue.offer(first)
        self.assertIsNone(queue.take())
        for sequence in range(2, 100):
            queue.offer(self.request(sequence))
        self.assertEqual(queue.take().frame.sequence, 99)
        self.assertIsNone(queue.take())

    def test_resolution_and_source_changes_allow_same_frame_again(self):
        queue = LatestRequest()
        for request in (self.request(1), self.request(1, (48, 27)),
                        self.request(1, (48, 27), generation=1)):
            queue.offer(request)
            self.assertIs(queue.take(), request)

    def test_pair_receives_identical_array_once_and_keeps_native_dimensions(self):
        for fixed in (False, True):
            processors = RecordingProcessor(), RecordingProcessor()
            request = self.request(7, fixed=fixed)
            result = process_pair(processors, request)
            self.assertIs(result.request, request)
            self.assertIs(processors[0].inputs[0], processors[1].inputs[0])
            self.assertEqual([len(p.inputs) for p in processors], [1, 1])
            self.assertEqual(result.images[0].shape, (36, 64, 3))
            self.assertEqual(result.images[1].shape, (36, 64, 3))
            if fixed:
                self.assertEqual(result.reference.shape, result.images[0].shape)


if __name__ == "__main__":
    unittest.main()
