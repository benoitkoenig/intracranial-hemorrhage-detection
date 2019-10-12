import unittest

from intracranial_hemorrhage_detection.export_predictions.classifier_3D_by_chunks import split_ids_into_chunks, restore_predictions_from_chunks

class TestSplitIdsIntoChunks(unittest.TestCase):
    def test_with_simple_ids(self):
        len_slice = 20
        while len_slice <= 60:
            slice_ids = [i for i in range(len_slice)]
            chunks = split_ids_into_chunks(slice_ids)
            restoration = restore_predictions_from_chunks(chunks, len(slice_ids))
            self.assertEqual(len(slice_ids), len(restoration))
            for i in range(len(slice_ids)):
                self.assertEqual(slice_ids[i], restoration[i])
            len_slice += 1

if __name__ == "__main__":
    unittest.main()
