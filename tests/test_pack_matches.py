import unittest

import torch

from flexglue.utils.ops import pack_matches


class PackMatchesTest(unittest.TestCase):
    def test_float_match_values_preserve_selection_and_score_gradients(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            for values in ([2, -1, 0], [-1, -1, -1], [2, float("nan"), 0]):
                with self.subTest(dtype=dtype, values=values):
                    matches = torch.tensor([values], dtype=dtype)
                    scores = torch.tensor(
                        [[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True
                    )
                    result = pack_matches(
                        {"matches0": matches, "matching_scores0": scores}
                    )
                    valid = matches[0] > -1
                    expected = torch.stack(
                        (torch.where(valid)[0], matches[0][valid]), dim=-1
                    )
                    torch.testing.assert_close(
                        result["matches"][0], expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(result["scores"][0], scores[0][valid])
                    result["matches"][0].fill_(1)
                    result["scores"][0].sum().backward()
                    torch.testing.assert_close(
                        scores.grad, valid[None].to(scores.dtype)
                    )

    def test_different_batch_counts_keep_two_pass_truncation(self):
        for match_batches, score_batches in ((2, 3), (3, 2)):
            with self.subTest(matches=match_batches, scores=score_batches):
                matches = torch.tensor([[1, -1, 0]]).expand(match_batches, -1)
                scores = torch.arange(score_batches * 3).reshape(score_batches, 3)
                result = pack_matches({"matches0": matches, "matching_scores0": scores})
                self.assertEqual(len(result["matches"]), match_batches)
                self.assertEqual(
                    len(result["scores"]), min(match_batches, score_batches)
                )
                for i, selected in enumerate(result["scores"]):
                    torch.testing.assert_close(selected, scores[i][[0, 2]])

    def test_mismatched_row_widths_still_raise(self):
        matches = torch.tensor([[0, -1, -1]])
        for width in (1, 4):
            with self.subTest(width=width), self.assertRaises(IndexError):
                pack_matches(
                    {"matches0": matches, "matching_scores0": torch.ones(1, width)}
                )

    def test_returned_match_mutation_preserves_score_backward(self):
        for indices in ([2, -1, 0], [2, -1, -1]):
            for numpy, column in ((False, 0), (False, 1), (True, 0)):
                with self.subTest(indices=indices, numpy=numpy, column=column):
                    scores = torch.tensor(
                        [[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True
                    )
                    output = pack_matches(
                        {
                            "matches0": torch.tensor([indices]),
                            "matching_scores0": scores,
                        }
                    )
                    matches = output["matches"][0]
                    if numpy:
                        matches = matches.numpy()
                    matches[:, column] = 1
                    output["scores"][0].sum().backward()
                    expected = scores.new_tensor([[int(i >= 0) for i in indices]])
                    torch.testing.assert_close(scores.grad, expected)


if __name__ == "__main__":
    unittest.main()
