import unittest

import torch

from flexglue.utils.ops import pack_matches


class PackMatchesTest(unittest.TestCase):
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
