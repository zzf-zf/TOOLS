import unittest

from extractor import EvidenceExtractor


class EvidenceExtractorTest(unittest.TestCase):
    def test_extract_multiline_block(self):
        extractor = EvidenceExtractor()
        response = """
        Reasoning...
        <original_evidence>
        Albert Einstein was born in Ulm.
        Ulm is in Germany.
        </original_evidence>
        <answer>Ulm</answer>
        """

        result = extractor.extract(response)

        self.assertTrue(result.valid)
        self.assertEqual(
            result.last_text,
            "Albert Einstein was born in Ulm.\n        Ulm is in Germany.",
        )

    def test_reject_missing_closing_tag(self):
        result = EvidenceExtractor().extract(
            "<original_evidence>Unclosed evidence"
        )
        self.assertFalse(result.valid)
        self.assertIn("Malformed", result.error)

    def test_multiple_blocks_are_configurable(self):
        response = (
            "<evidence>First fact</evidence>"
            "<evidence>Second fact</evidence>"
        )
        self.assertFalse(EvidenceExtractor(tag="evidence").extract(response).valid)
        self.assertTrue(
            EvidenceExtractor(
                tag="evidence", allow_multiple=True
            ).extract(response).valid
        )

    def test_grounding(self):
        result = EvidenceExtractor.check_grounding(
            ["Einstein was born in Ulm."],
            ["Biography: Einstein   was born in ULM."],
        )
        self.assertTrue(result.grounded)
        self.assertEqual(result.source_indices, (0,))

    def test_ungrounded_evidence(self):
        result = EvidenceExtractor.check_grounding(
            ["Einstein was born in Berlin."],
            ["Einstein was born in Ulm."],
        )
        self.assertFalse(result.grounded)
        self.assertEqual(result.source_indices, (None,))


if __name__ == "__main__":
    unittest.main()
