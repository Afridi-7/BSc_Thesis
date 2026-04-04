from src.classification.classifier import WBCClassifier


def test_uncertainty_summary_has_consistent_keys():
    classifier = WBCClassifier.__new__(WBCClassifier)

    summary = classifier.create_uncertainty_summary(
        [
            {"uncertainty_level": "LOW", "confidence": 0.9, "entropy": 0.1, "variance": 0.01, "flagged": False},
            {"uncertainty_level": "HIGH", "confidence": 0.4, "entropy": 0.8, "variance": 0.2, "flagged": True},
        ]
    )

    assert summary["flagged_count"] == 1
    assert summary["flagged_samples"] == 1
    assert summary["total_samples"] == 2
