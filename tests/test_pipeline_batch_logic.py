from src.utils.pipeline_helpers import collect_wbc_crops


class _DetectorStub:
    def extract_wbc_crops(self, image_result):
        return image_result.get("crops", [])


def test_collect_wbc_crops_across_all_images():
    detector = _DetectorStub()

    per_image = [
        {
            "image_path": "img1.jpg",
            "crops": [{"image": "crop1", "index": 0}],
        },
        {
            "image_path": "img2.jpg",
            "crops": [{"image": "crop2", "index": 3}, {"image": "crop3", "index": 5}],
        },
    ]

    crops = collect_wbc_crops(detector, per_image)

    assert len(crops) == 3
    assert crops[0]["source_image_path"] == "img1.jpg"
    assert crops[1]["source_image_path"] == "img2.jpg"
    assert crops[2]["source_image_index"] == 1
