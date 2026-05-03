import numpy as np

from baseball_pose.preprocessing.image_proposal import create_center_motion_grabcut_proposal


def test_image_proposal_selects_center_foreground_blob():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:100, 65:95] = 255
    previous = np.zeros_like(image)

    proposal = create_center_motion_grabcut_proposal(
        image=image,
        previous_image=previous,
        background_subtractor=None,
        center_width_ratio=0.5,
        min_area_ratio=0.002,
        grabcut_iterations=1,
    )

    assert proposal.mask.sum() > 0
    assert 50 <= proposal.roi.x <= 80
    assert proposal.roi.width < 80
