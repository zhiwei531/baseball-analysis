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


def test_image_proposal_trims_people_attached_to_center_body():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    image[25:145, 88:112] = 255
    image[45:135, 45:75] = 255
    image[45:135, 125:155] = 255
    image[80:92, 45:155] = 255
    previous = np.zeros_like(image)

    proposal = create_center_motion_grabcut_proposal(
        image=image,
        previous_image=previous,
        background_subtractor=None,
        center_width_ratio=0.7,
        min_area_ratio=0.001,
        grabcut_iterations=1,
        vertical_body_width_ratio=0.18,
    )

    assert proposal.mask[:, 90:110].sum() > 0
    assert proposal.mask[:, :65].sum() == 0
    assert proposal.mask[:, 135:].sum() == 0
    assert proposal.roi.width < 90


def test_image_proposal_tightens_lower_left_attached_person():
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    image[20:150, 96:122] = 255
    image[25:75, 70:100] = 255
    image[85:160, 50:90] = 255
    image[100:112, 50:122] = 255
    previous = np.zeros_like(image)

    proposal = create_center_motion_grabcut_proposal(
        image=image,
        previous_image=previous,
        background_subtractor=None,
        center_width_ratio=0.7,
        min_area_ratio=0.001,
        grabcut_iterations=1,
        vertical_body_width_ratio=0.2,
    )

    assert proposal.mask[25:75, 76:96].sum() > 0
    assert proposal.mask[105:150, :82].sum() == 0
