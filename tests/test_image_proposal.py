import numpy as np

from baseball_pose.preprocessing.image_proposal import (
    ImageProposal,
    ImageProposalTracker,
    create_center_motion_grabcut_proposal,
)
from baseball_pose.preprocessing.image_proposal_config import image_proposal_roi_config
from baseball_pose.preprocessing.roi import RoiBox


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


def test_image_proposal_tracker_follows_gradual_right_shift():
    tracker = ImageProposalTracker(
        initial_width_ratio=0.56,
        center_width_ratio=0.56,
        max_center_step=0.04,
        center_smoothing=1.0,
        max_width_step=0.04,
        width_smoothing=1.0,
        warmup_frames=0,
    )

    for center in [0.50, 0.52, 0.55, 0.58]:
        mask = np.zeros((120, 200), dtype=np.uint8)
        x = round(center * 200)
        mask[20:110, x - 12 : x + 12] = 255
        proposal = ImageProposal(
            mask=mask,
            roi=RoiBox(x - 20, 0, 40, 120),
            candidate_count=1,
            center_x=tracker.center_x,
            subject_center_x=center,
        )
        tracker.update(proposal)

    assert tracker.center_x > 0.55

    jump_mask = np.zeros((120, 200), dtype=np.uint8)
    jump_mask[20:110, 170:194] = 255
    tracker.update(
        ImageProposal(
            mask=jump_mask,
            roi=RoiBox(160, 0, 40, 120),
            candidate_count=1,
            center_x=tracker.center_x,
            subject_center_x=0.91,
        )
    )

    assert tracker.center_x < 0.63
    assert tracker.center_width_ratio >= 0.56


def test_image_proposal_tracker_ignores_roi_center_pulled_by_background():
    tracker = ImageProposalTracker(
        max_center_step=0.04,
        center_smoothing=1.0,
        warmup_frames=0,
    )
    mask = np.zeros((120, 220), dtype=np.uint8)
    mask[20:110, 85:115] = 255
    mask[0:120, 120:210] = 255

    proposal = ImageProposal(
        mask=mask,
        roi=RoiBox(85, 0, 125, 120),
        candidate_count=1,
        center_x=0.5,
        subject_center_x=0.5,
    )
    tracker.update(proposal)

    assert tracker.center_x < 0.53


def test_image_proposal_roi_config_applies_clip_override():
    config = {
        "conditions": {
            "image_center_motion_grabcut_pose": {
                "roi": {
                    "center_x": 0.5,
                    "center_width_ratio": 0.62,
                    "tracker_max_offset": 0.12,
                    "clip_overrides": {
                        "pitching_2": {
                            "center_x": 0.42,
                            "center_width_ratio": 0.76,
                        }
                    },
                }
            }
        }
    }

    batting_config = image_proposal_roi_config(config, "batting_1")
    pitching_config = image_proposal_roi_config(config, "pitching_2")

    assert "clip_overrides" not in pitching_config
    assert batting_config["center_x"] == 0.5
    assert pitching_config["center_x"] == 0.42
    assert pitching_config["center_width_ratio"] == 0.76
    assert pitching_config["tracker_max_offset"] == 0.12
