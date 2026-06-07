from baseball_pose.pose.rtmpose_pose import (
    _coco_wholebody133_to_halpe26,
    _fuse_coco17_body_with_wholebody_feet,
)


def test_coco_wholebody133_to_halpe26_projects_body_and_feet():
    keypoints = [[float(index), float(index + 100)] for index in range(133)]
    scores = [index / 100.0 for index in range(133)]

    projected_keypoints, projected_scores = _coco_wholebody133_to_halpe26(keypoints, scores)

    assert len(projected_keypoints) == 26
    assert len(projected_scores) == 26
    assert projected_keypoints[:17] == keypoints[:17]
    assert projected_keypoints[17] == keypoints[0]
    assert projected_keypoints[18] == [5.5, 105.5]
    assert projected_keypoints[19] == [11.5, 111.5]
    assert projected_keypoints[20:] == [
        keypoints[17],
        keypoints[20],
        keypoints[18],
        keypoints[21],
        keypoints[19],
        keypoints[22],
    ]


def test_fused_halpe26_anchors_implausible_wholebody_feet():
    body_keypoints = [[float(index), float(index + 100)] for index in range(17)]
    body_scores = [0.9 for _ in range(17)]
    wholebody_keypoints = [[float(index), float(index + 100)] for index in range(133)]
    wholebody_scores = [0.5 for _ in range(133)]
    wholebody_keypoints[17] = [500.0, 500.0]
    wholebody_keypoints[18] = [700.0, 700.0]
    wholebody_keypoints[19] = [900.0, 900.0]

    projected_keypoints, projected_scores = _fuse_coco17_body_with_wholebody_feet(
        body_keypoints,
        body_scores,
        wholebody_keypoints,
        wholebody_scores,
    )

    assert projected_keypoints[20] == body_keypoints[15]
    assert projected_keypoints[22] == body_keypoints[15]
    assert projected_keypoints[24] == body_keypoints[15]
    assert projected_scores[20] == 0.35
