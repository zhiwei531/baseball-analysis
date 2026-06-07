from baseball_pose.pose.rtmpose_pose import _coco_wholebody133_to_halpe26


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
