from baseball_pose.features.angles import angle_degrees


def test_angle_degrees_right_angle():
    assert angle_degrees((1, 0), (0, 0), (0, 1)) == 90
