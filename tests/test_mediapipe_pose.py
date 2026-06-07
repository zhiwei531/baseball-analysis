from baseball_pose.pose.mediapipe_pose import MEDIAPIPE_TO_CANONICAL, MEDIAPIPE_TO_FULL
from baseball_pose.pose.schema import CANONICAL_JOINTS, COCO17_JOINTS, HALPE26_JOINTS, MEDIAPIPE_JOINTS


def test_mediapipe_full_mapping_preserves_all_landmarks():
    assert len(MEDIAPIPE_TO_FULL) == 33
    assert tuple(MEDIAPIPE_TO_FULL.values()) == MEDIAPIPE_JOINTS
    assert set(MEDIAPIPE_TO_CANONICAL.values()) == set(CANONICAL_JOINTS)


def test_alternate_pose_schemas_keep_canonical_core_names():
    assert set(CANONICAL_JOINTS).issubset(set(COCO17_JOINTS))
    assert set(CANONICAL_JOINTS).issubset(set(HALPE26_JOINTS))
