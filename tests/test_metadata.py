from baseball_pose.io.metadata import load_clips


def test_clip_metadata_loads():
    clips = load_clips("data/metadata/clips.csv")

    assert {clip.clip_id for clip in clips} == {"batting_1", "batting_2", "pitching_1", "pitching_2"}
    assert {clip.action_type for clip in clips} == {"batting", "pitching"}
