from baseball_pose.config import load_config


def test_default_config_loads():
    config = load_config("configs/default.yaml")

    assert config.clip_ids == ["batting_1", "batting_2", "pitching_1", "pitching_2"]
    assert config.condition_ids == ["baseline_raw", "roi_clahe", "roi_clahe_smooth"]
