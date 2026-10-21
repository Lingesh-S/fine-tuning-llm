import pytest

from tamil_lora.config import TrainConfig, parse_args


def test_defaults_match_original_experiment():
    config = parse_args([])
    assert config.model_name == "distilgpt2"
    assert config.dataset == "tniranjan/aitamilnadu_tamil_stories_no_instruct"
    assert (config.lora_r, config.lora_alpha) == (8, 32)
    assert config.max_steps == -1


def test_overrides():
    config = parse_args(["--max-steps", "5", "--target-modules", "c_attn", "c_proj", "--push-to-hub", "me/model"])
    assert config.max_steps == 5
    assert config.target_modules == ["c_attn", "c_proj"]
    assert config.push_to_hub == "me/model"


@pytest.mark.parametrize("bad", [["--test-size", "0"], ["--test-size", "1.5"], ["--train-samples", "1"]])
def test_invalid_values_rejected(bad):
    with pytest.raises(SystemExit):
        parse_args(bad)


def test_config_serialises():
    assert TrainConfig().to_dict()["target_modules"] == ["c_attn"]
