import hashlib
from pathlib import Path

from methods.state_bridge import StateBridge, TASK_CONFIG
from prompts import EMBEDDING_HINT_MARKER, build_agent_message_embedding_mas


ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_4b_core_task_token_limits():
    assert {task: TASK_CONFIG[task]["max_new_tokens"] for task in (
        "arc_challenge", "medqa", "gsm8k", "mbppplus", "humanevalplus"
    )} == {
        "arc_challenge": 2048,
        "medqa": 8192,
        "gsm8k": 2048,
        "mbppplus": 4096,
        "humanevalplus": 4096,
    }


def test_released_method_defaults():
    defaults = StateBridge.__init__.__kwdefaults__
    assert defaults["temperature"] == 0.6
    assert defaults["top_p"] == 0.95
    assert defaults["max_prefix_tokens"] == 64
    assert defaults["adaptive_reg"] == 1e-3
    assert defaults["snap_ratio"] == 0.3
    assert defaults["use_hook"] is True


def test_receiver_prompt_contains_one_prefix_marker():
    class Args:
        model = "Qwen/Qwen3-4B"
        task = "medqa"

    messages = build_agent_message_embedding_mas(
        role="critic", question="example", args=Args(), has_prefix=True
    )
    assert messages[1]["content"].count(EMBEDDING_HINT_MARKER) == 1


def test_bundled_medqa_hash():
    digest = hashlib.sha256((ROOT / "data" / "medqa.json").read_bytes()).hexdigest()
    assert digest == "d4f09708b623a7750013b22c607495eb8872bbef7a3f50990b7279c8fd87ec8d"
