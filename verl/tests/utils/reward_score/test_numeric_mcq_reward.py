from verl.utils.reward_score.tool import compute_score_fns, compute_score_numeric_mcq


def test_numeric_mcq_reward_extracts_answer_tag_number():
    assert compute_score_numeric_mcq("<answer>3</answer>", "3") == 1
    assert compute_score_numeric_mcq("<answer>Option 3</answer>", "3") == 1
    assert compute_score_numeric_mcq("<answer>3. Gleason pattern</answer>", "3") == 1
    assert compute_score_numeric_mcq("<answer>The answer is option 3, not 4</answer>", "3") == 1
    assert compute_score_numeric_mcq("<answer>2</answer>", "3") == 0
    assert compute_score_numeric_mcq("3", "3") == 0


def test_numeric_mcq_reward_uses_last_answer_tag_from_last_assistant_turn():
    solution = (
        "<|im_start|>assistant\n"
        "<answer>1</answer><|im_end|>\n"
        "<|im_start|>tool\n"
        "observation<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<thinking>done</thinking><answer>4</answer><|im_end|>"
    )

    assert compute_score_numeric_mcq(solution, "4") == 1
    assert compute_score_numeric_mcq(solution, "1") == 0


def test_numeric_mcq_reward_is_registered():
    assert compute_score_fns["em_score_numeric_mcq"] is compute_score_numeric_mcq
