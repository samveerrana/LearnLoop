from learnloop.web_learning_cycle import compare


def test_compare_accepts_only_positive_weight_gain() -> None:
    result = compare({"correct": 18, "total": 59}, {"correct": 29, "total": 59})
    assert result["accepted"]
    assert result["absolute_gain_questions"] == 11
    assert result["percentage_point_gain"] == 18.64
    assert result["task_score_multiplier"] == 1.611


def test_compare_rejects_equal_score() -> None:
    assert not compare({"correct": 10, "total": 20}, {"correct": 10, "total": 20})["accepted"]
