from learnloop.reference_preflight import (
    MIN_PHYSICAL_MEMORY_BYTES,
    MODEL_BYTES,
    RESERVE_BYTES,
    assess,
)


def test_preflight_refuses_when_reserve_would_be_violated() -> None:
    report = assess(MODEL_BYTES + RESERVE_BYTES - 1)
    assert not report["safe_to_pull"]
    assert report["shortfall_bytes"] == 1


def test_preflight_accepts_exact_required_space() -> None:
    report = assess(MODEL_BYTES + RESERVE_BYTES)
    assert report["safe_to_pull"]
    assert report["safe_to_run_locally"]
    assert report["total_parameters_b"] == 21.0
    assert report["active_parameters_b"] == 3.6


def test_preflight_rejects_unsafe_16_gib_local_run_without_blocking_download() -> None:
    report = assess(
        MODEL_BYTES + RESERVE_BYTES,
        physical_bytes=16 * 1024**3,
    )
    assert report["safe_to_download"]
    assert report["safe_to_pull"]
    assert not report["safe_to_run_locally"]
    assert report["memory_shortfall_bytes"] == MIN_PHYSICAL_MEMORY_BYTES - 16 * 1024**3
