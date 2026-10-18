"""
Tests for technique-driven test design: boundary analysis, concrete data,
assertions on every case, and requirement coverage.
"""
import pytest
from state import create_initial_state
from agents.test_design_agent import (
    TestDesignAgent, validate_test_case, _extract_bounds, _design_for_requirement,
)

REQS = [
    {"id": "REQ-001", "description": "The user shall log in using email and password; the account locks after 5 failed attempts.", "category": "Security"},
    {"id": "REQ-002", "description": "The password shall be between 8 and 64 characters.", "category": "Security"},
    {"id": "REQ-003", "description": "The user may transfer an amount of at least 1 and at most 50000.", "category": "Functional"},
    {"id": "REQ-004", "description": "The system shall display the account summary page.", "category": "UI"},
]


def test_extract_bounds():
    assert _extract_bounds("between 8 and 64 characters") == (8, 64)
    assert _extract_bounds("at least 1 and at most 50000") == (1, 50000)
    assert _extract_bounds("up to 5 MB") == (None, 5)
    assert _extract_bounds("no numbers here") == (None, None)


def test_password_boundaries_have_exact_lengths():
    cases = _design_for_requirement(REQS[1], 2)
    pw = {c["title"]: c["test_data"]["password"] for c in cases if "minimum" in c["title"] or "maximum" in c["title"]}
    lengths = sorted(len(v) for v in pw.values())
    assert lengths == [7, 8, 65]


def test_numeric_bva_covers_n_minus_1_n_n_plus_1():
    values = {c["test_data"].get("value") for c in _design_for_requirement(REQS[2], 3)}
    assert {0, 1, 2, 50000, 50001} <= values


def test_account_word_does_not_trigger_numeric_rules():
    req = {"id": "REQ-9", "description": "The system shall show the account summary.", "category": "UI"}
    titles = " ".join(c["title"].lower() for c in _design_for_requirement(req, 9))
    assert "integer overflow" not in titles and "negative number" not in titles and "minimum" not in titles


@pytest.mark.asyncio
async def test_agent_output_is_valid_and_covers_every_requirement():
    state = create_initial_state("x", "https://demo.example.com")
    state["validated_requirements"] = REQS
    state = await TestDesignAgent().invoke(state)
    tcs = state["test_cases"]

    assert {t["requirement_id"] for t in tcs} == {r["id"] for r in REQS}
    for t in tcs:
        assert validate_test_case(t) == [], (t["id"], validate_test_case(t))
        assert t["id"] == t["tc_id"]
    assert len({t["id"] for t in tcs}) == len(tcs)
    for r in REQS:
        mine = [t for t in tcs if t["requirement_id"] == r["id"]]
        assert any(t["type"] == "Positive" or "Accept" in t["title"] for t in mine)
        assert any(t["type"] != "Positive" for t in mine)


@pytest.mark.asyncio
async def test_more_than_chunk_size_requirements_all_designed():
    state = create_initial_state("x", "https://demo.example.com")
    state["validated_requirements"] = [
        {"id": f"REQ-{i:03d}", "description": "The user shall enter a name.", "category": "Functional"} for i in range(1, 13)
    ]
    state = await TestDesignAgent().invoke(state)
    assert {t["requirement_id"] for t in state["test_cases"]} == {f"REQ-{i:03d}" for i in range(1, 13)}


def test_validator_rejects_generic_or_assertionless_cases():
    base = {"type": "Negative", "requirement_id": "REQ-1", "steps": ["a"], "test_data": {"x": "1"},
            "expected_result": "Rejected", "assertions": [{"type": "no_server_error"}]}
    assert validate_test_case(base) == []
    assert "no assertions" in validate_test_case({**base, "assertions": []})
    assert "generic expected_result" in validate_test_case({**base, "expected_result": "System shows an appropriate error"})
    assert "non-positive case without concrete test_data" in validate_test_case({**base, "test_data": {}})


@pytest.mark.asyncio
async def test_tier_weighting_and_sorting():
    state = create_initial_state("x", "https://demo.example.com")
    state["validated_requirements"] = [
        {"id": "REQ-001", "description": "The user shall log in using email and password.", "category": "Security"},
        {"id": "REQ-002", "description": "The system shall update the profile picture.", "category": "UI"},
        {"id": "REQ-003", "description": "The user may transfer an amount of at least 1 and at most 500.", "category": "Functional"},
    ]
    state["risk_matrix"] = [
        {"requirement_id": "REQ-001", "tier": 1}, {"requirement_id": "REQ-002", "tier": 3},
        {"requirement_id": "REQ-003", "tier": 2},
    ]
    tcs = (await TestDesignAgent().invoke(state))["test_cases"]

    assert [t["tier"] for t in tcs] == sorted(t["tier"] for t in tcs)          # Tier 1 first
    by_req = lambda rid: [t for t in tcs if t["requirement_id"] == rid]
    assert {"Positive", "Negative", "Security", "Boundary"} <= {t["type"] for t in by_req("REQ-001")}
    assert {t["type"] for t in by_req("REQ-002")} == {"Positive"}               # Tier 3 -> Positive only
    assert {t["type"] for t in by_req("REQ-003")} <= {"Positive", "Negative"}   # Tier 2 -> no Boundary/Security
    assert len(by_req("REQ-003")) >= 1
    assert all(t["tier"] == {"REQ-001": 1, "REQ-002": 3, "REQ-003": 2}[t["requirement_id"]] for t in tcs)


@pytest.mark.asyncio
async def test_tier_defaults_to_2_without_risk_data():
    state = create_initial_state("x", "https://demo.example.com")
    state["validated_requirements"] = REQS[:1]
    tcs = (await TestDesignAgent().invoke(state))["test_cases"]
    assert {t["tier"] for t in tcs} == {2}
