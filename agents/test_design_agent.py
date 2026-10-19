"""
Test Design Agent - Technique-driven test case design.

Instead of one generic positive + negative case per requirement, this agent
applies named test design techniques (equivalence partitioning, boundary value
analysis, error guessing, security payloads, state checks) and emits test cases
that carry concrete test data and machine-checkable assertions, so the script
generation stage can turn each case into a real pass/fail check.

Tier-aware: the risk agent's tier for each requirement decides which test types it
gets (TIER_TEST_TYPES), and the final suite is sorted Tier 1 first.

Test case schema:
    tier              1 | 2 | 3 (from the risk agent; 1 = highest priority)
    id / tc_id        "TC-001" (both keys kept for downstream compatibility)
    requirement_id    traceability back to the requirement
    type              Positive | Negative | Boundary | Security
    technique         e.g. "Boundary Value Analysis"
    title, preconditions, steps, expected_result, gherkin
    test_data         concrete inputs, e.g. {"password": "Ab1!xyz"}
    expected_error    short description of the rejection expected (or None)
    assertions        list of {"type": ..., "value": ...} - see ASSERTION_TYPES
"""
import json
import re
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any, Optional

CHUNK_SIZE = 5          # requirements per LLM call (keeps output quality up)
MAX_PER_REQ = 12        # cap on deterministic cases per requirement
TYPES = ("Positive", "Negative", "Boundary", "Security")

# Tier 1 -> 4 test types, Tier 2 -> 2, Tier 3 -> 1 (each type may hold several technique-driven cases)
TIER_TEST_TYPES: Dict[int, List[str]] = {
    1: ["Positive", "Negative", "Security", "Boundary"],
    2: ["Positive", "Negative"],
    3: ["Positive"],
}

# Assertion vocabulary shared with the script generation stage.
ASSERTION_TYPES = {
    "error_shown",       # value: list of keywords, any one must appear in a visible error/validation message
    "no_error_shown",    # no visible error/validation message
    "text_visible",      # value: text that must be visible on the page
    "url_contains",      # value: substring expected in the current URL
    "authenticated",     # user is logged in (e.g. logout control / dashboard visible)
    "not_authenticated", # no access granted (still on login / protected content absent)
    "no_server_error",   # no HTTP 5xx, stack trace or "Internal Server Error" text
    "no_dialog",         # no JavaScript alert/confirm/prompt fired (XSS check)
}

GENERIC_PHRASES = (
    "appropriate error", "descriptive error", "success status",
    "according to specification", "responds correctly", "works as expected",
)


# ---------------------------------------------------------------- assertions
def _err(*keywords: str) -> Dict[str, Any]:
    return {"type": "error_shown", "value": list(keywords)}

NO_ERR = {"type": "no_error_shown"}
NO_5XX = {"type": "no_server_error"}
NO_DIALOG = {"type": "no_dialog"}
AUTHED = {"type": "authenticated"}
NOT_AUTHED = {"type": "not_authenticated"}


def _url(fragment: str) -> Dict[str, Any]:
    return {"type": "url_contains", "value": fragment}


# ------------------------------------------------------- requirement analysis
def _extract_bounds(desc: str):
    """Pull stated numeric limits (min, max) out of requirement text."""
    d = desc.lower()
    lo = hi = None
    m = (re.search(r"between\s+(\d+)\s*(?:and|to|-)\s*(\d+)", d)
         or re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:characters|chars|digits)", d))
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:at least|minimum(?: of)?|min\.?)\s*(\d+)", d)
    if m and lo is None:
        lo = int(m.group(1))
    m = re.search(r"(?:at most|maximum(?: of)?|max\.?|up to|no more than|not exceed(?:ing)?)\s*(\d+)", d)
    if m and hi is None:
        hi = int(m.group(1))
    return lo, hi


def _detect(desc: str, category: str) -> Dict[str, bool]:
    d = f"{desc} {category}".lower()

    def has(*patterns: str) -> bool:
        return any(re.search(p, d) for p in patterns)

    return {
        "auth": has(r"\blog[- ]?in\b", r"\bsign[- ]?in\b", r"\bauthenticat"),
        "email": has(r"\be-?mail\b"),
        "password": has(r"\bpass(word|code|phrase)"),
        "numeric": has(r"\bamount\b", r"\bquantity\b", r"\bage\b", r"\bprice\b", r"\bbalance\b",
                       r"\blimit\b", r"\bnumber of\b", r"\bcount\b"),
        "text": has(r"\bname\b", r"\busername\b", r"\bsearch", r"\bcomment", r"\bmessage\b",
                    r"\baddress\b", r"\btitle\b", r"\btext\b", r"\binput"),
        "upload": has(r"\bupload", r"\battachment"),
        "session": has(r"\bsession", r"\btime[- ]?out", r"\bexpire", r"\bidle\b"),
        "lockout": has(r"\block(ed|out|s)?\b", r"\btoo many\b", r"\bfailed attempt"),
    }


def _short(v: Any, n: int = 40) -> str:
    s = str(v)
    return s if len(s) <= n else f"{s[:n]}... ({len(s)} chars)"


def _pw(n: int) -> str:
    """Password-like string of exactly n characters (mixed classes when n >= 3)."""
    return ("A1!" + "a" * n)[:n]


def _case(ctype, technique, title, data, expected, assertions, error=None):
    return {"type": ctype, "technique": technique, "title": title, "test_data": data,
            "expected_result": expected, "expected_error": error, "assertions": assertions}


# --------------------------------------------------------- case generators
def _auth_cases(flags):
    valid = {"email": "qa.user@example.com", "password": "ValidSecurePass123!"}
    cases = [
        _case("Positive", "Equivalence Partitioning", "Login with valid credentials", valid,
              "User is authenticated and reaches the post-login page", [AUTHED, NO_5XX]),
        _case("Negative", "Equivalence Partitioning", "Login with valid email and wrong password",
              {"email": valid["email"], "password": "WrongPass999!"},
              "Login is rejected and no session is created", [_err("invalid", "incorrect", "failed"), NOT_AUTHED],
              "Invalid credentials message"),
        _case("Negative", "Equivalence Partitioning", "Login with unregistered email",
              {"email": "nobody.registered@example.com", "password": "ValidSecurePass123!"},
              "Login is rejected and no session is created", [_err("invalid", "incorrect", "not found"), NOT_AUTHED],
              "Invalid credentials message"),
        _case("Negative", "Error Guessing", "Login with both fields empty", {"email": "", "password": ""},
              "Form is not submitted successfully; required-field feedback is shown",
              [_err("required", "enter", "empty", "invalid"), NOT_AUTHED], "Required field message"),
        _case("Negative", "Equivalence Partitioning", "Login with malformed email (missing @)",
              {"email": "userexample.com", "password": "ValidSecurePass123!"},
              "Malformed email is rejected before authentication", [_err("email", "invalid", "valid"), NOT_AUTHED],
              "Invalid email format message"),
        _case("Security", "Security Payload", "SQL injection in email field",
              {"email": "' OR '1'='1' --", "password": "x"},
              "Injection string is treated as data; login is rejected without a server error",
              [NOT_AUTHED, NO_5XX], "Invalid credentials message"),
        _case("Security", "Security Payload", "SQL injection in password field",
              {"email": valid["email"], "password": "' OR '1'='1"},
              "Injection string is treated as data; login is rejected without a server error",
              [NOT_AUTHED, NO_5XX], "Invalid credentials message"),
    ]
    if flags["lockout"]:
        cases.append(_case("Negative", "State Transition", "Account locks after repeated failed logins",
                           {"email": valid["email"], "password": "WrongPass999!", "attempts": 6},
                           "After repeated failures the account is locked or throttled, even for a later correct password",
                           [_err("locked", "too many", "try again", "attempts"), NOT_AUTHED],
                           "Account locked / too many attempts message"))
    return cases


def _email_cases():
    return [
        _case("Positive", "Equivalence Partitioning", "Accept well-formed email", {"email": "qa.user@example.com"},
              "Email is accepted", [NO_ERR, NO_5XX]),
        _case("Negative", "Equivalence Partitioning", "Reject email without @", {"email": "userexample.com"},
              "Email is rejected", [_err("email", "invalid", "valid")], "Invalid email format message"),
        _case("Negative", "Equivalence Partitioning", "Reject email without domain", {"email": "user@"},
              "Email is rejected", [_err("email", "invalid", "valid")], "Invalid email format message"),
        _case("Boundary", "Boundary Value Analysis", "Reject 255-character email local part",
              {"email": "a" * 255 + "@example.com"}, "Over-length email is rejected without a server error",
              [_err("email", "invalid", "long", "valid"), NO_5XX], "Invalid / too long message"),
    ]


def _password_cases(desc):
    lo, hi = _extract_bounds(desc)
    cases = [_case("Positive", "Equivalence Partitioning", "Accept strong password", {"password": "ValidSecurePass123!"},
                   "Password is accepted", [NO_ERR, NO_5XX]),
             _case("Negative", "Error Guessing", "Reject empty password", {"password": ""},
                   "Empty password is rejected", [_err("required", "enter", "password", "empty")], "Required field message"),
             _case("Negative", "Error Guessing", "Reject whitespace-only password", {"password": "        "},
                   "Whitespace-only password is rejected", [_err("password", "invalid", "required")], "Invalid password message")]
    if lo:
        cases.append(_case("Boundary", "Boundary Value Analysis", f"Reject password one below minimum ({lo - 1} chars)",
                           {"password": _pw(lo - 1)}, "Too-short password is rejected",
                           [_err("at least", "minimum", "short", "characters", "password")], "Minimum length message"))
        cases.append(_case("Boundary", "Boundary Value Analysis", f"Accept password at minimum length ({lo} chars)",
                           {"password": _pw(lo)}, "Minimum-length password is accepted", [NO_ERR, NO_5XX]))
    else:
        cases.append(_case("Boundary", "Error Guessing", "Reject single-character password", {"password": "a"},
                           "Trivially short password is rejected", [_err("password", "short", "characters", "weak")],
                           "Password too short message"))
    if hi:
        cases.append(_case("Boundary", "Boundary Value Analysis", f"Reject password one above maximum ({hi + 1} chars)",
                           {"password": _pw(hi + 1)}, "Over-length password is rejected",
                           [_err("at most", "maximum", "long", "characters", "password")], "Maximum length message"))
    return cases


def _numeric_cases(desc):
    lo, hi = _extract_bounds(desc)
    cases = []
    if lo is not None or hi is not None:
        if lo is not None:
            cases += [
                _case("Boundary", "Boundary Value Analysis", f"Reject value just below minimum ({lo - 1})", {"value": lo - 1},
                      "Out-of-range value is rejected", [_err("minimum", "at least", "invalid", "range")], "Out of range message"),
                _case("Boundary", "Boundary Value Analysis", f"Accept value at minimum ({lo})", {"value": lo},
                      "Minimum value is accepted", [NO_ERR, NO_5XX]),
                _case("Boundary", "Boundary Value Analysis", f"Accept value just above minimum ({lo + 1})", {"value": lo + 1},
                      "Value is accepted", [NO_ERR, NO_5XX])]
        if hi is not None:
            cases += [
                _case("Boundary", "Boundary Value Analysis", f"Accept value at maximum ({hi})", {"value": hi},
                      "Maximum value is accepted", [NO_ERR, NO_5XX]),
                _case("Boundary", "Boundary Value Analysis", f"Reject value just above maximum ({hi + 1})", {"value": hi + 1},
                      "Out-of-range value is rejected", [_err("maximum", "at most", "invalid", "range", "exceed")], "Out of range message")]
    else:
        cases.append(_case("Positive", "Equivalence Partitioning", "Accept typical valid number", {"value": 10},
                           "Value is accepted", [NO_ERR, NO_5XX]))
    cases += [
        _case("Negative", "Error Guessing", "Reject negative number", {"value": -1},
              "Negative value is rejected", [_err("invalid", "positive", "minimum", "greater")], "Invalid value message"),
        _case("Negative", "Error Guessing", "Reject non-numeric input", {"value": "abc"},
              "Non-numeric input is rejected", [_err("number", "numeric", "invalid")], "Invalid number message"),
        _case("Boundary", "Error Guessing", "Handle integer overflow value (2147483648)", {"value": 2147483648},
              "Value beyond 32-bit range is rejected or handled without a server error", [NO_5XX]),
    ]
    return cases


def _text_cases(desc):
    _, hi = _extract_bounds(desc)
    cases = [
        _case("Positive", "Equivalence Partitioning", "Accept typical text value", {"value": "Sample Value"},
              "Value is accepted", [NO_ERR, NO_5XX]),
        _case("Positive", "Equivalence Partitioning", "Accept unicode text", {"value": "Ünïcödé 名前"},
              "Unicode value is accepted and not corrupted", [NO_ERR, NO_5XX]),
        _case("Negative", "Error Guessing", "Reject empty required text", {"value": ""},
              "Empty value is rejected", [_err("required", "enter", "empty", "invalid")], "Required field message"),
        _case("Negative", "Error Guessing", "Reject whitespace-only text", {"value": "     "},
              "Whitespace-only value is rejected", [_err("required", "enter", "empty", "invalid")], "Required field message"),
        _case("Security", "Security Payload", "XSS payload is not executed", {"value": "<script>alert(1)</script>"},
              "Script is escaped or rejected; no JavaScript dialog fires", [NO_DIALOG, NO_5XX]),
        _case("Security", "Security Payload", "SQL injection payload is handled safely", {"value": "'; DROP TABLE users; --"},
              "Payload is treated as plain data without a server error", [NO_5XX]),
    ]
    if hi:
        cases.append(_case("Boundary", "Boundary Value Analysis", f"Reject text one over maximum ({hi + 1} chars)",
                           {"value": "a" * (hi + 1)}, "Over-length text is rejected",
                           [_err("maximum", "too long", "characters", "limit")], "Maximum length message"))
        cases.append(_case("Boundary", "Boundary Value Analysis", f"Accept text at maximum ({hi} chars)",
                           {"value": "a" * hi}, "Maximum-length text is accepted", [NO_ERR, NO_5XX]))
    else:
        cases.append(_case("Boundary", "Error Guessing", "Handle very long text (1001 chars)", {"value": "a" * 1001},
                           "Very long input is rejected or truncated without a server error", [NO_5XX]))
    return cases


def _upload_cases(desc):
    _, hi = _extract_bounds(desc)
    return [
        _case("Positive", "Equivalence Partitioning", "Accept valid file", {"file_name": "sample.pdf", "size_bytes": 20480},
              "File is uploaded", [NO_ERR, NO_5XX]),
        _case("Negative", "Equivalence Partitioning", "Reject disallowed file type", {"file_name": "payload.exe", "size_bytes": 20480},
              "Executable file is rejected", [_err("type", "format", "not allowed", "invalid", "unsupported")], "Unsupported file type message"),
        _case("Negative", "Error Guessing", "Reject empty (0 byte) file", {"file_name": "empty.pdf", "size_bytes": 0},
              "Empty file is rejected", [_err("empty", "invalid", "size")], "Empty file message"),
        _case("Boundary", "Boundary Value Analysis", "Reject file over size limit",
              {"file_name": "big.pdf", "size_bytes": (hi + 1) * 1024 * 1024 if hi else 50 * 1024 * 1024},
              "Oversized file is rejected without a server error", [_err("size", "large", "limit", "exceed"), NO_5XX], "File too large message"),
    ]


def _session_cases(desc):
    lo, _ = _extract_bounds(desc)
    return [_case("Negative", "State Transition", "Expired session forces re-authentication",
                  {"idle_minutes": (lo + 1) if lo else 31, "action": "perform any protected action"},
                  "User is sent back to login and the action is not performed",
                  [NOT_AUTHED, _url("login")], "Session expired message")]


def _generic_cases():
    return [
        _case("Positive", "Equivalence Partitioning", "Valid usage of the feature", {},
              "Feature completes without errors", [NO_ERR, NO_5XX]),
        _case("Negative", "Error Guessing", "Submit with no input", {"value": ""},
              "Missing input is rejected or handled without a server error", [NO_5XX]),
        _case("Security", "Security Payload", "Special characters are handled safely", {"value": "<>\"'%;--"},
              "Special characters do not break the page", [NO_5XX, NO_DIALOG]),
        _case("Boundary", "Boundary Value Analysis", "Very long input (1001 chars)", {"value": "a" * 1001},
              "Over-length input is rejected or truncated without a server error", [NO_5XX]),
    ]


# ------------------------------------------------------------ test assembly
def _steps_for(rid: str, data: Dict[str, Any]) -> List[str]:
    steps = [f"Open the screen/feature covered by {rid}"]
    steps += [f"Provide {k} = {_short(v)}" for k, v in data.items()]
    steps += ["Submit / perform the action", "Observe the application response"]
    return steps


def _describe(a: Dict[str, Any]) -> str:
    t, v = a.get("type"), a.get("value")
    return {
        "error_shown": f"an error message containing one of {v} is shown",
        "no_error_shown": "no error message is shown",
        "text_visible": f"the text '{v}' is visible",
        "url_contains": f"the URL contains '{v}'",
        "authenticated": "the user is authenticated",
        "not_authenticated": "no access is granted",
        "no_server_error": "no server error is shown",
        "no_dialog": "no JavaScript dialog appears",
    }.get(t, str(t))


def _gherkin(tc: Dict[str, Any]) -> str:
    pre = tc.get("preconditions")
    pre = "; ".join(pre) if isinstance(pre, list) else (pre or "the application is reachable")
    lines = [f"Scenario: {tc['title']}", f"  Given {pre}"]
    items = list((tc.get("test_data") or {}).items())
    for i, (k, v) in enumerate(items):
        lines.append(f"  {'When' if i == 0 else 'And'} the user enters {k} \"{_short(v)}\"")
    lines.append(f"  {'And' if items else 'When'} the user submits")
    lines.append(f"  Then {tc['expected_result']}")
    for a in tc.get("assertions", []):
        lines.append(f"  And {_describe(a)}")
    return "\n".join(lines)


def _finalize(stub: Dict[str, Any], rid: str) -> Dict[str, Any]:
    tc = dict(stub)
    tc["requirement_id"] = rid
    tc.setdefault("preconditions", "Application is accessible and in a ready state")
    tc["steps"] = _steps_for(rid, tc.get("test_data") or {})
    tc["gherkin"] = _gherkin(tc)
    return tc


def _expects_success(case: Dict[str, Any]) -> bool:
    """True if the case expects the action to succeed (e.g. accept-at-boundary)."""
    kinds = {a["type"] for a in case["assertions"]}
    return not kinds & {"error_shown", "not_authenticated"} and case["type"] in ("Positive", "Boundary")


def _design_for_requirement(req: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    rid = req.get("id", f"REQ-{idx:03d}")
    desc = req.get("description", "")
    flags = _detect(desc, req.get("category", ""))

    stubs: List[Dict[str, Any]] = []
    if flags["auth"]:
        stubs += _auth_cases(flags)
    else:
        if flags["email"]:
            stubs += _email_cases()
        if flags["password"]:
            stubs += _password_cases(desc)
    if flags["numeric"]:
        stubs += _numeric_cases(desc)
    if flags["upload"]:
        stubs += _upload_cases(desc)
    if flags["session"]:
        stubs += _session_cases(desc)
    if flags["text"] and not (flags["auth"] or flags["email"] or flags["password"]):
        stubs += _text_cases(desc)
    if not stubs:
        stubs = _generic_cases()

    # Every requirement needs at least one happy path and one rejection/robustness case.
    if not any(_expects_success(s) for s in stubs):
        stubs.insert(0, _generic_cases()[0])
    if all(_expects_success(s) for s in stubs):
        stubs.append(_generic_cases()[1])

    seen, unique = set(), []
    for s in stubs:
        if s["title"] not in seen:
            seen.add(s["title"])
            unique.append(s)
    return [_finalize(s, rid) for s in unique[:MAX_PER_REQ]]


# ------------------------------------------------------ validation / repair
def validate_test_case(tc: Dict[str, Any]) -> List[str]:
    """Return a list of problems; an empty list means the test case is usable."""
    issues = []
    if tc.get("type") not in TYPES:
        issues.append(f"invalid type {tc.get('type')!r}")
    if not tc.get("requirement_id"):
        issues.append("missing requirement_id")
    if not tc.get("steps"):
        issues.append("no steps")
    asserts = tc.get("assertions") or []
    if not asserts:
        if tc.get("applicable") is not False:
            issues.append("no assertions")
    elif any(a.get("type") not in ASSERTION_TYPES for a in asserts):
        issues.append("unknown assertion type")
    if tc.get("type") != "Positive" and not tc.get("test_data"):
        issues.append("non-positive case without concrete test_data")
    if any(p in str(tc.get("expected_result", "")).lower() for p in GENERIC_PHRASES):
        issues.append("generic expected_result")
    return issues


def _normalize(tc: Any, valid_ids: set) -> Optional[Dict[str, Any]]:
    """Coerce an LLM-produced test case into the schema; None if unusable."""
    if not isinstance(tc, dict) or tc.get("requirement_id") not in valid_ids:
        return None
    out = dict(tc)
    t = str(out.get("type", "")).strip().title()
    out["type"] = t
    out["technique"] = out.get("technique") or "Error Guessing"
    out["test_data"] = out["test_data"] if isinstance(out.get("test_data"), dict) else {}
    if isinstance(out.get("steps"), str):
        out["steps"] = [out["steps"]]
    asserts = out.get("assertions")
    if asserts is None and isinstance(out.get("assertion"), dict):
        asserts = [out["assertion"]]
    out["assertions"] = [a for a in (asserts or []) if isinstance(a, dict) and a.get("type") in ASSERTION_TYPES]
    out.setdefault("expected_error", None)
    out.setdefault("title", "Untitled test")
    out.setdefault("expected_result", "")
    if not out.get("gherkin"):
        out["gherkin"] = _gherkin(out)
    return out


# --------------------------------------------------------------- tiering
def _apply_tier(cases: List[Dict[str, Any]], tier: int, rid: str) -> List[Dict[str, Any]]:
    """Keep only the test types allowed for the tier; make sure each allowed type is present."""
    allowed = TIER_TEST_TYPES.get(tier, ["Positive"])
    kept = [c for c in cases if c["type"] in allowed]
    present = {c["type"] for c in kept}
    generic = {g["type"]: g for g in _generic_cases()}
    for ttype in allowed:
        if ttype not in present and ttype in generic:
            kept.append(_finalize(generic[ttype], rid))
    if not kept and cases:
        kept = [next((c for c in cases if _expects_success(c)), cases[0])]
    for c in kept:
        c["tier"] = tier
    return kept


class TestDesignAgent(BaseAgent):
    __test__ = False

    def __init__(self, model_client=None):
        super().__init__("Test_Design", model_client)

    # ------------------------------------------------------------------
    # Build a {requirement_id: tier} lookup from the risk agent's output
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tier_lookup(state: AgentState) -> Dict[str, int]:
        """Return {req_id: tier_int} from the risk_matrix produced by RiskCoveragePlanningAgent."""
        lookup: Dict[str, int] = {}

        # Primary source: risk_matrix entries with explicit requirement_id + tier
        for entry in state.get("risk_matrix", []):
            rid = entry.get("requirement_id", "")
            tier = entry.get("tier")
            if rid and tier is not None:
                lookup[rid] = int(tier)

        # Fallback: testing_priorities (tier text → tier int mapping)
        tier_label_map = {
            "Tier 1": 1, "Tier 2": 2, "Tier 3": 3,
            "Tier 1 (Smoke / Blocker)": 1,
            "Tier 2 (Core Business Logic)": 2,
            "Tier 3 (Edge Cases & Security)": 3,
            "Tier 3 (Edge Cases & UI)": 3,
        }
        for p in state.get("testing_priorities", []):
            tier_str = p.get("tier", "")
            tier_int = None
            for label, val in tier_label_map.items():
                if label in tier_str:
                    tier_int = val
                    break
            if tier_int is None:
                tier_int = 2  # default
            for rid in p.get("target_requirements", []):
                if rid not in lookup:
                    lookup[rid] = tier_int

        return lookup

    @staticmethod
    def _grounding_text(profile: Dict[str, Any], url: str) -> str:
        """Compact description of the live target page, taken from the DOM probe (empty if no probe ran)."""
        if not profile or not (profile.get("inputs") or profile.get("buttons")):
            return ""
        inputs = [{k: i.get(k) for k in ("type", "name", "label", "placeholder") if i.get(k)} for i in profile.get("inputs", [])[:15]]
        buttons = [b.get("text") for b in profile.get("buttons", [])[:12] if b.get("text")]
        return f"""
Live page inventory of the application under test ({url}), read from its real DOM:
- input fields: {json.dumps(inputs)}
- buttons: {json.dumps(buttons)}
- headings: {json.dumps(profile.get("headings", [])[:6])}
Design cases that can be exercised on THESE elements. test_data keys the executor understands:
email, password, value (any other text field), file_name + size_bytes, attempts (repeat count).
If a requirement cannot be exercised on this page, still return its case but set "applicable": false and "assertions": []
(it will be reported as SKIPPED - never invent assertions that would pass without testing anything).
"""

    def _prompt(self, chunk: List[Dict[str, Any]], tier_lookup: Dict[str, int]) -> str:
        tiers = {r.get("id"): tier_lookup.get(r.get("id", ""), 2) for r in chunk}
        return f"""Design test cases for these requirements:
{json.dumps(chunk, indent=2)}

Risk tier per requirement (1 = highest priority): {json.dumps(tiers)}
Only use the test types allowed for a requirement's tier:
- Tier 1: Positive, Negative, Security, Boundary
- Tier 2: Positive, Negative
- Tier 3: Positive

Apply these techniques per requirement and name the one used in "technique":
- Equivalence Partitioning: one case per valid and invalid input class.
- Boundary Value Analysis: for every stated limit N test N-1, N, N+1 (lengths, counts, amounts, timeouts).
- Error Guessing: empty, whitespace-only, wrong type, special characters, overflow.
- Security Payload: SQL injection, XSS, auth bypass where the requirement has user input.
- State Transition: lockout after repeated failures, session expiry, permission changes.

Rules:
- Every requirement needs >=1 Positive case and, within its allowed types, several Negative/Boundary/Security cases when it takes input.
- Every case needs CONCRETE "test_data" (actual values, not descriptions like "invalid input").
- "expected_result" must say exactly what happens. Do not write "appropriate error" or "works as expected".
- Do not invent exact UI wording. Use "assertions" with keyword lists instead.
- Allowed assertion types: error_shown (value: list of keywords), no_error_shown, text_visible (value: str),
  url_contains (value: str), authenticated, not_authenticated, no_server_error, no_dialog.
{getattr(self, "_grounding", "")}
Return ONLY a JSON array. Each object:
{{"title": str, "type": "Positive|Negative|Boundary|Security", "technique": str, "requirement_id": str, "tier": 1|2|3,
  "preconditions": str, "test_data": {{...}}, "steps": [str], "expected_result": str,
  "expected_error": str or null, "assertions": [{{"type": str, "value": ...}}], "gherkin": str}}"""

    @staticmethod
    def _llm_tier(cases: List[Dict[str, Any]]) -> int:
        """Tier the LLM reported for a requirement, defaulting to 2."""
        for c in cases:
            try:
                t = int(c.get("tier"))
                if t in TIER_TEST_TYPES:
                    return t
            except (ValueError, TypeError):
                continue
        return 2

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Designing tier-weighted, technique-driven test cases (EP / BVA / error guessing / security)...")
        reqs = state.get("validated_requirements", state.get("requirements", []))
        if not reqs:
            self.log_error("No requirements found to design test cases")
            return state

        tier_lookup = self._build_tier_lookup(state)
        self._grounding = self._grounding_text(state.get("target_profile") or {}, state.get("target_url", ""))
        if self._grounding:
            self.log_info("Grounding test design in the live DOM of the target")

        all_cases: List[Dict[str, Any]] = []
        dropped = 0
        for start in range(0, len(reqs), CHUNK_SIZE):
            chunk = reqs[start:start + CHUNK_SIZE]
            ids = {r.get("id", f"REQ-{start + i + 1:03d}") for i, r in enumerate(chunk)}

            def _fallback(chunk=chunk, start=start) -> str:
                out = []
                for i, r in enumerate(chunk):
                    out += _design_for_requirement(r, start + i + 1)
                return json.dumps(out)

            chunk_cases: List[Dict[str, Any]] = []
            try:
                text = await self.call_llm(
                    prompt=self._prompt(chunk, tier_lookup),
                    system_instruction="You are a Principal Software Test Engineer. You design rigorous, data-driven tests.",
                    model_name=self.config.GEMINI_FLASH_MODEL,
                    mock_fallback=_fallback,
                )
                parsed = self._parse_json_response(text)
                if isinstance(parsed, dict):
                    parsed = parsed.get("test_cases", [])
                for raw in parsed if isinstance(parsed, list) else []:
                    tc = _normalize(raw, ids)
                    if tc and not validate_test_case(tc):
                        chunk_cases.append(tc)
                    else:
                        dropped += 1
            except Exception as e:
                self.log_error(f"Test design failed for requirements {sorted(ids)}: {e}")

            # Any requirement the LLM did not cover with valid cases gets deterministic design.
            covered = {tc["requirement_id"] for tc in chunk_cases}
            for i, r in enumerate(chunk):
                if r.get("id", f"REQ-{start + i + 1:03d}") not in covered:
                    chunk_cases += _design_for_requirement(r, start + i + 1)

            # Apply the risk tier per requirement: risk agent's tier wins, else the LLM's, else 2.
            for rid in ids:
                mine = [c for c in chunk_cases if c["requirement_id"] == rid]
                tier = tier_lookup.get(rid, self._llm_tier(mine))
                chunk_cases = [c for c in chunk_cases if c["requirement_id"] != rid] + _apply_tier(mine, tier, rid)
            all_cases += chunk_cases

        # Tier 1 first (stable, so requirement order is kept within a tier), then number sequentially.
        all_cases.sort(key=lambda tc: tc.get("tier", 2))
        for n, tc in enumerate(all_cases, 1):
            tc["id"] = tc["tc_id"] = f"TC-{n:03d}"

        state["test_cases"] = all_cases
        state["gherkin_scenarios"] = [tc["gherkin"] for tc in all_cases if tc.get("gherkin")]
        if "metrics" in state and isinstance(state["metrics"], dict):
            state["metrics"]["total_test_cases"] = len(all_cases)

        tier_counts: Dict[int, int] = {}
        for tc in all_cases:
            tier_counts[tc["tier"]] = tier_counts.get(tc["tier"], 0) + 1
        tier_summary = ", ".join(f"Tier {t}: {c}" for t, c in sorted(tier_counts.items()))
        self.log_success(f"Generated {len(all_cases)} test cases sorted by tier ({tier_summary}); "
                         f"{dropped} invalid LLM cases dropped")
        return state
