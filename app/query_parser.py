"""ATK constraint extraction from free-text search queries.

Extracts inclusive ATK thresholds/ranges written inside a natural-language
query ("more than 4000 ATK", "dragões com mais de 4000 de ataque",
"between 1000 and 2000 ATK", ">= 2500", "2500+") and returns them as
structured min_atk/max_atk bounds plus the query remainder (the constraint
text stripped) for embedding.

Only the matched constraint span is stripped: "dragões com mais de 4000 de
ataque" leaves "dragões" as the remainder, while a bare "com mais de 4000"
would consume the whole query. When the strip removes the entire query (or
nothing matched), the remainder falls back to the original query.

The mapping is *inclusive* on purpose, matching the existing structured
filters: "more than 4000 ATK" produces min_atk=4000 (>= semantics), i.e. the
threshold itself is allowed. This matches the acceptance criteria for the
feature.

Only ATK is parsed. DEF, level, attribute and card_type stay explicit
parameters. Pure stdlib (re + dataclasses + logging): no app dependencies, no
network, no LLM.
"""

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

_KEYWORD = r"atk|ataque|attack(?:\s+power)?|pontos\s+de\s+ataque"

# Equality-tier keyword list: bare "attack" is excluded so verb-sense phrases
# like "monsters that attack 2 times" are not parsed as an ATK equality.
_EQ_KEYWORD = r"atk|ataque|attack(?:\s+power)|pontos\s+de\s+ataque"

# Optional keyword trailer: "ATK", "de ataque", "of attack power", ...
_TRAILER = rf"(?:\s*(?:de|of)?\s*(?:{_KEYWORD}))?"

# Trailer that requires the ATK keyword to be present. Bare ranges (without a
# "between"/"entre" prefix) only parse when followed by an explicit keyword.
_TRAILER_REQUIRED = rf"\s*(?:de|of)?\s*(?:{_KEYWORD})"

_COMP_MIN_PHRASES = (
    r"more than|over|above|greater than|at least|no less than|not less than|"
    r"minimum(?:\s+of)?|exceeding|exceeds|"
    r"mais de|acima de|maior que|pelo menos|no mínimo|no minimo|"
    r"mínimo(?:\s+de)?|minimo(?:\s+de)?"
)

# "less than"/"menos de" are guarded against a preceding "no"/"not" so the
# min-tier forms ("no less than", "not less than") never double-match here.
_COMP_MAX_PHRASES = (
    r"(?<!no\s)(?<!not\s)(?:less than|menos de)|"
    r"under|below|at most|no more than|maximum(?:\s+of)?|"
    r"abaixo de|menor que|no máximo|no maximo|"
    r"máximo(?:\s+de)?|maximo(?:\s+de)?"
)

# Tier 1: RANGE (between/entre/N1-N2/N1 to N2/N1 a N2/N1 até N2/N1 and N2).
_RANGE = re.compile(
    rf"(?:\bbetween\s+|entre\s+)"
    rf"(?<![\d.])(?P<n1>\d+)(?![\d.])"
    rf"\s*(?:até|and|e|to|a|-)\s*"
    rf"(?<![\d.])(?P<n2>\d+)(?![\d.])"
    rf"{_TRAILER}"
    rf"|"
    rf"(?<![\d.])(?P<n1b>\d+)(?![\d.])"
    rf"\s*(?:até|and|e|to|a|-)\s*"
    rf"(?<![\d.])(?P<n2b>\d+)(?![\d.])"
    rf"{_TRAILER_REQUIRED}",
    re.IGNORECASE,
)

# Tier 2: OPERATOR MIN (">= N", "> N", "≥ N", postfix "N+").
_OP_MIN = re.compile(
    rf"(?:>=|>|≥)\s*(?<![\d.-])(?P<n>\d+)(?![\d.]){_TRAILER}"
    rf"|(?<![\d.-])(?P<n2>\d+)(?![\d.])\+{_TRAILER}",
    re.IGNORECASE,
)

# Tier 3: OPERATOR MAX ("<= N", "< N", "≤ N").
_OP_MAX = re.compile(
    rf"(?:<=|<|≤)\s*(?<![\d.-])(?P<n>\d+)(?![\d.]){_TRAILER}",
    re.IGNORECASE,
)

# Tier 4: COMPARATIVE MIN.
# The PT connector "com" is consumed when present ("dragões com mais de 4000"
# leaves "dragões" as remainder; a bare "com mais de 4000" would consume the
# whole query and fall back to the original). The EN "with" is deliberately
# NOT consumed so "dragons with more than 4000 ATK" keeps "dragons with" as
# remainder.
# "(?<!no\s)" stops "no more than 2500" from matching "more than 2500"
# (that phrase belongs to the max tier).
_COMP_MIN = re.compile(
    rf"(?:\bcom\s+)?"
    rf"(?:\b(?:{_KEYWORD})\s+)?"
    rf"(?<!no\s)\b(?:{_COMP_MIN_PHRASES})\s+"
    rf"(?<![\d.-])(?P<n>\d+)(?![\d.]){_TRAILER}"
    rf"|(?<![\d.-])(?P<n2>\d+)(?![\d.])"
    rf"\s+(?:(?:de|of)?\s*(?:{_KEYWORD})\s+)?(?:or more|ou mais){_TRAILER}",
    re.IGNORECASE,
)

# Tier 5: COMPARATIVE MAX.
_COMP_MAX = re.compile(
    rf"\b(?:{_COMP_MAX_PHRASES})\s+"
    rf"(?<![\d.-])(?P<n>\d+)(?![\d.]){_TRAILER}"
    rf"|(?<![\d.-])(?P<n2>\d+)(?![\d.])"
    rf"\s+(?:(?:de|of)?\s*(?:{_KEYWORD})\s+)?(?:or less|ou menos){_TRAILER}",
    re.IGNORECASE,
)

# Tier 6: EQUALITY (ATK keyword REQUIRED — false-positive guard).
_EQ = re.compile(
    rf"\bexactly\s+(?<![\d.-])(?P<n>\d+)(?![\d.]){_TRAILER}"
    rf"|\b(?:{_EQ_KEYWORD})\s+(?:igual a|equal to|equals?|de|of)?\s*"
    rf"(?<![\d.-])(?P<n2>\d+)(?![\d.])"
    rf"(?!\s+(?:or more|or less|ou mais|ou menos))"
    rf"|(?:\b(?:com|with)\s+)?"
    rf"(?<!up to\s)(?<!about\s)(?<!around\s)(?<!approximately\s)"
    rf"(?<!approx\s)(?<!~)(?<!~\s)"
    rf"(?<![\d.-])(?P<n3>\d+)(?![\d.])"
    rf"\s*(?:de|of)?\s*\b(?:{_KEYWORD})\b"
    rf"(?!\s+(?:or more|or less|ou mais|ou menos))",
    re.IGNORECASE,
)

# Ordered tiers: first match wins (precedence range > operator > comparative >
# equality).
_TIERS = (
    (_RANGE, "range"),
    (_OP_MIN, "op_min"),
    (_OP_MAX, "op_max"),
    (_COMP_MIN, "comp_min"),
    (_COMP_MAX, "comp_max"),
    (_EQ, "eq"),
)


@dataclass(frozen=True)
class AtkConstraint:
    """ATK bounds extracted from a free-text query.

    Attributes:
        min_atk: Inclusive lower bound (>= semantics), or None.
        max_atk: Inclusive upper bound (<= semantics), or None.
        remainder_query: The query with the matched constraint text stripped
            (whitespace-collapsed); the original query when nothing was
            stripped or when the strip consumed the whole query.
    """

    min_atk: int | None = None
    max_atk: int | None = None
    remainder_query: str = ""


def _first_number(match: re.Match) -> int:
    """Return the first captured number group of ``match``."""
    for name in ("n", "n2", "n3"):
        value = match.group(name)
        if value is not None:
            return int(value)
    raise AssertionError("matched pattern without a number group")


def extract_atk_constraint(query: str) -> AtkConstraint:
    """Extract the first ATK constraint from ``query``.

    Tiers are tried in precedence order (range > operator min > operator max >
    comparative min > comparative max > equality); the first match wins.
    Returns an :class:`AtkConstraint` with the parsed bounds and the query
    remainder (constraint text stripped).
    """
    for pattern, kind in _TIERS:
        match = pattern.search(query)
        if not match:
            continue
        if kind == "range":
            n1 = match.group("n1") or match.group("n1b")
            n2 = match.group("n2") or match.group("n2b")
            lo, hi = int(n1), int(n2)
            bounds = (min(lo, hi), max(lo, hi))
        elif kind == "eq":
            n = _first_number(match)
            bounds = (n, n)
        elif kind in ("op_min", "comp_min"):
            bounds = (_first_number(match), None)
        else:  # op_max, comp_max
            bounds = (None, _first_number(match))

        remainder = pattern.sub("", query, count=1)
        remainder = re.sub(r"\s+", " ", remainder).strip()
        if not remainder:
            remainder = query
        return AtkConstraint(
            min_atk=bounds[0],
            max_atk=bounds[1],
            remainder_query=remainder,
        )

    return AtkConstraint(min_atk=None, max_atk=None, remainder_query=query)


def merge_atk_bounds(
    explicit_min: int | None,
    explicit_max: int | None,
    extracted_min: int | None,
    extracted_max: int | None,
) -> tuple[int | None, int | None]:
    """Merge explicit and extracted ATK bounds (AND semantics).

    Effective min = the larger of the two (with None passthrough on each
    side); effective max = the smaller of the two (with None passthrough).
    """
    if explicit_min is None:
        eff_min = extracted_min
    elif extracted_min is None:
        eff_min = explicit_min
    else:
        eff_min = max(explicit_min, extracted_min)

    if explicit_max is None:
        eff_max = extracted_max
    elif extracted_max is None:
        eff_max = explicit_max
    else:
        eff_max = min(explicit_max, extracted_max)

    return eff_min, eff_max


def prepare_search_query(
    query: str,
    min_atk: int | None = None,
    max_atk: int | None = None,
) -> tuple[str, int | None, int | None]:
    """Extract ATK constraints from ``query`` and merge them with explicit bounds.

    Returns ``(remainder_query, eff_min_atk, eff_max_atk)``. This is the single
    entry point used by both the CLI (``app.search``) and the MCP tool
    (``app.mcp_server``).
    """
    parsed = extract_atk_constraint(query)
    eff_min, eff_max = merge_atk_bounds(min_atk, max_atk, parsed.min_atk, parsed.max_atk)
    if eff_min is not None and eff_max is not None and eff_min > eff_max:
        logger.warning(
            "Contradictory ATK bounds after merge: min_atk=%s > max_atk=%s "
            "(query=%r, explicit_min=%s, explicit_max=%s, "
            "extracted_min=%s, extracted_max=%s)",
            eff_min, eff_max, query, min_atk, max_atk,
            parsed.min_atk, parsed.max_atk,
        )
    return parsed.remainder_query, eff_min, eff_max
