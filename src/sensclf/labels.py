"""Document-level sensitivity categories and the explicit raw-label -> category tables.

Task: multi-label classification of a whole document. A document is positive for a
category iff it contains at least one valid gold span whose RAW dataset label maps to that
category. The six categories come only from public dataset labels and from the special
categories of personal data of GDPR Article 9 (see docs/TASK.md).

Every raw label of each pinned dataset is listed explicitly (55 Nemotron-PII, 29 Gretel),
mapped to one category or to ``None`` (NONE), each with a one-line justification. An
unlisted label raises :class:`UnknownRawLabelError`: a new upstream label must be decided on,
never absorbed silently. The judgement calls are explained in docs/TASK.md ("Decisions").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from privgate.data import RawSpan

IDENTITY = "IDENTITY"
CONTACT = "CONTACT"
FINANCIAL = "FINANCIAL"
HEALTH = "HEALTH"
CREDENTIALS = "CREDENTIALS"
SPECIAL_CATEGORY = "SPECIAL_CATEGORY"
CATEGORIES: tuple[str, ...] = (IDENTITY, CONTACT, FINANCIAL, HEALTH, CREDENTIALS,
                               SPECIAL_CATEGORY)

NEMOTRON = "nemotron-pii"   # privgate's dataset names
GRETEL = "gretel-finance-en"


class UnknownRawLabelError(KeyError):
    """A raw label that the dataset's table does not list."""


@dataclass(frozen=True)
class Rule:
    """Where one raw label goes (``None`` = NONE) and why."""

    category: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.category is not None and self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if not self.reason.strip():
            raise ValueError("every rule needs a justification")


def _table(rows: dict[str, tuple[str | None, str]]) -> Mapping[str, Rule]:
    return MappingProxyType({raw: Rule(cat, why) for raw, (cat, why) in sorted(rows.items())})


_NAME = "personal name part: directly identifies a person"
_ADDR = "address part: locates where a person can be reached"
_NET = "network/device identifier: no category covers online identifiers (Decisions)"
_DATE = "a date or time alone is not tied to a person"
_ORG = "names an organisation, not a person"
_REC = "organisation-internal record key, not an identity document (Decisions)"
_FIN = "financial account, instrument or payment authentication datum"
_CRED = "authentication secret or login identifier"
_ART9 = "GDPR Art. 9(1) special category"

NEMOTRON_TABLE: Mapping[str, Rule] = _table({
    "account_number": (FINANCIAL, _FIN),
    "age": (None, "quasi-identifier, not Art. 9 and not an identifier (Decisions)"),
    "api_key": (CREDENTIALS, _CRED),
    "bank_routing_number": (FINANCIAL, "identifies the bank of a person's account (spec: routing)"),
    "biometric_identifier": (SPECIAL_CATEGORY, f"{_ART9}: biometric data (Decisions)"),
    "blood_type": (HEALTH, "health data (Art. 4(15)): a physiological attribute"),
    "certificate_license_number": (IDENTITY, "licence number issued to a person (Decisions)"),
    "city": (CONTACT, _ADDR),
    "company_name": (None, _ORG),
    "coordinate": (CONTACT, "geographic coordinates of a place (spec: coordinates)"),
    "country": (CONTACT, _ADDR + " (Decisions)"),
    "county": (CONTACT, _ADDR),
    "credit_debit_card": (FINANCIAL, _FIN),
    "customer_id": (None, _REC),
    "cvv": (FINANCIAL, _FIN),
    "date": (None, _DATE),
    "date_of_birth": (IDENTITY, "date of birth: core identity attribute"),
    "date_time": (None, _DATE),
    "device_identifier": (None, _NET),
    "education_level": (None, "socio-economic attribute, not Art. 9"),
    "email": (CONTACT, "e-mail address: a contact channel"),
    "employee_id": (None, _REC),
    "employment_status": (None, "socio-economic attribute, not Art. 9"),
    "fax_number": (CONTACT, "fax number: a contact channel"),
    "first_name": (IDENTITY, _NAME),
    "gender": (None, "not among the Art. 9 categories (Decisions)"),
    "health_plan_beneficiary_number": (HEALTH, "health insurance identifier (spec)"),
    "http_cookie": (CREDENTIALS, "session token: grants authenticated access"),
    "ipv4": (None, _NET),
    "ipv6": (None, _NET),
    "language": (None, "spoken language, not Art. 9"),
    "last_name": (IDENTITY, _NAME),
    "license_plate": (None, "identifies a vehicle, not a person directly (Decisions)"),
    "mac_address": (None, _NET),
    "medical_record_number": (HEALTH, "medical record identifier: links to health data"),
    "national_id": (IDENTITY, "government-issued national identifier"),
    "occupation": (None, "job title, not Art. 9 (Decisions)"),
    "password": (CREDENTIALS, _CRED),
    "phone_number": (CONTACT, "telephone number: a contact channel"),
    "pin": (FINANCIAL, "PIN; the spec lists PIN under FINANCIAL (Decisions)"),
    "political_view": (SPECIAL_CATEGORY, f"{_ART9}: political opinions"),
    "postcode": (CONTACT, _ADDR),
    "race_ethnicity": (SPECIAL_CATEGORY, f"{_ART9}: racial or ethnic origin"),
    "religious_belief": (SPECIAL_CATEGORY, f"{_ART9}: religious beliefs"),
    "sexuality": (SPECIAL_CATEGORY, f"{_ART9}: sex life or sexual orientation"),
    "ssn": (IDENTITY, "social security number: government identifier"),
    "state": (CONTACT, _ADDR),
    "street_address": (CONTACT, _ADDR),
    "swift_bic": (FINANCIAL, "SWIFT/BIC of a person's bank (spec: SWIFT)"),
    "tax_id": (IDENTITY, "government tax identifier"),
    "time": (None, _DATE),
    "unique_id": (None, "opaque technical identifier of unknown kind (Decisions)"),
    "url": (None, "web address: not tied to a category"),
    "user_name": (CREDENTIALS, "login identifier (spec: user name; Decisions)"),
    "vehicle_identifier": (None, "identifies a vehicle, not a person directly (Decisions)"),
})

GRETEL_TABLE: Mapping[str, Rule] = _table({
    "account_pin": (FINANCIAL, "account PIN (spec: PIN)"),
    "api_key": (CREDENTIALS, _CRED),
    "bank_routing_number": (FINANCIAL, "identifies the bank of a person's account (spec: routing)"),
    "bban": (FINANCIAL, "basic bank account number (spec: BBAN)"),
    "company": (None, _ORG),
    "credit_card_number": (FINANCIAL, _FIN),
    "credit_card_security_code": (FINANCIAL, "card security code (spec: CVV)"),
    "customer_id": (None, _REC),
    "date": (None, _DATE),
    "date_of_birth": (IDENTITY, "date of birth: core identity attribute"),
    "date_time": (None, _DATE),
    "driver_license_number": (IDENTITY, "government-issued driver licence number"),
    "email": (CONTACT, "e-mail address: a contact channel"),
    "employee_id": (None, _REC),
    "first_name": (IDENTITY, _NAME),
    "iban": (FINANCIAL, "international bank account number (spec: IBAN)"),
    "ipv4": (None, _NET),
    "ipv6": (None, _NET),
    "last_name": (IDENTITY, _NAME),
    "local_latlng": (CONTACT, "latitude/longitude of a place (spec: coordinates)"),
    "name": (IDENTITY, "full personal name: directly identifies a person"),
    "passport_number": (IDENTITY, "government-issued passport number"),
    "password": (CREDENTIALS, _CRED),
    "phone_number": (CONTACT, "telephone number: a contact channel"),
    "ssn": (IDENTITY, "social security number: government identifier"),
    "street_address": (CONTACT, _ADDR),
    "swift_bic_code": (FINANCIAL, "SWIFT/BIC of a person's bank (spec: SWIFT)"),
    "time": (None, _DATE),
    "user_name": (CREDENTIALS, "login identifier (spec: user name; Decisions)"),
})

TABLES: Mapping[str, Mapping[str, Rule]] = MappingProxyType({
    NEMOTRON: NEMOTRON_TABLE,
    GRETEL: GRETEL_TABLE,
})


def category_of(dataset: str, raw_label: str) -> str | None:
    """Category of ``raw_label`` in ``dataset`` (``None`` = NONE); unlisted labels raise."""
    try:
        table = TABLES[dataset]
    except KeyError:
        raise ValueError(f"no label table for dataset {dataset!r}") from None
    try:
        return table[raw_label].category
    except KeyError:
        raise UnknownRawLabelError(f"{dataset}: raw label {raw_label!r} is not in the table") \
            from None


def is_valid_gold_span(raw: RawSpan, text: str) -> bool:
    """privgate's per-span checks, applied to a raw span (adapter; see docs/DATA.md).

    Valid iff the offsets are ints with ``0 <= start < end <= len(text)``, the covered text
    equals the stored text up to letter case (when the dataset stores it), and the covered
    text is not blank. Overlap resolution and merging are deliberately NOT applied: at
    document level they could only remove a category whose span overlaps another one.
    """
    start, end = raw.start, raw.end
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        return False
    covered = text[start:end]
    if raw.text is not None and covered.casefold() != raw.text.casefold():
        return False
    return bool(covered.strip())


def labels_from_raw_labels(dataset: str, raw_labels: Iterable[str]) -> dict[str, int]:
    """6-bit label vector (category -> 0/1) of a document with these raw span labels."""
    present = {category_of(dataset, raw) for raw in raw_labels}
    return {c: int(c in present) for c in CATEGORIES}


def document_labels(dataset: str, text: str, raw_spans: Iterable[RawSpan]) -> dict[str, int]:
    """Labels of a document from its raw gold spans; invalid spans count for nothing."""
    return labels_from_raw_labels(
        dataset, (s.label for s in raw_spans if is_valid_gold_span(s, text))
    )


def table_markdown() -> str:
    """The raw-label table of docs/TASK.md, generated from the tables above."""
    lines = ["| dataset | raw label | category | justification |", "|---|---|---|---|"]
    for dataset, table in TABLES.items():
        for raw, rule in table.items():
            lines.append(f"| {dataset} | `{raw}` | {rule.category or 'NONE'} | {rule.reason} |")
    return "\n".join(lines)
