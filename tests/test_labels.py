from collections import Counter

import pytest
from privgate.data import LABEL_MAPS, RawSpan, build_document
from privgate.data.documents import LoadOptions, LoadTally

from sensclf import labels as L


def _words(text: str) -> list[str]:
    return text.split()


# Every raw label observed in the pinned files (privgate docs/DATA.md at 498adcc), written out
# by hand so that the table is checked against an independent list.
NEMOTRON_RAW = _words("""
account_number age api_key bank_routing_number biometric_identifier blood_type
certificate_license_number city company_name coordinate country county credit_debit_card
customer_id cvv date date_of_birth date_time device_identifier education_level email
employee_id employment_status fax_number first_name gender health_plan_beneficiary_number
http_cookie ipv4 ipv6 language last_name license_plate mac_address medical_record_number
national_id occupation password phone_number pin political_view postcode race_ethnicity
religious_belief sexuality ssn state street_address swift_bic tax_id time unique_id url
user_name vehicle_identifier
""")
GRETEL_RAW = _words("""
account_pin api_key bank_routing_number bban company credit_card_number
credit_card_security_code customer_id date date_of_birth date_time driver_license_number email
employee_id first_name iban ipv4 ipv6 last_name local_latlng name passport_number password
phone_number ssn street_address swift_bic_code time user_name
""")


def test_fixture_lists_have_the_documented_sizes():
    assert len(set(NEMOTRON_RAW)) == len(NEMOTRON_RAW) == 55
    assert len(set(GRETEL_RAW)) == len(GRETEL_RAW) == 29


@pytest.mark.parametrize("dataset, expected", [(L.NEMOTRON, NEMOTRON_RAW),
                                               (L.GRETEL, GRETEL_RAW)])
def test_table_lists_exactly_every_raw_label(dataset, expected):
    assert set(L.TABLES[dataset]) == set(expected)
    assert set(L.TABLES[dataset]) == set(LABEL_MAPS[dataset])  # privgate's maps agree


def test_every_rule_has_a_valid_category_and_a_reason():
    for table in L.TABLES.values():
        for rule in table.values():
            assert rule.category in (*L.CATEGORIES, None)
            assert rule.reason.strip()


def test_rule_rejects_unknown_category_and_empty_reason():
    with pytest.raises(ValueError):
        L.Rule("SECRET", "x")
    with pytest.raises(ValueError):
        L.Rule(L.IDENTITY, "  ")


@pytest.mark.parametrize("dataset, raw, category", [
    (L.NEMOTRON, "first_name", L.IDENTITY),
    (L.NEMOTRON, "ssn", L.IDENTITY),
    (L.NEMOTRON, "street_address", L.CONTACT),
    (L.NEMOTRON, "coordinate", L.CONTACT),
    (L.NEMOTRON, "cvv", L.FINANCIAL),
    (L.NEMOTRON, "pin", L.FINANCIAL),
    (L.NEMOTRON, "medical_record_number", L.HEALTH),
    (L.NEMOTRON, "blood_type", L.HEALTH),
    (L.NEMOTRON, "password", L.CREDENTIALS),
    (L.NEMOTRON, "user_name", L.CREDENTIALS),
    (L.NEMOTRON, "religious_belief", L.SPECIAL_CATEGORY),
    (L.NEMOTRON, "biometric_identifier", L.SPECIAL_CATEGORY),
    (L.NEMOTRON, "gender", None),
    (L.NEMOTRON, "age", None),
    (L.NEMOTRON, "occupation", None),
    (L.NEMOTRON, "customer_id", None),
    (L.GRETEL, "name", L.IDENTITY),
    (L.GRETEL, "passport_number", L.IDENTITY),
    (L.GRETEL, "local_latlng", L.CONTACT),
    (L.GRETEL, "iban", L.FINANCIAL),
    (L.GRETEL, "bban", L.FINANCIAL),
    (L.GRETEL, "swift_bic_code", L.FINANCIAL),
    (L.GRETEL, "api_key", L.CREDENTIALS),
    (L.GRETEL, "company", None),
])
def test_category_on_hand_examples(dataset, raw, category):
    assert L.category_of(dataset, raw) == category


def test_unknown_raw_label_and_dataset_raise():
    with pytest.raises(L.UnknownRawLabelError):
        L.category_of(L.NEMOTRON, "favourite_colour")
    with pytest.raises(ValueError):
        L.category_of("other-dataset", "email")


TEXT = "Ann Lee, ann@example.org, PIN 1234. Faith: Quaker."
BLANK = TEXT.index(" ann@")  # a single space character


def _at(sub, label, stored=None):
    start = TEXT.index(sub)
    return RawSpan(start, start + len(sub), label, sub if stored is None else stored)


def test_document_labels_from_spans():
    spans = [_at("Ann", "first_name"), _at("ann@example.org", "email"),
             _at("1234", "pin"), _at("Quaker", "religious_belief")]
    assert L.document_labels(L.NEMOTRON, TEXT, spans) == {
        L.IDENTITY: 1, L.CONTACT: 1, L.FINANCIAL: 1, L.HEALTH: 0, L.CREDENTIALS: 0,
        L.SPECIAL_CATEGORY: 1,
    }


def test_document_without_spans_or_with_none_labels_is_all_zero():
    zero = dict.fromkeys(L.CATEGORIES, 0)
    assert L.document_labels(L.NEMOTRON, TEXT, []) == zero
    assert L.document_labels(L.NEMOTRON, TEXT, [_at("Lee", "occupation")]) == zero


def test_one_span_is_enough_and_repeats_do_not_matter():
    spans = [_at("Ann", "first_name"), _at("Lee", "last_name")]
    assert L.document_labels(L.NEMOTRON, TEXT, spans)[L.IDENTITY] == 1


@pytest.mark.parametrize("raw", [
    RawSpan(0, 999, "first_name"),                 # out of bounds
    RawSpan(5, 5, "first_name"),                   # empty
    RawSpan(True, 3, "first_name"),                # bool offset
    RawSpan("0", 3, "first_name"),                 # non-int offset
    RawSpan(0, 3, "first_name", "Bob"),            # stored text differs
    RawSpan(BLANK, BLANK + 1, "first_name"),       # blank after stripping
])
def test_invalid_spans_do_not_label(raw):
    assert not L.is_valid_gold_span(raw, TEXT)
    assert L.document_labels(L.NEMOTRON, TEXT, [raw])[L.IDENTITY] == 0


def test_case_only_mismatch_is_valid_like_privgate():
    assert L.is_valid_gold_span(_at("Ann", "first_name", stored="ANN"), TEXT)


def test_span_validity_agrees_with_privgate_checks():
    """Adapter check == privgate's build_document with overlap resolution/merging off."""
    spans = [_at("Ann", "first_name"), _at("Ann", "first_name", stored="ANN"),
             _at("Lee", "last_name", stored="Leo"), RawSpan(0, 999, "email"),
             RawSpan(BLANK, BLANK + 1, "email"), RawSpan(0, 3, "first_name"), _at("1234", "pin")]
    tally = LoadTally(L.NEMOTRON, LoadOptions(merge_adjacent=False, overlap_policy="keep"))
    build_document("d", TEXT, spans, tally)
    assert tally.spans_kept == sum(L.is_valid_gold_span(s, TEXT) for s in spans) == 4


def test_table_markdown_has_one_row_per_raw_label():
    table = L.table_markdown()
    assert len(table.splitlines()[2:]) == 55 + 29
    assert "| nemotron-pii | `gender` | NONE |" in table


@pytest.mark.data
def test_every_raw_label_in_the_real_files_is_in_the_table():
    from privgate.data import gretel, nemotron
    from privgate.data._parquet import iter_rows

    from sensclf.rawdata import verified_raw_paths

    paths = verified_raw_paths(allow_download=False)
    sources = {
        "nemotron_train": (L.NEMOTRON, nemotron.parse_spans, "spans"),
        "nemotron_test": (L.NEMOTRON, nemotron.parse_spans, "spans"),
        "gretel_en_test": (L.GRETEL, gretel.parse_spans, "pii_spans"),
        "gretel_en_train": (L.GRETEL, gretel.parse_spans, "pii_spans"),
    }
    seen: dict[str, Counter] = {L.NEMOTRON: Counter(), L.GRETEL: Counter()}
    for name, (dataset, parse, column) in sources.items():
        for row in iter_rows(paths[name], [column]):
            seen[dataset].update(s.label for s in parse(row[column]))
    assert set(seen[L.NEMOTRON]) == set(NEMOTRON_RAW)
    assert set(seen[L.GRETEL]) == set(GRETEL_RAW)
