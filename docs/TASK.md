# Task definition

## Problem statement

Input: a document `x` (its full text, a Unicode string). Output: a label vector
`y ∈ {0,1}^6`, one bit per category, in this fixed order:

| k | category | meaning (definition source) |
|---|---|---|
| 1 | IDENTITY | names, date of birth, government/licence identifiers (dataset labels) |
| 2 | CONTACT | e-mail, phone, fax, street address and other address parts, coordinates (dataset labels) |
| 3 | FINANCIAL | account/card/CVV/routing/SWIFT/IBAN/BBAN/PIN (dataset labels) |
| 4 | HEALTH | medical record number, health plan beneficiary number, blood type (dataset labels; GDPR Art. 4(15)) |
| 5 | CREDENTIALS | password, API key, HTTP cookie, user name (dataset labels) |
| 6 | SPECIAL_CATEGORY | GDPR Art. 9(1): racial/ethnic origin, religious belief, political opinion, sex life/sexual orientation, biometric data for identification |

**Labelling rule.** `y_k = 1` iff the document contains at least one *valid* gold span
whose **raw** dataset label maps to category `k` (table below). A span is valid if its
offsets are integers inside the text, its covered text is not blank, and (when the dataset
stores it) the covered text equals the stored text up to letter case; these are privgate's
per-span checks. Overlap resolution and span merging are not applied: they cannot add a
category and could only remove one. A document may be positive for several categories or for
none.

The category set comes only from public dataset labels and from GDPR Article 9; it is
fixed now, before any model is trained, and is not tuned on results.

**Later milestones** (not started here) will learn scores `p̂_k(x) ∈ [0,1]`, study their
calibration, selective prediction (a classifier that may abstain, evaluated by
risk–coverage), and conformal prediction sets with finite-sample coverage guarantees
calibrated on the `calibration` split, in-domain and under domain shift.

## Evaluation sets and splits

All Nemotron splits are group-aware: both renderings of a persona (`us`, `intl`) share a
`group_id` (the Nemotron `uid`) and always land in the same split. With
`h = int(sha256(group_id), 16) mod 100` (privgate's hash, applied to privgate's split key):

| set | source | rule | role |
|---|---|---|---|
| `train` | Nemotron train file | `h ∈ [20, 100)` | model fitting |
| `calibration` | Nemotron train file | `h ∈ [10, 20)` | calibration and conformal prediction ONLY |
| `dev` | Nemotron train file | `h ∈ [0, 10)` | model selection (identical to privgate's dev split) |
| `test_shift` | Nemotron test file | all rows | domain-shifted test (same generator, mostly unseen domains) |
| `test_ood` | Gretel English test file | all rows | out-of-distribution test (other generator, finance) |
| `gretel_pool` | Gretel English train file minus texts that occur in the Gretel test file | all remaining rows | reserved for a later adaptation experiment; not used in Milestones 2–5 |

Sizes and label statistics are in [DATA.md](DATA.md) (generated). The processed sets are
`data/processed/<set>.jsonl` (one JSON object per line: `doc_id`, `text`, `labels` (category
→ 0/1), `meta`), and their SHA-256 hashes are recorded in `data/manifest.json`.

## Decisions

Judgement calls in the raw-label table, with the reason for each. They were fixed before any
statistic was computed.

1. **`biometric_identifier` → SPECIAL_CATEGORY.** Art. 9(1) lists "biometric data for the
   purpose of uniquely identifying a natural person". The values are references to
   biometric templates (`BIO-…`), not raw biometric measurements. A label that names a
   biometric *identifier* signals biometric processing for identification, which is the Art. 9
   case.
2. **`age` → NONE.** Age is a quasi-identifier, but it is neither an identifier nor an
   Art. 9 category. (A date of birth is an identifier and maps to IDENTITY.)
3. **`gender` → NONE.** Art. 9 covers "sex life or sexual orientation", not gender. Nemotron
   labels sexual orientation separately (`sexuality` → SPECIAL_CATEGORY).
4. **`occupation`, `education_level`, `employment_status` → NONE.** Socio-economic
   attributes that are not Art. 9 categories. Trade-union membership would be, but no dataset
   labels it.
5. **`customer_id`, `employee_id` → NONE.** Keys issued by one organisation for its own
   records. They are not identity documents, financial instruments or credentials. Mapping them
   to IDENTITY would make IDENTITY depend on how each generator formats internal record keys.
6. **`unique_id` → NONE.** Opaque technical identifiers (hashes, UUIDs, base64 strings) of
   unknown kind. There is no evidence for any one category.
7. **`certificate_license_number` → IDENTITY.** A licence or certificate number issued to
   one person, the same kind of identifier as a driver licence number (which the spec puts
   under IDENTITY).
8. **`license_plate`, `vehicle_identifier` → NONE.** They identify a vehicle; the link to a
   person needs a registry lookup.
9. **`ipv4`, `ipv6`, `mac_address`, `device_identifier`, `url` → NONE.** These are online or
   device identifiers. They can be personal data (GDPR Recital 30), but none of the six
   categories covers them, and adding a seventh category is out of scope.
10. **`city`, `county`, `state`, `country`, `postcode` → CONTACT.** The spec puts "street
    address and other address parts" under CONTACT. A country alone (often "USA") is a weak
    signal, so this choice raises CONTACT prevalence; the statistics show by how much.
11. **`user_name` → CREDENTIALS.** The spec lists user name under CREDENTIALS: it is half of
    a login.
12. **`pin`, `account_pin`, `cvv`, `credit_card_security_code` → FINANCIAL.** The spec lists
    PIN and CVV under FINANCIAL (payment authentication), not under CREDENTIALS.
13. **`bank_routing_number`, `swift_bic`, `swift_bic_code` → FINANCIAL.** They identify an
    institution, not a person, but the spec lists routing and SWIFT under FINANCIAL. In these
    documents they appear as the bank details of a person's account.
14. **`date`, `date_time`, `time`, `company_name`, `company`, `language` → NONE.** Not tied
    to a person by themselves.

## Raw-label table

Generated from `sensclf.labels` by `python scripts/make_task_tables.py`. A test fails if
any raw label of either dataset (55 Nemotron-PII, 29 Gretel) is missing.

<!-- labels:begin -->
| dataset | raw label | category | justification |
|---|---|---|---|
| nemotron-pii | `account_number` | FINANCIAL | financial account, instrument or payment authentication datum |
| nemotron-pii | `age` | NONE | quasi-identifier, not Art. 9 and not an identifier (Decisions) |
| nemotron-pii | `api_key` | CREDENTIALS | authentication secret or login identifier |
| nemotron-pii | `bank_routing_number` | FINANCIAL | identifies the bank of a person's account (spec: routing) |
| nemotron-pii | `biometric_identifier` | SPECIAL_CATEGORY | GDPR Art. 9(1) special category: biometric data (Decisions) |
| nemotron-pii | `blood_type` | HEALTH | health data (Art. 4(15)): a physiological attribute |
| nemotron-pii | `certificate_license_number` | IDENTITY | licence number issued to a person (Decisions) |
| nemotron-pii | `city` | CONTACT | address part: locates where a person can be reached |
| nemotron-pii | `company_name` | NONE | names an organisation, not a person |
| nemotron-pii | `coordinate` | CONTACT | geographic coordinates of a place (spec: coordinates) |
| nemotron-pii | `country` | CONTACT | address part: locates where a person can be reached (Decisions) |
| nemotron-pii | `county` | CONTACT | address part: locates where a person can be reached |
| nemotron-pii | `credit_debit_card` | FINANCIAL | financial account, instrument or payment authentication datum |
| nemotron-pii | `customer_id` | NONE | organisation-internal record key, not an identity document (Decisions) |
| nemotron-pii | `cvv` | FINANCIAL | financial account, instrument or payment authentication datum |
| nemotron-pii | `date` | NONE | a date or time alone is not tied to a person |
| nemotron-pii | `date_of_birth` | IDENTITY | date of birth: core identity attribute |
| nemotron-pii | `date_time` | NONE | a date or time alone is not tied to a person |
| nemotron-pii | `device_identifier` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| nemotron-pii | `education_level` | NONE | socio-economic attribute, not Art. 9 |
| nemotron-pii | `email` | CONTACT | e-mail address: a contact channel |
| nemotron-pii | `employee_id` | NONE | organisation-internal record key, not an identity document (Decisions) |
| nemotron-pii | `employment_status` | NONE | socio-economic attribute, not Art. 9 |
| nemotron-pii | `fax_number` | CONTACT | fax number: a contact channel |
| nemotron-pii | `first_name` | IDENTITY | personal name part: directly identifies a person |
| nemotron-pii | `gender` | NONE | not among the Art. 9 categories (Decisions) |
| nemotron-pii | `health_plan_beneficiary_number` | HEALTH | health insurance identifier (spec) |
| nemotron-pii | `http_cookie` | CREDENTIALS | session token: grants authenticated access |
| nemotron-pii | `ipv4` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| nemotron-pii | `ipv6` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| nemotron-pii | `language` | NONE | spoken language, not Art. 9 |
| nemotron-pii | `last_name` | IDENTITY | personal name part: directly identifies a person |
| nemotron-pii | `license_plate` | NONE | identifies a vehicle, not a person directly (Decisions) |
| nemotron-pii | `mac_address` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| nemotron-pii | `medical_record_number` | HEALTH | medical record identifier: links to health data |
| nemotron-pii | `national_id` | IDENTITY | government-issued national identifier |
| nemotron-pii | `occupation` | NONE | job title, not Art. 9 (Decisions) |
| nemotron-pii | `password` | CREDENTIALS | authentication secret or login identifier |
| nemotron-pii | `phone_number` | CONTACT | telephone number: a contact channel |
| nemotron-pii | `pin` | FINANCIAL | PIN; the spec lists PIN under FINANCIAL (Decisions) |
| nemotron-pii | `political_view` | SPECIAL_CATEGORY | GDPR Art. 9(1) special category: political opinions |
| nemotron-pii | `postcode` | CONTACT | address part: locates where a person can be reached |
| nemotron-pii | `race_ethnicity` | SPECIAL_CATEGORY | GDPR Art. 9(1) special category: racial or ethnic origin |
| nemotron-pii | `religious_belief` | SPECIAL_CATEGORY | GDPR Art. 9(1) special category: religious beliefs |
| nemotron-pii | `sexuality` | SPECIAL_CATEGORY | GDPR Art. 9(1) special category: sex life or sexual orientation |
| nemotron-pii | `ssn` | IDENTITY | social security number: government identifier |
| nemotron-pii | `state` | CONTACT | address part: locates where a person can be reached |
| nemotron-pii | `street_address` | CONTACT | address part: locates where a person can be reached |
| nemotron-pii | `swift_bic` | FINANCIAL | SWIFT/BIC of a person's bank (spec: SWIFT) |
| nemotron-pii | `tax_id` | IDENTITY | government tax identifier |
| nemotron-pii | `time` | NONE | a date or time alone is not tied to a person |
| nemotron-pii | `unique_id` | NONE | opaque technical identifier of unknown kind (Decisions) |
| nemotron-pii | `url` | NONE | web address: not tied to a category |
| nemotron-pii | `user_name` | CREDENTIALS | login identifier (spec: user name; Decisions) |
| nemotron-pii | `vehicle_identifier` | NONE | identifies a vehicle, not a person directly (Decisions) |
| gretel-finance-en | `account_pin` | FINANCIAL | account PIN (spec: PIN) |
| gretel-finance-en | `api_key` | CREDENTIALS | authentication secret or login identifier |
| gretel-finance-en | `bank_routing_number` | FINANCIAL | identifies the bank of a person's account (spec: routing) |
| gretel-finance-en | `bban` | FINANCIAL | basic bank account number (spec: BBAN) |
| gretel-finance-en | `company` | NONE | names an organisation, not a person |
| gretel-finance-en | `credit_card_number` | FINANCIAL | financial account, instrument or payment authentication datum |
| gretel-finance-en | `credit_card_security_code` | FINANCIAL | card security code (spec: CVV) |
| gretel-finance-en | `customer_id` | NONE | organisation-internal record key, not an identity document (Decisions) |
| gretel-finance-en | `date` | NONE | a date or time alone is not tied to a person |
| gretel-finance-en | `date_of_birth` | IDENTITY | date of birth: core identity attribute |
| gretel-finance-en | `date_time` | NONE | a date or time alone is not tied to a person |
| gretel-finance-en | `driver_license_number` | IDENTITY | government-issued driver licence number |
| gretel-finance-en | `email` | CONTACT | e-mail address: a contact channel |
| gretel-finance-en | `employee_id` | NONE | organisation-internal record key, not an identity document (Decisions) |
| gretel-finance-en | `first_name` | IDENTITY | personal name part: directly identifies a person |
| gretel-finance-en | `iban` | FINANCIAL | international bank account number (spec: IBAN) |
| gretel-finance-en | `ipv4` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| gretel-finance-en | `ipv6` | NONE | network/device identifier: no category covers online identifiers (Decisions) |
| gretel-finance-en | `last_name` | IDENTITY | personal name part: directly identifies a person |
| gretel-finance-en | `local_latlng` | CONTACT | latitude/longitude of a place (spec: coordinates) |
| gretel-finance-en | `name` | IDENTITY | full personal name: directly identifies a person |
| gretel-finance-en | `passport_number` | IDENTITY | government-issued passport number |
| gretel-finance-en | `password` | CREDENTIALS | authentication secret or login identifier |
| gretel-finance-en | `phone_number` | CONTACT | telephone number: a contact channel |
| gretel-finance-en | `ssn` | IDENTITY | social security number: government identifier |
| gretel-finance-en | `street_address` | CONTACT | address part: locates where a person can be reached |
| gretel-finance-en | `swift_bic_code` | FINANCIAL | SWIFT/BIC of a person's bank (spec: SWIFT) |
| gretel-finance-en | `time` | NONE | a date or time alone is not tied to a person |
| gretel-finance-en | `user_name` | CREDENTIALS | login identifier (spec: user name; Decisions) |
<!-- labels:end -->
