
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from agents import function_tool


# --------------------------------------------------
# File paths and configuration
# --------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_cleaned"

SUMMARY_FILE = DATA_DIR / "house_representative_analysis_2027.csv"
DONATIONS_FILE = DATA_DIR / "house_individual_donations_pre_request.csv"
EARMARKS_FILE = HERE / "earmark_sponsors.csv"

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITIES = (
    "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
)

HEADERS = {
    "User-Agent": "CongressionalResearchStudentProject/1.0 (educational research)"
}


def _json(data):
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _clean(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


@lru_cache(maxsize=1)
def _summary():
    return pd.read_csv(
        SUMMARY_FILE,
        dtype={
            "bioguide_id": str,
            "principal_committee_id": str,
        },
    )


def _find_representative(name):
    name = str(name).strip()
    if not name:
        raise ValueError("Provide a representative name or Bioguide ID.")
    df = _summary()
    by_id = df[df["bioguide_id"].astype(str).str.casefold() == name.casefold()]
    if len(by_id) == 1:
        return by_id.iloc[0]
    exact = df[df["full_name"].fillna("").str.casefold() == name.casefold()]
    matches = exact if not exact.empty else df[df["full_name"].fillna("").str.contains(name, case=False, regex=False)]
    if len(matches) != 1:
        raise ValueError(f"Expected one representative matching {name!r}; found {len(matches)}. Matches: {matches['full_name'].head(12).tolist()}")
    return matches.iloc[0]


# --------------------------------------------------
# Tool 1: Representative summary
# --------------------------------------------------

@function_tool
def get_representative_summary(name: str) -> str:
    """Get an incumbent House representative's donation and FY2027 earmark summary.

    Args:
        name: Full or partial representative name.
    """
    try:
        row = _find_representative(name)

        return _json({
            "name": row["full_name"],
            "bioguide_id": row["bioguide_id"],
            "state": row["state"],
            "district": _clean(row["district"]),
            "party": row["party"],
            "principal_committee_id": _clean(
                row["principal_committee_id"]
            ),
            "committee_status": _clean(
                row["committee_status"]
            ),
            "total_individual_contributions": _clean(
                row["total_individual_contributions"]
            ),
            "contribution_transactions": _clean(
                row["contribution_transactions"]
            ),
            "earmark_project_count": _clean(
                row["earmark_project_count"]
            ),
            "total_sponsored_request_amount": _clean(
                row["total_sponsored_request_amount"]
            ),
            "fractional_request_amount": _clean(
                row["fractional_request_amount"]
            ),
            "requested_earmarks": bool(
                row["requested_earmarks"]
            ),
            "donation_period": "2025-01-01 to 2026-02-28",
            "earmark_period": "FY2027 requests, not awards",
            "source": str(SUMMARY_FILE.relative_to(HERE)),
        })

    except (ValueError, OSError, KeyError) as exc:
        return _json({"error": str(exc)})


# --------------------------------------------------
# Tool 2: Top individual campaign donors
# --------------------------------------------------

@function_tool
def get_top_donors(
    representative: str,
    limit: int = 10,
) -> str:
    """Rank itemized individual campaign contributors for one House representative.

    Groups contributions by disclosed name, employer, occupation,
    city, and state. Identical names do not necessarily identify
    the same person.

    Args:
        representative: House representative's name.
        limit: Number of donor groups to return, from 1 to 25.
    """
    try:
        row = _find_representative(representative)

        committee_id = _clean(
            row["principal_committee_id"]
        )

        if not committee_id:
            return _json({
                "error": "No principal committee ID for this representative."
            })

        limit = max(1, min(int(limit), 25))

        cols = [
            "committee_id",
            "contributor_name",
            "employer",
            "occupation",
            "city",
            "state",
            "amount",
            "donation_date",
        ]

        grouped = {}
        matched_transactions = 0
        matched_total = 0.0

        # Process the large FEC CSV in chunks.
        for chunk in pd.read_csv(
            DONATIONS_FILE,
            usecols=cols,
            dtype={"committee_id": str},
            chunksize=100_000,
            low_memory=False,
        ):
            part = chunk.loc[
                chunk["committee_id"] == committee_id
            ].copy()

            if part.empty:
                continue

            part["amount"] = pd.to_numeric(
                part["amount"],
                errors="coerce",
            )

            part = part.loc[part["amount"] > 0]

            matched_transactions += len(part)
            matched_total += float(part["amount"].sum())

            keys = [
                "contributor_name",
                "employer",
                "occupation",
                "city",
                "state",
            ]

            for key in keys:
                part[key] = (
                    part[key]
                    .fillna("UNKNOWN")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

            agg = (
                part.groupby(keys, dropna=False)["amount"]
                .agg(["sum", "count"])
            )

            for key, values in agg.iterrows():
                old = grouped.setdefault(key, [0.0, 0])

                old[0] += float(values["sum"])
                old[1] += int(values["count"])

        donors = sorted(
            grouped.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:limit]

        return _json({
            "representative": row["full_name"],
            "committee_id": committee_id,
            "committee_status": _clean(
                row["committee_status"]
            ),
            "matching_transactions": matched_transactions,
            "matching_contributions_total": round(
                matched_total, 2
            ),
            "donors": [
                {
                    "contributor_name": key[0],
                    "employer": key[1],
                    "occupation": key[2],
                    "city": key[3],
                    "state": key[4],
                    "total_amount": round(values[0], 2),
                    "transaction_count": values[1],
                }
                for key, values in donors
            ],
            "source": str(
                DONATIONS_FILE.relative_to(HERE)
            ),
            "limitations": [
                "These are itemized individual contributions, not all campaign receipts.",
                "A donor with changed disclosure details may appear in multiple groups.",
                "Different people with identical disclosure fields may be grouped together.",
                "Employer information is self-reported and does not imply employer-directed donations.",
                "Historical committee matches require caution.",
            ],
        })

    except (ValueError, OSError, KeyError) as exc:
        return _json({"error": str(exc)})


# --------------------------------------------------
# Tool 3: Individual FY2027 earmark projects
# --------------------------------------------------

EARMARKS_ATTRIBUTED = DATA_DIR / "earmarks_by_bioguide.csv"
EARMARKS_UNRESOLVED = DATA_DIR / "earmarks_unresolved.csv"

@function_tool
def get_earmark_projects(representative: str, limit: int = 30) -> str:
    """List conservatively attributed FY2027 earmark requests by Bioguide ID.

    Args:
        representative: Full representative name or Bioguide ID.
        limit: Maximum project rows to return, 1 to 100.
    """
    try:
        row = _find_representative(representative)
        member_id = str(row["bioguide_id"])
        limit = max(1, min(int(limit), 100))
        if not EARMARKS_ATTRIBUTED.exists() or not EARMARKS_UNRESOLVED.exists():
            return _json({"status":"data_not_built","representative":row["full_name"],
                          "note":"do not retry", "message":"Run python congress_tools.py --build-earmarks offline first."})
        attributed = pd.read_csv(EARMARKS_ATTRIBUTED, dtype={"bioguide_id":str},low_memory=False)
        unresolved = pd.read_csv(EARMARKS_UNRESOLVED,dtype={"possible_bioguide_ids":str},low_memory=False)
        projects = attributed.loc[attributed.bioguide_id == member_id].copy()
        projects["amount"] = pd.to_numeric(projects["amount"],errors="coerce")
        projects = projects.sort_values("amount",ascending=False,na_position="last")
        possible = unresolved.loc[unresolved.possible_bioguide_ids.fillna("").map(
            lambda ids: member_id in ids.split(";"))].copy()
        possible["amount"] = pd.to_numeric(possible["amount"],errors="coerce")
        fields = [c for c in ["project_id","project_name","recipient","agency","account","appropriations_bill","location","state","amount","requestors","attribution"] if c in projects.columns]
        records = [{k:_clean(r[k]) for k in fields} for _,r in projects.head(limit).iterrows()]
        return _json({
            "status":"partial" if len(possible) else "success",
            "note":"do not retry" if len(possible) else "Attribution based on roster name and optional representative state; not independently verified",
            "representative":row["full_name"],"bioguide_id":member_id,
            "matching_project_rows":len(projects),"returned_project_rows":len(records),
            "projects":records,
            "unattributed_possible_projects":{
                "count":len(possible),"requested_amount_sum_not_attributable":_clean(possible.amount.sum(min_count=1)),
                "project_ids":possible.project_id.astype(str).head(30).tolist()},
            "source":str(EARMARKS_ATTRIBUTED.relative_to(HERE)),
            "unresolved_source":str(EARMARKS_UNRESOLVED.relative_to(HERE)),
            "limitations":["The project state is not evidence of the sponsoring member's identity.",
                           "Name and representative-state matching is provisional attribution, not independent verification.",
                           "Unresolved possible projects are not included in confirmed member totals.",
                           "Amounts are requests, not awards; shared projects may be counted for multiple requestors."]})
    except (ValueError,OSError,KeyError) as exc:
        return _json({"error":str(exc),"note":"do not retry without changing underlying data"})


# --------------------------------------------------
# Tool 4: Donor background research using Wikidata
# --------------------------------------------------

@function_tool
def get_donor_affiliations(
    donor_name: str,
    employer: str = "",
    occupation: str = "",
) -> str:
    """Search Wikidata for possible donor identities and public affiliations.

    Wikidata name matches are not proof that an FEC contributor
    is the same person. Independent verification is required.

    Args:
        donor_name: Contributor's disclosed name.
        employer: Employer listed in the FEC record, if available.
        occupation: Occupation listed in the FEC record, if available.
    """
    donor_name = donor_name.strip()

    if len(donor_name) < 3:
        return _json({
            "error": "Provide a donor name of at least three characters."
        })

    try:
        # Search Wikidata for people with matching names.
        response = requests.get(
            WIKIDATA_SEARCH,
            params={
                "action": "wbsearchentities",
                "search": donor_name,
                "language": "en",
                "format": "json",
                "type": "item",
                "limit": 5,
            },
            headers=HEADERS,
            timeout=12,
        )

        response.raise_for_status()

        candidates = []

        for match in response.json().get("search", []):
            qid = match["id"]

            item = {
                "wikidata_id": qid,
                "label": match.get("label"),
                "description": match.get("description"),
                "url": f"https://www.wikidata.org/wiki/{qid}",
            }

            try:
                # Retrieve structured Wikidata statements.
                detail = requests.get(
                    WIKIDATA_ENTITIES.format(qid),
                    headers=HEADERS,
                    timeout=12,
                )

                detail.raise_for_status()

                entity = detail.json()["entities"][qid]
                claims = entity.get("claims", {})

                # Wikidata property identifiers.
                properties = {
                    "P108": "employer",
                    "P106": "occupation",
                    "P463": "member_of",
                    "P39": "position_held",
                    "P69": "educated_at",
                }

                relations = {}

                for pid, label in properties.items():
                    ids = []

                    for claim in claims.get(pid, [])[:8]:
                        value = (
                            claim.get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value")
                        )

                        if (
                            isinstance(value, dict)
                            and value.get("id")
                        ):
                            ids.append(value["id"])

                    if ids:
                        relations[label] = [
                            {
                                "wikidata_id": rid,
                                "url": (
                                    "https://www.wikidata.org/wiki/"
                                    f"{rid}"
                                ),
                            }
                            for rid in ids
                        ]

                item["relationships"] = relations

            except (
                requests.RequestException,
                KeyError,
                ValueError,
            ):
                item["relationships_error"] = (
                    "Could not retrieve detailed Wikidata statements."
                )

            candidates.append(item)

        return _json({
            "searched_name": donor_name,
            "fec_employer_for_comparison": employer,
            "fec_occupation_for_comparison": occupation,
            "possible_wikidata_candidates": candidates,
            "identity_status": (
                "UNVERIFIED - candidates are name-search "
                "results only"
            ),
            "source": "https://www.wikidata.org/",
            "warning": (
                "Never attribute an affiliation to the FEC "
                "donor without independently confirming that "
                "the Wikidata person is the same individual. "
                "A shared name or employer alone is not definitive."
            ),
        })

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        return _json({
            "error": (
                f"Wikidata lookup unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        })


# Offline earmark attribution builder (run once locally).
import re
import unicodedata
ROSTER = SUMMARY_FILE
RAW = EARMARKS_FILE
DATA = DATA_DIR
SUFFIXES = {'jr','sr','ii','iii','iv','v'}

def _earmark_norm(value):
    if pd.isna(value): return ''
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii','ignore').decode().lower()
    return ' '.join(re.sub(r'[^a-z0-9]+',' ',value).split())


# Multiword surnames appearing in the House roster.
COMPOUND_SURNAMES = (
    "wasserman schultz", "watson coleman", "miller meeks",
    "leger fernandez", "kamlager dove", "mcclain delaney",
    "mcdonald rivet", "diaz balart", "de la cruz",
    "van duyne", "van orden", "king hinds", "ocasio cortez",
    "van drew",
)


def _earmark_surname(name):
    raw = str(name).strip()
    # Only treat a comma as Last, First if the trailing part isn't a suffix.
    if "," in raw:
        before, after = raw.split(",", 1)
        after_words = _earmark_norm(after).split()
        if after_words and after_words[0] not in SUFFIXES:
            raw = before

    words = _earmark_norm(raw).split()
    while words and words[-1] in SUFFIXES:
        words.pop()
    normalized = " ".join(words)
    for compound in sorted(COMPOUND_SURNAMES, key=len, reverse=True):
        if normalized == compound or normalized.endswith(" " + compound):
            return compound
    return words[-1] if words else ""


def _parse_earmark_representative(value):
    text = str(value).strip()
    match = re.fullmatch(r"(.*?)\s*\(([A-Za-z]{2})\)\s*", text)
    if match:
        name, state = match.group(1).strip(), match.group(2).upper()
    else:
        name, state = text, None
    # 'Scott, Austin (GA)' -> surname Scott, given name Austin.
    # A given name provides stronger evidence than surname + state alone.
    given_name = None
    if "," in name:
        before, after = name.split(",", 1)
        after_words = _earmark_norm(after).split()
        if after_words and after_words[0] not in SUFFIXES:
            name = before.strip()
            given_name = after_words[0]
    return _earmark_surname(name), state, given_name


def build_earmark_data():
    roster = pd.read_csv(ROSTER, dtype={"bioguide_id": str}, low_memory=False)
    em = pd.read_csv(RAW, low_memory=False)

    for col in ["full_name", "bioguide_id", "state"]:
        if col not in roster.columns:
            raise ValueError(f"Roster missing {col}")

    for col in ["project_id", "representative", "amount", "state"]:
        if col not in em.columns:
            raise ValueError(f"Earmarks missing {col}")

    if roster.bioguide_id.isna().any() or roster.bioguide_id.duplicated().any():
        raise ValueError("Roster has missing or duplicate Bioguide IDs")

    roster["_surname"] = roster["full_name"].map(_earmark_surname)
    roster["_state"] = roster["state"].astype(str).str.strip().str.upper()

    em = em.drop_duplicates(
        subset=["project_id", "representative"]
    ).copy()

    parsed = em["representative"].map(_parse_earmark_representative)
    em["_surname"] = parsed.map(lambda x: x[0])
    em["_rep_state"] = parsed.map(lambda x: x[1])
    em["_given_name"] = parsed.map(lambda x: x[2])

    assigned = []
    unresolved = []

    for record in em.to_dict("records"):
        surname = record.pop("_surname")
        rep_state = record.pop("_rep_state")
        given_name = record.pop("_given_name")

        candidates = roster[roster["_surname"] == surname]

        if rep_state:
            candidates = candidates[candidates["_state"] == rep_state]

        if given_name:
            candidates = candidates[
                candidates["full_name"].map(
                    lambda n: given_name in _earmark_norm(n).split()
                )
            ]

        if len(candidates) == 1:
            member = candidates.iloc[0]

            assigned.append({
                **record,
                "bioguide_id": str(member["bioguide_id"]),
                "full_name": member["full_name"],
                "attribution": (
                    "surname_state_and_given_name"
                    if rep_state and given_name
                    else "surname_and_representative_state"
                    if rep_state
                    else "surname_and_given_name"
                    if given_name
                    else "unique_roster_surname"
                )
            })
        else:
            ids = candidates["bioguide_id"].astype(str).tolist()

            unresolved.append({
                **record,
                "possible_bioguide_ids": ";".join(ids),
                "reason": (
                    "no_roster_match"
                    if not ids
                    else "ambiguous_representative"
                ),
                "candidate_count": len(ids)
            })

    DATA.mkdir(parents=True, exist_ok=True)

    original_cols = [
        c for c in em.columns
        if c not in ["_surname", "_rep_state", "_given_name"]
    ]

    assigned_cols = original_cols + [
        "bioguide_id", "full_name", "attribution"
    ]
    unresolved_cols = original_cols + [
        "possible_bioguide_ids", "reason", "candidate_count"
    ]

    a = pd.DataFrame(assigned, columns=assigned_cols)
    u = pd.DataFrame(unresolved, columns=unresolved_cols)

    a.to_csv(DATA / "earmarks_by_bioguide.csv", index=False)
    u.to_csv(DATA / "earmarks_unresolved.csv", index=False)

    if not a.empty:
        a["amount"] = pd.to_numeric(a["amount"], errors="coerce")

        stats = a.groupby("bioguide_id").agg(
            attributed_project_count=("project_id", "nunique"),
            attributed_request_amount=("amount", "sum")
        )
    else:
        stats = pd.DataFrame(
            columns=["attributed_project_count", "attributed_request_amount"]
        )

    checks = roster.set_index("bioguide_id")[
        [
            "full_name",
            "earmark_project_count",
            "total_sponsored_request_amount"
        ]
    ].join(stats)

    checks["attributed_project_count"] = (
        checks["attributed_project_count"].fillna(0)
    )
    checks["attributed_request_amount"] = (
        checks["attributed_request_amount"].fillna(0)
    )

    checks["project_count_difference"] = (
        checks["attributed_project_count"]
        - pd.to_numeric(checks["earmark_project_count"], errors="coerce")
    )
    checks["amount_difference"] = (
        checks["attributed_request_amount"]
        - pd.to_numeric(
            checks["total_sponsored_request_amount"], errors="coerce"
        )
    )

    checks.to_csv(DATA / "earmarks_reconciliation.csv")

    mismatch = checks[
        (checks["project_count_difference"] != 0)
        | (checks["amount_difference"].abs() > 0.01)
    ]

    print(
        f"{len(assigned)} attributed sponsor rows; "
        f"{len(unresolved)} unresolved; "
        f"{len(mismatch)} summary mismatches"
    )
    print(mismatch.head(20).to_string())
    print(
        "WARNING: Mismatches require investigation. "
        "Project state is NOT evidence of member identity."
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--build-earmarks":
        build_earmark_data()
    else:
        print("Usage: python congress_tools.py --build-earmarks")