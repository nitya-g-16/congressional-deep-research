
"""Local donor–earmark analysis, including automatic cross-industry discovery."""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from agents import function_tool

HERE = Path(__file__).resolve().parent
DONATIONS = HERE / "data_cleaned" / "house_individual_donations_pre_request.csv"
EARMARKS = HERE / "earmark_sponsors.csv"
SUMMARY = HERE / "data_cleaned" / "house_representative_analysis_2027.csv"

# Categories are used to organize findings, not to restrict direct matching.
INDUSTRIES = {
    "construction": {
        "aliases": ["construction", "building", "engineering", "infrastructure"],
        "donor": r"construction|contractor|architect|engineer|building|infrastructure",
        "project": r"construction|renovat|building|repair|moderniz|infrastructure|redevelop",
    },
    "housing_real_estate": {
        "aliases": ["housing", "real estate", "property", "development"],
        "donor": r"real estate|realtor|property|housing|developer|development",
        "project": r"housing|residential|affordable|shelter|redevelop|property",
    },
    "healthcare": {
        "aliases": ["healthcare", "health", "medical", "hospital"],
        "donor": r"hospital|health|medical|physician|nurs|pharma|biotech|clinic",
        "project": r"hospital|health|medical|clinic|patient|substance|treatment",
    },
    "education": {
        "aliases": ["education", "school", "university", "college"],
        "donor": r"school|university|college|teacher|professor|education",
        "project": r"school|university|college|education|student|classroom",
    },
    "energy_environment": {
        "aliases": ["energy", "environment", "climate", "renewable", "conservation"],
        "donor": r"energy|solar|renewable|environment|utility|conservation",
        "project": r"energy|solar|environment|climate|conservation|electric",
    },
    "transportation": {
        "aliases": ["transportation", "transit", "roads", "rail"],
        "donor": r"transport|transit|rail|aviation|logistics|shipping",
        "project": r"transport|transit|rail|road|bridge|ferry|airport|traffic",
    },
    "technology": {
        "aliases": ["technology", "tech", "software", "ai", "telecom"],
        "donor": r"software|technology|computer|telecom|semiconductor|artificial intelligence",
        "project": r"technology|digital|broadband|computer|cyber|telecom",
    },
    "finance": {
        "aliases": ["finance", "banking", "insurance", "investment"],
        "donor": r"bank|finance|investment|insurance|capital management|venture capital",
        "project": r"finance|banking|financial|insurance|economic development",
    },
    "public_safety": {
        "aliases": ["public safety", "police", "fire", "emergency"],
        "donor": r"police|fire department|security|emergency|public safety",
        "project": r"police|fire|emergency|public safety|law enforcement",
    },
    "arts_culture": {
        "aliases": ["arts", "culture", "museum", "entertainment"],
        "donor": r"artist|museum|music|theater|theatre|entertainment|arts",
        "project": r"museum|arts|culture|theater|theatre|historic preservation",
    },
    "social_services": {
        "aliases": ["social services", "nonprofit", "community services"],
        "donor": r"nonprofit|social work|community services|human services",
        "project": r"social services|food bank|homeless|community center|youth services",
    },
    "water_utilities": {
        "aliases": ["water", "utilities", "wastewater", "sewer"],
        "donor": r"water utility|water district|wastewater|utilities",
        "project": r"water|wastewater|sewer|drainage|stormwater",
    },
    "agriculture": {
        "aliases": ["agriculture", "farming", "food"],
        "donor": r"agricultur|farm|food production|agribusiness",
        "project": r"agricultur|farm|food system|irrigation",
    },
}

INVALID_EMPLOYERS = {
    "", "none", "unknown", "not employed", "unemployed",
    "retired", "self employed", "self", "n a", "not applicable",
    "information requested", "homemaker", "student",
}

ORG_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "co", "company", "pllc",
}

def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()

def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()

def canonical_org(value):
    words = norm(value).split()
    while words and words[-1] in ORG_SUFFIXES:
        words.pop()
    return " ".join(words)

def valid_org(value):
    name = canonical_org(value)
    return len(name) >= 5 and name not in INVALID_EMPLOYERS

def industry_for_text(value, field):
    text = clean(value)
    return {
        name
        for name, spec in INDUSTRIES.items()
        if re.search(spec[field], text, re.IGNORECASE)
    }

def requested_industries(industry):
    query = norm(industry)
    if query in {"", "all", "any", "all industries", "automatic", "general"}:
        return None
    selected = set()
    for name, spec in INDUSTRIES.items():
        terms = [name.replace("_", " ")] + spec["aliases"]
        if any(norm(term) == query or
               (len(norm(term)) > 3 and norm(term) in query)
               for term in terms):
            selected.add(name)
    # Unknown industry: still run exact organization matching across all
    # records, but do not invent a classification for the requested term.
    return selected if selected else set()

def resolve_rep(representative):
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing representative summary: {SUMMARY}")

    df = pd.read_csv(SUMMARY, low_memory=False)
    required = {"full_name", "principal_committee_id"}
    if not required.issubset(df.columns):
        raise ValueError("Summary CSV lacks full_name or principal_committee_id")

    query = norm(representative)
    if not query:
        raise ValueError("Representative name is required")

    exact = df[df["full_name"].map(norm) == query]
    if len(exact) != 1:
        partial = df[
            df["full_name"].map(norm).map(
                lambda name: query in name if query else False
            )
        ]
        if len(partial) != 1:
            raise ValueError(
                f"Representative not uniquely identified: {representative}"
            )
        exact = partial

    row = exact.iloc[0]
    return clean(row["full_name"]), clean(row["principal_committee_id"])

def get_projects(representative):
    df = pd.read_csv(EARMARKS, low_memory=False)
    required = {"project_id", "representative", "project_name",
                "recipient", "amount"}
    if not required.issubset(df.columns):
        raise ValueError("Earmark CSV is missing required columns")

    # The local file stores the sponsor surname in representative.
    # Handle compound surnames such as Ocasio-Cortez.
    surname = norm(representative).split()
    last = surname[-1]
    if len(surname) >= 2 and surname[-2] == "ocasio":
        last = "ocasio cortez"

    df = df[df["representative"].map(norm) == last]
    df = df.drop_duplicates(subset=["project_id"])

    projects = []
    for row in df.to_dict("records"):
        project_text = " ".join(
            clean(row.get(column))
            for column in ["project_name", "account", "agency"]
        )
        categories = industry_for_text(project_text, "project")
        amount = pd.to_numeric(row.get("amount"), errors="coerce")

        projects.append({
            "project_id": clean(row.get("project_id")),
            "project_name": clean(row.get("project_name")),
            "recipient": clean(row.get("recipient")),
            "amount_requested": float(amount) if pd.notna(amount) else None,
            "agency": clean(row.get("agency")),
            "account": clean(row.get("account")),
            "location": clean(row.get("location")),
            "industries": sorted(categories),
        })

    return projects

def get_donors(committee_id):
    required = [
        "committee_id", "entity_type", "contributor_name",
        "employer", "occupation", "amount",
    ]
    optional = ["fec_record_id", "memo_code", "transaction_id"]
    columns = required + optional

    donors = defaultdict(lambda: {
        "contributor_name": "",
        "employer": "",
        "occupation": "",
        "net_contributions": 0.0,
        "transactions": 0,
        "industries": set(),
    })

    seen_records = set()
    matching_transactions = 0

    for chunk in pd.read_csv(
        DONATIONS,
        usecols=lambda col: col in columns,
        chunksize=100000,
        low_memory=False,
    ):
        if not set(required).issubset(chunk.columns):
            raise ValueError("Donation CSV is missing required columns")

        subset = chunk[
            (chunk["committee_id"].astype(str).str.strip() == committee_id)
            & (chunk["entity_type"].astype(str).str.upper() == "IND")
        ]

        for row in subset.to_dict("records"):
            record_id = clean(row.get("fec_record_id"))
            if record_id:
                if record_id in seen_records:
                    continue
                seen_records.add(record_id)

            amount = pd.to_numeric(row.get("amount"), errors="coerce")
            if pd.isna(amount):
                continue

            name = clean(row.get("contributor_name"))
            employer = clean(row.get("employer"))
            occupation = clean(row.get("occupation"))

            categories = industry_for_text(
                employer + " " + occupation, "donor"
            )

            key = (norm(name), canonical_org(employer), norm(occupation))
            donor = donors[key]
            donor["contributor_name"] = name
            donor["employer"] = employer
            donor["occupation"] = occupation
            donor["net_contributions"] += float(amount)
            donor["transactions"] += 1
            donor["industries"].update(categories)
            matching_transactions += 1

    result = []
    for donor in donors.values():
        donor["net_contributions"] = round(
            donor["net_contributions"], 2
        )
        donor["industries"] = sorted(donor["industries"])
        result.append(donor)

    result.sort(key=lambda d: d["net_contributions"], reverse=True)
    return result, matching_transactions

def organization_matches(donors, projects):
    # An exact normalized name is a candidate match, not verified affiliation.
    employer_index = defaultdict(list)

    for donor in donors:
        employer = donor["employer"]
        if valid_org(employer):
            employer_index[canonical_org(employer)].append(donor)

    matches = []
    for project in projects:
        recipient = project["recipient"]
        if not valid_org(recipient):
            continue

        for donor in employer_index.get(canonical_org(recipient), []):
            matches.append({
                "match_type": "exact_normalized_employer_recipient",
                "confidence": "candidate_requires_verification",
                "contributor_name": donor["contributor_name"],
                "donor_employer": donor["employer"],
                "donor_net_contributions": donor["net_contributions"],
                "project_name": project["project_name"],
                "earmark_recipient": recipient,
                "requested_amount": project["amount_requested"],
                "explanation": (
                    "The disclosed donor employer and earmark recipient "
                    "have the same normalized organization name. This "
                    "does not prove identity, influence, or financial benefit."
                ),
            })

    matches.sort(
        key=lambda x: abs(x["donor_net_contributions"]),
        reverse=True,
    )
    return matches

def analyze(representative, industry="all", limit=15):
    if not DONATIONS.exists():
        raise FileNotFoundError(f"Missing donation CSV: {DONATIONS}")
    if not EARMARKS.exists():
        raise FileNotFoundError(f"Missing earmark CSV: {EARMARKS}")

    name, committee_id = resolve_rep(representative)
    projects = get_projects(name)
    donors, transaction_count = get_donors(committee_id)

    selected = requested_industries(industry)
    all_matches = organization_matches(donors, projects)

    if selected is None:
        filtered_donors = donors
        filtered_projects = projects
        direct_matches = all_matches
        mode = "all_industries"
    elif selected:
        filtered_donors = [
            d for d in donors
            if selected.intersection(d["industries"])
        ]
        filtered_projects = [
            p for p in projects
            if selected.intersection(p["industries"])
        ]
        relevant_projects = {
            p["project_id"] for p in filtered_projects
        }
        direct_matches = [
            m for m in all_matches
            if any(
                p["project_name"] == m["project_name"]
                and p["project_id"] in relevant_projects
                for p in projects
            )
        ]
        mode = "selected_industries"
    else:
        # For an unrecognized category, exact matches are still meaningful.
        filtered_donors = []
        filtered_projects = []
        direct_matches = all_matches
        mode = "unrecognized_industry_exact_matches_only"

    # Broad industry overlap is separate from organization-level matches.
    donor_categories = set().union(
        *(set(d["industries"]) for d in donors)
    ) if donors else set()

    project_categories = set().union(
        *(set(p["industries"]) for p in projects)
    ) if projects else set()

    shared_categories = sorted(donor_categories & project_categories)
    if selected is not None:
        shared_categories = [
            c for c in shared_categories if c in selected
        ]

    industry_summary = []
    for category in shared_categories:
        category_donors = [
            d for d in donors if category in d["industries"]
        ]
        category_projects = [
            p for p in projects if category in p["industries"]
        ]
        industry_summary.append({
            "industry": category,
            "donor_groups": len(category_donors),
            "donation_net_total": round(
                sum(d["net_contributions"] for d in category_donors), 2
            ),
            "earmark_project_count": len(category_projects),
            "earmark_requested_total": round(
                sum(p["amount_requested"] or 0 for p in category_projects),
                2,
            ),
        })

    industry_summary.sort(
        key=lambda item: item["earmark_requested_total"],
        reverse=True,
    )

    limit = max(1, min(int(limit), 30))

    return {
        "representative": name,
        "principal_committee_id": committee_id,
        "requested_industry": industry,
        "analysis_mode": mode,
        "recognized_industries": sorted(INDUSTRIES.keys()),
        "donation_period": "2025-01-01 to 2026-02-28",
        "earmark_period": "FY2027 requests, not awards",
        "individual_donation_transactions_processed": transaction_count,
        "total_donor_groups": len(donors),
        "total_earmark_projects": len(projects),
        "direct_organization_candidate_count": len(direct_matches),
        "direct_organization_candidates": direct_matches[:50],
        "industry_overlap_summary": industry_summary,
        "matching_donor_groups": len(filtered_donors),
        "matching_donation_net_total": round(
            sum(d["net_contributions"] for d in filtered_donors), 2
        ),
        "top_matching_donors": filtered_donors[:limit],
        "matching_earmark_projects": filtered_projects[:50],
        "matching_earmark_requested_total": round(
            sum(p["amount_requested"] or 0 for p in filtered_projects),
            2,
        ),
        "limitations": [
            "Donor employers and occupations are self-reported.",
            "Individual contributions are not corporate contributions.",
            "Employer-recipient matches are unverified name matches.",
            "An earmark recipient is not necessarily the final contractor.",
            "Broad industry categories are heuristic, not official classifications.",
            "A donor may appear in multiple industry categories.",
            "Industry totals can overlap and must not be summed across categories.",
            "No subcontractors, ownership networks, or indirect benefits are verified.",
            "A lack of exact matches does not rule out indirect relationships.",
            "Donation data covers only the stated partial period.",
            "Earmark amounts are requests, not approved funding.",
            "Records may contain intermediary, memo, or refund transactions.",
            "A shared industry or organization name does not establish causation.",
            "Surname-based earmark matching may be ambiguous for some members.",
        ],
    }

@function_tool
def analyze_industry_overlap(
    representative: str,
    industry: str = "all",
    limit: int = 15,
) -> str:
    """Analyze individual campaign donors and FY2027 earmark requests.

    Args:
        representative: Full name of the House representative.
        industry: Industry to investigate, or 'all' for automatic discovery.
        limit: Maximum number of donor groups to return.
    """
    try:
        return json.dumps(
            analyze(representative, industry, limit),
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "representative": representative,
            "industry": industry,
        })