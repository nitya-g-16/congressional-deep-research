
import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from agents import RunContextWrapper, function_tool


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data_cleaned"

CANDIDATE_PATH = DATA_DIR / "house_candidate_lookup_2026.csv"
PAC_SUMMARY_PATH = DATA_DIR / "house_pac_summary_2026.csv"
PAC_TRANSACTIONS_PATH = DATA_DIR / "house_pac_transactions_2026.csv"
REPRESENTATIVE_SUMMARY_PATH = DATA_DIR / "house_representative_analysis_2027.csv"


# --------------------------------------------------
# LOAD LOCAL DATA
# --------------------------------------------------

@lru_cache(maxsize=1)
def _load_candidates():
    df = pd.read_csv(
        CANDIDATE_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "candidate_id",
        "candidate_name",
        "principal_committee_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate lookup missing columns: {sorted(missing)}"
        )

    return df


@lru_cache(maxsize=1)
def _load_transactions():
    df = pd.read_csv(
        PAC_TRANSACTIONS_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "candidate_id",
        "committee_id",
        "committee_name",
        "amount",
        "date",
        "review_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"PAC transactions missing columns: {sorted(missing)}"
        )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return df


# --------------------------------------------------
# CANDIDATE NAME MATCHING
# --------------------------------------------------

def _normalize_name(name):
    """
    Normalize names such as:
    JOHNSON, MIKE -> mike johnson
    Mike Johnson -> mike johnson
    """

    name = str(name).strip()

    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"

    name = name.casefold()
    name = re.sub(r"[^a-z0-9\s]", " ", name)

    return " ".join(name.split())


@lru_cache(maxsize=1)
def _load_representatives():
    df = pd.read_csv(REPRESENTATIVE_SUMMARY_PATH, dtype=str, keep_default_na=False)
    required = {"full_name", "principal_committee_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Representative summary missing columns: {sorted(missing)}")
    return df


def _find_candidate(representative):
    """Resolve incumbent name, principal committee ID, or FEC candidate ID.

    Incumbent names come from the same roster used by congress_tools.py.
    Committee IDs are matched to the FEC candidate master without guessing
    which historical candidate is meant.
    """
    query = str(representative).strip()
    if not query:
        raise ValueError("Representative cannot be empty")

    candidates = _load_candidates()
    direct = candidates[candidates["candidate_id"].str.casefold() == query.casefold()]
    if len(direct) == 1:
        return direct.iloc[0]["candidate_id"]

    reps = _load_representatives()
    committee_id = query if re.fullmatch(r"C\d{8}", query, flags=re.I) else None
    if committee_id is None:
        normalized = _normalize_name(query)
        matches = reps[reps["full_name"].map(_normalize_name) == normalized]
        if matches.empty:
            matches = reps[reps["full_name"].map(_normalize_name).map(
                lambda n: normalized in n
            )]
        matches = matches.drop_duplicates(subset=["full_name", "principal_committee_id"])
        if len(matches) != 1:
            raise ValueError(
                f"Representative not uniquely found in incumbent roster: {representative} "
                f"({len(matches)} matches)"
            )
        committee_id = matches.iloc[0]["principal_committee_id"].strip()
        if not committee_id:
            raise ValueError(f"No principal committee ID for {representative}")

    matches = candidates[
        candidates["principal_committee_id"].str.casefold() == committee_id.casefold()
    ].drop_duplicates(subset=["candidate_id"])
    if len(matches) != 1:
        raise ValueError(
            f"Principal committee {committee_id} maps to {len(matches)} "
            "candidate IDs in the FEC candidate master; cannot safely identify "
            "a single candidate. An authoritative candidate-committee linkage "
            "is needed if this is missing or ambiguous."
        )
    return matches.iloc[0]["candidate_id"]


def find_committee(representative):
    candidate_id = _find_candidate(representative)
    matches = _load_candidates()
    matches = matches[matches["candidate_id"] == candidate_id]
    committee_id = matches.iloc[0]["principal_committee_id"].strip()
    if not committee_id:
        raise ValueError(f"No principal committee ID for {representative}")
    return committee_id


# --------------------------------------------------
# PAC CONTRIBUTION ANALYSIS
# --------------------------------------------------

def _fetch_pac_data_sync(representative, limit=10, cycle=2026):
    """
    Analyze locally cleaned FEC committee-to-candidate
    contribution records.

    PAC rankings are provisional:
    - Exclude joint fundraising records.
    - Exclude transactions flagged for large-amount review.
    - Preserve negative amounts when calculating net totals.
    """

    try:
        cycle = int(cycle)
        limit = max(1, min(int(limit), 25))

        if cycle != 2026:
            raise ValueError(
                "The cleaned dataset is prepared for the "
                "2025-2026 election cycle only."
            )

        candidate_id = _find_candidate(representative)

        candidates = _load_candidates()
        candidate = candidates[
            candidates["candidate_id"] == candidate_id
        ].iloc[0]

        recipient_committee_id = (
            candidate["principal_committee_id"]
        )

        transactions = _load_transactions()

        candidate_transactions = transactions[
            transactions["candidate_id"] == candidate_id
        ].copy()

        cycle_start = pd.Timestamp(cycle - 1, 1, 1)
        cycle_end = pd.Timestamp(cycle, 12, 31)

        candidate_transactions = candidate_transactions[
            candidate_transactions["date"].between(
                cycle_start,
                cycle_end,
            )
        ].copy()

        # Amount alone is not a reason to exclude an FEC transaction.
        # Keep standard and large-review records in provisional rankings;
        # joint fundraising and unrecognized classifications stay separate.
        included_statuses = {"standard_record", "large_transaction_review"}
        eligible = candidate_transactions[
            candidate_transactions["review_status"].isin(included_statuses)
        ].copy()
        excluded = candidate_transactions[
            ~candidate_transactions["review_status"].isin(included_statuses)
        ].copy()
        large_review_records = eligible[
            eligible["review_status"] == "large_transaction_review"
        ]

        group_columns = [
            "committee_id",
            "committee_name",
            "committee_type",
            "designation",
            "organization_type",
            "connected_organization",
        ]

        for col in group_columns:
            if col not in eligible.columns:
                eligible[col] = ""

        summary = (
            eligible.groupby(
                group_columns,
                dropna=False,
            )
            .agg(
                total_contributions=("amount", "sum"),
                transaction_count=("amount", "size"),
            )
            .reset_index()
        )

        summary = summary.sort_values(
            [
                "total_contributions",
                "committee_name",
            ],
            ascending=[False, True],
        )

        top_pacs = []

        for _, row in summary.head(limit).iterrows():
            top_pacs.append(
                {
                    "contributor_name": row["committee_name"],
                    "contributor_committee_id": row["committee_id"],
                    "net_contributions": round(
                        float(row["total_contributions"]), 2
                    ),
                    "transactions": int(
                        row["transaction_count"]
                    ),
                    "committee_type": row["committee_type"],
                    "designation": row["designation"],
                    "organization_type": row["organization_type"],
                    "connected_organization": (
                        row["connected_organization"]
                    ),
                }
            )

        review_counts = (
            candidate_transactions["review_status"]
            .value_counts()
            .to_dict()
        )

        return {
            "representative": representative,
            "candidate_name": candidate["candidate_name"],
            "candidate_id": candidate_id,
            "recipient_committee_id": recipient_committee_id,
            "cycle": cycle,
            "source": (
                "FEC committee-to-candidate bulk data "
                "(itpas2.txt), enriched with committee "
                "master (cm.txt) and candidate master (cn.txt)"
            ),
            "category": "Provisional PAC contribution rankings",
            "records_seen": int(len(candidate_transactions)),
            "records_in_ranking": int(len(eligible)),
            "records_excluded_from_ranking": int(len(excluded)),
            "large_transactions_included_for_review": int(len(large_review_records)),
            "review_status_counts": {
                str(key): int(value)
                for key, value in review_counts.items()
            },
            "total_net_ranked_contributions": round(
                float(eligible["amount"].sum()), 2
            ),
            "complete": False,
            "error": None,
            "top_pacs": top_pacs,
            "warning": (
                "This is a provisional ranking from a local "
                "FEC data snapshot, not an audited total. "
                "Large transactions previously flagged for review "
                "are included, not discarded based on amount. "
                "Joint fundraising and other excluded statuses "
                "remain outside the direct PAC ranking. "
                "Large records may need transaction-type and "
                "recipient verification. Amended filings "
                "and conduit transactions have not been fully "
                "reconciled. Records are associated with the "
                "candidate ID, not necessarily only the "
                "principal campaign committee. The candidate "
                "lookup includes historical candidates."
            ),
        }

    except Exception as exc:
        return {
            "representative": representative,
            "cycle": cycle,
            "complete": False,
            "error": str(exc),
            "top_pacs": [],
        }


async def fetch_pac_data(representative, limit=10, cycle=2026):
    """
    Async-compatible wrapper used by the agent.
    """

    return await asyncio.to_thread(
        _fetch_pac_data_sync,
        representative,
        limit,
        cycle,
    )


# --------------------------------------------------
# AGENT TOOL
# --------------------------------------------------

@function_tool
async def get_top_pac_contributors(
    ctx: RunContextWrapper,
    representative: str,
    limit: int = 10,
    cycle: int = 2026,
) -> str:
    """
    Find the top PAC contributors to a U.S. House candidate.

    Uses locally cleaned FEC committee-to-candidate data,
    rather than making live FEC API requests.

    Returns provisional net contribution totals, committee
    classifications, and data-quality warnings.
    """

    research_run = ctx.context

    async with research_run.pac_lock:
        # Cache results by query so different representatives
        # and limits do not accidentally reuse the same result.
        cache = getattr(research_run, "pac_results", None)

        if cache is None:
            cache = {}
            research_run.pac_results = cache

        cache_key = (
            str(representative).strip().casefold(),
            int(limit),
            int(cycle),
        )

        if cache_key in cache:
            return cache[cache_key]

        try:
            result = await fetch_pac_data(
                representative=representative,
                limit=limit,
                cycle=cycle,
            )

        except Exception as exc:
            result = {
                "representative": representative,
                "complete": False,
                "error": f"PAC retrieval failed: {exc}",
                "top_pacs": [],
            }

        serialized = json.dumps(result)

        cache[cache_key] = serialized

        # Preserve the original context attribute for
        # compatibility with existing agent code.
        research_run.pac_result = serialized

        return serialized