
import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from congress_tools import (
    get_representative_summary,
    _find_representative,
    get_top_donors,
    get_earmark_projects,
    get_donor_affiliations,
)
from pac_tools import get_top_pac_contributors
from industry_tools import analyze_industry_overlap
from context_panel import AgentContexts, PanelHooks, add_context_panel


HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")
set_tracing_disabled(True)

MODEL = LitellmModel(
    model="vertex_ai/gemini-3.5-flash",
    api_key="unused",
)

RESEARCHER_MODEL = LitellmModel(
    model="vertex_ai/gemini-3.5-flash-lite",
    api_key="unused",
)

SETTINGS = ModelSettings(
    extra_args={
        "vertex_location": "global",
        "timeout": 90,
        "num_retries": 2,
    }
)

MAX_RESEARCHERS = 6
MAX_RESEARCH_ROUNDS = 3


@dataclass
class Research:
    hooks: PanelHooks
    researchers: int = 0
    findings: list[str] = field(default_factory=list)
    assigned_tasks: set[tuple[str, str]] = field(default_factory=set)
    research_tasks: list[asyncio.Task] = field(default_factory=list)
    round_number: int = 1
    pac_result: str | None = None
    pac_results: dict = field(default_factory=dict)
    pac_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


RESEARCHER_INSTRUCTIONS = """
You are a congressional research analyst.

Complete ONLY your assigned task using the available tools.
Return factual findings, relevant figures, and limitations.

TASK TYPES:

INDIVIDUAL:
- Call get_top_donors.
- Report individual contributions, not organizational donations.
- Data covers January 1, 2025 through February 28, 2026.
- This is not the complete election cycle.
- Employer and occupation fields are self-reported.


PAC:
- Call get_top_pac_contributors exactly once.
- The tool uses locally cleaned FEC contribution data, not a live API.
- Report top PAC contributors and net contributions for the available cycle.
- Preserve the contributor_committee_id for EVERY PAC returned.
- Preserve the candidate_id and recipient_committee_id from the tool output.
- Preserve the exact source dataset names and election cycle.
- Include committee IDs alongside PAC names and contribution amounts.
- Preserve transaction counts, warnings, and errors.
- Clearly distinguish PAC contributions from individual donations.
- Do not describe provisional totals as audited totals.
- Do not invent missing committee IDs or transaction identifiers.

EARMARK:
- Call get_earmark_projects.
- These are FY2027 requests, not awarded funds.
- Report project names, recipients, and requested amounts.

AFFILIATION:
- Call get_donor_affiliations.
- Do not treat similar names as verified identities.
- Distinguish confirmed information from possible relationships.

INDUSTRY:
- Call analyze_industry_overlap exactly once.
- For broad donor-earmark connection questions, use industry="all".
- For a specific industry question, use the named industry.
- The tool uses local donation and earmark CSV files.
- It compares all available individual donor employers with earmark
  recipients, not just the top 10 donors.
- Report exact normalized organization-name matches separately
  from broad industry overlaps.
- Explain that matching names do not prove identity or influence.
- Do not invent organizational affiliations, contractors,
  beneficiaries, or financial relationships.
- If the tool returns an error, report it rather than inventing results.

Always describe the evidence and its limitations accurately.
"""


RESEARCH_TOOLS = {
    "individual": [
        get_representative_summary,
        get_top_donors,
    ],
    "pac": [
        get_representative_summary,
        get_top_pac_contributors,
    ],
    "earmark": [
        get_representative_summary,
        get_earmark_projects,
    ],
    "affiliation": [
        get_donor_affiliations,
    ],
    "industry": [
        analyze_industry_overlap,
    ],
}


@function_tool
async def research(
    ctx: RunContextWrapper[Research],
    representative: str,
    task_type: str,
    task: str,
) -> str:
    """Launch a specialist researcher to run concurrently with other researchers.

    Args:
        representative: Full name of the House representative.
        task_type: One of individual, pac, earmark, affiliation, industry.
        task: Specific research question and relevant industry if applicable.
    """
    run = ctx.context
    kind = task_type.strip().casefold()

    if kind not in RESEARCH_TOOLS:
        return (
            "Invalid task_type. Choose individual, pac, "
            "earmark, affiliation, or industry."
        )

    # Permit follow-up work in the same specialty when the question differs.
    # Prevent identical repeat assignments across all rounds.
    task_key = (kind, " ".join(task.casefold().split()))
    if task_key in run.assigned_tasks:
        return f"That {kind} research question was already assigned. Do not repeat it."

    if run.researchers >= MAX_RESEARCHERS:
        return "Researcher limit reached. Reply DONE."

    run.assigned_tasks.add(task_key)
    run.researchers += 1

    number = run.researchers

    agent = Agent(
        name=f"Researcher {number} ({kind})",
        instructions=RESEARCHER_INSTRUCTIONS,
        model=RESEARCHER_MODEL,
        model_settings=SETTINGS,
        tools=RESEARCH_TOOLS[kind],
    )

    prompt = (
        f"Task type: {kind}\n"
        f"Representative: {representative}\n"
        f"Task: {task}\n\n"
        "Complete only the assigned task.\n"
        "For an industry task, call analyze_industry_overlap exactly once.\n"
        "If the question does not specify an industry, use industry='all'.\n"
    )

    async def execute_research() -> str:
        try:
            result = await Runner.run(
                agent,
                prompt,
                context=run,
                hooks=run.hooks,
                max_turns=12,
            )
            finding = str(result.final_output)
        except Exception as exc:
            finding = (
                f"{kind} research failed: "
                f"{type(exc).__name__}: {exc}"
            )
        return f"Task type: {kind}\nTask: {task}\n\n{finding}"

    # Schedule the researcher without waiting for it here.
    # Other research() calls can launch additional researchers concurrently.
    run.research_tasks.append(asyncio.create_task(execute_research()))
    return f"Researcher {number} ({kind}) launched in round {run.round_number}. Continue delegating, then reply DONE."



LEAD_INSTRUCTIONS = """
You are the Lead agent for Congressional Deep Research.

Your job is to interpret the user's question, identify the representative,
and delegate only the research tasks necessary to answer it.

FIRST:
- Identify the representative using get_representative_summary.
- Use the full representative name in all research calls.

RESEARCH ROUTING:

1. DONOR-EARMARK CONNECTIONS:

If the user asks whether campaign donors, donor employers, industries,
organizations, or PACs are connected to earmark requests, assign an
INDUSTRY researcher.

Examples:
- "Are there any connections between AOC's donors and earmarks?"
- "Compare Pelosi's campaign contributions with her earmark requests."
- "Which industries donated to this representative and also
   appear in their earmark projects?"
- "Are any donor employers also earmark recipients?"
- "Investigate potential financial beneficiaries of earmarks."

For these questions:
- Assign research with task_type="industry".
- If no industry is specified, instruct the researcher to use "all".
- If an industry is specified, instruct the researcher to use that industry.
- Do not separately assign individual and earmark researchers just
  to perform the same comparison.

2. INDUSTRY-SPECIFIC QUESTIONS:

Examples:
- "Nancy Pelosi donations and earmarks in construction"
- "Compare healthcare donations and healthcare earmarks"
- "Are there technology-sector donor connections to earmarks?"

Assign exactly one INDUSTRY researcher.
Include the industry in the task.

3. LARGEST DONOR AND THEIR INDUSTRY:

For questions like:
- "Who is AOC's largest donor, what industry are they in,
   and are there earmarks in that industry?"

Assign:
- INDIVIDUAL to identify the actual largest individual donor.
- INDUSTRY with industry="all" to examine donor industries and earmarks.

Do not assume the largest donor has a known industry.
Do not substitute the largest industry-classifiable donor for the
largest overall donor without clearly explaining the distinction.

4. GENERAL DONOR QUESTIONS:

For questions like:
- "Who are the top 10 donors?"
- "Who contributes to this representative?"

Assign INDIVIDUAL and PAC research tasks.

5. OTHER QUESTIONS:

- Earmark-only question: assign EARMARK.
- Individual-only question: assign INDIVIDUAL.
- PAC-only question: assign PAC.
- Affiliation-only question: assign AFFILIATION.

IMPORTANT:
- Do not repeat the same research question. A follow-up may use the same
  task type only when it addresses a genuinely different unresolved question.
- Do not make unnecessary FEC API calls.
- Use the local industry tool for donor-earmark comparisons.
- Do not claim that donor-employer relationships establish influence.
- If the question explicitly requests additional distinct analyses,
  assign the necessary extra researchers.
- In the first round, launch all needed research tasks, then reply DONE.
- On subsequent rounds, read the findings provided to you. If they fully
  answer the question, reply DONE without calling research.
- If a material gap remains and a permitted tool can help, delegate ONLY
  focused follow-up research to resolve that gap; do not repeat earlier work.
- Do not invent new research needs merely to fill all rounds.
- If evidence is unavailable or cannot be established with the tools,
  acknowledge the limitation instead of repeating a failed search.
- Research tasks run concurrently; the system waits for them before writing.
"""


lead = Agent(
    name="Lead",
    instructions=LEAD_INSTRUCTIONS,
    model=MODEL,
    model_settings=SETTINGS,
    tools=[
        get_representative_summary,
        research,
    ],
)



WRITER_INSTRUCTIONS = """
You are the Writer agent for Congressional Deep Research.

Your job is to produce a clear, accurate, evidence-based Markdown research report that directly answers the user's question.

Use ONLY the research findings supplied by the Lead and Researcher agents. Do not invent figures, sources, affiliations, identities, or relationships.

REPORT REQUIREMENTS:

1. Answer the user's actual question.
   - Include only sections relevant to the requested research.
   - Do not include individual donor sections for PAC-only questions.
   - Do not include PAC or earmark sections unless they are relevant.
   - Avoid generic filler or repetitive explanations.

2. Start with a concise executive summary.
   - Identify the representative.
   - Summarize the most relevant findings.
   - Clearly distinguish provisional findings from independently verified facts.
   - Do not describe locally processed data as official, audited, or verified.

3. Present quantitative findings accurately.
   - Preserve exact amounts, dates, counts, and identifiers.
   - Use tables when comparing contributors or projects.
   - Never present a provisional local dataset total as an official FEC total.
   - Distinguish transaction counts from unique contributors.
   - Distinguish individual contributions from PAC contributions.
   - Do not combine contribution categories unless the tool provides a compatible combined total.
   - Do not calculate new totals, percentages, or rankings.
   - Do not imply that a ranked subset represents all campaign receipts.
   - Include all available entries up to the requested top-N limit.
   - Treat PACs with equal contribution totals as tied rather than assigning meaningful ranks.
   - State when a list is truncated or incomplete.

4. Interpret findings conservatively.
   - Describe industry representation without claiming dominance unless supported by aggregate industry-level evidence.
   - Do not assign industry classifications independently; use only classifications returned by tools.
   - Do not infer standardized donation behavior from similar transaction amounts.
   - Do not claim that campaign contributions caused or influenced earmark requests.
   - Clearly distinguish observed relationships, potential overlaps, and speculation.
   - Do not infer donor identity or affiliations solely from matching names.
   - Industry overlaps are heuristic unless independently verified.
   - Only describe the specific comparisons performed by the tools.
   - Do not introduce unsupported historical, biographical, legislative, or organizational context.

5. Handle donor identity and attribution carefully.
   - Treat individual donor groups as disclosure-field groups, not verified unique people.
   - Do not independently merge or consolidate donor groups based on names, addresses, employers, or occupations.
   - If the tool provides provisional consolidated groupings, preserve its uncertainty warnings.
   - If different findings provide conflicting donor totals, identify the discrepancy rather than silently choosing one.
   - Preserve donor names, employers, occupations, and locations as reported.
   - Employer and occupation information is self-reported and does not establish a verified affiliation.

6. Handle data limitations.
   - Clearly state the data source and coverage period when provided.
   - Identify provisional or unreconciled figures.
   - Explain unresolved representative or project attribution when relevant.
   - Do not silently assign ambiguous earmark projects.
   - If a tool returns status "partial", report attributed and unresolved projects separately when those figures are provided.
   - Do not claim unresolved projects are confirmed matches because their amounts reconcile with summary totals.
   - If a representative cannot be identified or a tool returns "not found", do not interpret missing data as zero.
   - Do not imply that FY2027 earmark requests were approved or awarded.
   - Preserve material warnings about amendments, conduits, unusually large contributions, and incomplete records.

7. Cite evidence accurately and consistently.
   - Every substantive factual claim, quantitative finding, and table must be supported by a citation.
   - Place citations immediately after the relevant sentence, figure, or table entry rather than only at the end of the report.
   - Use the following citation format:
     [Source: dataset name; research tool; record identifier if available]
   - For example:
     [Source: house_pac_transactions_2026.csv; get_top_pac_contributors; committee_id=C00123456]
   - The example above illustrates formatting only. Never reuse its values unless they actually appear in the supplied findings.
   - Use the exact dataset names, tool names, and identifiers provided by the Researchers.
   - Do not invent dataset names, record IDs, URLs, or verification steps.
   - If record-level identifiers are unavailable, cite the dataset and research tool when known.
   - For tables, include a Source/Record ID column when individual rows have distinct identifiers. Otherwise, cite the table immediately below it.
   - Do not cite the Lead or Researcher agents themselves as primary data sources. Cite the underlying tools and datasets.
   - Distinguish locally processed datasets from official external sources.
   - If external verification has not been performed, state that clearly.
   - If the findings do not identify a source, explicitly note that the source could not be established rather than inventing a citation.
   - End with a "Sources and Limitations" section identifying the datasets used, their coverage periods, and relevant limitations.
   - For PAC contribution tables, include a "FEC Committee ID" column using contributor_committee_id from the research findings.
   - Cite PAC tables using the source datasets (itpas2.txt, cm.txt, cn.txt), the candidate_id, and the contributor committee IDs.
   - Do not substitute the recipient_committee_id for a contributor_committee_id.
   - Preserve committee IDs exactly as returned by the PAC research tool.

8. Organize the report dynamically.
   - Use a descriptive title.
   - Include an executive summary.
   - Include relevant research sections with tables when useful.
   - Include a brief key findings or observations section only if it adds value.
   - End with a "Sources and Limitations" section.
   - Preserve the depth and organization of a professional congressional research report.
   - Do not use a fixed report template that creates irrelevant sections.
   - Do not shorten the report unnecessarily.
   - Do not include empty sections or tables.

9. Maintain a professional, neutral, analytical tone.
   - Avoid exaggerated claims such as "dominance" or "strong influence" without supporting evidence.
   - Do not overstate conclusions.
   - Do not fabricate missing information.
   - Distinguish documented facts from possible relationships.
   - If the evidence is insufficient, explicitly explain what cannot be established.

For multi-agent research, synthesize findings across researchers rather than simply concatenating their responses.

If the findings contain conflicting figures, ambiguous identities, unresolved project attribution, or incomplete research, explain those limitations instead of guessing.

If a tool returns an error or cannot find the requested representative, provide a concise explanation rather than generating a speculative report.
"""




writer = Agent(
    name="Writer",
    instructions=WRITER_INSTRUCTIONS,
    model=MODEL,
    model_settings=SETTINGS,
)


# Resolve the person named by the user before allowing any research agents to run.
# The model extracts a name; the local roster is the authority for validation.
NAME_EXTRACTOR = Agent(
    name="Representative name extractor",
    instructions=(
        "Extract the PERSON'S NAME mentioned in the user's question, even if "
        "you do not recognize them as a politician or member of Congress. "
        "Your task is text extraction, NOT determining whether they serve in Congress. "
        "For example, 'Who are Emily Nguyen's top donors?' -> Emily Nguyen; "
        "'Emily Nguyen' -> Emily Nguyen; 'Nancy Pelosi' -> Nancy Pelosi. "
        "Copy the name exactly as written. Never substitute a different person. "
        "Output only the name, or NONE if no person's name is present."
    ),
    model=MODEL,
    model_settings=SETTINGS,
)


def _name_fallback(question: str) -> str | None:
    """Conservative fallback for explicit first-and-last names when AI returns NONE."""
    # Supports plain names and common questions, without guessing a politician.
    text = question.strip()
    patterns = [
        r"^([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)$",
        r"\b(?:are|is|about|for|of|did|does)\s+([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _appears_in_question(name: str, question: str) -> bool:
    # Prevent the extractor from inventing or substituting a different name.
    normalize = lambda text: re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return f" {normalize(name)} " in f" {normalize(question)} "


async def validate_representative(question: str, representative: str = "") -> tuple[str | None, str | None]:
    # Validate the ORIGINAL user text before any AI extraction or optional
    # frontend representative field can expand a first name into a person.
    raw_question = question.strip().strip("\"'` ")
    if (re.fullmatch(r"[A-Za-z][A-Za-z'-]*", raw_question)
            and raw_question.casefold() != "aoc"):
        return None, (
            f"Could you provide the full name of the representative "
            f"you mean by '{raw_question}'?"
        )

    supplied = representative.strip()
    # The optional field must not silently override a different name typed
    # in the question. Only trust it if it appears verbatim in the question,
    # or if the question contains no named person (legacy form behavior).
    if supplied and raw_question and not _appears_in_question(supplied, raw_question):
        supplied = ""
    if supplied:
        requested = supplied
    elif re.search(r"\bAOC\b", question, re.IGNORECASE):
        requested = "Alexandria Ocasio-Cortez"
    else:
        result = await Runner.run(NAME_EXTRACTOR, question, max_turns=3)
        requested = str(result.final_output).strip().strip('"\'`')
        if not requested or requested.upper() == "NONE":
            requested = _name_fallback(question)
            if not requested:
                return None, "Which U.S. House representative would you like to research? Please provide their name."
        if not _appears_in_question(requested, question):
            return None, "I couldn't confidently identify the representative in your question. Please provide their full name."

    # Never infer an identity from a single first name or surname.
    # AOC has already been expanded to its full name above.
    # Bioguide IDs are unambiguous and may be supplied directly.
    if len(requested.split()) == 1 and not re.fullmatch(r"[A-Za-z]\d{6}", requested):
        return None, (
            f"Could you provide the full name of the representative "
            f"you mean by '{requested}'?"
        )

    try:
        row = _find_representative(requested)
    except ValueError as exc:
        message = str(exc)
        if "found 0" in message:
            return None, (f"I couldn't find a U.S. House representative named '{requested}' "
                          "in the available dataset. Please check the name or provide another representative.")
        return None, (f"I couldn't uniquely identify '{requested}' in the available dataset. "
                      "Please provide the representative's full name or Bioguide ID.")
    return str(row["full_name"]), None


async def deep_research(
    question: str,
    representative: str,
    hooks: PanelHooks,
) -> str:

    validated_name, validation_error = await validate_representative(question, representative)
    if validation_error:
        return validation_error
    representative = validated_name

    research_run = Research(hooks=hooks)

    prompt = (
        f"Question: {question}\n"
        f"Starting representative: {representative}\n"
    )

    try:
        for round_number in range(1, MAX_RESEARCH_ROUNDS + 1):
            research_run.round_number = round_number
            previous_count = len(research_run.research_tasks)

            if round_number == 1:
                lead_prompt = prompt + (
                    "\nRound 1: Delegate the initial research needed to answer the question. "
                    "Do not write the final report.\n"
                )
            else:
                previous_findings = "\n\n---\n\n".join(research_run.findings)
                lead_prompt = (
                    f"Original question: {question}\n"
                    f"Representative: {representative}\n"
                    f"Research round: {round_number} of {MAX_RESEARCH_ROUNDS}\n"
                    f"Completed findings from earlier rounds:\n{previous_findings}\n\n"
                    "Review whether the original question has material unanswered "
                    "parts that the available tools can investigate. If not, reply DONE "
                    "without calling research. Otherwise delegate only specific, "
                    "nonduplicative follow-up tasks, then reply DONE.\n"
                )

            lead_result = await Runner.run(
                lead,
                lead_prompt,
                context=research_run,
                hooks=hooks,
                max_turns=12,
            )

            newly_launched = research_run.research_tasks[previous_count:]
            if not newly_launched:
                if round_number == 1:
                    return str(lead_result.final_output)
                break

            # Each round fans out in parallel. The Lead reviews completed
            # findings before deciding whether another round is warranted.
            round_findings = await asyncio.gather(*newly_launched)
            research_run.findings.extend(round_findings)

    finally:
        # Cancel outstanding researchers on errors or request cancellation.
        for task in research_run.research_tasks:
            if not task.done():
                task.cancel()
        if research_run.research_tasks:
            await asyncio.gather(*research_run.research_tasks, return_exceptions=True)

    findings = "\n\n---\n\n".join(research_run.findings)

    writer_prompt = (
        f"Question: {question}\n\n"
        f"Research findings:\n\n{findings}"
    )

    report = await Runner.run(
        writer,
        writer_prompt,
        hooks=hooks,
    )

    return str(report.final_output)


sessions: dict[str, AgentContexts] = {}

app = FastAPI()

add_context_panel(app, sessions)


@app.get("/")
def index():
    return FileResponse(
        HERE / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/run")
async def run(
    question: str,
    url: str = "",
    representative: str = "",
):
    run_id = str(uuid.uuid4())

    sessions[run_id] = AgentContexts()

    queue: asyncio.Queue = asyncio.Queue()

    def emit(event):
        if event.get("type") == "context":
            event = {
                **event,
                "events": list(sessions[run_id].events),
            }

        queue.put_nowait(event)

    hooks = PanelHooks(
        sessions[run_id],
        emit,
    )

    async def work():
        try:
            report = await deep_research(
                question,
                representative or url,
                hooks,
            )

            queue.put_nowait({
                "type": "done",
                "markdown": report,
            })

        except Exception as exc:
            queue.put_nowait({
                "type": "error",
                "message": (
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                ),
            })

    task = asyncio.create_task(work())

    async def events():
        try:
            yield (
                "data: "
                + json.dumps({
                    "type": "start",
                    "run_id": run_id,
                })
                + "\n\n"
            )

            while True:
                event = await queue.get()

                yield (
                    "data: "
                    + json.dumps(event, default=str)
                    + "\n\n"
                )

                if event["type"] in ("done", "error"):
                    break

        finally:
            task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )