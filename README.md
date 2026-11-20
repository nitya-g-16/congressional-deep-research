
# Congressional Deep Research

## Project Overview

Congressional Deep Research is a multi-agent AI research assistant that investigates U.S. congressional campaign contributions, political action committee (PAC) support, earmark requests, and potential connections between political donors and organizations receiving federal funding.

Instead of relying solely on a language model's existing knowledge, the system retrieves information from Federal Election Commission (FEC) records, congressional earmark datasets, and Wikidata. Multiple research agents gather evidence, which is then synthesized into a structured research report.

### Target Users

The project is designed for:
Those investigating campaign finance and government spending, political researchers and policy analysts, students studying congressional funding and political contributions, members of the public interested in government transparency.

The goal is to make complex political-finance datasets more accessible and allow users to investigate relationships between congressional representatives, campaign donors, and federal funding recipients.

## System Architecture

The application follows a multi-agent architecture:

1. **Lead Agent:** Interprets the user's question and divides it into research tasks.
2. **Research Agents:** Work in parallel, calling specialized tools to retrieve campaign-finance records, earmark information, and organizational affiliations.
3. **Writer Agent:** Combines the findings into a coherent research report.

### Research Tools

| Tool | Purpose |
|---|---|
| `get_representative_summary` | Retrieves information about a congressional representative |
| `get_top_pac_contributors` | Identifies PAC contributions to a representative |
| `get_earmark_projects` | Retrieves congressional earmark projects |
| `get_donor_affiliations` | Investigates organizational affiliations of potential donors |
| `analyze_industry_overlap` | Examines overlap between contribution industries and earmark-related sectors |

## Data Sources and Loading

### 1. Federal Election Commission (FEC)

**Source:** https://www.fec.gov/data/browse-data/?tab=bulk-data

The project uses FEC bulk campaign-finance data to identify candidates, political committees, and reported PAC contributions.

Relevant files:

| File | Contents |
|---|---|
| `cn.txt` | Candidate information and identifiers |
| `cm.txt` | Political committee information and identifiers |
| `itpas2.txt` | Itemized PAC and party contributions |

**Loading process:**

1. Download the appropriate election-cycle bulk datasets from the FEC website.
2. Extract the compressed files.
3. Load the pipe-delimited text files into Python using pandas or into a relational database.
4. Match candidate and committee identifiers.
5. Filter and aggregate contributions for the requested election cycle.

For example, Nancy Pelosi's FEC candidate identifier is `H8CA05035`.

### 2. Congressional Earmark Data

The project uses a locally prepared dataset of FY2027 congressional earmark and Community Project Funding records.

The records include information such as:

- Congressional representative
- Project name
- Recipient organization
- Requested or reported funding amount
- Appropriations account
- Project identifier

**Loading process:**

The locally prepared earmark dataset is loaded by the backend and queried through `get_earmark_projects`.

To recreate the data, congressional request disclosures and appropriations tables can be collected and standardized into a structured dataset.

Funding requests, committee-listed amounts, and enacted appropriations are different stages and should not be treated as interchangeable.

### 3. Wikidata

**Source:** https://www.wikidata.org/

**SPARQL endpoint:** https://query.wikidata.org/

Wikidata provides entity information for identifying organizations and investigating possible affiliations.

The research tools use entity matching to investigate whether individuals or organizations appearing in earmark records have identifiable relationships to campaign-finance records.

A matching organization name or affiliation does not independently establish that a contribution occurred.

---

## Sample Queries

### Query 1: Top PAC Contributors

**Sample query:**

> What are the top PAC contributors to Nancy Pelosi during the 2026 election cycle?

#### Research Agent Answer

The research agent identifies Nancy Pelosi using FEC candidate ID `H8CA05035` and retrieves PAC contribution records for the 2026 election cycle.

The reported leading contributors include:

| PAC | Committee ID | Reported Contributions |
|---|---|---:|
| American Crystal Sugar PAC | C00110338 | $5,000 |
| AFSCME PEOPLE | C00011114 | $5,000 |
| Machinists Non-Partisan Political League | C00002469 | $5,000 |
| American Sugar Cane League PAC | C00081414 | $3,000 |
| AFT PAC | C00028860 | $2,500 |

These amounts reflect the tool's contribution aggregation and should be interpreted according to its reporting cutoff and filters.

**Tool used:** `get_top_pac_contributors`

**Underlying data:**
- FEC `itpas2.txt`: PAC contribution transactions
- FEC `cm.txt`: Committee information
- FEC `cn.txt`: Candidate information

#### Baseline Model Answer

The baseline responded:

> As of early 2024, data for the 2026 election cycle is not yet available. In U.S. federal politics, the 2026 cycle officially begins after the conclusion of the November 2024 general election.

It instead discussed historical contributors from the 2024 election cycle, including:

- AFSCME
- International Brotherhood of Electrical Workers
- Teamsters Union
- United Auto Workers
- Akin Gump Strauss Hauer & Feld
- Skadden, Arps et al.
- Alphabet, Adobe, and Salesforce

---

### Query 2: FY2027 Congressional Earmark Funding

**Sample query:**

> What FY2027 earmark funding is associated with Nancy Pelosi's Pier 15 Modernization and Upgrades project?

#### Research Agent Answer

The agent identifies the following project:

| Field | Value |
|---|---|
| Project | Pier 15 Modernization and Upgrades |
| Representative | Nancy Pelosi |
| Recipient | Port of San Francisco |
| Fiscal Year | FY2027 |
| Project ID | 4792 |
| Appropriations Account | HUD Economic Development Initiatives (EDI) |
| Locally Recorded Amount | $1,850,000 |

The project is associated with modernization work at Pier 15 in San Francisco.

An original request of approximately $3 million was also identified during earlier research. The original requested amount and the $1.85 million recorded amount represent different stages of the funding process.

The $1.85 million figure should not automatically be described as enacted funding.

**Tool used:** `get_earmark_projects`

**Underlying data:** Locally loaded FY2027 congressional earmark dataset, project ID `4792`.

#### Baseline Model Answer

The baseline responded:

> As of early 2024, there is no earmark funding associated with FY2027. This is because the congressional budget process for Fiscal Year 2027 will not begin until early 2026.

It instead discussed older Pier 15 projects:

**FY2025:**
- Project: Exploratorium Pier 15 Decarbonization and Energy Efficiency
- Requested amount: $1,142,500
- Purpose: Water heating and building management system improvements

**FY2024:**
- Project: Exploratorium for Pier 15 Modernization and Upgrades
- Reported allocation: $2,000,000
- Purpose: HVAC and Bay Water Heating/Cooling system improvements
- Account: HUD Economic Development Initiatives

The baseline concluded that FY2027 requests had not yet been made.

---

### Query 3: Donor Connections and Earmark Recipients

**Sample query:**

> Are there any verified campaign contribution connections between Nancy Pelosi's FY2027 Pier 15 earmark request and the Port of San Francisco or Exploratorium?

#### Research Agent Answer

The research agent investigated potential relationships involving:

- Nancy Pelosi
- Port of San Francisco
- Exploratorium
- The FY2027 Pier 15 earmark project

The agent identified the relevant organizations but did not establish a verified donor-affiliation-to-contribution connection through the searches performed.

This does not establish that no affiliated individual has ever contributed to Pelosi.

To verify such a connection, the research would need:

1. A named individual associated with the recipient organization
2. Evidence confirming that individual's affiliation
3. A matching FEC contribution record
4. Sufficient identity verification to avoid confusing individuals with similar names

The agent therefore avoids asserting a financial relationship without adequate supporting records.

**Tools used:**
- `get_donor_affiliations`
- `get_earmark_projects`
- `get_top_pac_contributors` where applicable

**Underlying data:**
- Wikidata entity and affiliation records
- FY2027 earmark dataset, project ID `4792`
- FEC campaign-finance records

#### Baseline Model Answer

The baseline began by stating that the FY2027 earmark cycle did not yet exist.

It instead discussed an FY2025 Pier 15 project and claimed:

**Port of San Francisco:**
- The Port is a government agency and cannot directly contribute to political campaigns.

**Exploratorium:**
- The Exploratorium is a 501(c)(3) nonprofit and cannot directly participate in political campaigns.

**Individual connections:**
- The baseline asserted that Exploratorium-associated individuals were longtime donors to Nancy Pelosi.
- It specifically named George Cogan and Maryellen Cogan as significant donors.
- It also made broader claims about contributions from members of the organization's board and leadership.

The baseline concluded:

> Direct Connections: None from the Port or the Exploratorium as institutions. Indirect Connections: Significant. Multiple members of the Exploratorium's leadership and board are verified, long-term donors to Nancy Pelosi's campaigns.

The baseline did not provide matching FEC transaction records or citations establishing these individual connections.

---

## Running the Application

The research agent is implemented in Python and FastAPI and deployed using Google Cloud Run.

To run it locally:

1. Clone the repository.
2. Install the dependencies specified in the project configuration.
3. Configure the required model credentials and environment variables.
4. Ensure the local FEC and earmark datasets are available at the paths expected by the backend.
5. Start the FastAPI application using the entry point defined in the repository.

**GitHub repository:**

https://github.com/nitya-g-16/congressional-deep-research

## Limitations

- Campaign contributions do not independently establish improper influence or a quid pro quo.
- Organizational affiliations require verification and can be incorrectly matched.
- FEC contribution totals depend on reporting dates, committee boundaries, amendments, and transaction filters.
- Earmark requests, committee recommendations, and enacted appropriations are different funding stages.
- Failure to identify a contribution relationship does not prove that no relationship exists.
- Research findings may change as datasets are updated.
- The agent's conclusions should be checked against the underlying records.

## Conclusion

Congressional Deep Research demonstrates how multi-agent orchestration and external data retrieval can improve the reliability of AI-generated political-finance research.

The three example queries illustrate the value of grounding language-model responses in structured evidence. Compared with the standalone baseline, the research agent can retrieve specific PAC contributions, identify FY2027 earmark records, and avoid making unsupported claims about political donor relationships.

The project aims to make complex congressional finance and spending information easier to investigate while maintaining appropriate caution about the conclusions that can be drawn from the data.
