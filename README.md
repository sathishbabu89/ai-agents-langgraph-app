# Banking Assistant: MCP, Langfuse Prompt Management & Evaluation

A learning/demo project demonstrating a Streamlit banking assistant that
uses **Model Context Protocol (MCP)** tools, retrieves versioned prompts
from **Langfuse**, traces execution, and evaluates prompt versions with
a Langfuse dataset and Gemini as an LLM-as-a-Judge.

> **Educational demo only.** It uses hard-coded sample account data and
> is not connected to a real banking system. Do not use it for real
> financial transactions or customer data.

## What this project demonstrates

-   **MCP server:** Exposes banking capabilities over Streamable HTTP.
-   **MCP client / host:** Discovers tools and invokes them in response
    to model tool calls.
-   **Tool calling:** The Groq model selects tools; the application
    executes them and returns results to the model.
-   **Langfuse observability:** Traces LLM and MCP activity.
-   **Prompt management:** Retrieves the Langfuse prompt
    `banking-assistant` by version so V1 and V2 can be compared.
-   **Evaluation:** Runs a Langfuse dataset against prompt versions and
    uses Gemini as an LLM-as-a-Judge.
-   **LangSmith:** Can be configured for tracing if enabled in the
    application.

## Architecture

``` text
User
 |
 v
Streamlit application (app.py)
 |
 +--> Langfuse: retrieve banking-assistant prompt version
 |
 +--> Groq chat model: select a tool or formulate an answer
 |       |
 |       +--> MCP client call
 |                |
 |                v
 |          MCP server (server.py)
 |             +-- get_account_balance
 |             +-- get_recent_transactions
 |
 +--> Langfuse tracing and prompt/version metadata

Evaluation scripts
 |
 +--> Langfuse dataset: banking-assistant-evaluation
 +--> Prompt V1 / Prompt V2
 +--> Groq + MCP application task
 +--> Gemini judge
 +--> Langfuse experiment runs and scores
```

## Project files

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `app.py`                            Main Streamlit banking assistant.
                                      Connects the LLM to MCP tools and
                                      integrates Langfuse prompt
                                      management/tracing.

  `app_v1.py`                         Earlier/alternate application
                                      version retained for comparison or
                                      reference.

  `server.py`                         MCP server exposing
                                      `get_account_balance` and
                                      `get_recent_transactions` over
                                      Streamable HTTP.

  `client.py`                         Standalone MCP client demo for
                                      discovering and calling tools
                                      directly.

  `experiment_v1.py`                  Runs the evaluation dataset with
                                      Prompt V1.

  `experiment_v2.py`                  Runs the evaluation dataset with
                                      Prompt V2.

  `evaluate_experiments.py`           Runs prompt-version evaluations and
                                      uses Gemini as an LLM-as-a-Judge;
                                      creates new scored experiment runs.

  `requirements.txt`                  Python dependencies.

  `.env.example`                      Template for configuration. It must
                                      contain placeholders, never real
                                      credentials.
  -----------------------------------------------------------------------

The PDF/RAG policy-search tool is intentionally outside this MCP version
of the demo.

## Prerequisites

-   Python version supported by the dependencies (Python 3.10+
    recommended)
-   Git
-   Groq API key
-   Langfuse project and API keys
-   Gemini API key to run `evaluate_experiments.py`

## Setup on Windows PowerShell

### 1. Clone the GitHub repository

After creating the repository:

``` powershell
git clone https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPOSITORY-NAME>.git
cd <YOUR-REPOSITORY-NAME>
```

If you are setting up from your existing local project folder, skip
cloning and continue below.

### 2. Create and activate a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the template and edit the local `.env` file:

``` powershell
Copy-Item .env.example .env
```

Example variables (use the exact variable names your code expects):

``` dotenv
GROQ_API_KEY=replace-with-your-groq-key
GROQ_MODEL=openai/gpt-oss-20b
MCP_SERVER_URL=http://127.0.0.1:8000/mcp

LANGFUSE_SECRET_KEY=replace-with-your-langfuse-secret-key
LANGFUSE_PUBLIC_KEY=replace-with-your-langfuse-public-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development

# Required only for evaluate_experiments.py
GEMINI_API_KEY=replace-with-your-gemini-api-key
GEMINI_MODEL=replace-with-a-model-available-to-your-account
```

If the script uses `GOOGLE_API_KEY` instead of `GEMINI_API_KEY`,
configure the variable it expects. Use a Gemini model identifier
available to your API key.

If LangSmith tracing is enabled in your code, its environment variables
may include:

``` dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=replace-with-your-langsmith-key
LANGSMITH_PROJECT=bank-ai-tool-calling
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

**Never commit `.env`, API keys, access tokens, or other secrets.**
`.env.example` must contain placeholders only.

## Run the demo

Use two PowerShell terminals in the project directory.

### Terminal 1: Start the MCP server

``` powershell
python .\server.py
```

The default endpoint is `http://127.0.0.1:8000/mcp`. Keep the server
running.

### Terminal 2: Run the standalone MCP client (optional)

``` powershell
python .\client.py
```

This demonstrates tool discovery and direct tool invocation without an
LLM.

### Terminal 2: Start the Streamlit application

``` powershell
streamlit run .\app.py
```

Open the local URL printed by Streamlit. The MCP server must be running
for MCP-backed operations to work.

## Prompt management in Langfuse

The prompt used in the demo is named `banking-assistant`. The project
has used V1 and V2 to demonstrate version retrieval and comparison.

Suggested lifecycle:

1.  Keep the current production-labelled prompt stable.
2.  Create a new prompt version.
3.  Test the new version against the same dataset.
4.  Inspect outputs, traces, and scores.
5.  Promote only after reviewing the results.

Do not promote a prompt solely because of an aggregate score; inspect
financial and refusal cases individually.

## Run experiments and evaluation

``` powershell
python .\experiment_v1.py
python .\experiment_v2.py
python .\evaluate_experiments.py
```

The exact prompt version and dataset behavior depend on each script's
configuration. `evaluate_experiments.py` creates new experiment runs; it
does not retroactively modify earlier runs. It requires the MCP server,
Groq credentials, Langfuse configuration, and a Gemini API key.

The Langfuse dataset used in the demo is named
`banking-assistant-evaluation`. It includes account-balance questions,
recent transactions, Amazon spending, a definition question, and a
money-transfer request. The transfer request should be refused because
this demo has no transfer capability.

LLM-as-a-Judge scores are useful signals, not absolute truth. Verify
amounts, debit/credit signs, transaction dates, and refusal behavior
against the actual tool results. Provider errors such as HTTP 503 can
leave items unscored; retry failed cases before comparing prompt
versions.

## Sample data and safety

The project uses fixed demo data in code. Account identifiers and
balances are examples only. Tool names and model tool calls do not
authorize real banking actions.

Before adapting this for a real environment:

-   Replace hard-coded data with approved service integrations.
-   Add authentication and authorization.
-   Validate tool arguments on the server.
-   Avoid exposing sensitive information in traces and logs.
-   Add appropriate guardrails, tests, and error handling.
-   Follow your organization's security and data-handling requirements.

## Troubleshooting

-   **MCP connection refused:** Confirm `server.py` is running and
    `MCP_SERVER_URL` points to the correct endpoint.
-   **Missing API key:** Check the local `.env` and exact variable
    names; restart the process after changing variables.
-   **Gemini 503 / 429:** The provider may be overloaded or
    rate-limiting requests. Retry later or configure a model available
    to your key.
-   **Prompt not linked to a trace:** Confirm the prompt is retrieved
    through Langfuse and metadata is attached as implemented in
    `app.py`.
-   **No experiment results:** Check the dataset name, service
    availability, and script output for failed items.

## Git hygiene

Keep these local-only items out of Git:

-   `.env`
-   `.venv/` and `venv/`
-   Python cache files
-   local vector databases, generated data, logs, and machine-specific
    editor settings

Review `git status` and inspect staged files before each push.

## License

Add a license if you intend to share or reuse this project publicly.
Until then, all rights remain with the repository owner.
