# Banking Assistant: MCP, Langfuse Observability, Prompt Management & Evaluation

A hands-on Python project demonstrating how to build and inspect a
tool-enabled banking assistant using **Streamlit**, **Groq**, **Model
Context Protocol (MCP)**, and **Langfuse**. The project also
demonstrates prompt versioning, dataset-driven experiments, and
LLM-as-a-Judge evaluation using Gemini.

> **Educational demo only.** The application uses hard-coded sample
> banking data. It is not connected to a real bank and must not be used
> for real financial transactions, production customer data, or
> financial advice.

------------------------------------------------------------------------

## Table of contents

-   [Project goals](#project-goals)
-   [Capabilities](#capabilities)
-   [Architecture](#architecture)
-   [Repository structure](#repository-structure)
-   [Technology stack](#technology-stack)
-   [How MCP and tool calling work
    together](#how-mcp-and-tool-calling-work-together)
-   [Langfuse observability](#langfuse-observability)
-   [Langfuse prompt management](#langfuse-prompt-management)
-   [Datasets and experiments](#datasets-and-experiments)
-   [LLM-as-a-Judge evaluation](#llm-as-a-judge-evaluation)
-   [Setup](#setup)
-   [Configuration](#configuration)
-   [Run the application](#run-the-application)
-   [Run experiments](#run-experiments)
-   [Interpreting evaluation results](#interpreting-evaluation-results)
-   [Security and production
    considerations](#security-and-production-considerations)
-   [Troubleshooting](#troubleshooting)
-   [Suggested demo walkthrough](#suggested-demo-walkthrough)

## Project goals

This repository brings several important LLM application engineering
practices together in one small, understandable project:

1.  Expose application capabilities as MCP tools rather than embedding
    every capability in the LLM prompt.
2.  Let the model decide when a tool is appropriate while keeping tool
    execution under application control.
3.  Manage system prompts outside the application code using Langfuse
    Prompt Management.
4.  Trace model calls and tool execution to understand application
    behavior.
5.  Compare prompt versions against a reusable dataset instead of
    relying only on manual testing.
6.  Use an LLM judge as one evaluation signal, while validating
    important financial facts against expected results and tool outputs.

## Capabilities

### Banking assistant

The Streamlit application supports a simple conversational banking
experience with two MCP-backed tools:

-   `get_account_balance` --- returns demo account balance information.
-   `get_recent_transactions` --- returns recent demo transactions.

The model can request a tool call, the application executes the
requested MCP tool, and the tool result is sent back to the model so it
can produce a natural-language response.

The demo intentionally does **not** provide a money-transfer tool. A
request to transfer money should therefore be declined rather than
presented as completed.

### MCP

The project demonstrates:

-   An MCP server using Streamable HTTP.
-   Tool discovery from an MCP client.
-   Tool invocation through the MCP protocol.
-   The distinction between the model requesting a tool and the
    application executing it.
-   A standalone client for exploring MCP independently of the
    LLM-powered UI.

### Langfuse

The project demonstrates three complementary Langfuse capabilities:

-   **Observability:** inspect traces, generations, tool execution, and
    prompt metadata.
-   **Prompt Management:** retrieve a centrally managed prompt and
    compare versions.
-   **Evaluation:** run a dataset against prompt versions and attach
    scores and explanations to experiment items.

### Evaluation

The evaluation scripts use a Langfuse dataset containing representative
banking questions. The application response is judged using Gemini, and
the results are recorded in Langfuse experiments for comparison.

------------------------------------------------------------------------

## Architecture

``` text
                              +----------------------+
                              |        User          |
                              +----------+-----------+
                                         |
                                         v
                              +----------------------+
                              | Streamlit: app.py    |
                              | Banking assistant    |
                              +----------+-----------+
                                         |
                      +------------------+------------------+
                      |                                     |
                      v                                     v
           +----------------------+             +----------------------+
           | Langfuse Prompt      |             | Groq chat model      |
           | Management           |             | Tool calling         |
           +----------------------+             +----------+-----------+
                                                           |
                                                  Model requests a tool
                                                           |
                                                           v
                                                +----------------------+
                                                | MCP client in app    |
                                                +----------+-----------+
                                                           |
                                                           v
                                                +----------------------+
                                                | MCP server           |
                                                | server.py            |
                                                +----------+-----------+
                                                           |
                                           +---------------+---------------+
                                           |                               |
                                           v                               v
                                  get_account_balance          get_recent_transactions
                                                           |
                                                           v
                                                Tool result returned
                                                to the model for answer

Langfuse observability records supported LLM / application spans and
prompt-version metadata for inspection.

Evaluation workflow:
Langfuse dataset -> Prompt V1 / Prompt V2 -> Groq + MCP task
                 -> Gemini judge -> scores and experiment comparison
```

**Important distinction:** tool calling and MCP solve related but
different problems. Tool calling lets the model express which capability
it wants to use. MCP provides a standard way for the application to
discover and call tools. Langfuse then helps inspect what happened,
which prompt was used, and how the result scored.

## Repository structure

``` text
mcp_banking_demo/
├── app.py
├── app_v1.py
├── client.py
├── server.py
├── experiment_v1.py
├── experiment_v2.py
├── evaluate_experiments.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `app.py`                            Main Streamlit host application.
                                      Uses Groq tool calling,
                                      communicates with the MCP server,
                                      and retrieves the managed Langfuse
                                      prompt.

  `app_v1.py`                         Earlier/alternate application
                                      version retained for reference or
                                      comparison.

  `server.py`                         MCP server exposing the demo
                                      banking tools over Streamable HTTP.

  `client.py`                         Standalone MCP client used to
                                      discover tools and call them
                                      without an LLM.

  `experiment_v1.py`                  Runs the Langfuse evaluation
                                      dataset using Prompt V1.

  `experiment_v2.py`                  Runs the Langfuse evaluation
                                      dataset using Prompt V2.

  `evaluate_experiments.py`           Runs prompt-version experiments and
                                      evaluates outputs with Gemini as an
                                      LLM-as-a-Judge. Creates new scored
                                      experiment runs.

  `requirements.txt`                  Python dependencies for the
                                      application, MCP, Langfuse, and
                                      Gemini evaluation.

  `.env.example`                      Safe configuration template
                                      containing placeholders only.

  `.gitignore`                        Excludes secrets, virtual
                                      environments, generated files, and
                                      local-only artifacts.
  -----------------------------------------------------------------------

The earlier PDF / RAG policy-search demonstration is intentionally not
part of the MCP-backed application flow described here.

------------------------------------------------------------------------

## Technology stack

-   **Python** --- application and experiment scripts.
-   **Streamlit** --- interactive UI.
-   **LangChain Core** --- chat messages, prompt templates, and
    model/tool abstractions.
-   **LangChain Groq** --- connection to the Groq-hosted chat model.
-   **MCP Python SDK** --- server/client communication and tool
    discovery.
-   **Langfuse** --- tracing, prompt management, datasets, experiments,
    and scores.
-   **Google Gen AI SDK** --- Gemini-based LLM-as-a-Judge in
    `evaluate_experiments.py`.
-   **python-dotenv** --- local environment configuration.
-   **LangSmith (optional)** --- tracing can also be enabled if
    configured in the application.

Exact compatibility depends on the installed package versions and
provider APIs. After validating the environment, consider pinning
versions for repeatable setup.

------------------------------------------------------------------------

## How MCP and tool calling work together

The flow for a request such as "What is my current account balance?" is:

1.  The application obtains the configured prompt from Langfuse.
2.  The user question and prompt are sent to the Groq chat model, with
    available tool schemas.
3.  The model may return a tool-call request for `get_account_balance`.
4.  The application validates/sets the demo account identifier and calls
    the MCP server.
5.  The MCP server executes the tool and returns the result.
6.  The application adds the tool result to the conversation and asks
    the model to formulate the final response.
7.  The response is displayed in Streamlit, while configured tracing
    allows the execution to be inspected.

The model does not directly execute Python functions or contact the MCP
server. The host application is responsible for executing the requested
tool and returning its result.

The standalone `client.py` demonstrates MCP discovery and invocation
separately from the LLM, helping make the protocol's role easier to
understand.

## Langfuse observability

Observability helps answer questions such as:

-   Which model call ran, and what did it return?
-   Which prompt version was associated with the generation?
-   Which tool was requested and what result did the application
    receive?
-   Where did latency or an error occur?
-   What happened during a particular user request or evaluation item?

### Configuration

The application uses Langfuse credentials from environment variables.
The typical configuration in this project is:

``` dotenv
LANGFUSE_SECRET_KEY=replace-with-your-secret-key
LANGFUSE_PUBLIC_KEY=replace-with-your-public-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

The Langfuse SDK uses these settings to connect to the configured
Langfuse project. Use the base URL appropriate for your Langfuse
deployment.

### Tracing the LLM and MCP work

The application integrates Langfuse with the LangChain model call using
the Langfuse LangChain callback handler. The MCP tool invocation is
explicitly instrumented so that the tool operation can be observed as
part of the workflow.

Conceptually, a request may contain:

``` text
Banking assistant request
├── LLM call: decide whether to call a tool
├── MCP tool execution: get_account_balance
└── LLM call: formulate the final response
```

The exact span names and nesting depend on how the installed SDK and
application instrumentation represent the calls. The important practice
is to correlate the model request, tool execution, and final answer
rather than inspecting only the final text.

### What to inspect in Langfuse

1.  Open the project's Traces view.
2.  Open a trace created by a test question.
3.  Inspect the model generation and any instrumented MCP operation.
4.  Check the prompt name/version associated with the generation.
5.  Review latency, inputs, outputs, and errors where available.

Avoid sending real customer data, credentials, account details, or other
sensitive information to traces. Apply appropriate redaction and access
controls before using observability in a real environment.

## Langfuse prompt management

The managed chat prompt used in this demo is named:

``` text
banking-assistant
```

The project uses versions V1 and V2 to demonstrate how prompt text can
evolve independently from the application code.

### Why manage prompts outside the code?

Centralized prompt management can help teams:

-   Version system instructions and review changes.
-   Keep application code separate from prompt wording.
-   Test candidate prompts before promotion.
-   Associate a generation with the prompt version that produced it.
-   Roll back or select a previously approved version through the
    configured label/version strategy.

### Prompt versions in this demo

**V1** establishes the basic banking assistant behavior: use the
appropriate banking tools, do not invent account information, and
respond naturally after receiving tool results.

**V2** strengthens the customer-facing instructions: answer directly
using tool results, present financial amounts clearly, keep the answer
concise, and avoid exposing internal tool or MCP implementation details.

These are intended to be compared through the same evaluation dataset,
not judged only by reading the prompt text.

### Version selection and lifecycle

During testing, the application can retrieve a specific prompt version,
such as V1 or V2. A deployment can instead retrieve a prompt using a
label such as `production`, provided that the application is configured
for label-based retrieval.

A recommended lifecycle is:

1.  Keep the approved production label on the current stable prompt.
2.  Create a new prompt version.
3.  Run the new version against a known dataset.
4.  Inspect the outputs, traces, scores, and failure cases.
5.  Promote only after reviewing the evidence.
6.  Keep a rollback path to the prior version.

**Do not move the production label simply because a candidate version
has a higher score on a tiny dataset.** First verify critical cases and
make sure the score is complete and trustworthy.

## Datasets and experiments

A dataset is a reusable collection of evaluation examples. Each item
contains an input and, where appropriate, an expected output. Reusing
the same dataset makes prompt comparisons more consistent.

### Dataset used

The Langfuse dataset in this demo is named:

``` text
banking-assistant-evaluation
```

It contains five representative questions:

  -----------------------------------------------------------------------
  Test scenario           Example question        What to verify
  ----------------------- ----------------------- -----------------------
  Definition              "What is an account     The explanation is
                          balance?"               accurate and
                                                  understandable.

  Unsupported operation   "Can you transfer       The assistant refuses
                          £5,000 to another       and does not claim a
                          account?"               transfer occurred.

  Merchant spending       "How much did I spend   The amount matches the
                          at Amazon?"             transaction data.

  Recent transactions     "Show me my recent      Transactions, amounts,
                          transactions."          debit/credit signs, and
                                                  any displayed dates are
                                                  supported by tool
                                                  results.

  Current balance         "What is my current     The balance matches the
                          account balance?"       tool result exactly.
  -----------------------------------------------------------------------

Expected outputs should describe the required facts and safety behavior.
They should not accidentally force a single wording or Markdown format
if several equivalent responses are acceptable.

### What is an experiment?

An experiment runs a task over the dataset and records the outputs.
Evaluators can then score those outputs. In this project, an experiment
is used to compare the banking assistant while retrieving different
versions of the `banking-assistant` prompt.

The scripts are:

``` powershell
python .\experiment_v1.py
python .\experiment_v2.py
```

The experiment run records the generated answers and provides a way to
inspect item-level outputs and traces in Langfuse.

For prompt comparisons, keep the dataset, application behavior, model
configuration, and evaluator criteria as consistent as possible.
Otherwise, differences may come from more than the prompt itself.

## LLM-as-a-Judge evaluation

`evaluate_experiments.py` uses Gemini to judge whether each generated
response is correct relative to the user question and expected answer.
It records a numeric score and a short explanation for each successfully
evaluated item.

The demo uses this scoring convention:

-   `1.0` --- judge considers the answer correct.
-   `0.0` --- judge considers the answer incorrect.

The script runs experiments for Prompt V1 and Prompt V2 and prints a
summary of scored items. It creates **new experiment runs**; it does not
retroactively score or modify previous runs.

### Why use an LLM judge?

Exact-string comparison is too strict for natural-language answers. For
example, these answers convey the same amount:

-   `Your current account balance is £85,420.50.`
-   `Your current account balance is **£85,420.50**.`

An LLM judge can accept harmless differences in wording and formatting
while considering whether the answer addresses the question.

### Limitations and responsible interpretation

An LLM judge is another model and can make mistakes. It may incorrectly
accept an unsupported claim or reject a valid answer. The application
model and the judge can also fail independently because of rate limits,
provider availability, or malformed responses.

For banking-related examples, verify at least:

-   Account balances and transaction amounts.
-   Whether debit/credit signs are preserved.
-   Whether displayed dates exist in the tool result.
-   Whether a requested unsupported action is clearly refused.
-   Whether the response answers the user's actual question.

A provider error such as HTTP 503 is **not** a pass or a fail. It is an
evaluation failure and should be retried or recorded as unscored.
Compare prompt versions only after checking that the required items were
evaluated successfully.

### Run the combined evaluation

``` powershell
python .\evaluate_experiments.py
```

Prerequisites:

-   `server.py` is running.
-   Groq credentials and model configuration are valid.
-   Langfuse credentials are valid.
-   The `banking-assistant-evaluation` dataset exists.
-   Prompt versions V1 and V2 exist in Langfuse.
-   A Gemini API key and an available model are configured.

After the run, open the printed Langfuse dataset-run links. Inspect each
item's input, expected output, actual output, score, judge explanation,
and associated trace. Compare the two runs only after accounting for any
unscored items.

## Setup

### Prerequisites

-   Python version compatible with the dependencies (Python 3.10+ is a
    practical starting point; confirm package requirements in your
    environment).
-   Git.
-   A Groq API key.
-   A Langfuse project and API keys.
-   A Gemini API key to run the judge-based evaluation.

### 1. Clone the repository

After creating the GitHub repository:

``` powershell
git clone https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPOSITORY-NAME>.git
cd <YOUR-REPOSITORY-NAME>
```

If you are working in the existing project folder, skip cloning.

### 2. Create a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
```

### 4. Configure environment variables

Copy the template:

``` powershell
Copy-Item .env.example .env
```

Edit `.env` with your own credentials. Never commit this file.

## Configuration

Variable names should match the ones used by the local scripts. The
following is a template of the configuration used in this project:

``` dotenv
# Groq model used by the banking assistant
GROQ_API_KEY=replace-with-your-groq-key
GROQ_MODEL=openai/gpt-oss-20b

# MCP endpoint
MCP_SERVER_URL=http://127.0.0.1:8000/mcp

# Langfuse: tracing and prompt management
LANGFUSE_SECRET_KEY=replace-with-your-langfuse-secret-key
LANGFUSE_PUBLIC_KEY=replace-with-your-langfuse-public-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development

# Gemini judge for evaluate_experiments.py
GEMINI_API_KEY=replace-with-your-gemini-api-key
GEMINI_MODEL=replace-with-a-model-available-to-your-account
```

If the evaluation script reads `GOOGLE_API_KEY` instead of
`GEMINI_API_KEY`, set the variable expected by that script. The
configured Gemini model identifier must be available to your API key.

If LangSmith tracing is also enabled in your code, configure the
variables it expects, for example:

``` dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=replace-with-your-langsmith-key
LANGSMITH_PROJECT=bank-ai-tool-calling
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Never paste actual credentials into source code, README files,
screenshots, or Git commits. `.env.example` must contain placeholders
only.

## Run the application

Open two terminals in the project directory.

### Terminal 1 --- MCP server

``` powershell
python .\server.py
```

The default endpoint is:

``` text
http://127.0.0.1:8000/mcp
```

Keep the server running.

### Terminal 2 --- Streamlit application

``` powershell
streamlit run .\app.py
```

Open the local URL printed by Streamlit. The application requires the
MCP server to be available for banking tool calls.

### Optional --- standalone MCP client

With the server running, execute:

``` powershell
python .\client.py
```

This demonstrates tool discovery and direct tool invocation without the
LLM.

## Troubleshooting

  -----------------------------------------------------------------------
  Symptom                             Things to check
  ----------------------------------- -----------------------------------
  MCP connection refused              Confirm `server.py` is running and
                                      `MCP_SERVER_URL` matches the
                                      endpoint.

  API key missing                     Check `.env`, variable names, and
                                      that the process was restarted
                                      after configuration changes.

  Langfuse traces absent              Check credentials, base URL, SDK
                                      integration, and that the code path
                                      actually invokes the instrumented
                                      model/tool calls.

  Prompt name/version absent from a   Confirm the prompt is fetched
  trace                               through Langfuse and the prompt
                                      metadata is attached to the
                                      LangChain prompt/generation as
                                      implemented by the app.

  Dataset not found                   Confirm the dataset name is exactly
                                      `banking-assistant-evaluation` in
                                      the selected Langfuse project.

  Gemini 503 / 429                    The provider may be overloaded or
                                      rate-limiting requests. Retry with
                                      appropriate backoff or configure a
                                      model available to your API key. Do
                                      not count failed judge calls as
                                      passes.

  V1/V2 item counts differ            Inspect evaluator errors and
                                      unscored items. Compare only after
                                      identifying missing scores.
  -----------------------------------------------------------------------

## Security and production considerations

This is a teaching project, not a production banking service. Before
adapting it for real use:

-   Replace hard-coded demo data with approved, authenticated service
    integrations.
-   Enforce authentication, authorization, and server-side validation
    for every tool call.
-   Never treat model-generated tool arguments as trusted.
-   Ensure unsupported operations are refused and cannot be invoked
    through another route.
-   Redact sensitive data from traces and logs; set appropriate
    retention and access policies.
-   Protect API keys with a secret manager in deployed environments.
-   Add unit, integration, security, and regression tests.
-   Evaluate important financial facts deterministically wherever
    possible; use LLM judging as a complementary signal.
-   Follow organizational security, privacy, compliance, and
    data-handling policies.

## Suggested demo walkthrough

For a short technical presentation, demonstrate the concepts in this
order:

1.  **MCP:** Show `server.py`, the available tools, and `client.py`
    discovering/calling them.
2.  **Tool calling:** Ask for a balance in the Streamlit app and follow
    the model's tool request through the MCP server.
3.  **Observability:** Open the corresponding Langfuse trace and inspect
    model activity, tool execution, and timing.
4.  **Prompt Management:** Show `banking-assistant` V1 and V2 and
    explain how the application retrieves a chosen version.
5.  **Dataset:** Open `banking-assistant-evaluation` and explain inputs
    and expected outputs.
6.  **Experiments:** Run the V1 and V2 scripts and compare item-level
    outputs.
7.  **Evaluation:** Run `evaluate_experiments.py`, review scores and
    judge explanations, and discuss why missing scores or judge mistakes
    must be handled explicitly.

## Git hygiene

Keep these local-only items out of version control:

-   `.env` and any file containing real credentials.
-   `.venv/`, `venv/`, and Python cache files.
-   Local vector databases, generated data, and logs.
-   Machine-specific editor settings.

Before pushing, run `git status` and inspect staged changes. If a secret
was ever committed, removing it in a later commit is not sufficient;
rotate the exposed credential and follow the provider's remediation
guidance.

## License

Add an appropriate license if you plan to distribute or reuse the
project publicly. Until then, the repository owner retains all rights.
