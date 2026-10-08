# LangGraph vs. Claude Code + CLAUDE.md + Fire

Grounded in building the same kind of thing (a multi-step LLM agent routed through
Fire) with LangGraph just now, compared to how this whole session was actually run:
Claude Code, following instructions, hitting Fire directly when it needs an LLM call.

**Workflow representation.** LangGraph makes the pipeline an explicit object: a
`StateGraph` with typed state, named nodes, and declared edges
(`decompose → answer → synthesize`) that you can call `.draw_mermaid()` on and get a
diagram back. It is code, versioned, and reviewable as a diff. The Claude Code +
CLAUDE.md approach has no such object — the "workflow" for this build task lived in
the prompt itself (12 numbered sections) and in my own sequencing of tool calls. That
sequencing isn't inspectable as a graph after the fact; it only exists as this
conversation transcript.

**Agent orchestration.** LangGraph's orchestration is deterministic and the framework
owns it: given the same state, the same nodes run in the same order, every time.
Claude Code's orchestration is me, deciding turn by turn what to do next based on
reasoning over the current context — flexible (I could react to the pip/venv failure
and route around it without anyone having coded that branch in advance) but not
reproducible in the same mechanical sense.

**State management.** LangGraph's state is a typed, explicit dict
(`AgentState(TypedDict)`) threaded through nodes by the framework, with well-defined
merge semantics per node's return value. Claude Code's "state" is the conversation
context plus whatever is actually written to disk (files, `.env`, git commits) —
real and durable, but not a single structured object a second process could resume
from the middle of.

**Reusable workflows.** The LangGraph graph I wrote is a reusable artifact: the same
`graph.py` runs for any question, unattended, and could be imported into another
service. What I did as Claude Code to scaffold the project (bootstrap pip, write
files, test, audit) was reusable only as *narrative* — someone would re-read this
report or the transcript to redo it, not import it as a module.

**Debugging.** LangGraph's failures surfaced as structured exceptions
(`FireAPIError`) at a specific named node, with `usage`/`call_id` in the logs for
every Fire call — debugging means reading one node's output. Debugging the Claude
Code side meant reading raw command output and log files myself (e.g. the venv
`ensurepip` failure, the killed background process) and reasoning about root cause
in natural language; there's no structured node boundary to point at.

**Control vs. complexity.** LangGraph trades upfront complexity (define state, nodes,
edges, a wrapper class satisfying `BaseChatModel`'s interface) for tight, predictable
control at runtime — useful exactly because the resulting graph needs to run the same
way thousands of times with no one watching. Claude Code + CLAUDE.md trades that
structure for adaptability: zero workflow code to write, but every run depends on a
reasoning model making good decisions live, which is harder to pin down in advance
but far cheaper to stand up for a one-off task like this build.

**When each is useful.** LangGraph is the right tool once the workflow is known,
needs to run unattended or at volume, and must behave identically on the hundredth
run as on the first — exactly what the shipped demo app now does every time someone
submits a question. Claude Code + CLAUDE.md + Fire is the right tool for the *meta*
task of getting there: reading unfamiliar docs, fixing a broken Python environment,
writing and testing new code, auditing for leaked secrets — open-ended work where the
next step genuinely depends on what the last command just printed, which is precisely
what happened repeatedly in this session (the capabilities fetch, the venv failure,
the killed server process, the false-positive secret-grep) and would have been
awkward to pre-encode as a fixed graph.
