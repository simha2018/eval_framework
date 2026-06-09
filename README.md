# GenAI Eval Framework

Enterprise evaluation framework for LangGraph multi-agent systems.
Supports **Claude** and **GPT** as judge models — swap with one config line.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...      # only if using OpenAI judge
```

## Quick Start

```python
from genai_eval import EvalHarness, EvalConfig, RAGInput, AgentTrace, SafetyInput

# Default: Claude Sonnet as judge
harness = EvalHarness(EvalConfig())

# --- RAG ---
report = harness.evaluate(rag=[
    RAGInput(
        question="What is Planisware used for?",
        context_chunks=["Planisware is a PPM platform..."],
        generated_answer="Planisware manages project portfolios.",
        ground_truth="Planisware is a PPM platform for R&D and IT project management.",
    )
])
print(report.summary())
```

## Switch Judge Model

```python
# Use GPT-4o as judge
EvalConfig(judge_provider="openai")

# Use Claude Opus for richer judgment
EvalConfig(anthropic_model="claude-opus-4-6")

# Run only selected dimensions
EvalConfig(active_dimensions=["rag", "agent", "safety"])
```

## LangGraph Integration

### From `graph.invoke()` (no checkpointer needed)

```python
from genai_eval.integrations.langgraph import trace_from_invoke_result

result = graph.invoke({"messages": [HumanMessage(content=task)]})

trace = trace_from_invoke_result(
    task=task,
    result=result,
    tools_available=["planisware_get_project", "cdocs_search", "kg_query"],
    model_used="claude-sonnet-4-6",
    latency_ms=1500.0,
)

report = EvalHarness().evaluate(agent=[trace])
```

### From state history (with checkpointer)

```python
from genai_eval.integrations.langgraph import traces_from_graph_history

traces = traces_from_graph_history(
    graph, {"configurable": {"thread_id": "t1"}}, task=task
)
report = EvalHarness().evaluate(agent=traces)
```

## Dimensions & Schemas

| Dimension | Input Schema | Key Metrics |
|-----------|-------------|-------------|
| `rag` | `RAGInput` | faithfulness, answer_relevance, context_recall |
| `agent` | `AgentTrace` | task_completion, step_efficiency, hallucinated_actions |
| `multi_agent` | `MultiAgentTrace` | e2e_success, routing_accuracy, loop_detection |
| `kg` | `KGInput` | kg_query_accuracy, entity_f1, schema_adherence |
| `safety` | `SafetyInput` | pii_leakage, jailbreak_resistance, over_refusal |
| `perf` | `PerfSample` | latency_p99, cost_per_query, cache_hit_rate |
| `llm_quality` | `LLMQualityInput` | correctness, hallucination_rate, consistency |
| `prompt` | `PromptQualityInput` | format_compliance, instruction_adherence |

## Architecture

```
genai_eval/
├── config.py                  # EvalConfig — judge provider & model
├── harness.py                 # EvalHarness — main entry point
├── models/
│   ├── anthropic_provider.py  # Claude wrapper
│   └── openai_provider.py     # GPT wrapper
├── judges/
│   └── llm_judge.py           # LLM-as-judge with 25+ built-in rubrics
├── schemas/
│   ├── inputs.py              # Typed input schemas (Pydantic)
│   └── outputs.py             # EvalResult, DimensionReport, EvalReport
├── evaluators/                # One evaluator per dimension
│   ├── rag_pipeline.py
│   ├── agent_behavior.py
│   ├── multi_agent.py
│   ├── knowledge_graph.py
│   ├── safety.py
│   ├── system_performance.py
│   ├── llm_quality.py
│   └── prompt_quality.py
├── integrations/
│   └── langgraph.py           # LangGraph trace helpers
└── reporters/
    └── report.py              # JSON / Markdown output
```

## Examples

```bash
python examples/eval_rag.py               # RAG with Claude vs GPT judge
python examples/eval_agent_langgraph.py   # LangGraph agent eval
python examples/eval_multi_agent.py       # Multi-agent orchestration eval
```

## SLA Defaults (system_performance)

Tune these in `evaluators/system_performance.py`:

| Metric | Default |
|--------|---------|
| p99 latency | ≤ 5 000 ms |
| Cold start | ≤ 3 000 ms |
| Cache hit rate | ≥ 30% |
| Error rate | ≤ 5% |
| Cost per query | ≤ $0.05 |
