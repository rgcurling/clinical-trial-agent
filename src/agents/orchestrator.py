"""
TrialMatch AI — Agentic Orchestrator

Claude drives the full pipeline autonomously using tool use.
The agent decides which tools to call, in what order, and synthesises
a final natural-language answer with ranked trial recommendations.

Usage:
    from src.agents.orchestrator import TrialMatchAgent

    agent = TrialMatchAgent()
    for event in agent.run("58yo female, Stage IIIB NSCLC, EGFR negative..."):
        print(event)          # stream agent thinking + tool calls to terminal
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Generator

import anthropic

from src.agents.tools import TOOL_SCHEMAS, ToolExecutor

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TURNS = 20  # safety ceiling on agentic loop

_SYSTEM_PROMPT = """\
You are TrialMatch AI, an expert clinical trial matching agent. Your goal is to \
find the most relevant recruiting clinical trials for a patient and explain their \
eligibility clearly.

You have access to these tools (use them in this general order):
1. extract_patient_profile   — parse the patient note into structured data
2. search_clinical_trials    — find candidate trials from ClinicalTrials.gov
3. rank_trials_by_relevance  — use BiomedBERT to surface the most relevant trials
4. evaluate_eligibility      — assess the patient against each top trial's criteria
5. get_critic_review         — have GPT-4o independently validate your top assessments
6. generate_patient_explanation — write a plain-English explanation for each final trial

Guidelines:
- Always extract the profile first.
- Search with the primary condition. Use the most specific condition term.
- After ranking, evaluate the top 5–7 trials (skip any with overall_score likely < 0.3).
- Run critic review on trials with score ≥ 0.5.
- Generate explanations for trials you will recommend to the patient.
- In your final answer, list trials from highest to lowest final score.
- Be concise in your reasoning. The patient wants clear, actionable information.
- Never fabricate trial data — use only what the tools return.
- If a trial has hard_exclusion=true, do not recommend it.
"""


@dataclass
class AgentEvent:
    """A single event streamed from the agent loop."""
    type: str          # "thinking" | "tool_call" | "tool_result" | "answer" | "error"
    content: str
    metadata: dict | None = None


class TrialMatchAgent:
    """
    Agentic orchestrator: Claude + tool use loop.

    Streams AgentEvents so callers can display progress in real time.
    """

    def __init__(self, model: str = _MODEL):
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def run(self, patient_text: str, *, location: str | None = None) -> Generator[AgentEvent, None, None]:
        """
        Run the agent for a patient query.

        Yields AgentEvents as the agent thinks, calls tools, and formulates its answer.
        """
        executor = ToolExecutor()
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Please find the best matching clinical trials for this patient"
                    + (f" near {location}" if location else "")
                    + f":\n\n{patient_text}"
                ),
            }
        ]

        for turn in range(_MAX_TURNS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Collect text and tool_use blocks from this response
            tool_calls = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    yield AgentEvent(type="thinking", content=block.text.strip())
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # Add assistant turn to history
            messages.append({"role": "assistant", "content": response.content})

            # Agent finished — no more tool calls
            if response.stop_reason == "end_turn" or not tool_calls:
                # Extract final answer from last text block
                final = next(
                    (b.text for b in reversed(response.content) if b.type == "text"),
                    "No response generated.",
                )
                yield AgentEvent(type="answer", content=final)
                return

            # Execute tool calls and feed results back
            tool_results = []
            for tc in tool_calls:
                yield AgentEvent(
                    type="tool_call",
                    content=f"→ {tc.name}({json.dumps(tc.input, ensure_ascii=False)[:120]}…)",
                    metadata={"tool": tc.name, "input": tc.input},
                )

                result_str = executor.execute(tc.name, tc.input)
                result_data = json.loads(result_str)

                yield AgentEvent(
                    type="tool_result",
                    content=_summarise_result(tc.name, result_data),
                    metadata={"tool": tc.name, "result": result_data},
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        yield AgentEvent(type="error", content="Agent reached maximum turn limit without finishing.")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _summarise_result(tool_name: str, result: dict) -> str:
    if "error" in result:
        return f"✗ Error: {result['error']}"
    if tool_name == "extract_patient_profile":
        return f"✓ Profile: {result.get('conditions')} | age={result.get('age')} | stage={result.get('stage')}"
    if tool_name == "search_clinical_trials":
        return f"✓ Found {result.get('n_found', 0)} trials"
    if tool_name == "rank_trials_by_relevance":
        top = result.get("ranked_trials", [])[:3]
        ids = [f"{t['nct_id']} ({t['similarity']:.2f})" for t in top]
        return f"✓ Top ranked: {', '.join(ids)}"
    if tool_name == "evaluate_eligibility":
        return (
            f"✓ {result.get('nct_id')}: score={result.get('overall_score'):.2f} "
            f"excluded={result.get('hard_exclusion')}"
        )
    if tool_name == "get_critic_review":
        return (
            f"✓ Critic: agree={result.get('agree')} "
            f"rec={result.get('recommendation')} "
            f"final_score={result.get('final_score', 0):.2f}"
        )
    if tool_name == "generate_patient_explanation":
        text = result.get("explanation", "")[:80]
        return f"✓ Explanation: \"{text}…\" (FK={result.get('fk_grade')})"
    return f"✓ Done"
