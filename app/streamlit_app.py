"""DejaFail web UI: stateless vs memory triage, the learning curve, and what the agent learned."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dejafail.config import ConfigError, load_config
from dejafail.llm import GroqLLM
from dejafail.memory import HindsightStore, MemoryUnavailable
from dejafail.models import VERDICT_KINDS, CIRun, Diagnosis, Outcome
from dejafail.replay import load_runs, load_steps, rolling_accuracy
from dejafail.triage import Triage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "shopfront"
KIND_COLORS = {
    "flaky": "#b45309",
    "regression": "#b91c1c",
    "dependency": "#6d28d9",
    "infra": "#1d4ed8",
    "unknown": "#4b5563",
}
PASTE = "(paste your own log)"

st.set_page_config(page_title="DejaFail", layout="wide")


@st.cache_resource
def services():
    cfg = load_config()
    store = HindsightStore.from_config(cfg)
    store.ensure_bank()
    return cfg, store, Triage(GroqLLM(cfg.groq_api_key, cfg.groq_model), store, cfg.repo)


@st.cache_data
def samples() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted((DATA / "demo_logs").glob("*.log")):
        out[f"new today · {path.stem}"] = path.read_text(encoding="utf-8")
    for run in reversed(load_runs(DATA / "runs.jsonl")):
        out[f"{run.run_id} · {run.started_at[:10]} · {run.branch}"] = run.log
    return out


def verdict_card(title: str, diagnosis: Diagnosis) -> None:
    v = diagnosis.verdict
    color = KIND_COLORS.get(v.kind, KIND_COLORS["unknown"])
    st.markdown(f"#### {title}")
    st.markdown(
        f"<span style='background:{color};color:#fff;padding:4px 12px;border-radius:6px;"
        f"font-weight:700;letter-spacing:.04em'>{v.kind.upper()}</span>"
        f"&nbsp;&nbsp;confidence {v.confidence:.0%}",
        unsafe_allow_html=True,  # safe: kind is validated against VERDICT_KINDS
    )
    st.write(v.summary)
    st.markdown(f"**Next action:** {v.next_action}")
    if v.memory_error:
        st.warning(f"Memory unavailable: {v.memory_error}")
    elif v.used_memory:
        if v.evidence:
            st.caption(
                f"Seen before in {v.seen_before_count} earlier run(s). Evidence the model cited from Hindsight:"
            )
            for e in v.evidence:
                st.markdown(f"- `{e.date or 'undated'}` {e.text}")
        elif diagnosis.memories:
            st.caption(
                f"Seen before in {v.seen_before_count} earlier run(s). "
                "Recalled from Hindsight (not cited by the model):"
            )
            recalled = sorted(diagnosis.memories, key=lambda m: not m.exact)[:5]  # stable: exact matches first
            for m in recalled:
                st.markdown(f"- `{m.date or 'undated'}` {m.text}")
        else:
            st.caption("Not seen before: Hindsight has no matching memories.")


st.title("DejaFail")
st.caption("The CI triage agent that remembers every red build. Memory by Hindsight.")

try:
    cfg, store, triage = services()
except (ConfigError, MemoryUnavailable) as exc:
    st.error(str(exc))
    st.stop()

tab_triage, tab_curve, tab_learned = st.tabs(["Triage", "Learning curve", "What it learned"])

with tab_triage:
    options = samples()
    choice = st.selectbox("Failing CI run", [PASTE, *options])
    log = st.text_area("CI log", value="" if choice == PASTE else options[choice], height=240)
    if st.button("Diagnose", type="primary", disabled=not log.strip()):
        with st.spinner("Recalling earlier runs from Hindsight and asking the model..."):
            st.session_state["result"] = (
                log,
                triage.diagnose(log, use_memory=False),
                triage.diagnose(log, use_memory=True),
            )
    result = st.session_state.get("result")
    if result and result[0] == log:
        _, stateless, remembered = result
        left, right = st.columns(2)
        with left:
            verdict_card("Stateless LLM (no memory)", stateless)
        with right:
            verdict_card("DejaFail + Hindsight memory", remembered)
        st.divider()
        st.subheader("Teach DejaFail what really happened")
        with st.form("feedback"):
            label = st.selectbox("Real cause", VERDICT_KINDS, index=VERDICT_KINDS.index(remembered.verdict.kind))
            rerun = st.checkbox("A rerun passed with no code change")
            note = st.text_input("Fix or note", placeholder="e.g. pinned pydantic<2")
            if st.form_submit_button("Teach"):
                run = CIRun.adhoc(log)
                try:
                    store.record_failure(run, remembered.sig)
                    store.record_outcome(run, remembered.sig, Outcome(rerun, label, note))
                    st.success(
                        f"Learned: {remembered.sig.test_id or 'job failure'} -> {label}. "
                        "Diagnose a similar failure to see it recalled."
                    )
                except MemoryUnavailable as exc:
                    st.error(str(exc))

with tab_curve:
    results_path = DATA / "replay_results.json"
    if not results_path.exists():
        st.info("No replay yet. Run `python -m dejafail replay` to generate the learning curve.")
    else:
        try:
            steps = load_steps(results_path)
        except (ValueError, TypeError, KeyError):
            steps = None
            st.warning("Replay results file is incomplete or malformed; rerun the replay.")
        if steps is not None and not steps:
            st.info("Replay has no steps yet.")
        elif steps:
            total = max(len(steps), 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("CI runs replayed", len(steps))
            c2.metric("Correct with memory", f"{sum(s.memory_correct for s in steps) / total:.0%}")
            c3.metric("Correct stateless", f"{sum(s.stateless_correct for s in steps) / total:.0%}")
            frame = pd.DataFrame(rolling_accuracy(steps)).set_index("run")
            frame = frame.rename(columns={"memory": "With Hindsight memory", "stateless": "Stateless LLM"})
            st.line_chart(frame, x_label="CI run (chronological)", y_label="Accuracy, last 8 runs")
            flaky = [s for s in steps if s.truth == "flaky"]
            if flaky:
                st.caption(
                    f"Flaky-test failures: memory got {sum(s.memory_correct for s in flaky)}/{len(flaky)}, "
                    f"stateless got {sum(s.stateless_correct for s in flaky)}/{len(flaky)}. "
                    "Synthetic history of a fictional repository; see README."
                )
            st.dataframe(
                pd.DataFrame([
                    {"run": s.run_id, "date": s.started_at[:10], "test": s.test_id or "(job)", "truth": s.truth,
                     "memory": s.memory_kind, "stateless": s.stateless_kind, "seen before": s.seen_before}
                    for s in steps
                ]),
                hide_index=True,
            )

with tab_learned:
    st.subheader("Flaky ledger")
    st.caption("A Hindsight mental model: a consolidated summary the agent keeps up to date from its memories.")
    try:
        summary = store.learned_summary()
    except MemoryUnavailable as exc:
        summary = ""
        st.error(str(exc))
    st.markdown(summary or "_Nothing consolidated yet. Seed or replay history, then refresh._")
    if st.button("Refresh mental model"):
        try:
            store.refresh_summary()
            st.info("Refresh requested. It can take a minute; reopen this tab to see the new version.")
        except MemoryUnavailable as exc:
            st.error(str(exc))
    st.subheader("Ask the ledger")
    question = st.text_input("Question", placeholder="Which tests should we quarantine, and why?")
    if st.button("Ask", disabled=not question.strip()):
        with st.spinner("Reflecting over memory..."):
            try:
                st.markdown(store.ask(question))
            except MemoryUnavailable as exc:
                st.error(str(exc))
