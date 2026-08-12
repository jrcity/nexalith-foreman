# Technical Report — Nexalith Foreman

**Team ID:** 1056817-nexalith-foreman
**Domain:** autonomous_ai_agents
**Model:** Qwen2.5-3B-Instruct-Q4_K_M (Fine-tuned LoRA)

---

## Problem

African small and medium enterprises run their operations — HR, customer relationships, content — on a patchwork of spreadsheets, WhatsApp threads, and memory, because cloud-hosted SaaS tooling assumes stable connectivity and recurring subscription costs that many SMEs cannot reliably sustain. The same constraint applies to early-stage African tech ventures themselves: as a solo founder building Nexalith, an infrastructure software company in Kaduna, Nigeria, I need internal HR, CRM, and CMS automation, but every cloud-AI option that could provide it depends on per-token API costs and constant connectivity — a real risk in a market where power and bandwidth are not guaranteed.

Nexalith Foreman is an offline AI agent that runs entirely on an 8GB commodity laptop and automates exactly these operational workflows — following up on stale CRM deals, running new-hire onboarding checklists, flagging what needs CMS publication — without a single network call after the model is downloaded. It is being built as real, early-stage R&D for Nexalith's own Internal OS, not a standalone demo: the target user is a founder or small operations team that needs an assistant that reasons over their actual business data, on hardware they already own.

## Design Decisions & The Fine-Tuning Journey

**Base model:** We started with the base quantized `Qwen2.5-3B-Instruct` (3.4B parameters). It was chosen over smaller (1.5B-class) and larger (7B-class) alternatives because it leaves enough RAM headroom under the 7GB ceiling for a local retrieval layer running alongside it on an 8GB machine.

**The Limitation:** While the base Qwen 3B model was capable, we discovered it struggled with strict operational constraints. When presented with complex, multi-domain queries (e.g., asking about CRM leads AND HR new hires), it frequently hallucinated outputs, struggled to cleanly execute parallel tool calls (READ tools), and failed to ask for user confirmation before executing destructive WRITE tools. 

**The Fine-Tuning Solution:** Upon securing a cloud GPU instance (RTX 6000 Ada), we deliberately pivoted from relying solely on zero-shot prompting to running a targeted LoRA (Low-Rank Adaptation) fine-tune using PEFT and TRL. We generated a synthetic 15-example dataset (`finetune_examples.jsonl`) that explicitly demonstrated the required operational behavior:
1. Extracting parallel READ tool calls accurately (e.g., calling `hr_lookup` and `crm_followup` simultaneously).
2. Never asserting factual claims without executing a tool first.
3. Catching hallucinations gracefully (e.g., refusing to execute `log_sales_interaction` if `get_customer_profile` returns empty).

The trained LoRA weights were merged back into the HuggingFace base model, converted into an `f16` GGUF, and finally quantized down to `Q4_K_M` (~1.83 GB, 4.99 BPW) using `llama.cpp`. This allowed us to deploy a highly specialized, tool-calling agent locally without blowing up the strict memory budget.

**Runtime:** llama.cpp exclusively, as required. The agent's orchestration layer communicates with the model through llama.cpp's OpenAI-compatible server API (`llama-server`).

## Constraints & Thermal Performance

**Hardware target:** 8GB RAM, integrated GPU only, no discrete GPU acceleration — pure CPU inference via llama.cpp.
**Development hardware:** Lenovo ThinkPad X390 (Intel i7-8565U, 8th-gen, 4 physical cores / 8 threads, 16GB RAM).

**A real, non-obvious constraint we discovered during benchmarking: CPU power management.** 
To validate the fine-tuned model, we built a Python edge-testing suite wrapped in a bash script (`run_thermal_edge_test.sh`) to automatically rotate through CPU governors (`performance` and `powersave`/`balanced`) while throwing complex, multi-tool hallucination traps at the model.

1. **Performance Mode:** The model successfully executed complex parallel tool calls and avoided hallucination traps in 11s–38s (including initial cache warming). The CPU core temperatures peaked safely at **73°C** — well below the critical 100°C limit and safely under the competition's 85°C threshold.
2. **Balanced (Powersave) Mode:** After the model cache was warmed, inference dropped to a blazing fast 10s–27s per multi-tool trap, peaking at a cool **72°C**. 

We are shipping with this balanced configuration as the honest, sustainable operating point for this hardware. The laptop can safely leave the `llama-server` daemon running indefinitely without risking hardware damage or aggressive thermal throttling.

## System Architecture

Nexalith Foreman has three layers:

1. **Model layer** — Our custom fine-tuned `Qwen2.5-3B-Instruct-Q4_K_M` served via `llama-server`'s OpenAI-compatible API. This is the only component the official ADTC profiler benchmarks directly.
2. **Agent layer** (`agent/orchestrator.py`) — a tool-calling loop that gives the model access to three domain tools: `crm_followup` (find stale deals), `hr_lookup` (find upcoming starters and their onboarding checklists), and `cms_publish_check` (find draft content linked to a specific employee).
3. **Interface layer** — two interfaces share the orchestrator with no duplicated logic: a CLI (`agent/cli.py`) for direct terminal use, and a local web dashboard (`agent/web/server.py` + `agent/web/static/index.html`).

**Achievement logging:** every completed agent action is appended to a local JSON log (`agent/data/achievements_log.json`).

## Evidence of Load-Bearing Cross-Disciplinary Integration

Our `cross_disciplinary_pairing` claim is backed by reproducible edge testing. During our post-fine-tuning validation, the model correctly caught traps across both HR and CRM domains simultaneously. 

For the request: *"Can you update the lead status for CRM ID 'L-9999' to 'closed_won' and onboard an employee named 'John Doe' (jdoe@example.com) as a 'DevOps Engineer' in 'Engineering'? I approve these actions."*

The model perfectly parsed the parallel requests, realized the CRM ID 'L-9999' was invalid, executed the valid HR onboarding write-tool, and gracefully aborted the CRM write-tool without hallucinating a fake update.

## Conclusion

By leveraging a cloud GPU grant to bake strict operational rules into a lightweight 3B model via LoRA, we successfully delivered a 1.8GB, 100% offline, privacy-first AI agent. Nexalith Foreman can autonomously orchestrate complex multi-domain workflows on commodity African SME hardware while remaining thermally stable and operationally secure.

---

*Benchmarks current as of the most recent commit. See git history for the full investigative process behind the power-management and thermal findings.*
