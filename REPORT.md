# Technical Report — Nexalith Foreman

**Team ID:** 1056817-nexalith-foreman
**Domain:** autonomous_ai_agents
**Model:** Qwen2.5-3B-Instruct-Q4_K_M

---

## Problem

African small and medium enterprises run their operations — HR, customer relationships, content — on a patchwork of spreadsheets, WhatsApp threads, and memory, because cloud-hosted SaaS tooling assumes stable connectivity and recurring subscription costs that many SMEs cannot reliably sustain. The same constraint applies to early-stage African tech ventures themselves: as a solo founder building Nexalith, an infrastructure software company in Abuja, Nigeria, I need internal HR, CRM, and CMS automation, but every cloud-AI option that could provide it depends on per-token API costs and constant connectivity — a real risk in a market where power and bandwidth are not guaranteed.

Nexalith Foreman is an offline AI agent that runs entirely on an 8GB commodity laptop and automates exactly these operational workflows — following up on stale CRM deals, running new-hire onboarding checklists, flagging what needs CMS publication — without a single network call after the model is downloaded. It is being built as real, early-stage R&D for Nexalith's own Internal OS, not a standalone demo: the target user is a founder or small operations team that needs an assistant that reasons over their actual business data, on hardware they already own.

## Design Decisions

**Base model:** Qwen2.5-3B-Instruct (3.4B parameters), chosen over smaller (1.5B-class) and larger (7B-class) alternatives for three reasons: it produces reliable structured/tool-calling output, which is essential for an orchestration agent deciding which workflow to execute; it leaves enough RAM headroom under the 7GB ceiling for a local retrieval layer running alongside it; and unlike a narrow fine-tune, it generalizes well to prompts outside its exact training distribution — important given the evaluation explicitly tests two organizer-generated hidden prompts in addition to our own two.

**Quantization:** GGUF Q4_K_M, ~1.95GB on disk. We evaluated more aggressive quantization (Q4_0) purely as a thermal mitigation (see Constraints below) but did not adopt it, since Q4_K_M's quality retention was preferred once a safe power-management configuration was found that kept Q4_K_M within thermal limits.

**Why RAG and tool-calling over fine-tuning:** Rather than fine-tuning the base model on synthetic HR/CRM/CMS data, Foreman uses structured retrieval and tool-calling over a local operational dataset. This was a deliberate choice: fine-tuning on a small synthetic corpus risks the model memorizing surface patterns rather than reasoning over data, which would likely fail exactly the kind of hidden, out-of-distribution prompt the evaluation is designed to catch. Tool-calling keeps the model's reasoning general while grounding its answers in real, retrievable, structured facts.

We deliberately chose structured (SQL/keyword-style) retrieval over embeddings-based semantic search. Our underlying data — HR records, CRM deals, CMS drafts — is structured with well-defined fields, not free-text prose, so the model's real job is deciding *which query to run* (a genuinely agentic decision, made via tool-calling) rather than performing fuzzy semantic matching over vectors. This is both more accurate for this data shape and avoids the added RAM and latency cost of loading a separate embedding model under the 7GB ceiling.

**Runtime:** llama.cpp exclusively, as required. The agent's orchestration layer communicates with the model through llama.cpp's OpenAI-compatible server API (`llama-server`), meaning the model is the literal decision-making component every tool-call and retrieval step routes through — not a decorative chatbot layered on top of a CRM.

**Alternatives considered and rejected:**
- *Llama 3.2-3B-Instruct* — comparable size and speed, but weaker structured/tool-calling output in our testing.
- *7B-class models (e.g. Mistral-7B-Instruct)* — roughly double the RAM and disk footprint for marginal quality gains in our domain; would have left little headroom for the RAG layer under the 7GB ceiling.
- *Fine-tuning instead of tool-calling/RAG* — rejected for the overfitting risk described above. We remain open to layering a light fine-tune on top of this working system later (see Future Work) if compute access allows, but the system does not depend on it.

## Constraints

**Hardware target:** 8GB RAM, integrated GPU only, no discrete GPU acceleration — pure CPU inference via llama.cpp, per the ADTC Standard Laptop profile.

**Development hardware:** Lenovo ThinkPad X390 (Intel i7-8565U, 8th-gen, 4 physical cores / 8 threads, 16GB RAM, Intel UHD 620 integrated graphics) — an older CPU generation than the reference profile (10th–12th gen Intel / Ryzen 5 3000–5000 series), making it a conservative testbed: performance observed here is a reasonable lower bound for what the actual judging hardware should achieve.

**A real, non-obvious constraint we discovered during benchmarking: CPU power management.** Our development laptop's `power-profiles-daemon` defaulted to a `power-saver` profile that silently capped CPU frequency to 800MHz regardless of the kernel's CPU governor setting — a separate mechanism from the standard `cpupower`/governor layer most benchmarking guidance assumes. This produced generation speeds as low as 3.7–4.5 tokens/sec that had nothing to do with the model or quantization, and would have led us to wrongly conclude the model was too slow for this hardware class.

Correcting the power profile to fully unrestricted (`performance`) raised throughput to 13.5 tokens/sec, but also drove sustained core temperatures to 97°C with confirmed thermal throttling — a direct violation of the competition's 85°C thermal threshold. We tested a middle setting (`balanced` profile) and found it sustains 8.5–10.2 tokens/sec at a peak core temperature of 73–80°C with no throttling observed across multiple controlled benchmark runs — comfortable margin under the threshold. We are shipping with this configuration as the honest, sustainable operating point for this hardware, and have documented all three states (capped, unrestricted, balanced) as a deliberate finding rather than picking only the most flattering number.

**Connectivity:** Nexalith Foreman makes zero network calls during inference. The only network dependency is the one-time model download via `download_model.sh`, executed before evaluation begins, consistent with the competition's offline-execution requirement.

**Data constraint:** The RAG corpus is a synthetic but structurally realistic dataset of HR, CRM, and CMS records modeled on real SME operational patterns, since no production Nexalith Internal OS data yet exists at this early R&D stage.

## System Architecture

Nexalith Foreman has three layers:

1. **Model layer** — Qwen2.5-3B-Instruct-Q4_K_M served via `llama-server`'s OpenAI-compatible API. This is the only component the official ADTC profiler benchmarks directly.
2. **Agent layer** (`agent/orchestrator.py`) — a tool-calling loop that gives the model access to three domain tools: `crm_followup` (find stale deals), `hr_lookup` (find upcoming starters and their onboarding checklists), and `cms_publish_check` (find draft content linked to a specific employee). The model decides which tool(s) to call based on the user's natural-language request; the orchestrator executes the real Python function against the local dataset and returns structured results for the model to reason over and respond from.
3. **Interface layer** — two interfaces share the orchestrator with no duplicated logic: a CLI (`agent/cli.py`) for direct terminal use, and a local web dashboard (`agent/web/server.py` + `agent/web/static/index.html`) that adds a live status board (open CRM deals tagged stale/active, upcoming onboarding with checklist progress, and a running achievements feed).

**Achievement logging:** every completed agent action is appended to a local JSON log (`agent/data/achievements_log.json`) with a timestamp and which tool(s) were used. This is not cosmetic — it is the audit trail that lets us verify, after the fact, exactly which tools the model actually invoked for a given request, independent of what its prose response claims.

## Evidence of Load-Bearing Cross-Disciplinary Integration

Our `cross_disciplinary_pairing` claim (enterprise operations + autonomous agent orchestration, `load_bearing: true`) is not just architectural intent — we have direct, logged evidence of it working. For the request *"We just hired a new employee starting Monday. Set up their onboarding checklist, and let me know what CMS content needs to be published for the new-hire announcement,"* the achievement log recorded:

```json
{
  "timestamp": "2026-06-24T12:48:25",
  "action": "hr_lookup,cms_publish_check",
  "summary": "Handled request: We just hired a new employee starting Monday. Set up their onboarding checklist,"
}
```

The model called `hr_lookup`, extracted the returned employee ID (`emp_001`) from the result, and used that ID to call `cms_publish_check` — chaining two tools across two domains (HR and CMS) for one request, without being told the employee ID explicitly. This is a timestamped, independently verifiable record of cross-domain tool chaining, not a description of intended design.

In separate testing, a broader, less-scripted request ("what do I have on CRM, CMS and HR todo") caused the model to call all three tools in one turn and organize the combined results into clearly labeled sections unprompted — demonstrating the agent generalizes beyond our two submitted test prompts to reasonable variations in phrasing and scope.

We also observed the model correctly decline to fabricate an answer when asked about something outside its real tools ("what do you have currently on tasks") — it asked for clarification rather than inventing a plausible-sounding but fictional response, which we view as an important grounding property for a tool that real operators would rely on.

## Benchmarks

All figures below were captured using the official `adtc-profiler` tool in participant mode (`--skip-accuracy`), on the development hardware described above, with the `balanced` power profile active — our shipping configuration.

| Metric | Value |
|---|---|
| Machine | ThinkPad X390 — Intel i7-8565U, 16GB RAM, Intel UHD 620 |
| Power profile | `balanced` (see Constraints) |
| Generation speed (controlled benchmark) | 8.5–10.2 tokens/sec |
| Peak RAM (profiler-measured) | ~3.46 GB |
| Peak core temperature (controlled benchmark) | 73–80°C |
| Thermal throttling | None observed under `balanced` profile |
| CPU utilization (p99) | 83–89% |

These are self-reported development benchmarks. Official scores are measured by the ADTC profiler on the standard evaluation machine.

**Sustained real-world use.** Beyond controlled benchmarks, we ran `llama-server` continuously through many real, varied-length agent conversations over several hours. Generation speed across these real requests settled consistently at 8.1–9.6 tokens/sec — in line with the controlled benchmark range — and peak core temperature during active conversational use measured 53°C with the cooling fan at 0 RPM, well under the 85°C threshold even under realistic, non-synthetic load. This gives us confidence that our controlled benchmark numbers are representative of genuine usage rather than an artifact of the benchmark methodology itself.

**A note on methodology.** Earlier in development, inconsistent readings (ranging from 3.65 to 13.5 tokens/sec across nominally identical runs) led us to discover the power-management issue described above, as well as a secondary finding that back-to-back benchmark runs without a cooldown period produce inflated thermal readings unrepresentative of a single real inference session. All controlled-benchmark figures in the table above reflect isolated, cooled-start runs under the final shipping power configuration.

## Future Work

If GPU compute access becomes available (we have applied for the ADTC GPU compute grant), we plan to layer a light fine-tune on top of this working RAG/tool-calling system to improve domain fluency and tone consistency — strictly additive to the current architecture, not a replacement for it. The system ships as a complete, working product without this; fine-tuning is a refinement, not a dependency.

---

*Benchmarks current as of the most recent commit. See git history for the full investigative process behind the power-management and thermal findings above.*
