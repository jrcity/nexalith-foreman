# Nexalith Foreman

Offline AI ops agent for HR, CRM, and CMS workflows. Runs entirely on an 8GB laptop — no cloud, no API calls during inference.

Built for the **Africa Deep Tech Challenge 2026**, Autonomous AI Agents track. This is early-stage R&D for [Nexalith](https://nexalith.co)'s own Internal OS.

See `REPORT.md` for the full technical writeup (problem, design decisions, constraints, benchmarks).

---

## Quick Start

### 1. Download the model

```bash
bash download_model.sh
```

This fetches `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB) to `model/`. Safe to re-run — it skips the download if the file already exists.

### 2. Build llama.cpp (if you don't already have it)

```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

### 3. Start the model server

From the `nexalith-foreman` repo root:

```bash
/path/to/llama.cpp/build/bin/llama-server \
  -m model/qwen2.5-3b-instruct-q4_k_m.gguf \
  --ctx-size 4096 --threads 4 --port 8080
```

Leave this running in its own terminal. Wait for `server is listening on http://127.0.0.1:8080` before continuing.

**Linux power-management note:** if generation speed seems far below ~8 tokens/sec, check your CPU power profile — see "A note on CPU power management" below before assuming the model itself is slow.

### 4. Build the local database

```bash
python3 agent/rag/migrate_to_sqlite.py
```

This builds `agent/data/foreman.db` from the synthetic seed dataset (`agent/data/seed_corpus.json`). Safe to re-run — it rebuilds the tables from scratch each time.

### 5. Run the agent

**Option A — CLI:**

```bash
pip install -r agent/requirements.txt --break-system-packages
python3 agent/cli.py
```

Type a request at the `foreman>` prompt, e.g.:
```
One of our sales reps has three deals that have had no activity in over two weeks. Find them and draft a short follow-up message for each.
```

Type `exit` to quit.

**Option B — Web dashboard:**

```bash
pip install -r agent/requirements.txt --break-system-packages
python3 agent/web/server.py
```

Open `http://localhost:8090` in a browser. The dashboard shows a chat panel alongside a live status board (open CRM deals, upcoming onboarding, and a running achievements feed that updates after each request).

Both interfaces call the exact same orchestrator logic (`agent/orchestrator.py`) — neither duplicates the other's behavior.

---

## What it does

Nexalith Foreman has three tools the model can call, based on the user's natural-language request:

- **`crm_followup`** — finds CRM deals with no activity in N+ days, for drafting follow-up messages
- **`hr_lookup`** — finds employees starting within N days and their onboarding checklist status
- **`cms_publish_check`** — finds draft CMS content linked to a specific employee (e.g. a new-hire announcement still in draft)

The model decides which tool(s) to call, in what order, and chains them across domains when a request calls for it — e.g. a new-hire request triggers both `hr_lookup` and `cms_publish_check` in one turn, with the employee ID flowing from the first tool's result into the second's call, without being told explicitly. Every completed action is logged to `agent/data/achievements_log.json` with a timestamp and which tools fired — a verifiable audit trail, not just a prose claim.

The underlying data (`agent/data/seed_corpus.json`) is a synthetic but structurally realistic HR/CRM/CMS dataset, since no production Nexalith Internal OS data exists yet at this early R&D stage.

---

## A note on CPU power management

During development we found that Linux power-management daemons (`power-profiles-daemon`, `TLP`) can silently cap CPU frequency well below the chip's real maximum, independent of the kernel's CPU governor setting. If you're benchmarking or running this on a laptop and generation speed seems unexpectedly low, check:

```bash
powerprofilesctl get
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq   # compare to cpuinfo_max_freq
```

We ship and benchmark under the `balanced` profile specifically because the `performance` profile, while faster (~13.5 t/s), drove sustained core temperatures to 97°C with confirmed throttling on our hardware — over the competition's 85°C thermal threshold. See `REPORT.md` → Constraints for the full investigation.

---

## Repository structure

```
nexalith-foreman/
├── metadata.json              # required by ADTC profiler
├── download_model.sh          # downloads the .gguf model weight
├── REPORT.md                  # full technical writeup
├── model/                     # downloaded model lives here (gitignored)
├── agent/
│   ├── orchestrator.py        # core tool-calling loop
│   ├── cli.py                 # interactive terminal interface
│   ├── tools/                 # crm_followup, hr_lookup, cms_publish, achievement_log
│   ├── rag/retriever.py       # structured retrieval over the local dataset
│   ├── data/
│   │   ├── seed_corpus.json       # synthetic HR/CRM/CMS records
│   │   └── achievements_log.json  # append-only action audit trail (generated at runtime)
│   └── web/
│       ├── server.py          # FastAPI backend (wraps orchestrator.py, no duplicated logic)
│       └── static/index.html  # dashboard frontend
└── requirements.txt
```

---

## Testing locally with the official profiler

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git" --break-system-packages
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```

See `REPORT.md` for our own recorded benchmark results and the power-management findings that shaped our final configuration choice.
