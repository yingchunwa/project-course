# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This repo has **three active code areas** plus an archived lab. Each has its own environment, commands, and conventions — treat them as separate project contexts:

1. **`src/project_course/`** — the mainline Python 3.10.12 package (FastAPI service + camera CLI). This is the stable production scaffold.
2. **`frontend/dashboard/`** — React 18 + Vite + ECharts dashboard (TypeScript), consumes the FastAPI service.
3. **`motor_fault_pca_project/`** — standalone PCA-based motor-fault detector for Orange Pi edge deployment. Independent visual + vibration PCA branches; no fusion.
4. **`laboratory/global-camera/`** — active camera bring-up smoke-test (separate `uv` project).
5. **`laboratory/legacy/motion-amplifier/`** — archived visual-vibration prototype, kept for reference. Excluded from root lint.

The end goal is a dual-modal (camera + IMU) device-state monitoring system. The feature contract is `doc/feature_schema.md`: per-window rows with shared identity fields, `vision_*` and `sensor_*` prefixes, fused by a join on `sample_id + window_index` — **not** a custom reshape.

## Commands

### Root Python package (`src/project_course`)

All commands run from repo root using **`uv`**. Pinned to **Python 3.10.12** via `.python-version` and `requires-python == 3.10.12` — preserve exactly unless intentionally migrating.

```bash
uv sync --all-groups                       # install with dev deps (ruff, pytest, httpx)
uv run --group dev ruff check .            # lint (selects E,W,F,I; line length 88)
uv run --group dev pytest                  # full root test suite
uv run --group dev pytest tests/test_api.py::test_name   # single test
uv build                                   # build the package

uv run project-course-api                  # FastAPI server (port 8000)
uv run project-course-api-dev              # FastAPI with hot reload
uv run project-course-camera list          # discover V4L2 devices
uv run project-course-camera probe --device 0 --backend v4l2 --fourcc YUYV --width 1280 --height 720 --fps 60
uv run python scripts/generate_demo_samples.py   # seed data/samples/ for the dashboard
```

### Frontend dashboard (`frontend/dashboard`)

```bash
npm install
npm run dev        # Vite dev server at 0.0.0.0:5173 (CORS-whitelisted by API)
npm run build      # tsc -b && vite build
npm run preview
```

### Global-camera laboratory (`laboratory/global-camera`)

Separate `uv` project — has its own `pyproject.toml` and lockfile.

```bash
uv sync
uv run global-camera-lab --device 2 --backend v4l2 --save-frame
uv run global-camera-qt-preview --device 2 --backend v4l2 --fourcc YUYV --width 1280 --height 720 --fps 60
```

### Motor-fault PCA project (`motor_fault_pca_project`)

Uses plain `pip` (Orange Pi target), not `uv`.

```bash
pip install -r requirements.txt              # offline train/test
pip install -r requirements-realtime.txt     # realtime camera + I2C
python src/run_all.py                        # train+test both branches
python src/realtime_detect.py --visual --vibration --vibration-source i2c
```

## High-level architecture

### FastAPI service (`src/project_course/api`)

- `app.py` — `create_app()` factory; lifespan: `db.init_db()` → mark orphaned-running tasks failed → `scan_directory()` ingests `data/samples/` → start `simulator_lifespan()`.
- `config.py` — `pydantic_settings` reads env vars with `PROJECT_COURSE_` prefix. Default window 0.5s/0.25s hop, IMU 1680 Hz, camera `YUY2_160x140_420fps`, analysis 420 fps.
- `storage/db.py` — SQLite at `data/project_course.sqlite3`. Three tables: `tasks`, `window_samples` (live per-window payload JSON), `history_samples` (offline-ingested CSV/Parquet metadata).
- `storage/ingest.py` — scans `data/samples/` for `.csv`/`.parquet`, validates required columns (`sample_id`, `window_index`, `center_time_s`), splits `vision_*` vs `sensor_*` populated columns, upserts into `history_samples`. Files must contain exactly one `sample_id`.
- `live/state.py` + `live/simulator.py` — `LIVE_STATE` is a single-task in-memory ring buffer (`settings.window_buffer_size`, default 240 ≈ 2 min at 0.5s hop). The simulator is the **default data source** when no real pipeline is attached; it produces synthetic per-window dual-modal features for a profile (normal/unbalance/loose/misaligned). Real feature pipelines should call `live.publish_window` and `live.record_sync_quality` — the same hooks the simulator uses.
- `routers/` — `tasks` (POST creates and **auto-stops any active task** — explicit "one-click reset" semantics), `dashboard` (live state), `history` (offline samples), `health`.

API base path: `/api/v1/`. CORS whitelist defaults to `http://localhost:5173` / `127.0.0.1:5173`.

### Dashboard (`frontend/dashboard/src`)

Routes (`App.tsx`): `/` LiveDashboardPage, `/history` HistoryListPage, `/history/:sampleId` HistoryDetailPage. Polls the FastAPI service via `axios`; renders charts with `echarts-for-react`. The dashboard expects feature rows to follow `doc/feature_schema.md`.

### Camera mainline (`src/project_course/camera`)

Reusable V4L2/OpenCV helpers — discovery (`v4l2.py`), capture config + probing (`core.py`), CLI (`cli.py`). The mainline flow is **list first, probe second**. Other backends (Qt preview) live under `laboratory/global-camera/` and reuse the same OpenCV setup; do **not** import laboratory code from `src/`.

### Sampling + fusion design (per `specs/001-dual-modal-monitoring/plan.md`)

- Hardware: USB UVC camera `YUY2 160x140@420fps`; STM32 six-axis IMU @ 1680 Hz over USB CDC `921600 bps` (≈62% serial utilization).
- Time sync: linear fit `t_host = a · t_imu_tick_us + b` over a sliding 4.0s window, re-fit every 1.0s; require R² ≥ 0.995 or flag window as `sync_fit_failed`. MCU tick is 32-bit @ 1 MHz with host-side unwrap.
- Window: default `window_size_s=0.25`, `window_hop_s=0.05` (legacy-validated); stable-demo mode `1.0`/`0.5`. Note `config.py` ships `0.5`/`0.25` as defaults — the plan and the API config differ on purpose; the API runs at the demo cadence and accepts overrides per-task.
- Fusion is a **join on `sample_id + window_index`**: build `vision_*` row, build `sensor_*` row, join, feed to RF/XGBoost/SVM. `fused_*` fields are display-only.
- Incremental update gate: ≥ 90 windows from ≥ 3 independent tasks before retraining.
- Backend rejects tasks whose STM32 handshake (`protocol_version`, `imu_sample_rate_hz`, `axis_order`, `tick_hz`, `frame_format`) disagrees with task config.

## Key conventions

- **Three Python environments, do not mix them.** Root uses `uv` with `pyproject.toml`; `laboratory/global-camera/` is its own `uv` project; `motor_fault_pca_project/` uses `pip`.
- **Schema-first.** New vision features must use `vision_*` prefix; sensor features `sensor_*`; identity fields (`sample_id`, `window_index`, `center_time_s`) must be preserved so fusion stays a join. See `doc/feature_schema.md`.
- **Laboratory ↔ mainline separation.** `laboratory/` is for exploration; do not import laboratory modules into `src/project_course`. Promote capabilities by reimplementing through clean interfaces.
- **Ruff config excludes** `laboratory/legacy/motion-amplifier` — that tree is archived.
- **Hardware assets** belong under `assets/hardware/{datasheets,wiring,calibration,reference-images}/`. Do not scatter them under `doc/` or `laboratory/`.
- **ROI is mandatory** for the archived motion-analysis pipeline — full-frame tracking captures global scene motion, not target vibration.
- **Don't trust container FPS** in the archived scripts; use `--fps-override` and derive the real rate from acquisition context (`sr600`/`sr400` in source video names).
- The archived experiment workflow sets `MPLCONFIGDIR=.mplconfig` and `UV_CACHE_DIR=.uv-cache` so caches stay in-project.
- `laboratory/legacy/motion-amplifier/scripts/analyze_guitar.py` is despite its name a **generic one-off video analyzer** — used for both guitar and car-engine experiments.

## Speckit

The active spec is `specs/001-dual-modal-monitoring/` (plan, research, data-model, contracts, tasks, quickstart). When the user asks about spec-driven work, that is the source of truth and `AGENTS.md` points to it. The `/speckit-plan` workflow fills in `plan.md` from `.specify/templates/`.

## Doc-context index

`SKILL.md` defines a project-specific skill: when archiving research material or summarizing a phase, run `python3 scripts/update_doc_context_index.py` to refresh `doc/context/key-context-index.md` before committing.
