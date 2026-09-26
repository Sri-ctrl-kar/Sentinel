# Sentinel frontend (M0.9)

The incident-intelligence command center: upload surveillance footage and watch
Sentinel reveal, in order, **what it sees**, **what is happening**, **what is
likely to happen next**, **why it believes that**, and **how much warning time
remains**.

It is a pure client of the M0.9 API. It computes nothing about perception,
tracking, risk or grounding — it renders what `app/api/` returns, and where a
field is absent it says so rather than filling the gap.

```
┌──────────────────────────────┬────────────────────────┐
│ SEE      video + track overlay│ UNDERSTAND  risk, TTR  │
│ EVENT TIMELINE (click → seek) │ PREDICT     incident   │
│ AI INTERPRETATION + grounding │ WHY THIS ALERT?        │
│                               │ LIFECYCLE / PROVENANCE │
└──────────────────────────────┴────────────────────────┘
```

## Install

```bash
cd frontend
npm install
```

Node 20+ (developed on Node 22).

## Run it

Two processes, independent of each other. **Backend** — from the repository
root:

```bash
pip install -r requirements-api.txt
uvicorn app.api.server:app --reload --port 8000
```

**Frontend** — from `frontend/`:

```bash
npm run dev          # http://127.0.0.1:5173
```

Open http://127.0.0.1:5173 and upload a video. Either process can be restarted
without the other; there is no proxy and no shared state.

## Commands

| Command | What it does |
| --- | --- |
| `npm run dev` | Vite dev server on :5173, hot reload |
| `npm run build` | Typecheck (`tsc --noEmit`) then production build to `dist/` |
| `npm run preview` | Serve the production build on :5173 |
| `npm test` | Run the test suite once (Vitest + Testing Library) |
| `npm run test:watch` | Watch mode |

## API base URL

The frontend talks to the API over CORS. The URL is read from Vite's
environment at build time:

```bash
cp .env.example .env.local     # then edit
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Default when unset: `http://127.0.0.1:8000`. If the API runs on another host or
port, set this and rebuild (`npm run build`) or restart `npm run dev`.

The API allows `http://localhost:5173` and `http://127.0.0.1:3000` (and the
`:3000` equivalents) by default. Serving the frontend from a different origin
means telling the API about it:

```bash
SENTINEL_API_CORS_ORIGINS=https://sentinel.example uvicorn app.api.server:app
```

## How it talks to FastAPI

`src/api/client.ts` is the only module that knows about HTTP. Every response
type in `src/api/types.ts` mirrors `app/api/schemas.py` field for field.

| Endpoint | Used for |
| --- | --- |
| `GET /api/health` | connection chip in the header |
| `GET /api/device` | device chip; AMD ROCm / HIP shown when reported |
| `POST /api/analyze` | multipart upload → `analysis_id` (202) |
| `GET /api/analyze/{id}` | polled until `complete` or `failed`; progress, pipeline summary, error |
| `GET /api/analyze/{id}/events` | the event timeline and entity states |
| `GET /api/analyze/{id}/risk` | risk timeline, worst moment, score disclaimer |
| `GET /api/analyze/{id}/timeline` | per-frame tracks for the overlay |
| `GET /api/analyze/{id}/incident` | evidence → explanation → grounding |
| `GET /api/analyze/{id}/video` | the original upload, played in a `<video>` |

`src/hooks/useAnalysis.ts` owns the lifecycle: `idle → uploading → queued →
running → loading → complete | failed`. Polling stops on a terminal state.

## What it will not do

* **Invent data.** Missing time-to-risk reads "unavailable" with the engine's
  own reason. A clip with no incident says so. Absent metadata shows "—" or
  "not reported", never a plausible default.
* **Turn a risk score into a probability.** The score is rendered as `n/100`
  with the disclaimer the API sends beside it.
* **Show a prediction as an event.** The tense badge is driven by the engine's
  `time_to_risk.status` and lifecycle state: `PREDICTED — has not happened`
  versus `CURRENT — happening now`.
* **Draw on the video.** The server returns the original bytes; boxes are SVG
  drawn client-side from timeline coordinates.
* **Fake progress.** The bar is proportional only when the API reports a
  percentage; otherwise it is indeterminate and the frame count is shown.

## Video playback

The player needs a codec the browser can decode — in practice **H.264/MP4**.
Sentinel's pipeline reads more than that (anything OpenCV opens), so a clip can
analyse perfectly and still not render: the clips produced by
`scripts/generate_demo_video.py` and `scripts/make_benchmark_clip.py` use
`mp4v`, which Chrome will not play. When that happens the stage says so and the
rest of the dashboard — overlay, risk, timeline, evidence — keeps working, with
the scrubber driving the analysis clock.

## Tests

```bash
npm test
```

90 tests over five files, no network and no backend required — the fixtures in
`src/test/fixtures/` were **captured from the real API** rather than written by
hand.

| File | Covers |
| --- | --- |
| `format.test.ts` | severity → state mapping, missing-value formatting, predicted-vs-current logic |
| `timeline.test.ts` | frame lookup, step-function risk selection, marker building |
| `panels.test.tsx` | overlay geometry, risk states, time-to-risk variants, evidence, lifecycle, provenance, AI narrative, unplayable video |
| `EventTimeline.test.tsx` | marker rendering, click-to-seek, accessible names |
| `App.test.tsx` | upload, polling through queued → running → complete, failure, missing fields, seeking, AMD device chip |

## Accessibility

Every panel is a labelled `section`. Severity and lifecycle are conveyed by
text as well as colour. Timeline markers are real buttons with visually-hidden
names. Controls are keyboard reachable with a visible focus ring, and
`prefers-reduced-motion` disables the transitions.

## Layout

Desktop first — a two-column workspace at ≥1180px that collapses to a single
column below it, tested down to 1280×880.
