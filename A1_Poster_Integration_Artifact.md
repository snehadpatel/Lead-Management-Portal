# A1 Poster — What to Keep (Integration Artifact)

**Audience:** Industry evaluators · **Goal:** Show **one integrated system**, not three separate assignments  
**Format:** A1 · **Text rule:** concise **bullet points only** (no paragraphs)  
**Visual rule:** aim for **~60% visuals + diagrams**, **~20% text**, **~20% white space**

**Template:** use `A1_Studio_Project_Poster_Template.docx` as the layout shell.

---

## 1. Global rules (apply to whole poster)

- **One hero diagram** dominates the board — the **tri-domain architecture** (see §4).
- **Every technical claim** in Results ties to a **figure, metric, or screenshot** (prefer plots over words).
- **Consistent color coding** across the poster (example): green = Big Data / data platform, orange = AI/ML, blue = deployment / cloud / APIs.
- **Readable from ~1–2 m:** title ~34–40 pt, section headings ~24–30 pt, body ~16–20 pt.
- **Cap bullets:** ~4–6 bullets per section box; if it does not fit, **cut text** or **merge into a diagram**.

---

## 2. How to demonstrate “integration of three domains”

Evaluators look for **explicit links**, not a list of topics.

Keep these **three integration anchors** visible:

| Anchor | What to show |
|--------|----------------|
| **Data → model** | Same curated layer (e.g. lake/Parquet/features) feeds **multiple** models or analytics |
| **Model → service** | Trained artifacts power **API / UI / batch scoring** — not “notebook only” |
| **Validation → trust** | Metrics + plots prove the pipeline works **end-to-end** for the stated problem |

Optional **one-line caption** under the architecture:  
`Sources → scalable processing → model layer → deployed prototype → measured outcomes`

---

## 3. Section-by-section — what to keep

### Header section

**Purpose:** Identity + credibility in 5 seconds.

**Keep (bullets):**

- Official **project title** (specific, not generic).
- **Course integration** line (e.g. AI + Big Data Analytics + deployment/cloud strand).
- **Team names**, **enrollment numbers**, **guide/mentor**, **institution/program**.
- **Optional:** QR to repo / demo / short video; **institution logo**.

**Visual:** logo strip + QR; avoid clutter.

---

### Introduction & problem statement

**Purpose:** Why this project exists — **business + technical** problem.

**Keep (bullets):**

- **Domain context** (1–2 bullets: who suffers, what system/process).
- **Problem** — scale, silos, latency, cost, or decision quality (pick **one main pain**).
- **Scope** — data types involved (tabular / text / time-series / logs).
- **Objectives** — 3 bullets max, each **action verbs** (ingest, model, deploy, validate).

**Visual:** tiny schematic of **data modalities** or stakeholder → system arrow.

---

### Tri-course integration architecture (**main focus**)

**Purpose:** Prove integration — **largest panel on the poster**.

**Keep (diagram + minimal labels):**

- **Big Data Analytics layer:** ingestion, storage/format, processing (batch/stream), quality/feature steps you **actually** use.
- **Artificial Intelligence layer:** named tasks (e.g. classify / cluster / NLP / forecast) → **named model families** (not a lecture).
- **Third domain — deployment / cloud:** APIs, containers, orchestration, runtime (Colab/local/cloud), observability (**health** endpoint counts).

**Avoid:** dumping tool logos without arrows; disconnected boxes with no data flow.

**Visual budget:** this section alone should consume **~35–50%** of poster area (diagram + arrows + legend).

---

### Methodology & prototype implementation

**Purpose:** Show **how** you built it — reproducible engineering path.

**Keep (bullets):**

- **Data path:** where data lives → how it moves → intermediate outputs (e.g. Parquet, features).
- **Training path:** command-level reality (“train job / script / notebook”) — **one** line each for train vs inference if different.
- **Prototype path:** API + dashboard / Docker / Compose — what an evaluator **could run or see**.
- **Reproducibility:** fixed metrics source (JSON/logs), versioning, or single “ground truth” eval folder — **one** bullet.

**Visual:** horizontal **4-step** strip: *Ingest → Transform → Train → Serve* aligned under the architecture.

---

### Results & validation

**Purpose:** Evidence for industry reviewers — **numbers + pictures**.

**Keep:**

- **Table or small chart** of **primary KPIs** per model/task (accuracy/F1/AUC, silhouette, RMSE, etc. — match what you actually computed).
- **2–4 plots:** confusion matrix, ROC or PR, cluster projection, forecast overlay, dashboard screenshot.
- **Validation method** in **2 bullets max:** train/test or CV; what split or leakage controls you respected.

**Avoid:** orphan metrics with no plot; claims that are not on the poster as a figure or table.

**Visual:** this is the second-largest visual block after architecture.

---

### Conclusion & future scope

**Purpose:** Close the story — **outcomes + maturity path**.

**Keep (bullets):**

- **3 outcomes** tied to integration (e.g. “unified lake + multi-model + served API”).
- **1 bullet** on limitations or assumptions (honest — builds trust).
- **3 future bullets:** streaming/real-time, MLOps/registry, scaling, security, cost, or better models.

**Visual:** three checkmarks or “now → next” mini roadmap.

---

## 4. Suggested layout on A1 (60 / 20 / 20)

| Zone | Rough share | Content |
|------|-------------|---------|
| Center | **40–50%** | Tri-course architecture diagram |
| Right or below arch | **15–25%** | Results plots + KPI mini-table |
| Left column | **10–15%** | Intro + methodology strips |
| Bottom strip | **8–12%** | Conclusion + future |
| Margins & gutters | **15–20%** | Intentional whitespace |

---

## 5. Pre-print checklist

- [ ] Architecture shows **continuous flow** across **three** domains (not three isolated lists).
- [ ] Every model/task has **≥1 metric** and **≥1 visual**.
- [ ] Prototype is visible (**API / UI / container** — screenshot counts).
- [ ] No paragraph blocks; **bullets only**.
- [ ] Font sizes readable at distance; **title** stands out.
- [ ] Colors mean the same thing everywhere (legend on architecture).
- [ ] Names, enrollments, mentor, title are **correct** in the header.

---

## 6. Project-specific paste content (optional)

For the **Lume AI** codebase, pre-filled bullets, metrics, and figure paths live in:

`A1_Studio_Project_Poster_Content.md`

Use that file for **exact text and numbers**; use **this file** as the **rubric and layout contract** while you fill `A1_Studio_Project_Poster_Template.docx`.
