# ◈ GOIES — Geopolitical Open Intelligence & Extraction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-ff6b35?style=flat-square)
![NetworkX](https://img.shields.io/badge/NetworkX-3.2+-ff7043?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-00d4ff?style=flat-square)

**AI-powered geopolitical intelligence platform.**
Transforms raw text, PDFs, URLs and RSS feeds into an interactive, queryable knowledge graph.
Runs entirely offline. No data ever leaves your machine.

[Quick Start](#-quick-start) · [Features](#-features) · [API Reference](#-api-reference) · [Deploy](#-deployment) · [Architecture](#-architecture)

</div>

---

## Table of Contents

1. [What is GOIES?](#1-what-is-goies)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [File Structure](#4-file-structure)
5. [Quick Start](#5-quick-start)
6. [Docker](#6-docker)
7. [Deployment — Railway](#7-deployment--railway)
8. [Intelligence Pipeline](#8-intelligence-pipeline)
9. [Knowledge Graph Engine](#9-knowledge-graph-engine)
10. [Geo-Positional System](#10-geo-positional-system)
11. [Simulation Engine](#11-simulation-engine)
12. [Crisis Forecasting Engine](#12-crisis-forecasting-engine)
13. [Strategic Analyst Chat (GraphRAG)](#13-strategic-analyst-chat-graphrag)
14. [OSINT Engine](#14-osint-engine)
15. [Semantic Embeddings](#15-semantic-embeddings)
16. [Graph Query Language (GQL)](#16-graph-query-language-gql)
17. [Report Generation](#17-report-generation)
18. [API Reference](#18-api-reference)
19. [Environment Variables](#19-environment-variables)
20. [Frontend — Dashboard](#20-frontend--dashboard)
21. [Roadmap](#21-roadmap)
22. [Contributing](#22-contributing)

---

## 1. What is GOIES?

GOIES is an **open-source geopolitical intelligence platform** that transforms unstructured text — news articles, diplomatic cables, OSINT reports, RSS feeds — into an interactive, queryable knowledge graph.

It enables analysts to:

- **Extract** entity networks (countries, people, organizations, treaties, events, resources, technology) from any text source using a local LLM
- **Visualize** the resulting network in an interactive graph or as a live world tension map
- **Query** the graph in plain English via a GraphRAG analyst chat interface
- **Simulate** policy scenarios and model cascading geopolitical effects
- **Forecast** crises using structural graph analysis combined with LLM reasoning
- **Monitor** the world through RSS feeds, GDELT, and Wikipedia enrichment
- **Export** the graph as JSON, CSV, GraphML, PDF reports, or Markdown briefs

The system is designed to run **entirely locally** — no data leaves your machine. All LLM inference runs via [Ollama](https://ollama.com). Privacy is a first-class constraint.

---

## 2. Features

### Core Intelligence
| Feature | Description |
|---|---|
| **Entity Extraction** | 7 entity classes with confidence scores — country, person, organization, technology, event, treaty, resource |
| **Relationship Extraction** | Typed, directed edges extracted simultaneously with entities |
| **Chunked Processing** | Sentence-boundary-aware chunking with 200-char overlap for long documents |
| **Fuzzy Deduplication** | SequenceMatcher-based entity resolution prevents "US", "USA", "United States" becoming 3 nodes |
| **Streaming Extraction** | Server-Sent Events (SSE) streaming — watch entities appear in real-time |
| **Multi-Source Ingestion** | Paste text · URL scraping · PDF upload · DOCX upload · RSS feeds |

### Graph Engine
| Feature | Description |
|---|---|
| **Interactive Graph** | vis.js network with physics, grouping, ego subgraphs |
| **Graph Analytics** | Degree centrality, betweenness centrality, density, connected components |
| **Path Finder** | Shortest path between any two nodes |
| **Node Merge** | Deduplicate nodes with canonical name resolution |
| **Graph Versioning** | Timestamped snapshots with timeline replay |
| **Export** | JSON · CSV · GraphML (Gephi/Cytoscape compatible) |

### Intelligence Analysis
| Feature | Description |
|---|---|
| **Policy Simulation** | Two-pass LLM pipeline: parse scenario → apply to graph copy → cascade analysis |
| **Crisis Forecasting** | Structural signal detection (hotspots, reciprocal hostility, instability triangles) + LLM narrative |
| **GraphRAG Chat** | BFS-expanded graph context + LLM for grounded, citation-backed answers |
| **Narrative Brief** | Auto-generated 3-paragraph intelligence summary of current graph state |

### Geo & OSINT
| Feature | Description |
|---|---|
| **Tension World Map** | Leaflet.js dark map with tension-scored country markers |
| **OSINT RSS Feeds** | Manage and ingest RSS/Atom feeds with deduplication |
| **GDELT Integration** | Query GDELT global news event database by entity |
| **Wikipedia Enrichment** | Auto-enrich nodes with Wikipedia background data |

### Semantic Intelligence
| Feature | Description |
|---|---|
| **Node2Vec Embeddings** | Train graph embeddings for structural similarity search |
| **Semantic Search** | Find entities by meaning, not just exact name |
| **K-Means Clustering** | Auto-cluster the graph into semantic community groups |
| **Similar Nodes** | Find structurally analogous entities (who acts like Iran?) |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GOIES — Full Stack                            │
├──────────────────┬────────────────────────┬─────────────────────────┤
│  Frontend (SPA)  │   FastAPI Backend       │   Intelligence Modules  │
│                  │                         │                         │
│  index.html      │  server.py (889 lines)  │  extractor.py    (231)  │
│  app.html        │                         │  simulator.py    (245)  │
│  js/api.js       │  /api/*  (40+ routes)   │  forecaster.py   (354)  │
│                  │                         │  geo.py          (260)  │
│  vis.js          │  FastAPI + uvicorn      │  query_engine.py (446)  │
│  Leaflet.js      │  Pydantic v2            │  osint_engine.py (434)  │
│  Chart.js        │  Async SSE streaming    │  embedding_engine(262)  │
│  Tom Select      │                         │  graph_algo.py          │
│                  │                         │  ingestor.py            │
│                  │                         │  reporter.py            │
│                  │                         │  utils.py        (358)  │
└────────┬─────────┴──────────┬─────────────┴──────────┬──────────────┘
         │                    │                          │
         ▼                    ▼                          ▼
  ┌─────────────┐   ┌──────────────────┐     ┌────────────────────┐
  │  Static     │   │  NetworkX Graph  │     │  Ollama REST API   │
  │  Assets     │   │  (in-memory +    │     │  :11434            │
  │  (vendored) │   │   JSON persist)  │     │  llama3.2 / any    │
  └─────────────┘   └──────────────────┘     └────────────────────┘
```

### Two Deployment Modes

| Mode | Stack | Use Case |
|---|---|---|
| **Full Stack** | FastAPI + uvicorn · Python backend · all features | Teams, production, full analysis workflows |
| **Local Dev** | Same, but with `--reload` and direct Ollama on localhost | Development, personal use |

---

## 4. File Structure

```
GOIES/
│
├── server.py              ← FastAPI backend — all /api/* routes (889 lines)
├── extractor.py           ← LLM entity + relationship extraction engine (231)
├── utils.py               ← Graph helpers, chunking, fuzzy resolution, analytics (358)
├── graph_algo.py          ← Pathfinding, centrality, ego subgraphs
├── geo.py                 ← Geo-tension scoring, 80+ country coordinates (260)
├── simulator.py           ← Two-pass LLM policy simulation engine (245)
├── forecaster.py          ← Crisis forecasting: structural signals + LLM (354)
├── query_engine.py        ← GQL parser + executor (446)
├── embedding_engine.py    ← Node2Vec embeddings, semantic search, clustering (262)
├── osint_engine.py        ← RSS ingestion, GDELT, Wikipedia enrichment (434)
├── ingestor.py            ← PDF / DOCX / URL text extraction
├── reporter.py            ← PDF and Markdown intelligence report generation
│
├── requirements.txt       ← All Python dependencies (complete)
├── Dockerfile             ← Production container (Python 3.11-slim)
├── docker-compose.yml     ← Local dev stack: backend + Ollama sidecar
├── railway.toml           ← Railway.app one-click deployment config
├── .gitignore             ← Excludes __pycache__, runtime data, .env
├── .dockerignore
│
├── frontend/              ← All frontend assets (served by FastAPI at /)
│   ├── index.html         ← Landing page          → http://localhost:8000/
│   ├── app.html           ← Full dashboard         → http://localhost:8000/app.html
│   ├── js/
│   │   └── api.js         ← Centralized API client (40+ endpoint wrappers, ESM)
│   └── lib/               ← Vendored JS — works fully offline
│       ├── vis-9.1.2/     ← vis-network graph renderer
│       ├── tom-select/    ← Enhanced dropdowns
│       └── bindings/      ← vis.js utility bindings
│
├── goies_graph.json       ← Persisted graph state  [gitignored — runtime data]
├── goies_snapshots/       ← Auto-created timestamped graph versions [gitignored]
│
└── specs/                 ← Architecture & roadmap documentation
    ├── project.md         ← Full production architecture spec (Vol. 1)
    └── IMPROVEMENTS.md    ← Feature roadmap (Vol. 2 — 30 planned enhancements)
```

---

## 5. Quick Start

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running

### Install

```bash
# Clone the repo
git clone https://github.com/tanu-1403/GOIES.git
cd GOIES

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Pull the LLM

```bash
ollama pull llama3.2
```

> Any Ollama-compatible model works. Larger models (llama3.2:70b, mistral-large) produce better extractions.

### Run

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Open

| Page | URL |
|---|---|
| **Landing page** | http://localhost:8000/ |
| **Dashboard** | http://localhost:8000/app.html |
| **API docs** | http://localhost:8000/api/docs |

---

## 6. Docker

Run GOIES and Ollama together with a single command.

```bash
# Start both services
docker-compose up -d --build

# Pull the model inside the Ollama container (one-time, ~2GB download)
docker exec -it goies-ollama ollama pull llama3.2

# Open http://localhost:8000
```

### docker-compose.yml overview

```yaml
services:
  goies-backend:
    build: .
    ports: ["8000:8000"]
    environment:
      - OLLAMA_HOST=http://ollama:11434   # internal Docker network
    depends_on: [ollama]
    volumes:
      - ./goies_snapshots:/app/goies_snapshots   # persist snapshots

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama        # persist downloaded models
```

### Useful Docker commands

```bash
# View backend logs
docker logs goies-backend -f

# List available models
docker exec -it goies-ollama ollama list

# Pull additional model
docker exec -it goies-ollama ollama pull mistral

# Stop everything
docker-compose down

# Stop and wipe volumes (removes downloaded models)
docker-compose down -v
```

---

## 7. Deployment — Railway

Railway is the recommended cloud platform. It supports Docker natively, provides persistent volumes, and auto-deploys from GitHub on every push.

### Full Step-by-Step

**Step 1 — Push to GitHub**
```bash
cd GOIES
git init && git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/GOIES.git
git push -u origin main
```

**Step 2 — Create Railway project**

1. Go to **[railway.app](https://railway.app)** → Login with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `GOIES` repository
4. Railway auto-detects the `Dockerfile` → click **Deploy Now**

**Step 3 — Add Ollama service**

1. Inside your project, click **+ New Service** → **Docker Image**
2. Enter: `ollama/ollama:latest` → Deploy
3. Click the Ollama service → **Settings → Networking**
4. Note the **internal hostname** (e.g., `ollama.railway.internal`)

**Step 4 — Set environment variable**

1. Click your GOIES backend service → **Variables** tab
2. Add:
   ```
   OLLAMA_HOST = http://ollama.railway.internal:11434
   ```
3. Railway auto-redeploys.

**Step 5 — Add persistent volume**

1. Click GOIES backend → **Volumes** tab
2. Click **Mount a Volume** → set path: `/app/goies_snapshots`
3. Click **Create**

**Step 6 — Pull the LLM (one-time)**

1. Click the Ollama service → **Shell** (or use Railway CLI)
2. Run:
   ```bash
   ollama pull llama3.2
   ```

**Step 7 — Get your public URL**

1. Click GOIES backend → **Settings → Networking → Public Networking**
2. Click **Generate Domain**
3. Your app is live at `https://goies-production.up.railway.app`

### Cost Estimate

| Service | Monthly |
|---|---|
| GOIES backend (Hobby) | ~$5 |
| Ollama service (RAM-heavy) | ~$10–15 |
| **Total** | **~$15–20/mo** |

Railway provides **$5 free credit** to test before committing.

---

## 8. Intelligence Pipeline

### Entity Classes

| Class | Examples |
|---|---|
| `country` | Russia, Taiwan, European Union |
| `person` | Putin, Zelensky, Rafael Grossi |
| `organization` | NATO, IAEA, Wagner Group, IMF |
| `technology` | Javelin missile, Starlink, S-400 |
| `event` | Ukraine invasion, Taiwan Strait exercises |
| `treaty` | NPT, START-3, AUKUS agreement |
| `resource` | Black Sea grain, Nord Stream, Uranium stockpile |

### Extraction Flow

```
Text input (paste / URL / PDF / RSS)
        ↓
Sentence-boundary chunking (max 4000 chars, 200-char overlap)
        ↓
Per-chunk Ollama prompt → structured JSON
        ↓
Confidence filtering (drop < 0.5)
        ↓
Fuzzy entity resolution (SequenceMatcher, threshold 0.82)
        ↓
Graph update (add nodes + edges, merge duplicates)
        ↓
Analytics recompute (centrality, density, tension scores)
        ↓
SSE event stream → live UI update
```

### Extraction Prompt Schema

```json
{
  "entities": [
    {
      "id": "Russia",
      "group": "country",
      "confidence": 0.94,
      "attributes": {"capital": "Moscow", "leader": "Putin"}
    }
  ],
  "relationships": [
    {
      "from": "Russia",
      "to": "Ukraine",
      "label": "invades",
      "confidence": 0.97
    }
  ]
}
```

### Ingestion Sources

| Source | Endpoint | Status |
|---|---|---|
| Plain text paste | `POST /api/extract` | ✅ |
| URL scraping | `POST /api/ingest/url` | ✅ |
| PDF upload | `POST /api/ingest/file` | ✅ |
| DOCX upload | `POST /api/ingest/file` | ✅ |
| TXT / MD upload | `POST /api/ingest/file` | ✅ |
| SSE streaming | `POST /api/extract/stream` | ✅ |
| RSS feed | `POST /api/osint/ingest` | ✅ |
| GDELT query | `GET /api/osint/gdelt` | ✅ |
| Wikipedia enrichment | `POST /api/osint/enrich/{node_id}` | ✅ |

---

## 9. Knowledge Graph Engine

### Graph Data Model

**Node schema:**
```json
{
  "id": "Russia",
  "group": "country",
  "confidence": 0.94,
  "attributes": {"capital": "Moscow", "leader": "Putin"},
  "ingested_at": "2026-03-09T11:00:00Z",
  "source_count": 12,
  "tension_score": 84.3
}
```

**Edge schema:**
```json
{
  "source": "Russia",
  "target": "Ukraine",
  "label": "invades",
  "confidence": 0.97,
  "ingested_at": "2026-03-09T11:00:00Z"
}
```

### Graph Analytics

Computed automatically after every extraction via `utils.get_graph_analytics()`:

| Metric | Algorithm |
|---|---|
| Degree centrality | `networkx.degree_centrality` |
| Betweenness centrality | `networkx.betweenness_centrality` |
| Connected components | `networkx.weakly_connected_components` |
| Graph density | `2E / N(N-1)` |
| Group breakdown | Counts per entity class |
| Top degree nodes | Sorted degree dictionary |
| Top betweenness nodes | Sorted betweenness dictionary |

### Entity Group Colors

| Group | Color | Hex |
|---|---|---|
| country | Red-pink | `#ff7b72` |
| person | Orange | `#ffa657` |
| organization | Purple | `#d2a8ff` |
| technology | Blue | `#79c0ff` |
| event | Green | `#7ee787` |
| treaty | Yellow | `#f0e68c` |
| resource | Lime | `#56d364` |
| unknown | Grey | `#8b949e` |

### Export Formats

| Format | Endpoint | Use Case |
|---|---|---|
| JSON | `GET /api/export/json` | Full backup, re-import |
| CSV | `GET /api/export/csv` | Excel, pandas analysis |
| GraphML | `GET /api/export/graphml` | Gephi, Cytoscape, yEd |

---

## 10. Geo-Positional System

### Tension Score Algorithm

Each country node receives a tension score (0–100) based on its graph position:

```python
score = 0

# Outgoing hostile edges (aggressor)
for edge in outgoing_edges(country):
    score += edge_score(edge.label) * 1.2   # aggressor weighting

# Incoming hostile pressure (target)
for edge in incoming_edges(country):
    score += edge_score(edge.label) * 0.9   # target weighting

# Military event bonus
for event_node in connected_events(country):
    if is_military(event_node):
        score += 7.0

# Degree centrality multiplier (more connections = more weight)
score *= (1.0 + 0.04 * min(degree(country), 20))

final_score = normalize_to_100(score)
```

### Edge Score Vocabulary

| Score | Keywords |
|---|---|
| `+18` | sanction, attack, invade, bomb, missile, strike, kill, threaten, blockade, terrorize |
| `+9` | restrict, ban, expel, dispute, tension, pressure, cyber, confront |
| `−3` | cooperate, ally, partner, invest, aid, support, trade, treaty |
| `+2` | (default / unknown relationship) |

### Tension Color Bands

| Score | Color | Label |
|---|---|---|
| 75–100 | `#ff2244` | Critical |
| 50–75 | `#ff6b35` | High |
| 25–50 | `#ffaa40` | Medium |
| 10–25 | `#ffe066` | Low |
| 0–10 | `#00ff88` | Peaceful |

---

## 11. Simulation Engine

### How It Works

```
User: "The US lifts all sanctions on Iran"
         ↓
Pass 1 — LLM parses scenario into structured mutations:
  {
    "changes": [
      {"action": "remove_edge", "from": "US", "to": "Iran", "label": "sanctions"},
      {"action": "add_edge", "from": "US", "to": "Iran", "label": "diplomatic talks"}
    ],
    "base_risk": 35
  }
         ↓
Changes applied to a COPY of the live graph (live graph never mutated)
         ↓
Pass 2 — LLM cascade analysis using modified graph context:
  {
    "cascade_narrative": "...",
    "second_order_effects": ["Israel escalates", "Gulf states reassess", "..."],
    "risk_adjustment": +5
  }
         ↓
Output: SimulationResult with risk score, narrative, second-order effects
```

### Risk Scoring

| Label | Range | Color |
|---|---|---|
| LOW | 0–24 | `#00ff99` |
| MEDIUM | 25–49 | `#ffb347` |
| HIGH | 50–74 | `#ff6b35` |
| CRITICAL | 75–100 | `#ff3355` |

### Simulation Result Schema

```json
{
  "scenario": "US lifts all sanctions on Iran",
  "risk_score": 38.5,
  "risk_label": "MEDIUM",
  "cascade_narrative": "The removal of US sanctions would fundamentally alter...",
  "second_order": [
    "Israel likely to escalate military exercises near Iranian border",
    "Gulf states reassess US security guarantees",
    "Iranian oil exports resume, depressing global crude by 8-12%"
  ],
  "added_edges": [{"from": "US", "to": "Iran", "label": "diplomatic talks"}],
  "removed_edges": [{"from": "US", "to": "Iran", "label": "sanctions"}],
  "affected_nodes": ["US", "Iran", "Israel", "Saudi Arabia", "IAEA"],
  "model_used": "llama3.2"
}
```

---

## 12. Crisis Forecasting Engine

### Structural Signal Detection

Before any LLM call, the forecaster analyzes graph structure for objective warning signs:

| Signal | Detection Method |
|---|---|
| **Hotspot nodes** | Nodes with the highest count of hostile outgoing edges |
| **Reciprocal hostility** | A→B hostile AND B→A hostile simultaneously |
| **Instability triangles** | Three actors where 2 edges are hostile + 1 cooperative |
| **Conflict brokers** | High betweenness nodes embedded in hostile clusters |

### Instability Triangle Logic

A triangle of actors is **structurally unstable** when:
- 2 edges are hostile (sanctions, attacks, threatens)
- 1 edge is cooperative (ally, partner, trade)

The cooperative relationship is under stress from both sides — a classic balance-theory instability pattern.

### Forecast Output Schema

```json
{
  "global_risk": 67.0,
  "global_label": "HIGH",
  "hotspot_nodes": ["Russia", "Iran", "North Korea", "China", "Israel"],
  "forecasts": [
    {
      "rank": 1,
      "title": "Taiwan Strait Military Crisis",
      "actors": ["China", "Taiwan", "US"],
      "probability": 0.73,
      "severity": "CRITICAL",
      "timeframe": "near-term (0-3 months)",
      "structural_signal": "Reciprocal hostility between China and Taiwan with US as high-betweenness broker",
      "narrative": "Structural analysis reveals China and Taiwan in mutual hostile relationship...",
      "mitigation": "Activate US-China hotline; ASEAN mediation track recommended."
    }
  ]
}
```

---

## 13. Strategic Analyst Chat (GraphRAG)

### Architecture

```
User: "How are Russia and NATO connected?"
         ↓
1. Keyword matching against graph nodes
2. BFS expansion — 2 hops from matched nodes
3. Collect up to 25 edges as context
         ↓
LLM prompt:
  "You are a geopolitical analyst.
   Graph context: [BFS edges — 25 relationships]
   Question: [user query]
   Grounded answer:"
         ↓
Response streamed to chat window
```

All answers are grounded in the actual graph data — the LLM cannot hallucinate entities or relationships that don't exist in the current graph.

### Query Examples

```
"Who is most central to the Ukraine conflict?"
→ Structured answer citing degree + betweenness from graph

"What is the path from the US to North Korea?"
→ US → sanctions → North Korea (direct)
   US → South Korea → [ally] → South Korea → [border tension] → North Korea (2-hop)

"Which countries are at highest risk of escalation?"
→ Ranked list with structural justification from graph

"Summarize Russia's connections"
→ 18 direct connections enumerated with relationship types
```

---

## 14. OSINT Engine

### RSS Feed Management

```bash
# Add a feed
POST /api/osint/feeds
{ "url": "https://www.reuters.com/world/rss" }

# List all feeds
GET /api/osint/feeds

# Trigger ingestion (background task)
POST /api/osint/ingest

# Remove a feed
DELETE /api/osint/feeds?url=https://...
```

The OSINT engine tracks processed article URLs in `processed_urls.json` to prevent duplicate ingestion across restarts.

### GDELT Integration

Query the [GDELT](https://www.gdeltproject.org) global news event database for any entity in the graph:

```bash
GET /api/osint/gdelt?entity=Russia&days=7
```

Returns the top 20 recent news articles from 100+ countries' media mentioning the entity, automatically ingested into the graph.

### Wikipedia Enrichment

Enrich any node with background context from Wikipedia:

```bash
POST /api/osint/enrich/{node_id}
```

Extracts structured attributes: capital, leader, population, GDP, military budget, alliance memberships, founded year, and a 200-character summary — all added to the node's attribute set.

---

## 15. Semantic Embeddings

The embedding engine (powered by [Node2Vec](https://snap.stanford.edu/node2vec/)) trains graph embeddings that capture structural position, not just name similarity.

### Train Embeddings

```bash
POST /api/embed/train
{ "dimensions": 64, "walk_length": 30, "num_walks": 200 }
```

Training runs in the background. Check status:

```bash
GET /api/embed/status
```

### Semantic Search

```bash
GET /api/embed/search?q=nuclear+program&k=5
```

Returns the 5 nodes whose structural position in the graph is most similar to nodes associated with the query.

### Similar Nodes

```bash
GET /api/embed/similar/Iran?k=5
```

Returns the 5 nodes most structurally similar to Iran — not just connected to it, but playing a similar structural role in the network. May return: North Korea, Venezuela, Cuba — all with similar patterns of sanctions, isolation edges, and adversarial relationships.

### K-Means Clustering

```bash
GET /api/embed/clusters?n=5
```

Automatically clusters the graph into N semantic community groups based on structural position. Returns cluster assignments with suggested group names.

---

## 16. Graph Query Language (GQL)

GOIES includes a domain-specific query language for deterministic, exact-answer queries against the graph.

### Syntax Reference

```bash
GET /api/gql/help    # Full syntax reference
POST /api/gql        # Execute a query
{ "query": "find countries that sanction Iran" }
```

### Example Queries

```
find countries
→ All country nodes

find countries that sanction Iran
→ [US, EU, UK, Canada, Australia, ...]

show path from China to NATO
→ China → [trade] → Germany → [member of] → NATO

list edges where label contains missile
→ [North Korea → [test-fired missile] → Pacific Ocean, ...]

nodes with degree > 8
→ [Russia (18), US (15), China (14), ...]

count relationships between Russia and Ukraine
→ 7 edges

ego of Iran hops 1
→ [US, Israel, Saudi Arabia, IAEA, Hezbollah, Russia]
```

Unknown queries fall back to LLM interpretation automatically.

---

## 17. Report Generation

Generate professional intelligence reports from the current graph state.

### PDF Report

```bash
POST /api/report
{
  "format": "pdf",
  "entities": ["Russia", "Ukraine", "NATO", "China"],
  "include_graph": true,
  "include_forecast": true
}
```

Returns a downloadable PDF brief with: executive summary, entity profiles, relationship network diagram, geo tension assessment, and crisis forecasts.

### Markdown Brief

```bash
POST /api/report
{ "format": "md" }
```

Returns a structured Markdown document suitable for pasting into Notion, Obsidian, or any document editor.

### Narrative Summary

```bash
GET /api/narrative/summary?model=llama3.2
```

Returns a 3-paragraph natural language intelligence summary of the entire current graph state.

---

## 18. API Reference

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/extract` | Extract intelligence from text |
| `POST` | `/api/extract/stream` | Streaming extraction (SSE) |
| `POST` | `/api/ingest/url` | Fetch and extract from URL |
| `POST` | `/api/ingest/file` | Upload PDF / DOCX / TXT / MD |

### Graph

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/graph` | Full graph in vis.js format + analytics |
| `GET` | `/api/graph?ego={node}&hops={n}` | Ego subgraph |
| `DELETE` | `/api/graph` | Clear entire graph |
| `GET` | `/api/path?src={a}&tgt={b}` | Shortest path between two nodes |
| `POST` | `/api/node/merge` | Merge two nodes into one |
| `GET` | `/api/export/{fmt}` | Export as json / csv / graphml |

### Intelligence

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/query` | Natural language GraphRAG query |
| `POST` | `/api/simulate` | Policy scenario simulation |
| `POST` | `/api/forecast` | Crisis forecast |
| `GET` | `/api/narrative/summary` | Auto-generated graph brief |
| `POST` | `/api/report` | Generate PDF or Markdown report |

### Geo

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/geo` | Geo tension markers (Leaflet format) |

### Snapshots

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/snapshots` | List all snapshots |
| `GET` | `/api/snapshots/timeline` | Timeline data for slider |
| `GET` | `/api/snapshots/{id}` | Load a specific snapshot |

### OSINT

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/osint/status` | OSINT engine status |
| `GET` | `/api/osint/feeds` | List RSS feeds |
| `POST` | `/api/osint/feeds` | Add RSS feed |
| `DELETE` | `/api/osint/feeds?url={url}` | Remove feed |
| `POST` | `/api/osint/ingest` | Trigger background RSS ingestion |
| `POST` | `/api/osint/enrich/{node_id}` | Wikipedia enrichment |
| `GET` | `/api/osint/gdelt?entity={e}&days={n}` | GDELT news query |

### Embeddings

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/embed/train` | Train Node2Vec embeddings |
| `GET` | `/api/embed/status` | Embedding training status |
| `GET` | `/api/embed/similar/{node_id}?k={n}` | Structurally similar nodes |
| `GET` | `/api/embed/search?q={query}&k={n}` | Semantic search |
| `GET` | `/api/embed/clusters?n={k}` | K-means graph clusters |

### GQL

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/gql` | Execute GQL query |
| `GET` | `/api/gql/help` | GQL syntax reference |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Ollama status + available models |
| `GET` | `/api/models` | List available Ollama models |
| `GET` | `/api/docs` | Interactive Swagger UI |
| `POST` | `/api/watch_list` | Update alert thresholds |
| `POST` | `/api/simulate` | Policy simulation |

---

## 19. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL. Set to the internal hostname in Docker/Railway. |
| `PORT` | `8000` | Server port. Railway sets this automatically. |

### Setting for Docker Compose

```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434
```

### Setting for Railway

In the Railway dashboard → Backend service → Variables:
```
OLLAMA_HOST = http://ollama.railway.internal:11434
```

---

## 20. Frontend — Dashboard

The frontend is a single-page application with 12 analysis panels, all wired to the backend via `frontend/js/api.js`.

### Navigation Tabs

| Tab | Panel | Description |
|---|---|---|
| `EXTRACT` | Ingestion | Paste text, URL, or upload file. SSE streaming extraction log. |
| `GRAPH` | Network | vis.js interactive graph. Node inspector, path finder, ego mode. |
| `GEO MAP` | World Map | Leaflet dark map with tension markers. Click markers for profiles. |
| `SIMULATE` | Simulation | Policy scenario input. Risk gauge, cascade analysis, history. |
| `FORECAST` | Forecasting | Crisis forecasts with probability bars, hotspot badges. |
| `QUERY` | Chat | GraphRAG analyst chat. User/assistant bubbles, Enter-to-send. |
| `ANALYTICS` | Stats | Centrality charts, group distribution, health indicators. |
| `TIMELINE` | Snapshots | Load historical snapshots, vis.js snapshot viewer. |
| `REPORT` | Reports | Entity checklist, format selector, PDF/Markdown/GraphML export. |
| `GQL` | Query Language | Query textarea, quick-chips, results table. |
| `SEMANTIC` | Embeddings | Train, search, similar nodes, K-means clustering. |
| `OSINT` | OSINT | RSS feed manager, GDELT query, Wikipedia enrichment, article log. |

### Design System

```
Background:  #010508  (void black)
Accent:      #00d4ff  (intelligence cyan)
Danger:      #ff2244  (critical red)
Success:     #00ff88  (clear green)
Warning:     #ffaa00  (amber)
Purple:      #c084fc  (forecast)

Fonts:
  Orbitron      → Display / HUD labels
  Rajdhani      → Body / descriptions
  JetBrains Mono → Code / data fields
```

### API Client (`frontend/js/api.js`)

All backend calls are centralized in `api.js` as a clean ESM module:

```javascript
import API from './js/api.js';

// Extract intelligence
const result = await API.extract(text, model);

// Run simulation
const sim = await API.simulate(scenario, model);

// Get geo tension markers
const geo = await API.geo();

// Stream extraction with SSE
API.stream(text, model, (event) => console.log(event));
```

---

## 21. Roadmap

### Near-term (P0)

- [ ] **Multi-step simulation** — chain scenarios with compound risk timelines
- [ ] **Alert watch list** — browser notifications when entities cross tension thresholds
- [ ] **Tension choropleth map** — fill country polygons with tension colors
- [ ] **Contradiction detection UI** — flag + resolve opposing edge labels

### Medium-term (P1)

- [ ] **Browser extension** — right-click any selected text → Extract to GOIES
- [ ] **Graph health score** — entity diversity, source diversity, recency metrics
- [ ] **Relationship decay** — exponential confidence decay on old edges
- [ ] **Source credibility tracker** — per-domain reliability scoring
- [ ] **GQL saved queries** — named query library for monitoring workflows

### Long-term (P2+)

- [ ] **Node2Vec training** on larger corpora for improved similarity search
- [ ] **Multi-user graphs** — named workspaces with role-based access
- [ ] **Telegram channel monitoring** — OSINT from conflict-zone channels
- [ ] **UN Security Council feed** — voting pattern extraction
- [ ] **OFAC/EU sanctions sync** — authoritative sanctions relationship auto-mapping
- [ ] **Plugin system** — hooks for domain-specific enrichment (arms control, finance, etc.)
- [ ] **Python client library** — `pip install goies-client` for programmatic access
- [ ] **GraphQL API** — flexible schema for external dashboard integrations
- [ ] **Obsidian/Notion export** — entity pages with wiki-links

See [`specs/IMPROVEMENTS.md`](specs/IMPROVEMENTS.md) for the full 30-feature roadmap with implementation details.

---

## 22. Contributing

Contributions are welcome. The project is actively developed.

### Setup for Development

```bash
git clone https://github.com/tanu-1403/GOIES.git
cd GOIES
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

### Key Directories to Know

- `server.py` — Add new API endpoints here. Follow the existing pattern.
- `extractor.py` — Extraction prompt engineering. OLLAMA_BASE_URL is env-configurable.
- `frontend/js/api.js` — Add new endpoint wrappers here for any new backend routes.
- `frontend/app.html` — All panel logic lives here as self-contained `<script>` blocks.

### Pull Request Guidelines

- Keep PRs focused on one feature or fix
- Update `requirements.txt` if you add new Python dependencies
- Add new API endpoints to this README's API Reference table
- Test extraction with at least one real news article before submitting

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**GOIES** — *Because intelligence should be open.*

[GitHub](https://github.com/tanu-1403/GOIES) · [Issues](https://github.com/tanu-1403/GOIES/issues) · [API Docs](http://localhost:8000/api/docs)

</div>
