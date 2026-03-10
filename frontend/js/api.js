/**
 * api.js — GOIES Centralized API Client
 * All backend calls flow through this module.
 * Base URL auto-detects: dev (localhost:8000) vs production.
 */

const BASE = (() => {
  const { protocol, hostname, port } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return `${protocol}//${hostname}:${port || 8000}`;
  }
  return ''; // same-origin in production
})();

async function _handleResponse(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const API = {
  /** GET request */
  get: (path) => fetch(`${BASE}${path}`).then(_handleResponse),

  /** POST request with JSON body */
  post: (path, body) =>
    fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(_handleResponse),

  /** DELETE request */
  del: (path) => fetch(`${BASE}${path}`, { method: 'DELETE' }).then(_handleResponse),

  /** POST file upload (multipart/form-data) */
  upload: (path, formData) =>
    fetch(`${BASE}${path}`, { method: 'POST', body: formData }).then(_handleResponse),

  /** SSE streaming POST — returns raw Response for caller to read */
  stream: (path, body) =>
    fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // ── Convenience wrappers ──────────────────────────────────────────────────

  health:          ()        => API.get('/api/health'),
  models:          ()        => API.get('/api/models'),
  graph:           (ego, hops) => API.get(ego ? `/api/graph?ego=${encodeURIComponent(ego)}&hops=${hops||2}` : '/api/graph'),
  clearGraph:      ()        => API.del('/api/graph'),
  path:            (src,tgt) => API.get(`/api/path?src=${encodeURIComponent(src)}&tgt=${encodeURIComponent(tgt)}`),
  narrativeSummary:(model)   => API.get(`/api/narrative/summary?model=${model}`),
  geo:             ()        => API.get('/api/geo'),
  snapshots:       ()        => API.get('/api/snapshots'),
  snapshotTimeline:()        => API.get('/api/snapshots/timeline'),
  getSnapshot:     (id)      => API.get(`/api/snapshots/${encodeURIComponent(id)}`),
  simulations:     ()        => API.get('/api/simulations'),
  exportGraph:     (fmt)     => `${BASE}/api/export/${fmt}`,

  ingestUrl:       (url)           => API.post('/api/ingest/url', { url }),
  ingestFile:      (formData)      => API.upload('/api/ingest/file', formData),
  extract:         (text, model, persona) => API.post('/api/extract', { text, model, persona }),
  extractStream:   (text, model, persona) => API.stream('/api/extract/stream', { text, model, persona }),
  query:           (question, model, persona) => API.post('/api/query', { question, model, persona }),
  simulate:        (scenario, model) => API.post('/api/simulate', { scenario, model }),
  forecast:        (model, focus)  => API.post('/api/forecast', { model, focus }),
  report:          (entities, format, model) => API.post('/api/report', { entities, format, model }),
  watchList:       (thresholds)    => API.post('/api/watch_list', { thresholds }),
  mergeNodes:      (source, target) => API.post('/api/node/merge', { source, target }),

  gql:             (query)         => API.post('/api/gql', { query }),
  gqlHelp:         ()              => API.get('/api/gql/help'),

  embedTrain:      ()              => API.post('/api/embed/train', {}),
  embedStatus:     ()              => API.get('/api/embed/status'),
  embedSimilar:    (nodeId, k)     => API.get(`/api/embed/similar/${encodeURIComponent(nodeId)}?k=${k||8}`),
  embedSearch:     (q, k)          => API.get(`/api/embed/search?q=${encodeURIComponent(q)}&k=${k||8}`),
  embedClusters:   (n)             => API.get(`/api/embed/clusters?n=${n||5}`),

  osintStatus:     ()              => API.get('/api/osint/status'),
  osintFeeds:      ()              => API.get('/api/osint/feeds'),
  osintAddFeed:    (url, name)     => API.post('/api/osint/feeds', { url, name }),
  osintRemoveFeed: (url)           => API.del(`/api/osint/feeds?url=${encodeURIComponent(url)}`),
  osintIngest:     (model, n)      => API.post('/api/osint/ingest', { model, articles_per_feed: n }),
  osintEnrich:     (nodeId, model) => API.post(`/api/osint/enrich/${encodeURIComponent(nodeId)}?model=${model}`, {}),
  osintGdelt:      (entity, days)  => API.get(`/api/osint/gdelt?entity=${encodeURIComponent(entity)}&days=${days||7}`),
};

export default API;
