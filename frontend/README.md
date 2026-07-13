# MaintAI — Frontend

React + Vite UI for the MaintAI maintenance NLP workbench (zinc dark theme, shadcn/ui).

## Documentation

| Document | Contents |
|----------|----------|
| [../Readme.md](../Readme.md) | System design, architecture, engineer workflow features |
| [../docs/DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md) | Setup, API, session persistence, bulk review, active learning |

Recent workflow features (documented in README): session persistence, run history reopen, bulk review with undo, confusion matrix / label drift, active learning from corrections.

## Quick run

With Docker (recommended), from repo root:

```bash
docker compose up --build
```

Open http://localhost:5173

### Frontend only (local)

```bash
npm install
npm run dev
```

Ensure the backend is reachable at the proxy target (default `http://127.0.0.1:5050`).

## Structure

```text
src/
  pages/           Dashboard, Results, History, Health
  components/      Feature components + ui/ (shadcn)
  services/api.js  All backend HTTP calls
  context/         Shared app state
```

## Conventions

- No ML logic in the frontend.
- Use `@/` import alias for `src/`.
- Add API methods in `services/api.js`, not inline `fetch` in components.
