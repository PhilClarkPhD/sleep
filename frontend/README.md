# Mora frontend

The browser GUI for Mora Sleep Scoring. React 19 + TypeScript, built with Vite,
styled with Tailwind v4, state in Zustand, charts with react-plotly.js.

```bash
npm install          # once
npm run dev          # http://localhost:3000, forwards /api to the backend on :8000
npm run build        # type-check + production build into dist/
npm run lint
```

Needs the backend running on port 8000 for anything past the upload box:
`cd ../backend && uvicorn app.main:app --reload --port 8000`.

Configuration: one build-time variable, `VITE_API_URL` (backend base URL). Empty in
development (the dev server proxies), set to the Railway URL on Cloudflare Pages.
The API key, if the server requires one, is entered in the gear panel and stored in
`localStorage` under `mora_api_key`.

Docs: design and component map in `../docs/GUI_DESIGN.md`; how to make changes in
`../docs/guides/EDIT_THE_UI.md`; deployment in `../docs/guides/DEPLOY.md`.
