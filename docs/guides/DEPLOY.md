# Guide: host the app online (optional)

**Status: not currently hosted.** The Railway backend was shut down in November
2026 to stop the monthly cost, and the Cloudflare Pages frontend has nothing to
talk to without it. Everyone runs the app locally (`RUN_THE_APP.md`). Keep this
guide for the day someone wants a shared URL again; the config files
(`backend/Dockerfile`, `railway.json`, `docker-compose.yml`, `frontend/Dockerfile`)
are still in the repo and worked as of the shutdown.

When it was live, the flow was: **commit, push to `claude/refactor`, wait two
minutes.** Both hosts watched that branch. The rest of this page records the
settings that made that work.

## Where things run

| Piece | Host | URL | Rebuilds when |
|---|---|---|---|
| Backend (API + model) | Railway, project "sleep", Docker | https://sleep-production.up.railway.app | push to `claude/refactor` |
| Frontend (GUI) | Cloudflare Pages, project `sleep-cz3` | https://sleep-cz3.pages.dev | push to `claude/refactor` |

Cost: small monthly Railway fee for the backend; Cloudflare Pages is free.

Logins: the Cloudflare account is the one logged into **via GitHub** (the Drexel
address), not the Gmail one; Pages projects are under "Workers & Pages", not
"Websites". Railway login is also via GitHub.

## The normal deploy

```bash
git status                      # see what you changed
git add -A
git commit -m "Promote XGBoost_1.2.5"   # or whatever you did
git push origin claude/refactor
```

Then:

- Railway: open the project → Deployments. A new build starts within a minute and
  takes 3–5 minutes (it installs Python packages). Green check = live. Confirm with
  https://sleep-production.up.railway.app/api/v1/model/info showing the version you
  expect.
- Cloudflare Pages: Workers & Pages → `sleep-cz3` → Deployments. 1–2 minutes.
  Hard-refresh the site (Cmd+Shift+R) to bypass the browser cache.

## Deploying a new model specifically

The Docker image copies `backend/models/` into itself at build time, so the model
that is served is whatever `.pkl` is committed there. Follow `RETRAIN_MODEL.md`
step 4 (copy the new `.pkl` and `_ood.pkl` in, delete the old), commit, push.
Model files are 8–20 MB; that is fine for git here.

If the backend starts but `/health` says `degraded`: the `.pkl` isn't in
`backend/models/` in the commit you pushed, or the model's `feature_code_version`
has a different MAJOR than the code (check Railway's deploy logs for
"Incompatible feature code").

## Railway settings that must be right

Found under the service → Settings:

| Setting | Value | Why |
|---|---|---|
| Source → Root Directory | `/` (repo root) | The Dockerfile copies `sleep_features/` from the repo root. With root = `backend/`, the build fails on `COPY backend/requirements.txt` |
| Build → Dockerfile Path | `backend/Dockerfile` | Also set in `railway.json` at the repo root |
| Deploy → Healthcheck | `/api/v1/health` | In `railway.json` |
| Variables | `MORA_CORS_ORIGINS=https://sleep-cz3.pages.dev` (optional; `*` works), `MORA_API_KEYS` (only if enabling auth), `MORA_RATE_LIMIT_ENABLED` | `PORT` is set by Railway automatically |

Note (2026-11-13): the repo was changed so the build context is the repo root.
If the first deploy after that fails, the Root Directory setting is the cause.

## Cloudflare Pages settings that must be right

Workers & Pages → `sleep-cz3` → Settings → Builds & deployments:

| Setting | Value |
|---|---|
| Production branch | `claude/refactor` |
| Root directory | `frontend` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Environment variable | `VITE_API_URL=https://sleep-production.up.railway.app` |

There is no config file in the repo for Pages; these live in the dashboard.

Escape hatch if the GitHub integration misbehaves:

```bash
cd frontend && npm run build
npx wrangler pages deploy dist --project-name sleep-cz3
```

## Testing before you push

Run the full stack locally (`RUN_THE_APP.md`) and score a real file. For the
backend you can also build the Docker image exactly as Railway does:

```bash
docker build -f backend/Dockerfile -t mora-api .      # from the repo root
docker run -p 8000:8000 mora-api
```

or `docker compose up` for both halves.

## Rolling back

Railway and Pages both keep previous deployments; each has a "Redeploy" / "Rollback"
button on the older deployment. Or `git revert <commit>` and push.

## Branching

`claude/refactor` is production. Work on a branch if the change is risky, open a
pull request on GitHub, merge when happy. A plain push to `claude/refactor` deploys
immediately, so don't push half-finished work there.
