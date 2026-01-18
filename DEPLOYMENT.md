# Mora Sleep Scoring - Deployment Guide

This guide walks you through deploying the Mora Sleep Scoring app to the cloud.

**Architecture:**
- **Backend (FastAPI)** → Railway (~$5-15/month)
- **Frontend (React)** → Cloudflare Pages (free)

---

## Prerequisites

- GitHub account (your repo: `PhilClarkPhD/sleep`)
- Code pushed to `claude/refactor` branch
- Railway account (sign up with GitHub)
- Cloudflare account (free tier)

---

## Part 1: Deploy Backend to Railway

### Step 1: Create Railway Account

1. Go to **https://railway.app**
2. Click **"Login"** in the top right
3. Select **"Sign in with GitHub"**
4. Authorize Railway to access your repositories

### Step 2: Create New Project

1. From the Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Search for `sleep` and select **`PhilClarkPhD/sleep`**
4. When prompted for a branch, select **`claude/refactor`**

### Step 3: Configure the Service

Railway will create a service. You need to point it to the backend folder:

1. Click on your new service (it might be named after your repo)
2. Go to the **Settings** tab
3. Under **Source**, set:
   - **Root Directory**: `backend`
4. Railway will auto-detect the `Dockerfile` and `railway.json`

### Step 4: Deploy

1. Railway should automatically start building
2. Watch the build logs - it takes ~2-3 minutes
3. Look for "Deployment successful" message

### Step 5: Get Your Public URL

1. Go to **Settings** tab
2. Scroll to **Networking** section
3. Click **"Generate Domain"**
4. You'll get a URL like: `https://sleep-production-xxxx.up.railway.app`

### Step 6: Verify Deployment

Test your backend is working:

```bash
curl https://YOUR-RAILWAY-URL/api/v1/health
# Should return: {"status":"healthy","model_loaded":true,"version":"1.0.0"}

curl https://YOUR-RAILWAY-URL/api/v1/model/info
# Should return model metadata
```

**Save your Railway URL** - you'll need it for the frontend.

---

## Part 2: Deploy Frontend to Cloudflare Pages

### Step 1: Create Cloudflare Account

1. Go to **https://pages.cloudflare.com**
2. Click **"Sign up"** (it's free)
3. Verify your email

### Step 2: Connect GitHub

1. From the Cloudflare Pages dashboard, click **"Create a project"**
2. Click **"Connect to Git"**
3. Select **GitHub** as your git provider
4. Authorize Cloudflare to access your repositories
5. Select the **`PhilClarkPhD/sleep`** repository

### Step 3: Configure Build Settings

Set the following build configuration:

| Setting | Value |
|---------|-------|
| Production branch | `claude/refactor` |
| Root directory | `frontend` |
| Build command | `npm run build` |
| Build output directory | `dist` |

### Step 4: Set Environment Variable

**Important:** Before clicking deploy, set the API URL:

1. Expand **"Environment variables"** section
2. Click **"Add variable"**
3. Set:
   - **Variable name**: `VITE_API_URL`
   - **Value**: `https://YOUR-RAILWAY-URL` (the URL from Part 1, Step 5)

   Example: `https://sleep-production-abc123.up.railway.app`

### Step 5: Deploy

1. Click **"Save and Deploy"**
2. Wait for the build to complete (~1-2 minutes)
3. Cloudflare will give you a URL like: `https://sleep-xxx.pages.dev`

### Step 6: Test the App

1. Open your Cloudflare Pages URL in a browser
2. You should see the Mora Sleep Scoring interface
3. Try uploading a WAV file to test the full flow

---

## Troubleshooting

### Backend Issues

**"Model not loaded" error:**
- Check Railway logs for errors
- Verify the model file exists in `backend/models/`
- Check that the Dockerfile copies the models directory

**502 Bad Gateway:**
- The app might still be starting up (wait 30 seconds)
- Check Railway logs for Python errors
- Verify the start command in `railway.json`

### Frontend Issues

**"API Error: Failed to connect":**
- Verify `VITE_API_URL` is set correctly in Cloudflare
- Check that the Railway backend is running
- Check browser console for CORS errors

**CORS errors:**
- The backend has CORS enabled for all origins
- Make sure you're using HTTPS URLs

**Blank page:**
- Check Cloudflare build logs
- Verify build command and output directory are correct

---

## Local Development vs Production

| Aspect | Local Dev | Production |
|--------|-----------|------------|
| Backend URL | `http://localhost:8000` | `https://xxx.up.railway.app` |
| Frontend URL | `http://localhost:3000` | `https://xxx.pages.dev` |
| API proxy | Vite handles it | Direct to Railway URL |
| Model path | `model_artifacts/` | `backend/models/` |

---

## Cost Estimates

| Service | Free Tier | Paid Estimate |
|---------|-----------|---------------|
| Railway | $5 credit/month | ~$5-15/month |
| Cloudflare Pages | Unlimited | Free |
| **Total** | ~$0-5/month | ~$5-15/month |

Railway charges based on usage. A sleep scoring app with occasional use will likely stay under $10/month.

---

## Updating the Deployment

### Backend Updates

1. Push changes to `claude/refactor` branch
2. Railway auto-deploys on push (if enabled)
3. Or manually trigger deploy from Railway dashboard

### Frontend Updates

1. Push changes to `claude/refactor` branch
2. Cloudflare auto-deploys on push
3. Build takes ~1-2 minutes

---

## Custom Domain (Optional)

### Railway (Backend)
1. Go to Settings → Networking
2. Click "Custom Domain"
3. Add your domain (e.g., `api.mora.yourdomain.com`)
4. Add the CNAME record to your DNS

### Cloudflare Pages (Frontend)
1. Go to your project → Custom domains
2. Click "Set up a custom domain"
3. Add your domain (e.g., `mora.yourdomain.com`)
4. Cloudflare handles DNS automatically if domain is on Cloudflare
