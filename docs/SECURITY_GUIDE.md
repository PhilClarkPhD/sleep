# Security Guide for Mora Sleep Scoring

A beginner-friendly explanation of the security features.

---

## TL;DR - Do You Even Need This?

**Current state: Security is OFF by default.**

If you're running this for yourself and a few trusted lab members on a private URL, you might not need any security at all. The API currently works without authentication.

**When you DO need security:**
- The URL becomes public/discoverable
- You're worried about random people using your compute resources
- You want to track who's using the API
- You have compliance requirements (IRB, institutional policy)

---

## The Simple Mental Model

Think of the API like a **locked room with your scoring computer inside**:

```
WITHOUT SECURITY (current default):
┌─────────────────────────────────────────────────┐
│                   INTERNET                       │
│                                                 │
│   Anyone who knows the URL can walk in          │
│                     │                           │
│                     ▼                           │
│              ┌─────────────┐                    │
│              │  Your API   │  "Door is open,    │
│              │  (Railway)  │   come on in!"     │
│              └─────────────┘                    │
│                                                 │
└─────────────────────────────────────────────────┘

WITH API KEY SECURITY:
┌─────────────────────────────────────────────────┐
│                   INTERNET                       │
│                                                 │
│   People need a "key" to get in                 │
│                     │                           │
│                     ▼                           │
│              ┌─────────────┐                    │
│              │  🔐 LOCKED  │  "Show me your     │
│              │  Your API   │   key first!"      │
│              │  (Railway)  │                    │
│              └─────────────┘                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## What API Keys Are (And Are NOT)

### API Keys are NOT:
- ❌ User accounts with usernames/passwords
- ❌ Login sessions
- ❌ Connected to email addresses
- ❌ Managed through a user interface

### API Keys ARE:
- ✅ Simple secret passwords (like `abc123xyz789`)
- ✅ Shared manually (you text/email them to users)
- ✅ Included in every request to the API
- ✅ Checked by the server before processing

**Think of it like a shared WiFi password**, not like a bank account login.

---

## How It Works (Step by Step)

### Without Security (Current State)

```
1. User visits https://sleep-cz3.pages.dev
2. User uploads a WAV file
3. Frontend sends file to https://sleep-production.up.railway.app/api/v1/score
4. Backend processes it and returns scores
5. Done! No questions asked.
```

### With API Key Security (If Enabled)

```
1. You generate a key: "mora-lab-key-2024-secret"
2. You give this key to your lab members (email, Slack, etc.)
3. You configure Railway to require this key

Then when someone uses the app:

4. User visits https://sleep-cz3.pages.dev
5. User uploads a WAV file
6. Frontend sends file + key in header:
   POST /api/v1/score
   X-API-Key: mora-lab-key-2024-secret
   [file data]

7. Backend checks: "Is this key valid?"
   - YES → Process the file, return scores
   - NO  → Return "401 Unauthorized"
```

---

## The Three Security Features

### 1. API Key Authentication
**What:** Requires a secret key in every request
**Why:** Blocks random people from using your API
**Default:** OFF (no keys required)

### 2. Rate Limiting
**What:** Limits requests per minute per user (by IP address)
**Why:** Prevents someone from overwhelming your server
**Default:** OFF (unlimited requests)

### 3. Request Logging
**What:** Records who called the API and when
**Why:** Debugging and monitoring usage
**Default:** ON (always logs)

---

## Do You Need to Change Anything?

### Scenario A: Private Lab Use (5-10 trusted people)

**Recommendation: Leave security OFF**

Your Railway URL is not indexed by Google. Unless you share it publicly, random people won't find it. The small risk of someone stumbling onto it is probably not worth the hassle of managing keys.

```
Current setup:
- MORA_API_KEYS = (empty, auth disabled)
- MORA_RATE_LIMIT_ENABLED = false
- MORA_CORS_ORIGINS = "*"

→ No changes needed!
```

### Scenario B: Shared with Collaborators Outside Your Lab

**Recommendation: Add API keys**

You're sharing the URL more broadly. Add a key so only people you've given the key to can use it.

```
Recommended setup:
- MORA_API_KEYS = "your-secret-key-here"
- MORA_RATE_LIMIT_ENABLED = true
- MORA_CORS_ORIGINS = "https://sleep-cz3.pages.dev"
```

### Scenario C: Public/Published Tool

**Recommendation: API keys + rate limiting + restricted CORS**

If you publish a paper with this tool or make it widely available, you need all protections.

---

## How to Enable Security (If You Want It)

### Step 1: Generate a Strong Key

```bash
# Run this in terminal to generate a random key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Output example: Yx7kM9pQrS2tUvWxYz3aBcDeFgHiJkLmNoPq
```

### Step 2: Configure Railway

1. Go to https://railway.app and open your project
2. Click on the **backend** service
3. Go to **Variables** tab
4. Add these variables:

```
MORA_API_KEYS=Yx7kM9pQrS2tUvWxYz3aBcDeFgHiJkLmNoPq
MORA_RATE_LIMIT_ENABLED=true
MORA_RATE_LIMIT_PER_MINUTE=60
MORA_CORS_ORIGINS=https://sleep-cz3.pages.dev
```

5. Railway will automatically redeploy

### Step 3: Update the Frontend (Important!)

The web frontend needs to send the API key with requests. Currently it doesn't.

**Option A: Hardcode the key (simple but less secure)**

Edit `frontend/src/api/sleepApi.ts` to include the key:

```typescript
const headers: Record<string, string> = {
  'X-API-Key': 'your-key-here'  // Add this line
};
```

**Option B: Environment variable (better)**

Add to your frontend's environment:
```
VITE_API_KEY=your-key-here
```

Then use it in the code:
```typescript
const headers: Record<string, string> = {};
if (import.meta.env.VITE_API_KEY) {
  headers['X-API-Key'] = import.meta.env.VITE_API_KEY;
}
```

### Step 4: Give the Key to Your Users

For web app users: They don't need to do anything (the frontend handles it).

For CLI/API users: Tell them to include the header:
```bash
curl -X POST "https://sleep-production.up.railway.app/api/v1/score" \
  -H "X-API-Key: your-key-here" \
  -F "file=@recording.wav"
```

---

## Multiple Keys for Multiple Users

You can have different keys for different people/groups:

```
MORA_API_KEYS=lab-internal-key-abc123,collaborator-key-xyz789,reviewer-key-temp456
```

**Why do this?**
- If a key is compromised, you only revoke that one
- You can track usage by key (in logs)
- You can give temporary keys to reviewers

**Limitations:**
- No built-in way to see "which key was used" in the UI
- All keys have the same permissions (no admin vs. user)
- You manage keys manually (no database)

---

## What About User Logins?

The current system does NOT support:
- User registration
- Password reset
- User-specific data/history
- Permissions per user

**If you need real user accounts**, you'd need to add:
1. A database (PostgreSQL)
2. User model with hashed passwords
3. JWT tokens or session management
4. Registration/login pages

This is significantly more complex. For a small lab tool, API keys are usually sufficient.

---

## Quick Reference

### Environment Variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `MORA_API_KEYS` | "" (empty) | Comma-separated keys. Empty = no auth. |
| `MORA_RATE_LIMIT_ENABLED` | false | Enable request limiting |
| `MORA_RATE_LIMIT_PER_MINUTE` | 60 | Max requests per minute per IP |
| `MORA_CORS_ORIGINS` | "*" | Which domains can call the API |

### Error Messages You Might See

| Error | Meaning | Fix |
|-------|---------|-----|
| "API key required" | Auth is enabled but no key sent | Add `X-API-Key` header |
| "Invalid API key" | Key doesn't match server config | Check the key is correct |
| "Rate limit exceeded" | Too many requests | Wait 60 seconds |
| CORS error in browser | Frontend domain not allowed | Add domain to `MORA_CORS_ORIGINS` |

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY DECISION TREE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Is this just for you and trusted lab members?              │
│     YES → Leave everything as-is (no auth needed)           │
│     NO  ↓                                                   │
│                                                             │
│  Are you sharing the URL with external collaborators?       │
│     YES → Add API key (MORA_API_KEYS)                       │
│     NO  ↓                                                   │
│                                                             │
│  Is this going to be publicly accessible?                   │
│     YES → Add API keys + rate limiting + restrict CORS      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The bottom line: **For a handful of trusted users, you probably don't need to change anything.** The security features are there when you need them, but they're not required.
