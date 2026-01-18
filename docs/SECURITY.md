# Security Guide

This document covers the security features of the Mora Sleep Scoring API.

---

## Overview

The API includes three security layers, all optional and disabled by default for development convenience:

1. **API Key Authentication** - Protect endpoints from unauthorized access
2. **Rate Limiting** - Prevent abuse and ensure fair usage
3. **Request Logging** - Monitor API usage for security and debugging

---

## API Key Authentication

### How It Works

When enabled, all scoring endpoints require an `X-API-Key` header. Keys are hashed using SHA-256 before comparison (keys are never stored in plain text in memory after startup).

### Configuration

Set the `MORA_API_KEYS` environment variable with a comma-separated list of valid keys:

```bash
# Single key
export MORA_API_KEYS="my-secret-key-123"

# Multiple keys
export MORA_API_KEYS="key-for-user-1,key-for-user-2,shared-lab-key"
```

### Usage

Include the key in requests:

```bash
curl -X POST "https://sleep-production.up.railway.app/api/v1/score" \
  -H "X-API-Key: my-secret-key-123" \
  -F "file=@recording.wav"
```

In Python:

```python
import requests

response = requests.post(
    "https://sleep-production.up.railway.app/api/v1/score",
    headers={"X-API-Key": "my-secret-key-123"},
    files={"file": open("recording.wav", "rb")}
)
```

### Error Responses

| Status | Message | Cause |
|--------|---------|-------|
| 401 | "API key required" | Missing `X-API-Key` header when auth is enabled |
| 401 | "Invalid API key" | Key doesn't match any configured keys |

### Disabling Authentication

If `MORA_API_KEYS` is empty or not set, authentication is disabled and all requests are allowed. This is the default behavior.

---

## Rate Limiting

### How It Works

Rate limiting restricts how many requests a client can make per minute. Clients are identified by their IP address (supports `X-Forwarded-For` header for proxied requests).

### Configuration

```bash
# Enable rate limiting
export MORA_RATE_LIMIT_ENABLED=true

# Set requests per minute (default: 60)
export MORA_RATE_LIMIT_PER_MINUTE=30
```

### Error Response

When rate limit is exceeded:

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

Status code: `429 Too Many Requests`
Headers: `Retry-After: 60`

### Notes

- The current implementation uses in-memory storage, which resets when the server restarts
- Rate limits are per-client (based on IP address)
- For production with multiple instances, consider using Redis-based rate limiting

---

## Request Logging

All requests are logged automatically with the following information:

- HTTP method and path
- Response status code
- Request duration (ms)
- Client IP address
- User-Agent (truncated to 50 chars)

### Log Format

```
2024-01-15 10:30:45,123 - app.main - INFO - POST /api/v1/score - 200 - 1523.4ms - 192.168.1.100 - Mozilla/5.0...
```

### Log Levels

- `INFO`: Successful requests
- `WARNING`: Rate limit exceeded, invalid API keys
- `ERROR`: Processing failures

---

## CORS Configuration

Cross-Origin Resource Sharing (CORS) controls which domains can access the API.

### Configuration

```bash
# Allow all origins (default, for development)
export MORA_CORS_ORIGINS="*"

# Restrict to specific domains
export MORA_CORS_ORIGINS="https://sleep-cz3.pages.dev,https://yourdomain.com"
```

---

## Railway Deployment

To configure security on Railway:

1. Go to your Railway project dashboard
2. Click on the backend service
3. Navigate to **Variables** tab
4. Add the environment variables:

```
MORA_API_KEYS=your-secret-key-here
MORA_RATE_LIMIT_ENABLED=true
MORA_RATE_LIMIT_PER_MINUTE=60
MORA_CORS_ORIGINS=https://sleep-cz3.pages.dev
```

5. The service will automatically redeploy with the new settings

---

## Security Best Practices

### API Keys

1. **Generate strong keys**: Use at least 32 random characters
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Never commit keys**: Use environment variables, not config files

3. **Rotate keys periodically**: Generate new keys and update clients

4. **One key per user/service**: Easier to revoke access if compromised

### Production Checklist

- [ ] Set `MORA_API_KEYS` with strong, unique keys
- [ ] Enable rate limiting (`MORA_RATE_LIMIT_ENABLED=true`)
- [ ] Restrict CORS origins to your frontend domain
- [ ] Use HTTPS (Railway provides this automatically)
- [ ] Monitor logs for suspicious activity
- [ ] Set `MORA_DEBUG=false` in production

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MORA_API_KEYS` | "" (disabled) | Comma-separated API keys |
| `MORA_RATE_LIMIT_ENABLED` | false | Enable rate limiting |
| `MORA_RATE_LIMIT_PER_MINUTE` | 60 | Requests per minute per client |
| `MORA_CORS_ORIGINS` | "*" | Allowed CORS origins |
| `MORA_DEBUG` | false | Enable debug mode |

---

## Troubleshooting

### "API key required" but I don't want authentication

Make sure `MORA_API_KEYS` is empty or not set:
```bash
# Unset the variable
unset MORA_API_KEYS
```

### Rate limit too restrictive

Increase the limit:
```bash
export MORA_RATE_LIMIT_PER_MINUTE=120
```

Or disable rate limiting:
```bash
export MORA_RATE_LIMIT_ENABLED=false
```

### CORS errors in browser

Add your frontend domain to allowed origins:
```bash
export MORA_CORS_ORIGINS="http://localhost:5173,https://your-production-domain.com"
```
