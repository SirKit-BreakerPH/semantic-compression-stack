# WordPress iFrame Embedding Guide

## The short version

Once your Flask server is running (locally or hosted), add this to any
WordPress page using the **Custom HTML** block:

```html
<iframe
  src="http://YOUR-SERVER-URL:5000"
  width="100%"
  height="820"
  frameborder="0"
  style="border-radius:8px; border:1px solid #1e2230;"
  allow="clipboard-write"
  title="Semantic Compression Stack">
</iframe>
```

---

## Step 1 — Run the server

### Option A: Your own server / VPS (recommended)

```bash
# Install web deps
pip install flask flask-cors

# Set your free API keys (optional — only needed for Layer 5)
export GROQ_API_KEY="gsk_..."         # from console.groq.com (free)
export GEMINI_API_KEY="AIza..."       # from aistudio.google.com (free)

# Start (exposed to internet)
python webui/app.py --host 0.0.0.0 --port 5000
```

Then point your iFrame at: `http://your-server-ip:5000`

---

### Option B: Free hosting on Render.com (zero cost)

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your repo, set:
   - **Build command:** `pip install -r requirements.txt && pip install flask flask-cors`
   - **Start command:** `python webui/app.py --host 0.0.0.0 --port $PORT`
   - **Environment variables:** `GROQ_API_KEY`, `GEMINI_API_KEY`
4. Deploy — Render gives you a free `https://yourapp.onrender.com` URL
5. Point your iFrame at that URL

---

### Option C: Codespaces (development only)

When running in Codespaces:
1. Start the server: `python webui/app.py --port 5000`
2. Codespaces gives you a forwarded HTTPS URL like:
   `https://youruser-5000.app.github.dev`
3. Use that URL in your iFrame during development

---

## Step 2 — Add the iFrame to WordPress

### Method A: Block Editor (Gutenberg)

1. Edit the page/post where you want the tool
2. Click **+** → search for **Custom HTML**
3. Paste the iFrame code (adjust the `src` URL):

```html
<iframe
  src="https://YOUR-RENDER-URL.onrender.com"
  width="100%"
  height="860"
  frameborder="0"
  style="border-radius:8px; border:1px solid #222; max-width:1200px; display:block; margin:0 auto;"
  allow="clipboard-write"
  title="Semantic Compression Tool">
</iframe>
```

4. Publish or update the page.

---

### Method B: Classic Editor

Paste the iFrame HTML directly in the **Text** tab (not Visual).

---

### Method C: iFrame plugin (if your theme blocks raw HTML)

Some WordPress themes/hosts (e.g. WordPress.com) block raw HTML.
Install the free plugin **"Advanced iFrame"** or **"WP iFrame"**,
then use their shortcode:

```
[advanced_iframe src="https://YOUR-URL" width="100%" height="860"]
```

---

## Step 3 — HTTPS requirement

WordPress sites served over HTTPS **cannot** embed HTTP iFrames (mixed
content policy). Solutions:

| Your WordPress | Server needs |
|----------------|-------------|
| https://... | Your Flask server must also be HTTPS |
| http://... | Any URL works |

**Easiest fix:** Deploy to Render.com (free HTTPS) or use Cloudflare Tunnel
to put HTTPS in front of your local server.

---

## Security notes

- API keys are entered in the browser and sent **directly** to Groq/Gemini
  from the user's browser — they never touch your Flask server
- The Flask server has CORS set to `*` by default
  (restrict with `ALLOWED_ORIGINS=https://yoursite.com` env var)
- Layer 1–4 processing happens on **your server** (no external calls)
- Layer 5 calls Groq or Gemini APIs only when keys are provided

---

## Minimal iFrame embed (copy-paste ready)

```html
<!-- Paste this in WordPress Custom HTML block -->
<iframe
  src="https://YOUR-APP-URL"
  width="100%"
  height="860"
  frameborder="0"
  allow="clipboard-write"
  style="border:1px solid #1e2230; border-radius:8px; display:block;">
</iframe>

<p style="text-align:center; font-size:12px; color:#888; margin-top:8px;">
  Powered by Semantic Compression Stack · 
  <a href="https://console.groq.com" target="_blank">Get free Groq key</a> · 
  <a href="https://aistudio.google.com" target="_blank">Get free Gemini key</a>
</p>
```
