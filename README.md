# Graph RAG

Hybrid retrieval-augmented generation: **FAISS vector search + Neo4j knowledge
graph**, LLM-powered graph extraction, LangGraph orchestration, and an
OpenRouter-backed chat model, served through Streamlit.

## Configuration

Create a local environment file (never commit it — it is git-ignored):

```bash
cp .env.example .env
```

Set at least `OPENROUTER_API_KEY`. Get a key from
<https://openrouter.ai/keys>.

### Streamlit Cloud deployment

Do **not** rely on a committed `.env` on Streamlit Cloud. Set the key in the
dashboard instead:

**App menu (☰) → Settings → Secrets →**

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

Streamlit Secrets take precedence over any `.env`/environment value.

### API key resolution order

`Settings.resolve_openrouter_key()` picks the key in this order:

1. **Streamlit Secrets** (`st.secrets["OPENROUTER_API_KEY"]`)
2. **OS environment variable** (`OPENROUTER_API_KEY`)
3. **`.env` file** (loaded by pydantic-settings, local development only)

### Troubleshooting: `UnauthorizedResponseError: User not found`

OpenRouter returns HTTP 401 **"User not found"** when the API key it receives is
invalid, revoked, regenerated, or copied incorrectly. Fixes:

1. Create a **fresh** key at <https://openrouter.ai/keys> (old keys may have
   been deleted/revoked).
2. On Streamlit Cloud, paste it into **App settings → Secrets** as
   `OPENROUTER_API_KEY`, then **Reboot the app** (secrets are read at startup).
3. In the app, open **📊 RAG system health** and click **🔑 Test OpenRouter
   key** to confirm the key and see which source it is loaded from.
4. If the health panel says the key comes from the **".env file (committed)"**,
   a stale key checked into the repo is shadowing your secret — the committed
   `.env` has been removed; redeploy and use Streamlit Secrets.
5. A 402 error means the account is out of credits — add credits at
   <https://openrouter.ai/credits>.

> Security note: if a real key was ever committed to the repository, **rotate
> it** — treat it as compromised regardless of later removal from git history.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
