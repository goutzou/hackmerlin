# HackMerlin Agent

An autonomous agent that plays [hackmerlin.io](https://hackmerlin.io/) using Playwright and OpenAI models.

## Run with Docker

### 1. Build

```bash
docker build -t hackmerlin-agent .
```

### 2. Headless run (recommended in containers)

```bash
docker run --rm \
  -e OPENAI_API_KEY="sk-..." \
  -e OPENAI_MODEL="gpt-4o-mini" \
  -e HEADLESS=true \
  hackmerlin-agent
```

## Run Locally (macOS/Linux)

### 1. Clone & set up

```bash
git clone https://github.com/<your-username>/hackmerlin.git
cd hackmerlin
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install Playwright browsers

```bash
python -m playwright install chromium
```

### 3. Set environment

```bash
echo "OPENAI_API_KEY=sk-..." > .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env
echo "HEADLESS=false" >> .env             # set true if you don’t want a window
```

### 4. Run the agent

```bash
python agent_hackmerlin.py
```
