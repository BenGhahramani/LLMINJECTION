# LLM Injection Test Bench

A minimal chat-based web application with file upload and a SQLite database
backend, designed for researching prompt-injection attacks and defences against
a tool-calling LLM.

---

## Prerequisites

| Dependency | Version |
|------------|---------|
| Python     | 3.11+   |
| pip        | latest  |
| An OpenAI API key | — |

## Quick Start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env             # then edit .env with your real key

# 4. Run the development server
uvicorn backend.app:app --reload

# 5. Open http://localhost:8000 in your browser
```

## Project Structure

```
├── backend/
│   ├── __init__.py         # Package marker
│   ├── app.py              # FastAPI server — bridges UI ↔ LLM ↔ DB
│   └── database.py         # SQLite schema, seed data, exposed DB functions
├── frontend/
│   └── index.html          # Single-page chat UI (HTML + CSS + JS)
├── uploads/                # User-uploaded files land here (git-ignored)
│   └── .gitkeep
├── .env.example            # Template for environment variables
├── .gitignore
├── requirements.txt        # Pinned Python dependencies
├── LICENSE
└── README.md               # ← you are here
```

## Database Schema

Five tables are created on first startup and seeded with realistic fake data.

| Table            | Purpose                          | LLM Access   |
|------------------|----------------------------------|-------------- |
| `customers`      | Customer contact records         | Allowed       |
| `orders`         | Order history with status        | Allowed       |
| `documents`      | Uploaded-file metadata           | Allowed       |
| `internal_notes` | Confidential company notes       | **Restricted** |
| `audit_log`      | Auto-logged on every DB function | Admin only    |

## Exposed DB Functions (Attack Surface)

These functions are registered as OpenAI "tools" so the LLM can call them.
The system prompt *forbids* the restricted/dangerous ones — your attacks
attempt to bypass that instruction.

| Function                           | Access Level   | Description                     |
|------------------------------------|---------------|----------------------------------|
| `get_customer(customer_id)`        | Allowed       | Look up a customer by ID         |
| `search_orders(query)`             | Allowed       | Search orders by product/status  |
| `list_customers()`                 | Allowed       | List all customers               |
| `list_documents()`                 | Allowed       | List uploaded document metadata  |
| `get_internal_note(note_id)`       | **Restricted**| Read a confidential internal note|
| `update_internal_note(note_id, …)` | **Dangerous** | Modify a confidential note       |

## API Endpoints

| Method | Path         | Description                                          |
|--------|-------------|------------------------------------------------------|
| `GET`  | `/`          | Serves the frontend (`frontend/index.html`)         |
| `POST` | `/api/chat`  | Accepts `message` (form) + optional `file` (multipart) |
| `GET`  | `/api/audit` | Returns the last 100 audit-log entries as JSON       |

## What to Test

### Attacks

Try to make the LLM violate its system-prompt rules:

- **Direct leakage** — ask the LLM to read internal notes.
- **Indirect injection** — embed override instructions in an uploaded
  `.txt` or `.pdf` file so the LLM executes them.
- **Privilege escalation** — trick the LLM into calling
  `update_internal_note` to *modify* confidential data.

### Defences

Observe or extend the existing mitigations:

- **System-prompt instruction hierarchy** — rules 1–7 in `SYSTEM_PROMPT`.
- **Input sanitisation** — add pre-processing to strip suspicious patterns.
- **Structured tool restrictions** — enforce allow-lists in the backend
  before executing tool calls.
- **Audit logging** — inspect `GET /api/audit` to see exactly which tools
  the LLM invoked.

## Tech Stack

| Layer    | Technology        |
|----------|-------------------|
| Frontend | HTML + CSS + vanilla JS |
| Backend  | Python / FastAPI  |
| Database | SQLite            |
| LLM      | OpenAI GPT-4o-mini (via function calling) |

## Licence

See [LICENSE](./LICENSE).
