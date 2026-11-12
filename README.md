# Personal AI Knowledge Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Frontend: React + TS + Vite + Tailwind](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript%20%7C%20Vite%20%7C%20Tailwind-purple.svg)](frontend/)
[![Backend: FastAPI + Python](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.12-009688.svg)](backend/)
[![Database: Supabase + PostgreSQL](https://img.shields.io/badge/Database-Supabase%20%7C%20PostgreSQL%20%7C%20pgvector-3ECF8E.svg)](supabase_schema.sql)
[![Testing: Vitest (27/27 Passed)](https://img.shields.io/badge/Tests-27%2F27%20Passed-brightgreen.svg)](frontend/src/test/)

The **Personal AI Knowledge Assistant** is a document-centric AI application that enables individuals to upload, organize, and query their private knowledge bases using state-of-the-art Retrieval-Augmented Generation (RAG). Users can upload documents (PDF, DOCX, TXT), search their knowledge base semantically, and generate grounded answers, summaries, and study quizzes powered by Google Gemini and Supabase PostgreSQL with `pgvector`.

📄 **Full Technical Specification**: [`docs/Personal_AI_Knowledge_Assistant_Technical_Documentation.md`](docs/Personal_AI_Knowledge_Assistant_Technical_Documentation.md)

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Supabase Setup & Database Schema](#-supabase-setup--database-schema)
- [Environment Configuration](#-environment-configuration)
- [Local Development Guide](#-local-development-guide)
- [Automated Testing Suite](#-automated-testing-suite)
- [Implementation Roadmap](#-implementation-roadmap)
- [Security & Privacy](#-security--privacy)

---

## ✨ Key Features

- **🔐 Robust Supabase Authentication**:
  - Email and password registration and login.
  - Interactive password visibility toggles.
  - Comprehensive field validation with meaningful error notifications.
  - Session persistence via `localStorage` across page reloads.
  - Protected routes redirecting unauthenticated visitors to `/login`.
  - Header and sidebar user profile management with instant sign-out.

- **📊 Modern AI SaaS Dashboard**:
  - Live metric overviews: Total documents uploaded, aggregate storage usage, and active security status.
  - Embedded quick uploader with drag-and-drop support.
  - Recently uploaded documents table with direct download and deletion capabilities.
  - 5-section sidebar navigation: **Dashboard**, **My Documents**, **AI Assistant**, **Search**, and **Settings**.

- **📁 Knowledge Document Management**:
  - Support for **PDF**, **DOCX**, and **TXT** files (up to 25 MB per file).
  - Drag-and-drop or local file selection.
  - Private file storage in isolated Supabase Storage buckets.
  - Complete document metadata tracking (UUID, user ownership, file type, file size, storage path, status, timestamps).
  - Temporary signed URLs for secure on-demand downloads.
  - Confirmation modals for safe, permanent deletions.

- **🛡️ Row Level Security (RLS)**:
  - Strict database policies ensure users can only view, insert, update, or delete their own document metadata (`auth.uid() = user_id`).
  - Storage bucket isolation restricts access to `${user.id}/*` prefixes.

---

## 🏛 System Architecture

The architecture separates the presentation layer, application backend, RAG engine, model provider, database, and storage:

```text
                               +-------------------------------------+
                               |           AI / RAG ENGINE           |<=====>[ GEMINI API ]
                               |   (Retrieval + Prompt Generation)   |       (LLM + Embeddings)
                               +-------------------------------------+
                                        ^                |
                       Question + Context|                | top-k chunks
                                        |                v
[ USER ] ==========> [ PRESENTATION ] ===> [ APPLICATION ] ========> [ DATA LAYER ]
(Web / Mobile Browser) (React + Tailwind)  (FastAPI Backend)        (Supabase + pgvector)
                                                 |                          ^
                                                 | (file access/storage)     |
                                                 v                          |
                                       [ FILE STORAGE ] --------------------+
                                    (Supabase Private Storage)
```

---

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS | High-performance, modern UI with white & soft lavender/blue SaaS palette |
| **Routing** | React Router DOM v7 | Client-side routing with protected route guards |
| **Icons** | Lucide React | Clean, consistent icons |
| **Backend** | Python 3.12, FastAPI, Uvicorn | RESTful API orchestration and asynchronous processing |
| **Database** | Supabase PostgreSQL + `pgvector` | Relational document metadata, user auth, and vector index |
| **Object Storage**| Supabase Storage | Encrypted private document bucket with RLS |
| **Testing** | Vitest, React Testing Library, JSDOM | Comprehensive unit and integration test coverage |

---

## 🗄 Supabase Setup & Database Schema

The database migration script is provided in [`supabase_schema.sql`](supabase_schema.sql).

### Applying the Schema:
1. Open your [Supabase Dashboard](https://supabase.com/dashboard).
2. Navigate to the **SQL Editor** in the left sidebar.
3. Copy the contents of [`supabase_schema.sql`](supabase_schema.sql) and click **Run**.

### Schema Summary:
- **`public.documents` Table**:
  - `id`: UUID (Primary Key, `gen_random_uuid()`)
  - `user_id`: UUID (Foreign Key to `auth.users(id) ON DELETE CASCADE`)
  - `file_name`: TEXT
  - `file_type`: TEXT (`pdf`, `docx`, `txt`)
  - `file_size`: BIGINT
  - `storage_path`: TEXT
  - `status`: TEXT (`uploaded`, `processing`, `ready`, `failed`)
  - `created_at` / `updated_at`: TIMESTAMPTZ
- **Row Level Security (RLS)**:
  - Enabled on `public.documents` for `SELECT`, `INSERT`, `UPDATE`, and `DELETE`.
- **Storage Bucket (`documents`)**:
  - Created as a private bucket (`public = false`) with 50 MB limits.
  - Storage policies enforce folder isolation by `${auth.uid()}`.

---

## 🔑 Environment Configuration

Create a `.env` file in `frontend/` (see [`frontend/.env.example`](frontend/.env.example)):

```env
# Supabase Project URL
VITE_SUPABASE_URL=https://oeczcuvxbqdbepgzzohk.supabase.co

# Supabase Anonymous Public API Key (From Project Settings -> API)
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Backend FastAPI Service (Optional default: http://localhost:8000)
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🚀 Local Development Guide

### 1. Prerequisites
- Node.js 18+ and npm
- Python 3.10+ (for FastAPI backend)

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The application will launch at `http://localhost:5173`.

### 3. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API documentation available at `http://localhost:8000/docs`.

---

## 🧪 Automated Testing Suite

The project includes unit and integration tests powered by Vitest and React Testing Library:

```bash
cd frontend
npm test
```

### Test Coverage Highlights:
- **`src/test/supabase.test.ts`**: Validates Supabase client initialization and environment variable bindings.
- **`src/test/auth.test.tsx`**: Tests email/password validation, password visibility toggle, Supabase error handling, and protected route redirection.
- **`src/test/documents.test.ts`**: Tests file format validation (PDF/DOCX/TXT), 25 MB size limits, byte formatting, and mocked Supabase DB and storage operations.
- **`src/test/dashboard.test.tsx`**: Tests dashboard metric cards, quick uploader, recent documents table, and delete confirmation modals.
- **Results**: **27 / 27 tests passing**.

---

## 🗺 Implementation Roadmap

- [x] **Day 1: Foundation & Authentication**
  - Supabase integration, AuthContext, Protected Routes, Login/Register pages.
- [x] **Day 2: Dashboard & Document Management**
  - Private Supabase Storage bucket, PostgreSQL `documents` table with RLS, Drag & Drop upload (PDF, DOCX, TXT), document list, and deletion.
- [ ] **Day 3: Text Extraction & Embeddings**
  - PyMuPDF and python-docx parsing, text chunking with overlap, Gemini embedding generation, `pgvector` indexing.
- [ ] **Day 4: RAG Question Answering**
  - Vector similarity search, prompt construction with citations, Gemini conversational stream, chat history persistence.
- [ ] **Day 5: Summaries & Evaluation**
  - Document-level summaries, quiz generation, retrieval quality benchmarks, deployment.

---

## 🔒 Security & Privacy

1. **Credentials Isolation**: Service-role keys and LLM tokens are held exclusively on the backend; the frontend only uses the public Supabase `anon` key.
2. **Database Row Level Security**: All queries and mutations verify `auth.uid() = user_id`.
3. **Private File Storage**: Uploaded files can only be accessed by their owner via short-lived signed URLs.
4. **Input Sanitization**: File extensions and MIME types are validated client-side and server-side.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
