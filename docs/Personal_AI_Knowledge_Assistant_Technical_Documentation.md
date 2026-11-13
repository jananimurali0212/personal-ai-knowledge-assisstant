# PERSONAL AI KNOWLEDGE ASSISTANT
## Project Design & Technical Documentation

| Metadata Field | Specification Details |
| :--- | :--- |
| **Project category** | Artificial Intelligence / Generative AI |
| **System approach** | Retrieval-Augmented Generation (RAG) |
| **Proposed stack** | React, FastAPI, Gemini API, Supabase PostgreSQL + pgvector |
| **Development environment** | Google AI Studio and Antigravity IDE |
| **Document version** | 1.0 — Initial project specification |

**Purpose**: Define the product scope, architecture, data flow, technical components, implementation sequence, and validation approach for an AI assistant grounded in a user's own knowledge sources.

---

## 1. Executive Summary

The **Personal AI Knowledge Assistant** is a document-centric AI application that lets an individual upload, organize, and query their own knowledge. Users can ask natural-language questions across one or more uploaded files, request summaries, and generate study questions. The system retrieves relevant passages from a private knowledge base and passes those passages to a language model to produce grounded answers with source references.

---

## 2. Problem Statement

Information is often distributed across PDFs, lecture notes, reports, manuals, and personal files. Keyword search can miss relevant passages when the user's wording differs from the document wording. Reading entire documents to locate one fact is time-consuming. This project addresses the problem through semantic retrieval and contextual answer generation, while preserving links to the source material.

---

## 3. Objectives

- Provide a secure workspace for uploading and organizing personal documents.
- Extract and normalize text from supported file formats.
- Build a semantic index using vector embeddings.
- Retrieve relevant passages for natural-language questions.
- Generate answers grounded in retrieved content and show citations.
- Support summarization, quiz generation, and conversation history.
- Evaluate retrieval quality, answer grounding, latency, and reliability.

---

## 4. Scope and Core Features

- **Document management**: upload, list, preview, categorize, and delete supported documents.
- **AI question answering**: ask questions about one document or a selected collection.
- **Semantic search**: retrieve passages by meaning rather than exact keyword overlap.
- **Source citations**: show document name and page number or chunk reference where available.
- **Summarization**: produce document-level and section-level summaries.
- **Quiz generation**: create MCQs and short-answer questions from selected material.
- **Conversation history**: save threads and support contextual follow-up questions.

---

## 5. Proposed Technology Stack

- **Frontend**: React.js, TypeScript, and Tailwind CSS for a responsive, modern web interface.
- **Backend**: Python with FastAPI for authentication-aware APIs, processing jobs, and orchestration.
- **LLM**: Google Gemini API for answer generation and supported summarization tasks.
- **Embeddings**: Gemini embedding model; use the same compatible embedding model for document chunks and queries.
- **Database**: Supabase PostgreSQL for users, documents, conversations, and metadata.
- **Vector search**: `pgvector` extension with similarity search and user/document filters.
- **File storage**: Supabase Storage private object store.
- **Document parsing**: PyMuPDF for PDFs and python-docx for DOCX files; plain-text parser for TXT.
- **Development**: Google AI Studio for model prototyping, Antigravity IDE for implementation, Git/GitHub for version control.

---

## 6. High-Level Architecture

The high-level design separates the user interface, application services, RAG engine, model provider, database, and file storage. The frontend sends authenticated requests to FastAPI. The backend coordinates document ingestion and question answering. The RAG engine retrieves relevant vectors from pgvector and invokes Gemini for embeddings or answer generation. Original files remain in private object storage.

```
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

## 7. Low-Level Architecture (Ingestion & Query Pipelines)

The low-level design has two related pipelines: **document ingestion** and **query answering**. During ingestion, a file is validated, parsed, cleaned, chunked, embedded, and persisted with metadata. During question answering, the user's question is embedded, relevant chunks are retrieved with access-control filters, and a grounded prompt is sent to Gemini. The response is returned with source identifiers and saved to conversation history.

```
DOCUMENT INGESTION PIPELINE:
+-------------------+      +--------------------+      +--------------------+
|   1. UPLOAD API   | ===> | 2. TEXT EXTRACTION | ===> |    3. CHUNKING     |
| (Validate & Auth) |      | (PyMuPDF / docx)   |      | (Normalize/Overlap)|
+-------------------+      +--------------------+      +--------------------+
                                                                  |
                                                                  v
+-------------------+      +--------------------+      +--------------------+
|     5. PERSIST    | <=== |  STORED VECTORS    | <=== |   4. EMBEDDINGS    |
| (Supabase+pgvector)      | (pgvector index)   |      | (Gemini Embeddings)|
+-------------------+      +--------------------+      +--------------------+

QUERY & ANSWERING PIPELINE:
+-------------------+      +--------------------+      +--------------------+
|  6. QUESTION API  | ===> |  7. QUERY VECTOR   | ===> |    8. RETRIEVAL    |
| (Auth & Validate) |      | (Embed User Query) |      | (Similarity Search)|
+-------------------+      +--------------------+      +--------------------+
                                                                  |
                                                                  v
+-------------------+      +--------------------+      +--------------------+
|    12. HISTORY    | <=== |    11. RESPONSE    | <=== | 10. GENERATION &   |
| (Save Chat Thread)|      | (Answer + Sources) |      |   9. PROMPT BUILD  |
+-------------------+      +--------------------+      +--------------------+
```

---

## 8. Detailed Processing Workflow

### 8.1 Document Ingestion
1. Frontend requests an upload authorization or submits the file through the backend.
2. Backend validates file extension, MIME type, size, and authenticated user ownership.
3. Original file is stored in a private storage bucket; a document record is created with a processing status.
4. Parser extracts text and page boundaries. Scanned PDFs require an OCR extension and should be marked accordingly.
5. Text is normalized and split into chunks using a configurable chunk size and overlap.
6. Embeddings are generated in batches. Each chunk is stored with its document ID, user ID, page number, and embedding.
7. Processing status changes to completed or failed, with a safe error message.

### 8.2 Question Answering
1. Backend authenticates the user and checks which documents they are allowed to query.
2. Query text is embedded using the same embedding space as the indexed chunks.
3. Vector search retrieves top-k similar chunks, applying user and optional document filters.
4. Optional reranking or threshold filtering removes weak matches.
5. Prompt builder includes the question, retrieved passages, citation metadata, and instructions not to invent unsupported facts.
6. Gemini generates the response. The backend maps cited chunk identifiers to document names and page numbers.
7. Frontend displays the answer and source references; messages are saved in the conversation tables.

---

## 9. Data Model and Database Design

Use UUID primary keys and timestamps. Add indexes for user/document lookups and a vector index appropriate to the selected pgvector distance operator and embedding dimensions.

| Table | Important Columns | Purpose |
| :--- | :--- | :--- |
| `profiles` | `id`, `email`, `created_at` | Application user profile; linked to Supabase Auth. |
| `documents` | `id`, `user_id`, `file_name`, `file_type`, `storage_path`, `status`, `created_at` | Tracks uploaded files and processing state. |
| `document_chunks` | `id`, `document_id`, `user_id`, `chunk_text`, `page_number`, `chunk_index`, `embedding` | Stores searchable passages and vector representations. |
| `conversations` | `id`, `user_id`, `title`, `created_at` | Stores chat threads. |
| `messages` | `id`, `conversation_id`, `role`, `content`, `created_at` | Stores user and assistant messages. |
| `message_sources` | `message_id`, `chunk_id`, `relevance_score` | Links generated answers to retrieved evidence. |

> **Important**: Enforce Row Level Security (RLS) on all user-owned tables. Storage policies must also prevent cross-user file access. Do not rely only on frontend filtering.

---

## 10. API Design (Proposed)

- `POST /api/documents/upload` — validate and register a document upload.
- `GET /api/documents` — list the authenticated user's documents.
- `GET /api/documents/{document_id}` — retrieve document metadata and processing status.
- `DELETE /api/documents/{document_id}` — delete a document, its chunks, and stored file according to retention policy.
- `POST /api/chat` — submit a question with conversation ID and optional document filters.
- `GET /api/conversations` — list the user's conversations.
- `GET /api/conversations/{conversation_id}` — retrieve messages and source references.
- `POST /api/documents/{document_id}/summary` — request a document summary.
- `POST /api/documents/{document_id}/quiz` — generate questions from a document.

---

## 11. Functional Requirements

- **FR-01**: The user can register, log in, and log out.
- **FR-02**: The user can upload supported PDF, DOCX, and TXT documents.
- **FR-03**: The system validates files and reports processing status.
- **FR-04**: The system extracts, chunks, and embeds document text.
- **FR-05**: The user can query one or multiple authorized documents.
- **FR-06**: The assistant returns grounded answers with available source references.
- **FR-07**: The user can request summaries and quizzes.
- **FR-08**: The user can view and delete documents and conversation threads.

---

## 12. Non-Functional Requirements

- **Security**: authenticated access, RLS, private storage, secrets held server-side, and safe file validation.
- **Performance**: batch embedding requests, index vector columns, use asynchronous processing for large files, and measure retrieval and generation latency separately.
- **Reliability**: retries with limits, idempotent ingestion, clear processing states, and graceful API error handling.
- **Scalability**: separate ingestion workers from interactive chat requests as workload grows.
- **Usability**: responsive layout, accessible controls, clear empty/loading/error states.
- **Observability**: log request IDs, processing failures, token usage, and timing without logging sensitive document contents unnecessarily.

---

## 13. Security and Privacy Considerations

- Keep Gemini and Supabase service credentials in backend environment variables; never expose service-role keys in browser code.
- Apply RLS policies to documents, chunks, conversations, and messages.
- Verify ownership on every document ID supplied by a client.
- Use private storage buckets and short-lived signed URLs where needed.
- Restrict file size and accepted MIME types; consider malware scanning for public deployments.
- Treat retrieved document text as untrusted input. Prompt instructions must prevent document text from overriding system rules.
- Provide a deletion workflow that removes database records and stored files consistently.
- Define retention and backup policies before handling sensitive personal or institutional documents.

---

## 14. Testing and Evaluation Plan

- **Functional tests**: valid upload, invalid file, empty PDF, multi-page PDF, DOCX parsing, document deletion, chat history, and access isolation.
- **Retrieval tests**: known-answer questions, paraphrased queries, multi-document questions, no-match questions, and page citation accuracy.
- **Generation tests**: factual consistency with retrieved passages, refusal or uncertainty when evidence is insufficient, citation correctness, and resistance to prompt injection in documents.
- **Performance tests**: upload processing time, embedding throughput, retrieval latency, model response latency, and end-to-end latency.
- **Suggested evaluation metrics**: Recall@k, Precision@k, Mean Reciprocal Rank (MRR), answer faithfulness, citation precision, p50/p95 latency, and cost per query.

---

## 15. Five-Day Implementation Plan

- **Day 1 — Foundation**: Create repository, React application, FastAPI service, environment configuration, Supabase project, and authentication UI. *Deliverable: User can sign in and view the dashboard.*
- **Day 2 — Document Ingestion**: Implement upload, validation, private storage, PDF/DOCX/TXT extraction, and document status. *Deliverable: Uploaded text is extracted and associated with its owner.*
- **Day 3 — Retrieval Foundation**: Implement chunking, embedding generation, pgvector schema, similarity-search function, and access filters. *Deliverable: A test query retrieves relevant chunks.*
- **Day 4 — RAG Chat**: Implement prompt construction, Gemini generation, citations, conversation persistence, and follow-up context. *Deliverable: End-to-end grounded question answering.*
- **Day 5 — Quality and Release**: Add summaries and quiz generation, test security and edge cases, measure retrieval quality and latency, fix issues, and deploy. *Deliverable: A tested initial release.*

---

## 16. Risks and Mitigations

- **Scanned PDFs**: text extraction may return little or no text; detect this and add OCR as a planned capability.
- **Incorrect answers**: use retrieval thresholds, evidence-grounded prompts, citations, and an explicit insufficient-evidence response.
- **Large documents**: process asynchronously and batch embedding calls.
- **Cross-user data exposure**: enforce RLS, private storage policies, and backend ownership checks.
- **Model/API limits**: handle rate limits, retry transient errors, and expose clear status messages.
- **Embedding changes**: version the embedding model and re-index documents if the embedding space changes.

---

## 17. Expected Outcome

The completed system will let a user build a private searchable knowledge base and interact with it through natural-language chat. It will retrieve evidence from uploaded documents, generate context-aware answers, show sources, and support additional knowledge workflows such as summarization and quiz generation.

---

## 18. Future Enhancements

- OCR for scanned documents and image-based notes.
- Voice input and spoken answers.
- Website, URL, and YouTube transcript ingestion.
- Hybrid retrieval combining keyword and vector search.
- Reranking and query rewriting for improved retrieval quality.
- Knowledge graph and entity relationship exploration.
- Browser extension and local-model support.
- Advanced evaluation dashboard and user feedback loop.

---

## 19. Conclusion

This project combines document processing, vector search, retrieval-augmented generation, and conversational AI in a practical knowledge management application. Its modular architecture supports an achievable first release while leaving room for stronger retrieval, multimodal ingestion, richer memory, and production-grade observability.

