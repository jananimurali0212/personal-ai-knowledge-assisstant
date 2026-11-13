"""Main FastAPI application entrypoint for Personal AI Knowledge Assistant."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config.settings import settings
from .routes.health import router as health_router
from .routes.documents import router as documents_router
from .routes.search import router as search_router

app = FastAPI(
    title="Personal AI Knowledge Assistant API",
    description="Backend services for Personal AI Knowledge Assistant - Document Processing & RAG Engine",
    version=settings.VERSION,
)

# Enable CORS for local frontend development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health check endpoint: GET /health
app.include_router(health_router)

# Mount API routes under /api/v1 prefix
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(search_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Personal AI Knowledge Assistant API",
        "docs_url": "/docs",
        "health": "/health",
        "version": settings.VERSION,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
