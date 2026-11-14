"""API routes package."""
from .health import router as health_router
from .documents import router as documents_router
from .search import router as search_router
from .generation import router as generation_router
from .chat import router as chat_router
from .conversations import router as conversations_router
from .quizzes import router as quizzes_router

__all__ = [
    "health_router",
    "documents_router",
    "search_router",
    "generation_router",
    "chat_router",
    "conversations_router",
    "quizzes_router",
]
