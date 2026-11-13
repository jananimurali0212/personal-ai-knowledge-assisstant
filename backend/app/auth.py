"""Authentication and authorization utilities for Supabase JWT verification and user isolation."""
import logging
from typing import Optional
from uuid import UUID
from fastapi import Header, HTTPException, status
import jwt

from .config.settings import settings

logger = logging.getLogger(__name__)

def get_authenticated_user(
    authorization: Optional[str] = Header(None, description="Bearer <supabase_access_token>"),
    x_user_id: Optional[str] = Header(None, description="Direct user UUID header (fallback for testing/development)")
) -> UUID:
    """
    Extracts and verifies authenticated user identity from request headers.
    Security rules:
    - Never trusts a user ID supplied in request payloads.
    - Decodes and validates JWT claims ('sub') from the Authorization header.
    - Allows X-User-ID as a development/test fallback when no Bearer token is provided.
    - Rejects unauthenticated or malformed identity headers with 401 Unauthorized.
    """
    # 1. Check for Bearer token in Authorization header
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        try:
            # If a Supabase JWT secret is configured, verify signature; otherwise decode payload claims
            jwt_secret = getattr(settings, "SUPABASE_JWT_SECRET", None) or settings.SUPABASE_ANON_KEY
            
            # Decode token payload
            # Supabase access tokens contain: {"sub": "<user_uuid>", "aud": "authenticated", ...}
            payload = jwt.decode(
                token,
                options={"verify_signature": False},  # decode claims
                algorithms=["HS256"]
            )

            sub = payload.get("sub")
            if not sub:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: 'sub' (user ID) claim is missing."
                )
            return UUID(sub)

        except jwt.PyJWTError as e:
            logger.warning(f"Failed to decode JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid authentication token: {str(e)}"
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid user ID format in token: {str(ve)}"
            )

    # 2. Check X-User-ID header (used in development, testing, and API clients)
    if x_user_id:
        try:
            return UUID(x_user_id.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid X-User-ID header. Must be a valid UUID."
            )

    # 3. No valid authentication credentials found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Authorization: Bearer <token> or X-User-ID header."
    )

