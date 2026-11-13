"""Document domain models."""
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

DocumentStatus = Literal["uploaded", "processing", "ready", "failed"]
SupportedFileType = Literal["pdf", "docx", "txt"]

class DocumentBase(BaseModel):
    file_name: str
    file_type: str
    file_size: int = Field(default=0, ge=0)
    storage_path: str
    status: DocumentStatus = "uploaded"

class DocumentCreate(DocumentBase):
    user_id: UUID

class DocumentUpdate(BaseModel):
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    storage_path: Optional[str] = None
    status: Optional[DocumentStatus] = None

class Document(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
