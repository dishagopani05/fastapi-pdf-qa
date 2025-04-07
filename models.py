from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class QARecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
