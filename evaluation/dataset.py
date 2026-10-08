"""Evaluation dataset loading and validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .config import BASE_DIR
from .schemas import Expected

DEFAULT_DATASET = BASE_DIR / "evaluations" / "dataset" / "questions.json"


class EvalCase(BaseModel):
    id: str
    question: str
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    expected_answer: Optional[str] = None
    expected_sources: List[str] = Field(default_factory=list)
    expected_tools: List[str] = Field(default_factory=list)
    expect_refusal: bool = False

    def expected(self) -> Expected:
        return Expected(answer=self.expected_answer, sources=self.expected_sources, tools=self.expected_tools, refusal=self.expect_refusal)


class EvalDataset(BaseModel):
    version: str
    description: str = ""
    cases: List[EvalCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalDataset":
        ids = [c.id for c in self.cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate case ids: {sorted(dupes)}")
        return self


def load_dataset(path: Optional[Path] = None) -> EvalDataset:
    path = Path(path or DEFAULT_DATASET)
    return EvalDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
