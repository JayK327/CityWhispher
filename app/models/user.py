from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


CATEGORY_INDEX = {
    "historical": 0,
    "cultural":   1,
    "commercial": 2,
    "nature":     3,
    "food":       4,
}

DEFAULT_WEIGHTS = [0.4, 0.3, 0.1, 0.1, 0.1]  # bias toward historical + cultural


class UserPreferences(BaseModel):
    user_id:    str
    weights:    list[float] = Field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    tone:       str = "informative"
    updated_at: Optional[datetime] = None

    def dominant_category(self) -> str:
        idx = self.weights.index(max(self.weights))
        return list(CATEGORY_INDEX.keys())[idx]

    def weight_for(self, category: str) -> float:
        idx = CATEGORY_INDEX.get(category, -1)
        if idx == -1:
            return 0.1
        return self.weights[idx]
