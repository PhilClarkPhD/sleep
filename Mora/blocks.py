import pandas as pd
from pydantic import BaseModel, model_validator
from typing import Dict, Literal
from datetime import datetime, timedelta


class TimeBlockUnit(BaseModel):
    duration: timedelta
    phase: Literal["Light", "Dark"]

    @model_validator(mode="after")
    @classmethod
    def validate_timedelta_structure(cls, model):
        """
        Validate that the `duration` field is a valid `datetime.timedelta` and has a positive duration.
        """
        duration = getattr(model, "duration", None)

        # Ensure duration is a `timedelta` object
        if not isinstance(duration, timedelta):
            raise ValueError("Duration must be a `datetime.timedelta` object.")

        # Ensure duration is positive
        if duration.total_seconds() <= 0:
            raise ValueError("Duration must be greater than zero.")

        return model


class TimeBlocks(BaseModel):
    blocks: Dict[int, TimeBlockUnit]

    @model_validator(mode="after")
    @classmethod
    def validate_block_structure(cls, model):
        for key in model.blocks.keys():
            if not isinstance(key, int):
                raise ValueError(f"Block key '{key} must be an integer")
        return model

    def __repr__(self):
        return f"TimeBlocks(\n{self.blocks})"
