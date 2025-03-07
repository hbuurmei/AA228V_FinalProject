from enum import Enum
from typing import List, Union, Literal
from pydantic import BaseModel, Field, validator
import numpy as np


class ShapeType(str, Enum):
    """Enum for the shape of the synthetic data region."""
    RECTANGULAR = "rectangular"
    SPHERICAL = "spherical"


class ExtraDataConfig(BaseModel):
    """
    Configuration for generating synthetic data.
    
    Attributes:
        centroid: Center point of the shape
        eps: Radius (for spherical) or half-width (for rectangular)
        shape: Type of shape to sample from ('rectangular' or 'spherical')
        dim: Dimensionality of the data
        label: Label to assign to all generated points
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    centroid: List[float] = Field(..., description="Center point of the shape")
    eps: float = Field(..., gt=0, description="Radius (spherical) or half-width (rectangular)")
    shape: ShapeType = Field(default=ShapeType.SPHERICAL, description="Shape type to sample from")
    dim: int = Field(..., gt=0, description="Dimensionality of the data")
    label: int = Field(..., ge=0, description="Label to assign to all generated points")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    
    @validator('centroid')
    def validate_centroid_dim(cls, v, values):
        if 'dim' in values and len(v) != values['dim']:
            raise ValueError(f"Centroid dimension ({len(v)}) must match specified dimension ({values['dim']})")
        return v
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return self.dict()
