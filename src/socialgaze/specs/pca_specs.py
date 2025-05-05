# src/socialgaze/specs/pca_specs.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PCASpec:
    name: str
    fit_type: str  # 'avg' or 'trial_wise'
    fit_categories: Optional[List[str]] = None
    fit_interactive_only: Optional[bool] = None
    transform_type: str = "trial_wise"
    transform_categories: Optional[List[str]] = None
    transform_interactive_only: Optional[bool] = None


PCASPECS = [
    PCASpec(
        name="avg_face_vs_object_transform_trialwise",
        fit_type="avg",
        fit_categories=["face", "object"],
        fit_interactive_only=None,
        transform_type="trial_wise",
        transform_categories=["face", "object"],
        transform_interactive_only=None,
    ),
    PCASpec(
        name="avg_face_noninteractive_transform_face_interactive",
        fit_type="avg",
        fit_categories=["face"],
        fit_interactive_only=False,
        transform_type="trial_wise",
        transform_categories=["face"],
        transform_interactive_only=True,
    ),
]
