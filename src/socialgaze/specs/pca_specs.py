# src/socialgaze/specs/pca_specs.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PCAFitSpec:
    name: str
    trialwise: bool
    categories: Optional[List[str]] = None
    split_by_interactive: Optional[bool] = None


@dataclass(frozen=True)
class PCATransformSpec:
    name: str
    trialwise: bool
    categories: Optional[List[str]] = None
    split_by_interactive: Optional[bool] = None


FIT_SPECS = [
    # PCAFitSpec(
    #     name="fit_avg_face_obj",
    #     trialwise=False,
    #     categories=["face", "object"],
    #     split_by_interactive=None,
    # ),
    PCAFitSpec(
        name="fit_trialwise_face_obj",
        trialwise=True,
        categories=["face", "object"],
        split_by_interactive=None,
    ),
    PCAFitSpec(
        name="fit_avg_interactive_and_noninteractive_face_obj",
        trialwise=False,
        categories=["face", "object"],
        split_by_interactive=True,
    ),
    PCAFitSpec(
        name="fit_trialwise_interactive_and_noninteractive_face_obj",
        trialwise=True,
        categories=["face", "object"],
        split_by_interactive=True,
    ),
]

TRANSFORM_SPECS = [
    PCATransformSpec(
        name="transform_avg_face_obj",
        trialwise=False,
        categories=["face", "object"],
        split_by_interactive=None,
    ),
    PCATransformSpec(
        name="transform_trialwise_face_obj",
        trialwise=True,
        categories=["face", "object"],
        split_by_interactive=None,
    ),
    PCATransformSpec(
        name="transform_avg_interactive_and_noninteractive_face_obj",
        trialwise=False,
        categories=["face", "object"],
        split_by_interactive=True,
    ),
    PCATransformSpec(
        name="transform_trialwise_interactive_and_noninteractive_face_obj",
        trialwise=True,
        categories=["face", "object"],
        split_by_interactive=True,
    ),
]
