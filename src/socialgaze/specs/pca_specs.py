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
    PCAFitSpec(
        name="fit_avg_face_obj",
        trialwise=False,
        categories=["face", "object"],
        split_by_interactive=None,
    ),
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

def get_all_fit_transform_pairs():
    """
    Returns all (fit_spec, transform_spec) pairs by combining each fit spec
    with every transform spec. This is consistent with how projections are done
    in 02_pc_projection.py.
    """
    return [(fit, transform) for fit in FIT_SPECS for transform in TRANSFORM_SPECS]


