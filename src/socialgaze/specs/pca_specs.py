# src/socialgaze/specs/pca_specs.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PCAFitSpec:
    name: str
    categories: Optional[List[str]] = None
    split_by_interactive: Optional[bool] = None


@dataclass(frozen=True)
class PCATransformSpec:
    name: str
    categories: Optional[List[str]] = None
    split_by_interactive: Optional[bool] = None



FIT_SPECS = [
    PCAFitSpec(
        name="fit_avg_face",
        categories=["face"],
        split_by_interactive=None,
    ),
    PCAFitSpec(
        name="fit_int_non_int_face",
        categories=["face"],
        split_by_interactive=True,
    ),
    PCAFitSpec(
        name="fit_avg_obj",
        categories=["object"],
        split_by_interactive=None,
    ),
    PCAFitSpec(
        name="fit_avg_face_obj",
        categories=["face", "object"],
        split_by_interactive=None,
    ),
    PCAFitSpec(
        name="fit_int_non_int_face_obj",
        categories=["face", "object"],
        split_by_interactive=True,
    ),
]

TRANSFORM_SPECS = [
    PCATransformSpec(
        name="transform_avg_face",
        categories=["face"],
        split_by_interactive=None,
    ),
    PCATransformSpec(
        name="transform_int_non_int_face",
        categories=["face"],
        split_by_interactive=True,
    ),
    PCATransformSpec(
        name="transform_avg_obj",
        categories=["object"],
        split_by_interactive=None,
    ),
    PCATransformSpec(
        name="transform_avg_face_obj",
        categories=["face", "object"],
        split_by_interactive=None,
    ),
    PCATransformSpec(
        name="transform_int_non_int_face_obj",
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


