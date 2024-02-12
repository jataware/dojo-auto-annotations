from __future__ import annotations

from agent import Agent
from meta import Meta
import xarray as xr
from typing import TypeVar
from utils import enum_to_keys, ask_user


from MetadataSchema import (
    AnnotationSchema,
    GeoAnnotation,
    DateAnnotation,
    FeatureAnnotation,
    ColumnType,
    DateType,
    GeoType,
    FeatureType,
    CoordFormat,
    GadmLevel,
    TimeField,
    LatLong,
    TimeRange,
)

import pdb


def handle_netcdf(meta: Meta, agent: Agent) -> AnnotationSchema:
    data = xr.open_dataset(meta.path)
    pdb.set_trace()


def handle_geotiff(meta: Meta, agent: Agent) -> AnnotationSchema:
    data = xr.open_rasterio(meta.path)
    pdb.set_trace()
