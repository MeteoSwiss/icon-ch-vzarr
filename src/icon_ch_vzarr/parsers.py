import base64
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict
import logging

import earthkit.data as ekd
from earthkit.data.readers.grib.codes import GribField
import numpy as np
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import ObjectStoreRegistry
from virtualizarr.types import ChunkKey
from virtualizarr.manifests.utils import create_v3_array_metadata
import xarray as xr

from icon_ch_vzarr.codec import EccodesCodec


LOG = logging.getLogger(__name__)

DEFAULT_PROFILE = {
    "global_attrs": [{"Conventions": "CF-1.8"}, {"institution": "MeteoSwiss"}],
    "dim_roles": {"level_type": "typeOfLevel"},
    "level_dim_mode":"level_per_type",
    "lazy_load": True,
    "ensure_dims": "valid_time",
}

FILE_COORDS = [
    "valid_time",
    "forecast_reference_time",
    "step",
    "time",
    "latitude",
    "longitude",

]

LEVEL_COORDS = [
    "generalVertical",
    "generalVerticalLayer",
    "isobaricInhPa",
    "isobaricInPa",
]

class IconChParser:
    """Parser for ICON-CH data."""

    def __init__(self, filter_keys: dict[str, Any] | None = None, ekd_xr_profile: dict[str, Any] | None = None):
        """Initialize the parser.
        
        Parameters
        ----------
        filter_keys : dict[str, Any], optional
            A dictionary of GRIB key-value mappings to filter the data using earthkit data.
        ekd_xr_profile : dict[str, Any], optional
            The profile for earthkit data xarray engine.
        """

        self.filter_keys = filter_keys or {}
        self.ekd_xr_profile = DEFAULT_PROFILE | (ekd_xr_profile or {})


    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        
        # NOTE: we need to register this here when using multiprocessing
        # register_codec(CODEC_ID, EccodesCodec)

        # access the file's contents, e.g. using the ObjectStore instance in the registry
        store, path_in_store = registry.resolve(url)
        absolute_path = Path(store.prefix or "/") / path_in_store # NOTE: investigate if this is intended behavior

        # read the GRIB file using earthkit data and apply any filter keys
        source = ekd.from_source("file", absolute_path).sel(self.filter_keys)

        # scan and extract metadata from the GRIB file using earthkit data
        manifestgroup = extract_metadata(source, self.ekd_xr_profile)

        # construct the Manifeststore from the parsed metadata and the object store registry
        return ManifestStore(group=manifestgroup, registry=registry)
    

# TODO: only use eartkit-data to extract the information we need, without actually constructing the xarray object
# don't know if there is an internal API for this 
def extract_metadata(fieldlist: ekd.FieldList, earthkit_xarray_profile: dict[str, Any] | None = None) -> ManifestGroup:
    """Scan a GRIB file and return ManifestArray objects for each variable.
    
    Parameters
    ----------
    fieldlist : ekd.FieldList
        List of GRIB fields to process.
    earthkit_xarray_profile : dict[str, Any], optional
        Earthkit-data's xarray engine profile arguments.
    """
    
    # create arrays for coordinates that are unique to the file
    # in other words, they are the same for all fields in the fieldlist
    file_coords_arrays = create_file_coordinates_arrays(fieldlist)

    # create arrays for coordinates that are unique to the level type
    # in other words, they are the same for all fields with the same type of level
    # e.g. "isobaricInhPa", "generalVerticalLayer", etc.
    level_coords_arrays: dict[str, ManifestArray] = {}
    
    # we are going to store arrays separately for debugging purposes (see check below)
    variable_arrays_per_level_type: dict[str, dict[str, ManifestArray]] = defaultdict(dict)

    # iterate over all fields in the fieldlist, grouped by type of level
    grouped_fields: Iterable[ekd.FieldList] = fieldlist.group_by("typeOfLevel")
    for level_type_group in grouped_fields:

        # get the type of level from the metadata (e.g. "surface", "isobaricInhPa", "generalVerticalLayer")
        level_type = level_type_group.metadata("typeOfLevel")[0]

        # create arrays for all veriables with same level type
        level_type_arrays = create_variable_arrays(level_type_group, earthkit_xarray_profile)
        
        # check that there are no variables with more than a single level type
        for existing_level_type, existing_arrays in variable_arrays_per_level_type.items():
            conflicting_keys = set(existing_arrays.keys()) & set(level_type_arrays.keys())
            if conflicting_keys:
                raise ValueError(
                    f"Variables {conflicting_keys} are present in both level type '{existing_level_type}' and '{level_type}'. "
                    "This is currently not supported."
                )
        
        # add the arrays for this level type to the dictionary
        variable_arrays_per_level_type[level_type].update(level_type_arrays)

        
        # create arrays for vertical level coordinates (e.g. "isobaricInhPa", "generalVerticalLayer")
        if level_type not in LEVEL_COORDS:
            continue 

        level_coords_chunk_entries: dict[str, ChunkEntry] = {}
        for field in level_type_group:
            level = field.metadata("level")
            if level in level_coords_chunk_entries:
                continue
            level_coords_chunk_entries[level] = _chunk_entry_from_field(field)
        level_coords_arrays[level_type] = _level_coordinate_array(
            coord=level_type,
            chunk_entries=level_coords_chunk_entries,
            dtype=np.float64,  # assuming level coordinates are float64
        )
      
    # merge all variable arrays for each level type into a single dictionary
    variable_arrays: dict[str, ManifestArray] = {}
    for level_type, arrays in variable_arrays_per_level_type.items():
        variable_arrays.update(arrays)
    return ManifestGroup(variable_arrays | file_coords_arrays | level_coords_arrays)


def create_file_coordinates_arrays(source: ekd.FieldList) -> dict[str, ManifestArray]:
    """Generate ManifestArray objects for coordinates that are unique to the file."""
    
    # Extract the first field to get the metadata
    template_field = source[0]

    out = {}

    out["valid_time"] = _file_coordinate_array(
        param="valid_time",
        chunk_entry=_chunk_entry_from_field(template_field),
        shape=[1],
        dims=["valid_time"],
        dtype=np.dtype("datetime64[ns]"),
    )

    return out


def create_variable_arrays(
    fields: ekd.FieldList, earthkit_xarray_profile: dict[str, Any] | None = None
) -> dict[str, ManifestArray]:
    """Create a dictionary of ManifestArray objects for each variable in the provided FieldList.
    
    Groups the input fields by their 'shortName', constructs chunked array entries for each group,
    and generates ManifestArray instances with appropriate shape, chunking, and metadata attributes.
    The function also encodes the GRIB message in base64 and attaches it to the attributes,
    so that it is JSON serializable.
    
    Parameters
    ----------
    fields : ekd.FieldList
        List of GRIB fields to process, grouped by variable short name.
    earthkit_xarray_profile : dict[str, Any], optional
        Additional keyword arguments to pass to the `to_xarray` method for xarray conversion.
    Returns
    -------
    dict[str, ManifestArray]
        Dictionary mapping variable short names to their corresponding ManifestArray objects.
    """
    
    earthkit_xarray_profile = earthkit_xarray_profile or {}

    # TODO: make the logic more generic and fully depend on earthkit-data's xarray engine profile
    
    variable_arrays = {}
    for grouped_fields in fields.group_by("shortName"):
        grouped_fields: ekd.FieldList
        param = grouped_fields.metadata("shortName")[0]
        chunk_entries = {}
        for idx, field in enumerate(grouped_fields):
            field: GribField
            chunk_key = ChunkKey("0.0" if len(grouped_fields) == 1 else f"0.{idx}.0")
            chunk_entries[chunk_key] = _chunk_entry_from_field(field)
        
        shape = (1, *field.shape) if len(grouped_fields) == 1 else (1, len(grouped_fields), *field.shape)
        chunk_shape = shape if len(grouped_fields) == 1 else (1, 1, *field.shape)

        # construct the object using more than one vertical level, so we keep the dimension
        xr_obj: xr.DataArray = grouped_fields[:2].to_xarray(**earthkit_xarray_profile)[param]
        attrs = xr_obj.attrs.copy()
        message = attrs["_earthkit"].pop("message")
        attrs["_earthkit"]["b64message"] = base64.b64encode(message).decode("utf-8")
        LOG.info(f"Creating ManifestArray for variable '{param}' with shape {shape} and chunk shape {chunk_shape}")
        LOG.info(f"xarray object dims: {xr_obj.dims}, dtype: {xr_obj.dtype}, attrs: {attrs}")
        variable_arrays[param] = _variable_array(
            param=param,
            chunk_entries=chunk_entries,
            shape=shape,
            chunk_shape=chunk_shape,
            dims=xr_obj.dims,
            dtype=np.dtype("float64"),
            attrs=attrs,
        )
    
    return variable_arrays


def _variable_array(
    param: str,
    chunk_entries: dict[str, ChunkEntry],
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dims: Iterable[str],
    dtype: np.dtype,
    attrs: dict[str, Any],
) -> ManifestArray:
    """Create a ManifestArray for a variable.
    
    Parameters
    ----------
    param : str
        The name of the parameter (e.g., "T", "U").
    chunk_entries : dict[str, ChunkEntry]
        A dictionary mapping chunk keys to ChunkEntry objects.
    shape : tuple[int, ...]
        The shape of the data array.
    chunk_shape : tuple[int, ...]
        The shape of the chunks.
    dims : Iterable[str]
        The dimension names for the array.
    dtype : np.dtype
        The data type of the array.
    attrs : dict[str, Any]
        The attributes of the array.
    
    Returns
    -------
    ManifestArray
        The ManifestArray containing the variable data.
    """
    
    codec = EccodesCodec(var=param).to_dict()
    
    metadata = create_v3_array_metadata(
        shape=shape,
        chunk_shape=chunk_shape,
        data_type=dtype,
        codecs=[codec],
        dimension_names=list(dims),
        attributes=attrs,
    )

    chunk_manifest = ChunkManifest(entries=chunk_entries)
    
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)



def _file_coordinate_array(
    param: str,
    chunk_entry: ChunkEntry,
    shape: list[int],
    dims: list[str],
    dtype: np.dtype,
) -> ManifestArray:
    """Create a ManifestArray for a file coordinate.
    
    Because file coordinates are unique to the file, we create a ManifestArray with a single chunk.
    
    Parameters
    ----------
    param : str
        The name of the parameter (e.g., "valid_time").
    chunk_entry : ChunkEntry
        The ChunkEntry containing the path, offset, and length of the data.
    shape : list[int]
        The shape of the data array, e.g. [1] for a single coordinate.
    dims : list[str]
        The dimension names for the coordinate array, e.g. ['valid_time'], ['cell'] or ['latitude', 'longitude'].
    dtype : np.dtype
        The data type of the array.
    
    Returns
    -------
    ManifestArray
        The ManifestArray containing the file coordinate data.
    """
    codec = EccodesCodec(var=param).to_dict()
    metadata = create_v3_array_metadata(
        shape=tuple(shape),
        chunk_shape=tuple(shape),
        data_type=dtype,
        codecs=[codec],
        dimension_names=dims,
    )
    chunk_key = ChunkKey(".".join(["0"] * len(shape)))
    chunk_manifest = ChunkManifest(entries={chunk_key: chunk_entry})
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


def _level_coordinate_array(
    coord: str, chunk_entries: dict[str, ChunkEntry], dtype: np.dtype = np.int64
) -> ManifestArray:
    """Create a ManifestArray for a vertical level coordinate.
    
    We assume that a file can contain fields at multiple vertical levels, 
    so we create a ManifestArray with multiple chunks. By definition, 
    a vertical level coordinate is one-dimensional, so we create a 
    ManifestArray with a single dimension.
    
    Parameters
    ----------
    coord : str
        The name of the coordinate (e.g., "isobaricInhPa").
    chunk_entries : dict[str, ChunkEntry]
        A dictionary mapping coordinate values to ChunkEntry objects,
        where each ChunkEntry contains the path, offset, and length of the data.
    dtype : np.dtype, optional
        The data type of the array, default is np.float64.
    
    Returns
    -------
    ManifestArray
        The ManifestArray containing the vertical level coordinate data.
    """
    codec = EccodesCodec(var=coord).to_dict()
    sorted_coord_values = dict(
        sorted(chunk_entries.items(), key=lambda item: float(item[0]))
    )
    shape = [len(sorted_coord_values)]
    metadata = create_v3_array_metadata(
        shape=tuple(shape),
        chunk_shape=(1,),
        data_type=dtype,
        codecs=[codec],
        dimension_names=[coord],
    )

    entries: dict[ChunkKey, ChunkEntry] = {}
    for idx, coord_value in enumerate(sorted_coord_values):
        key = f"{idx}"
        chunk_key = ChunkKey(key)
        entries[chunk_key] = sorted_coord_values[coord_value]
    chunk_manifest = ChunkManifest(entries=entries)
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


def _chunk_entry_from_field(field: GribField) -> ChunkEntry:
    """Create a ChunkEntry from a GribField."""
    return ChunkEntry.with_validation(
        path=field.path,
        offset=field.offset,
        length=field._length
    )


__all__ = [
    "IconChParser",
    "extract_metadata",
    "create_file_coordinates_arrays",
    "create_variable_arrays",
]
