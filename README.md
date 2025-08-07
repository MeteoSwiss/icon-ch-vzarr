# ICON-CH-VZARR

VirtualiZarr parsers for MeteoSwiss ICON-CH GRIB files. 

> [!IMPORTANT]
> This software is an experimental proof-of-concept. Expect bugs and missing features.


This project explores using [earthkit-data](https://github.com/ecmwf/earthkit-data) for parsing GRIB data within use of [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr). Specifically, it focuses on using its xarray engine to control how GRIB metadata is used to create virtualized datasets, in other words, how dimensions, coordinates and attributes are defined. 

The goal is to delegate as much as possible to earthkit-data, and let the logic for mapping GRIB to the Zarr data model entirely depend on a declarative [profile](https://earthkit-data.readthedocs.io/en/latest/guide/xarray/profile.html). 

**Current status**:
- only a parser for KENDA-CH1 (NWP analysis) has been implemented
- we cannot yet persist virtualized datasets using kerchunk references or icechunk
- only works with explicit files, but eventually a FDB-based object store could be implemented

**References**
- VirtualiZarr's documentation on [custom parsers](https://virtualizarr.readthedocs.io/en/stable/custom_parsers.html)
- earthkit-data's [xarray engine](https://earthkit-data.readthedocs.io/en/latest/guide/xarray/overview.html)
- example implementation of a [custom parser for HRRR](https://github.com/virtual-zarr/hrrr-parser)