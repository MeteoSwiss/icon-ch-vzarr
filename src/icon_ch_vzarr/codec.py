from typing import Self

import numpy as np
import eccodes
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration

LEVEL_COORDINATES = {
    "isobaricInhPa",
    "isobaricInPa",
    "generalVerticalLayer",
    "generalVertical",
}
"""Coordinates for 3D variables"""

CODEC_ID = "eccodes_grib_codec"

# Time unit conversion constants
SECONDS_PER_UNIT = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400
}


class EccodesGribCodec(ArrayBytesCodec):
    """Transform GRIB2 bytes into zarr arrays using ecCodes."""

    var: str | None

    def __init__(self, var: str | None = None) -> None:
        object.__setattr__(self, "var", var)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, CODEC_ID, require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if not self.var:
            return {"name": CODEC_ID}
        else:
            return {"name": CODEC_ID, "configuration": {"var": self.var}}

    def _extract_coordinates(self, gid: int) -> np.ndarray:
        """Extract latitude or longitude coordinates from GRIB message."""
        lat_vals = eccodes.codes_get_array(gid, "latitudes")
        lon_vals = eccodes.codes_get_array(gid, "longitudes")
        
        if self.var == "latitude":
            return np.array(lat_vals)
        elif self.var == "longitude":
            return np.array(lon_vals)
        else:
            raise ValueError(f"Unexpected coordinate variable: {self.var}")

    def _extract_valid_time(self, gid: int) -> np.datetime64:
        """Extract valid time from GRIB message."""
        try:
            ref_date = eccodes.codes_get(gid, "validityDate")
            ref_time = eccodes.codes_get(gid, "validityTime")
            
            year = ref_date // 10000
            month = (ref_date // 100) % 100
            day = ref_date % 100
            hour = ref_time // 100
            minute = ref_time % 100
            
            datetime_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"
            return np.datetime64(datetime_str)
        except Exception as e:
            raise ValueError(f"Failed to extract valid_time: {e}") from e

    def _extract_reference_time(self, gid: int) -> np.datetime64:
        """Extract reference time from GRIB message."""
        try:
            ref_date = eccodes.codes_get(gid, "dataDate")
            ref_time = eccodes.codes_get(gid, "dataTime")
            
            # Convert dataDate to string and parse
            date_str = f"{ref_date:08d}"
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            hour = ref_time // 100
            minute = ref_time % 100
            
            datetime_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
            return np.datetime64(datetime_str)
        except Exception as e:
            raise ValueError(f"Failed to extract reference time: {e}") from e

    def _extract_forecast_step(self, gid: int) -> np.timedelta64:
        """Extract forecast step from GRIB message."""
        try:
            fcst_time = eccodes.codes_get(gid, "forecastTime")
            unit = eccodes.codes_get(gid, "stepUnits", default="h")
            
            scale = SECONDS_PER_UNIT.get(unit, 3600)
            return np.timedelta64(fcst_time * scale, "s")
        except Exception as e:
            raise ValueError(f"Failed to extract forecast step: {e}") from e

    def _extract_level_coordinate(self, gid: int) -> np.float64:
        """Extract level coordinate from GRIB message."""
        try:
            level_value = eccodes.codes_get(gid, "level")
            return np.float64(level_value)
        except Exception as e:
            raise ValueError(f"Failed to extract level coordinate: {e}") from e

    def _extract_field_values(self, gid: int) -> np.ndarray:
        """Extract field values from GRIB message."""
        try:
            values = eccodes.codes_get_values(gid)
            return np.array(values)
        except Exception as e:
            raise ValueError(f"Failed to extract field values: {e}") from e

    async def _decode_single(
        self,
        chunk_data: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        if not isinstance(chunk_data, Buffer):
            raise TypeError("chunk_data must be a Buffer instance")
        
        chunk_bytes = chunk_data.to_bytes()
        
        if not chunk_bytes:
            raise ValueError("Empty chunk data provided")

        # Create a temporary handle from bytes
        try:
            gid = eccodes.codes_new_from_message(chunk_bytes)
        except Exception as e:
            raise ValueError(f"Failed to create GRIB message from bytes: {e}") from e

        try:
            # Extract data based on variable type
            if self.var in ("latitude", "longitude"):
                data = self._extract_coordinates(gid)
            elif self.var == "valid_time":
                data = self._extract_valid_time(gid)
            elif self.var == "time":
                data = self._extract_reference_time(gid)
            elif self.var == "step":
                data = self._extract_forecast_step(gid)
            elif self.var in LEVEL_COORDINATES:
                data = self._extract_level_coordinate(gid)
            else:
                data = self._extract_field_values(gid)

            # Convert data type if necessary
            if chunk_spec.dtype != data.dtype:
                try:
                    data = data.astype(chunk_spec.dtype.to_native_dtype())
                except Exception as e:
                    raise ValueError(f"Failed to convert data type from {data.dtype} to {chunk_spec.dtype}: {e}") from e

            # Reshape data if necessary
            if data.shape != chunk_spec.shape:
                try:
                    data = data.reshape(chunk_spec.shape)
                except ValueError as e:
                    raise ValueError(f"Cannot reshape data from {data.shape} to {chunk_spec.shape}: {e}") from e

            return data

        finally:
            eccodes.codes_release(gid)

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # This is a read-only codec
        raise NotImplementedError("EccodesGribCodec is read-only and does not support encoding")

    def compute_encoded_size(
        self, input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError("EccodesGribCodec does not support size computation for encoding")
