"""Contains tests for the acquisition parameter validation MCP tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import imwrite

from cindra.layout import PARAMETERS_FILENAME
from cindra.interface.acquisition_tools import (
    _is_integer_list,
    _validate_acquisition_parameters,
    validate_recording_readiness_tool,
)

if TYPE_CHECKING:
    from pathlib import Path


def _mroi_parameters(roi_lines: object, roi_x_coordinates: object, roi_y_coordinates: object) -> dict[str, object]:
    """Builds a two-ROI acquisition parameter mapping around the supplied MROI fields."""
    return {
        "frame_rate": 30.0,
        "plane_number": 1,
        "channel_number": 1,
        "roi_number": 2,
        "roi_lines": roi_lines,
        "roi_x_coordinates": roi_x_coordinates,
        "roi_y_coordinates": roi_y_coordinates,
    }


class TestIntegerListGuard:
    """Tests the element check that the MROI validations share."""

    @pytest.mark.parametrize("value", [[0, 1, 2], []])
    def test_integer_lists_are_accepted(self, value: list[int]) -> None:
        """Verifies that a list holding integer elements alone passes the check."""
        assert _is_integer_list(value=value) is True

    @pytest.mark.parametrize("value", [["0"], [0.0], [True], (0, 1), "01", None])
    def test_every_other_value_is_rejected(self, value: object) -> None:
        """Verifies that a non-list, or a list holding a string, float, or boolean, fails the check."""
        assert _is_integer_list(value=value) is False


class TestAcquisitionParameterValidation:
    """Tests the shared validator that the generation, file-validation, and readiness tools all run."""

    @pytest.mark.parametrize(
        "roi_lines",
        [
            [["0", "1"], ["2", "3"]],
            [[0.0, 255.0], [256.0, 511.0]],
            [[True, False], [0, 1]],
        ],
    )
    def test_non_integer_roi_lines_are_reported(self, roi_lines: list[list[object]]) -> None:
        """Verifies that a roi_lines entry holding a string, float, or boolean element is reported as an error."""
        errors, _ = _validate_acquisition_parameters(
            data=_mroi_parameters(roi_lines=roi_lines, roi_x_coordinates=[0, 0], roi_y_coordinates=[0, 16])
        )

        assert "'roi_lines' must be a list of lists of integers." in errors

    @pytest.mark.parametrize("coordinates", [["0", "1"], [0.0, 16.0], [True, False]])
    def test_non_integer_roi_coordinates_are_reported(self, coordinates: list[object]) -> None:
        """Verifies that non-integer x and y ROI offsets are each reported as an error."""
        x_errors, _ = _validate_acquisition_parameters(
            data=_mroi_parameters(roi_lines=[[0, 1], [2, 3]], roi_x_coordinates=coordinates, roi_y_coordinates=[0, 16])
        )
        y_errors, _ = _validate_acquisition_parameters(
            data=_mroi_parameters(roi_lines=[[0, 1], [2, 3]], roi_x_coordinates=[0, 0], roi_y_coordinates=coordinates)
        )

        assert "'roi_x_coordinates' must be a list of integers." in x_errors
        assert "'roi_y_coordinates' must be a list of integers." in y_errors

    def test_integer_mroi_fields_validate(self) -> None:
        """Verifies that an MROI mapping holding integer line indices and offsets produces no errors."""
        errors, _ = _validate_acquisition_parameters(
            data=_mroi_parameters(roi_lines=[[0, 1], [2, 3]], roi_x_coordinates=[0, 0], roi_y_coordinates=[0, 16])
        )

        assert errors == []


class TestRecordingReadiness:
    """Tests the readiness gate that cross-validates the parameter file against the raw TIFF files."""

    @pytest.mark.parametrize("roi_lines", [[["0", "1"], ["2", "3"]], [[0.0, 1.0], [2.0, 3.0]]])
    def test_non_integer_roi_lines_report_an_invalid_recording(
        self, tmp_path: Path, roi_lines: list[list[object]]
    ) -> None:
        """Verifies that non-integer line indices produce an invalid verdict instead of raising a TypeError."""
        parameters = _mroi_parameters(roi_lines=roi_lines, roi_x_coordinates=[0, 0], roi_y_coordinates=[0, 16])
        (tmp_path / PARAMETERS_FILENAME).write_text(json.dumps(obj=parameters))
        imwrite(tmp_path / "recording.tif", np.zeros((4, 32, 32), dtype=np.uint16))

        result = validate_recording_readiness_tool(recording_directory=str(tmp_path))

        assert result["success"] is True
        assert result["valid"] is False
        assert "'roi_lines' must be a list of lists of integers." in result["errors"]
