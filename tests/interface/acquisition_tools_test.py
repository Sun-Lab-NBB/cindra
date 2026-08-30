"""Contains tests for the acquisition parameter generation and validation MCP tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import imwrite

from cindra.io import find_data_directory
from cindra.layout import PARAMETERS_FILENAME
from cindra.interface.acquisition_tools import (
    _is_integer_list,
    _summarize_roi_line_blocks,
    _validate_acquisition_parameters,
    validate_recording_readiness_tool,
    generate_acquisition_parameters_file_tool,
    validate_acquisition_parameters_file_tool,
)

if TYPE_CHECKING:
    from pathlib import Path


_MROI_GEOMETRY: dict[str, object] = {
    "frame_rate": 30.0,
    "roi_number": 2,
    "roi_x_coordinates": [0, 64],
    "roi_y_coordinates": [0, 0],
}
"""The two-region acquisition geometry the span expansion tests share."""


class TestIntegerListGuard:
    """Tests the element check that the MROI validations share."""

    @pytest.mark.parametrize("value", [[0, 1, 2], []])
    def test_integer_lists_are_accepted(self, value: list[int]) -> None:
        """Verifies that a list holding integer elements alone passes the check."""
        assert _is_integer_list(value=value)

    @pytest.mark.parametrize("value", [["0"], [0.0], [True], (0, 1), "01", None])
    def test_every_other_value_is_rejected(self, value: object) -> None:
        """Verifies that a non-list, or a list holding a string, float, or boolean, fails the check."""
        assert not _is_integer_list(value=value)


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


class TestRoiLineBlockSummaries:
    """Tests the per-ROI summaries that replace the line arrays in every validation response."""

    def test_contiguous_blocks_are_summarized_by_their_span(self) -> None:
        """Verifies that a duplicate-free block covering an unbroken run of rows reports a contiguous span."""
        summaries = _summarize_roi_line_blocks(roi_lines=[list(range(16)), list(range(16, 32))])

        assert summaries == [
            {"roi": 0, "line_count": 16, "span": [0, 15], "contiguous": True},
            {"roi": 1, "line_count": 16, "span": [16, 31], "contiguous": True},
        ]

    @pytest.mark.parametrize("lines", [[0, 1, 3], [0, 1, 1]])
    def test_broken_and_duplicated_blocks_are_not_contiguous(self, lines: list[int]) -> None:
        """Verifies that a block skipping a row, or repeating one, is reported as non-contiguous."""
        summaries = _summarize_roi_line_blocks(roi_lines=[lines])

        assert summaries[0]["contiguous"] is False
        assert summaries[0]["line_count"] == 3

    @pytest.mark.parametrize(
        "lines,line_count",
        [
            ([], 0),
            (["0", "1"], 2),
            ("01", 0),
        ],
    )
    def test_malformed_blocks_degrade_to_a_line_count(self, lines: object, line_count: int) -> None:
        """Verifies that an empty, non-integer, or non-list block reports an empty span and no contiguity."""
        summaries = _summarize_roi_line_blocks(roi_lines=[lines])

        assert summaries == [{"roi": 0, "line_count": line_count, "span": [], "contiguous": False}]


class TestAcquisitionParametersFileGeneration:
    """Tests the generation tool that writes the parameters file into the raw imaging directory."""

    def test_the_file_is_written_into_the_raw_data_path(self, tmp_path: Path) -> None:
        """Verifies that the tool writes the parameters file into the directory named by raw_data_path."""
        result = generate_acquisition_parameters_file_tool(raw_data_path=str(tmp_path), frame_rate=30.0)

        assert result["success"] is True
        assert result["file_path"] == str(tmp_path / PARAMETERS_FILENAME)
        assert json.loads((tmp_path / PARAMETERS_FILENAME).read_text())["frame_rate"] == 30.0

    def test_a_missing_raw_data_path_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a raw_data_path pointing at no directory reports an error naming that path."""
        missing = tmp_path / "absent"

        result = generate_acquisition_parameters_file_tool(raw_data_path=str(missing), frame_rate=30.0)

        assert result["success"] is False
        assert str(missing) in result["error"]


class TestRoiLineSpanExpansion:
    """Tests the span form of the generation tool, which names each region by its first and last row index."""

    def test_spans_expand_to_the_same_file_as_enumerated_lines(self, tmp_path: Path) -> None:
        """Verifies that a span request and an enumerated request describing the same regions write the same file."""
        span_directory = tmp_path / "spans"
        enumerated_directory = tmp_path / "enumerated"
        span_directory.mkdir()
        enumerated_directory.mkdir()

        span_result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(span_directory), roi_line_spans=[[0, 15], [24, 39]], **_MROI_GEOMETRY
        )
        enumerated_result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(enumerated_directory),
            roi_lines=[list(range(16)), list(range(24, 40))],
            **_MROI_GEOMETRY,
        )

        assert span_result["success"] is True
        assert enumerated_result["success"] is True
        written = json.loads((span_directory / PARAMETERS_FILENAME).read_text())
        assert written == json.loads((enumerated_directory / PARAMETERS_FILENAME).read_text())
        assert written["roi_lines"] == [list(range(16)), list(range(24, 40))]

    def test_the_response_summarizes_the_expanded_lines(self, tmp_path: Path) -> None:
        """Verifies that the response carries the per-region summary of the expanded spans."""
        result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(tmp_path), roi_line_spans=[[0, 15], [24, 39]], **_MROI_GEOMETRY
        )

        assert result["parameters"]["roi_lines"] == [
            {"roi": 0, "line_count": 16, "span": [0, 15], "contiguous": True},
            {"roi": 1, "line_count": 16, "span": [24, 39], "contiguous": True},
        ]

    def test_a_single_row_region_is_accepted(self, tmp_path: Path) -> None:
        """Verifies that inclusive bounds naming one row expand to that single row index."""
        result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(tmp_path), roi_line_spans=[[7, 7], [24, 39]], **_MROI_GEOMETRY
        )

        assert result["success"] is True
        assert json.loads((tmp_path / PARAMETERS_FILENAME).read_text())["roi_lines"][0] == [7]

    def test_supplying_both_forms_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that naming the regions twice reports an error rather than preferring one form."""
        result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(tmp_path),
            roi_lines=[list(range(16)), list(range(24, 40))],
            roi_line_spans=[[0, 15], [24, 39]],
            **_MROI_GEOMETRY,
        )

        assert result["success"] is False
        assert "'roi_lines' and 'roi_line_spans'" in result["error"]
        assert not (tmp_path / PARAMETERS_FILENAME).exists()

    @pytest.mark.parametrize(
        ("spans", "expected"),
        [
            ([[0, 15], [39, 24]], "precedes its first row"),
            ([[-1, 15], [24, 39]], "cannot be negative"),
            ([[0, 15, 20], [24, 39]], "exactly two integers"),
            ([[True, 15], [24, 39]], "exactly two integers"),
            ([[0, 15]], "must equal 'roi_number'"),
            ([], "non-empty list"),
        ],
    )
    def test_malformed_spans_are_rejected(self, tmp_path: Path, spans: object, expected: str) -> None:
        """Verifies that every malformed span request reports its own reason and writes no file."""
        result = generate_acquisition_parameters_file_tool(
            raw_data_path=str(tmp_path), roi_line_spans=spans, **_MROI_GEOMETRY
        )

        assert result["success"] is False
        assert any(expected in error for error in result["errors"])
        assert not (tmp_path / PARAMETERS_FILENAME).exists()


class TestAcquisitionParametersFileValidation:
    """Tests the file validator, its compacted parameter echo, and its line slice requests."""

    def test_roi_lines_are_replaced_by_per_roi_summaries(self, tmp_path: Path) -> None:
        """Verifies that the echoed parameters carry one summary entry per ROI."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path))

        assert result["valid"] is True
        assert result["parameters"]["roi_lines"] == [
            {"roi": 0, "line_count": 16, "span": [0, 15], "contiguous": True},
            {"roi": 1, "line_count": 16, "span": [16, 31], "contiguous": True},
        ]
        assert result["parameters"]["roi_x_coordinates"] == [0, 0]

    def test_a_requested_slice_returns_the_named_lines(self, tmp_path: Path) -> None:
        """Verifies that a valid slice request returns the half-open range of the named ROI's line list."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[1, 0, 3])

        assert result["roi_line_slice"] == {"roi": 1, "start": 0, "stop": 3, "lines": [16, 17, 18]}

    @pytest.mark.parametrize("request_triplet", [[0, 1], [0, 1, 2, 3], [0, 0, True]])
    def test_a_request_that_is_not_three_integers_is_rejected(
        self, tmp_path: Path, request_triplet: list[object]
    ) -> None:
        """Verifies that a slice request of the wrong length, or one holding a boolean, is rejected."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[[0, 1], [2, 3]])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=request_triplet)

        assert result["success"] is False
        assert result["error"].startswith("Unable to validate acquisition parameters file.")
        assert "must hold exactly three integers" in result["error"]

    @pytest.mark.parametrize("roi_index", [-1, 2])
    def test_an_out_of_range_roi_index_is_rejected(self, tmp_path: Path, roi_index: int) -> None:
        """Verifies that a slice request naming an ROI the parameters do not cover reports the valid range."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[[0, 1], [2, 3]])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[roi_index, 0, 1])

        assert result["success"] is False
        assert "ROI index must be in the range [0, 1]" in result["error"]

    @pytest.mark.parametrize("bounds", [[-1, 2], [1, 1], [0, 3]])
    def test_out_of_range_slice_bounds_are_rejected(self, tmp_path: Path, bounds: list[int]) -> None:
        """Verifies that a negative start, an empty range, and a stop past the last line are each rejected."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[[0, 1], [2, 3]])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[0, *bounds])

        assert result["success"] is False
        assert "bounds must satisfy 0 <= start < stop <= 2" in result["error"]

    def test_a_slice_of_a_non_integer_block_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that an ROI holding non-integer line indices cannot be sliced."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[["0", "1"], [2, 3]])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[0, 0, 1])

        assert result["success"] is False
        assert "ROI 0 does not hold a list of integer line indices" in result["error"]

    def test_a_slice_of_parameters_without_roi_lines_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a slice request against single-ROI parameters reports the absent line entries."""
        file_path = tmp_path / PARAMETERS_FILENAME
        file_path.write_text(json.dumps(obj={"frame_rate": 30.0, "plane_number": 1, "channel_number": 1}))

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[0, 0, 1])

        assert result["success"] is False
        assert "hold no 'roi_lines' entries to slice" in result["error"]

    def test_a_slice_wider_than_the_cap_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a slice covering more lines than a single request serves reports the cap."""
        file_path = _write_parameters_file(directory=tmp_path, roi_lines=[list(range(2500)), list(range(2500, 5000))])

        result = validate_acquisition_parameters_file_tool(file_path=str(file_path), roi_line_slice=[0, 0, 2100])

        assert result["success"] is False
        assert "exceeds the 2000 line cap" in result["error"]


class TestRecordingReadiness:
    """Tests the readiness gate that resolves the imaging directory beneath the named path, cross-validates its
    parameter file against the raw TIFF files, and serves ROI line slice requests."""

    def test_a_ready_recording_echoes_the_raw_data_path(self, tmp_path: Path) -> None:
        """Verifies that a directory holding the parameters file and a matching TIFF validates and echoes its path."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["success"] is True
        assert result["valid"] is True
        assert result["raw_data_path"] == str(tmp_path)
        assert result["tiff_file_count"] == 1
        assert "errors" not in result

    def test_roi_lines_are_replaced_by_per_roi_summaries(self, tmp_path: Path) -> None:
        """Verifies that the echoed acquisition parameters carry one summary entry per ROI."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["acquisition_parameters"]["roi_lines"] == [
            {"roi": 0, "line_count": 16, "span": [0, 15], "contiguous": True},
            {"roi": 1, "line_count": 16, "span": [16, 31], "contiguous": True},
        ]

    def test_a_requested_slice_returns_the_named_lines(self, tmp_path: Path) -> None:
        """Verifies that a valid slice request returns the requested line indices alongside the summaries."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path), roi_line_slice=[1, 0, 3])

        assert result["roi_line_slice"] == {"roi": 1, "start": 0, "stop": 3, "lines": [16, 17, 18]}

    def test_a_rejected_slice_carries_the_readiness_prefix(self, tmp_path: Path) -> None:
        """Verifies that a slice request the readiness tool refuses reports its own error prefix."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path), roi_line_slice=[5, 0, 1])

        assert result["success"] is False
        assert result["error"].startswith("Unable to validate recording readiness.")
        assert "ROI index must be in the range [0, 1]" in result["error"]

    def test_a_block_reaching_past_the_frame_height_is_reported(self, tmp_path: Path) -> None:
        """Verifies that a line index past the last row of the raw frame invalidates the recording."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(16, 48))], frame_height=32)

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["valid"] is False
        assert any(
            "ROI 1 roi_lines maximum (47) reaches past the last row of the raw frame, which holds 32 rows" in error
            for error in result["errors"]
        )

    def test_overlapping_blocks_are_reported_as_a_warning(self, tmp_path: Path) -> None:
        """Verifies that consecutive ROI blocks claiming the same raw rows produce an overlap warning."""
        _write_recording(directory=tmp_path, roi_lines=[list(range(16)), list(range(12, 28))], frame_height=32)

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["valid"] is True
        assert any("overlaps ROI 0" in warning for warning in result["warnings"])

    def test_a_gap_is_reported_without_a_readable_tiff(self, tmp_path: Path) -> None:
        """Verifies that the block cross-check reports a gap even when no TIFF file yields a frame height."""
        _write_parameters_file(directory=tmp_path, roi_lines=[[0, 1], [4, 5]])
        (tmp_path / "broken.tif").write_bytes(b"not a tiff at all")

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert "frame_height" not in result
        assert any("leaving 2 raw rows unassigned after ROI 0" in warning for warning in result["warnings"])

    @pytest.mark.parametrize("roi_lines", [[["0", "1"], ["2", "3"]], [[0.0, 1.0], [2.0, 3.0]]])
    def test_non_integer_roi_lines_report_an_invalid_recording(
        self, tmp_path: Path, roi_lines: list[list[object]]
    ) -> None:
        """Verifies that non-integer line indices produce an invalid verdict instead of raising a TypeError."""
        _write_recording(directory=tmp_path, roi_lines=roi_lines)

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["success"] is True
        assert result["valid"] is False
        assert "'roi_lines' must be a list of lists of integers." in result["errors"]

    def test_a_parameters_file_one_level_down_resolves_that_directory(self, tmp_path: Path) -> None:
        """Verifies that a recording root is validated through the imaging directory nested inside it."""
        raw_directory = tmp_path / "mesoscope_data"
        raw_directory.mkdir()
        _write_recording(directory=raw_directory, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["success"] is True
        assert result["valid"] is True
        assert result["raw_data_path"] == str(raw_directory)

    def test_several_parameters_files_resolve_the_one_the_conversion_reads(self, tmp_path: Path) -> None:
        """Verifies that a root holding several candidates resolves the same directory the conversion would."""
        for name in ("recording_a", "recording_b"):
            raw_directory = tmp_path / name / "mesoscope_data"
            raw_directory.mkdir(parents=True)
            _write_recording(directory=raw_directory, roi_lines=[list(range(16)), list(range(16, 32))])

        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["success"] is True
        assert result["raw_data_path"] == str(find_data_directory(data_path=tmp_path))

    def test_an_empty_subtree_asks_for_a_generated_file(self, tmp_path: Path) -> None:
        """Verifies that a directory whose subtree holds no parameters file points at the generation tool."""
        result = validate_recording_readiness_tool(raw_data_path=str(tmp_path))

        assert result["success"] is False
        assert "or anywhere in its subtree" in result["error"]
        assert "generate_acquisition_parameters_file_tool" in result["error"]


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


def _write_parameters_file(directory: Path, roi_lines: object) -> Path:
    """Writes a two-ROI acquisition parameters file carrying the supplied line blocks into the directory."""
    file_path = directory / PARAMETERS_FILENAME
    file_path.write_text(
        json.dumps(obj=_mroi_parameters(roi_lines=roi_lines, roi_x_coordinates=[0, 0], roi_y_coordinates=[0, 16]))
    )
    return file_path


def _write_recording(directory: Path, roi_lines: object, frame_height: int = 32) -> None:
    """Writes a two-ROI parameters file and a single-page TIFF of the requested frame height into the directory."""
    _write_parameters_file(directory=directory, roi_lines=roi_lines)
    imwrite(directory / "recording.tif", data=np.zeros((frame_height, 32), dtype=np.uint16))
