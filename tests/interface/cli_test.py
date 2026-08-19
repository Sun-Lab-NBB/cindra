"""Contains tests for the cindra command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from click.testing import Result, CliRunner

from cindra.interface.cli import cindra_cli, report_command_failure

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigureCommand:
    """Tests the destination the configuration generation command resolves from its name option."""

    def test_dotted_name_keeps_every_component(self, tmp_path: Path) -> None:
        """Verifies that a name carrying a dot is extended rather than truncated at its last dot."""
        result = _configure(output_path=tmp_path, name="mouse5_2024.03.01")

        assert result.exit_code == 0
        assert (tmp_path / "mouse5_2024.03.01.yaml").exists()
        assert not (tmp_path / "mouse5_2024.03.yaml").exists()

    def test_yml_name_is_normalized_to_yaml(self, tmp_path: Path) -> None:
        """Verifies that a '.yml' name resolves to the '.yaml' extension the pipeline loader requires."""
        result = _configure(output_path=tmp_path, name="session.yml")

        assert result.exit_code == 0
        assert (tmp_path / "session.yaml").exists()

    def test_omitted_name_uses_the_default(self, tmp_path: Path) -> None:
        """Verifies that the single-recording default name is used when no name is supplied."""
        result = _configure(output_path=tmp_path)

        assert result.exit_code == 0
        assert (tmp_path / "cindra_sd_conf.yaml").exists()

    def test_blank_name_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a whitespace-only name errors instead of writing a file beside the output directory."""
        result = _configure(output_path=tmp_path, name="   ")

        assert result.exit_code == 2
        assert "must carry at least one non-whitespace" in result.output
        assert "Traceback" not in result.output
        assert list(tmp_path.iterdir()) == []
        assert not (tmp_path.parent / f"{tmp_path.name}.yaml").exists()


class TestErrorReporting:
    """Tests the reporting of command failures through the console."""

    def test_library_error_is_reported_without_a_traceback(self) -> None:
        """Verifies that an exception raised by a command body is reported instead of reaching the interpreter."""

        @click.command("fail")
        @report_command_failure
        def _fail() -> None:
            """Raises a library error."""
            message = "The probe failed."
            raise RuntimeError(message)

        result = CliRunner().invoke(cli=_fail, args=[])

        assert result.exit_code == 0
        assert result.exception is None
        assert "Traceback" not in result.output

    def test_missing_configuration_file_is_reported(self, tmp_path: Path) -> None:
        """Verifies that a run against an absent configuration file reports a message instead of a traceback."""
        result = CliRunner().invoke(cli=cindra_cli, args=["run", "--input-path", str(tmp_path / "absent.yaml")])

        assert result.exit_code == 0
        assert result.exception is None
        assert "Traceback" not in result.output

    def test_usage_error_from_a_body_passes_through(self) -> None:
        """Verifies that a usage error a command body raises keeps Click's banner and exit status."""

        @click.command("fail")
        @report_command_failure
        def _fail() -> None:
            """Raises a usage error."""
            message = "The probe rejected its argument."
            raise click.UsageError(message)

        result = CliRunner().invoke(cli=_fail, args=[])

        assert result.exit_code == 2
        assert "The probe rejected its argument." in result.output

    def test_malformed_option_still_exits_two(self) -> None:
        """Verifies that Click parameter validation runs ahead of the wrapped body and keeps its own exit code."""
        result = CliRunner().invoke(cli=cindra_cli, args=["configure", "--pipeline", "not-a-pipeline"])

        assert result.exit_code == 2


def _configure(output_path: Path, name: str | None = None) -> Result:
    """Runs the 'configure' command against the output directory, optionally naming the generated file."""
    arguments = ["configure", "--pipeline", "single-recording", "--output-path", str(output_path)]
    if name is not None:
        arguments.extend(["--name", name])
    return CliRunner().invoke(cli=cindra_cli, args=arguments)
