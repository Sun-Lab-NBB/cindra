"""Contains tests for the cindra command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from click.testing import Result, CliRunner

from cindra.interface.cli import RoutedErrorGroup, cindra_cli

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


class TestErrorRouting:
    """Tests the conversion of subcommand errors into single-line terminal messages."""

    def test_library_error_becomes_a_single_line_message(self) -> None:
        """Verifies that an arbitrary library exception is reported as one error line instead of a traceback."""
        group = RoutedErrorGroup("probe")

        @group.command("fail")
        def _fail() -> None:
            """Raises a library error."""
            message = "The probe failed."
            raise RuntimeError(message)

        result = CliRunner().invoke(cli=group, args=["fail"])

        assert result.exit_code == 1
        assert "Unable to complete the 'fail' command. RuntimeError: The probe failed." in result.output
        assert "Traceback" not in result.output

    def test_abort_keeps_the_exit_code_click_assigns_it(self) -> None:
        """Verifies that an abort is re-raised rather than converted, since Click renders it itself."""
        group = RoutedErrorGroup("probe")

        @group.command("fail")
        def _fail() -> None:
            """Raises an abort."""
            raise click.Abort

        result = CliRunner().invoke(cli=group, args=["fail"])

        assert result.exit_code == 1
        assert "Aborted!" in result.output

    def test_explicit_exit_code_is_preserved(self) -> None:
        """Verifies that a subcommand exiting with its own code keeps that code."""
        group = RoutedErrorGroup("probe")

        @group.command("fail")
        def _fail() -> None:
            """Exits with a specific code."""
            raise SystemExit(3)

        result = CliRunner().invoke(cli=group, args=["fail"])

        assert result.exit_code == 3

    def test_missing_configuration_file_reports_a_message(self, tmp_path: Path) -> None:
        """Verifies that a run against an absent configuration file reports a message instead of a traceback."""
        result = CliRunner().invoke(cli=cindra_cli, args=["run", "--input-path", str(tmp_path / "absent.yaml")])

        assert result.exit_code == 1
        assert "Unable to complete the 'run' command." in result.output
        assert "Traceback" not in result.output


def _configure(output_path: Path, name: str | None = None) -> Result:
    """Runs the 'configure' command against the output directory, optionally naming the generated file."""
    arguments = ["configure", "--pipeline", "single-recording", "--output-path", str(output_path)]
    if name is not None:
        arguments.extend(["--name", name])
    return CliRunner().invoke(cli=cindra_cli, args=arguments)
