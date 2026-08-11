"""Contains tests for the cindra command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner, Result

from cindra.interface.cli import cindra_cli

if TYPE_CHECKING:
    from pathlib import Path


def _configure(output_path: Path, name: str | None = None) -> Result:
    """Runs the 'configure' command against the output directory, optionally naming the generated file."""
    arguments = ["configure", "--pipeline", "single-recording", "--output-path", str(output_path)]
    if name is not None:
        arguments.extend(["--name", name])
    return CliRunner().invoke(cli=cindra_cli, args=arguments)


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

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert list(tmp_path.iterdir()) == []
        assert not (tmp_path.parent / f"{tmp_path.name}.yaml").exists()
