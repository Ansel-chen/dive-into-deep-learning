from pathlib import Path

import pytest
from PIL import Image

from examples.generate_figures import generate_all_figures


@pytest.fixture
def figure_output_dir():
    output_dir = Path(__file__).resolve().parents[1] / "runs" / "figure-test"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    output_dir.rmdir()


def test_generate_all_figures_creates_readable_pngs(figure_output_dir: Path):
    figures = generate_all_figures(figure_output_dir)
    assert len(figures) == 4
    for figure in figures:
        path = Path(figure)
        assert path.suffix == ".png"
        assert path.exists()
        with Image.open(path) as image:
            image.verify()
