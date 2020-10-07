import pytest

from AFQ.viz.utils import Viz


def test_viz_name_errors():
    Viz("fury")

    with pytest.raises(
        TypeError,
        match="Visualization backend contain"
        + " either 'plotly' or 'fury'. "
            + "It is currently set to plotli"):
        Viz("plotli")
