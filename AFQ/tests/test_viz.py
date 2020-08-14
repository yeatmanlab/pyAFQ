from AFQ.viz.utils import Viz


def test_viz_name_errors():
    Viz("fury")

    try:
        Viz("plotlyy")
    except TypeError:
        pass
    else:
        AssertionError("Misnamed viz backend did not throw type error")
