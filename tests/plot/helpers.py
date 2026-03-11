import matplotlib
import numpy as np


def extract_heatmap_annotations(ax: matplotlib.axes.Axes, n: int) -> dict[tuple[int, int], str]:
    """Return a ``{(row, col): text}`` dict for the *n* x *n* heatmap cells."""
    result: dict[tuple[int, int], str] = {}
    for t in ax.texts:
        if t.get_transform() != ax.transData:
            continue
        x, y = t.get_position()
        col = round(x - 0.5)
        row = round(y - 0.5)
        if 0 <= row < n and 0 <= col < n:
            result[(row, col)] = t.get_text()
    return result


def extract_heatmap_texts(
    ax: matplotlib.axes.Axes,
    n: int,
) -> dict[tuple[int, int], matplotlib.text.Text]:
    """Return a ``{(row, col): text_artist}`` dict for the *n* x *n* heatmap cells."""
    result: dict[tuple[int, int], matplotlib.text.Text] = {}
    for t in ax.texts:
        if t.get_transform() != ax.transData:
            continue
        x, y = t.get_position()
        col = round(x - 0.5)
        row = round(y - 0.5)
        if 0 <= row < n and 0 <= col < n:
            result[(row, col)] = t
    return result


def extract_heatmap_collection(
    ax: matplotlib.axes.Axes,
    collection_index: int,
    n: int,
) -> np.ma.MaskedArray:
    """Return a heatmap collection as an ``n x n`` masked array."""
    return np.ma.asarray(ax.collections[collection_index].get_array()).reshape(n, n)
