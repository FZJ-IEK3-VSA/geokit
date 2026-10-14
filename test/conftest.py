"""Use matplotlib's non-interactive backend: the drawing tests must not need Tk or Qt."""

import matplotlib

matplotlib.use("Agg")
