"""Concrete detection backends.

Each module here adapts one inference library to the :class:`Detector`
interface. They are imported lazily by ``app.perception.detector`` so that a
missing optional dependency never breaks the rest of the package.
"""
