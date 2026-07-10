"""medal_predictor — Olympic medal prediction package.

End-to-end pipeline: ``data`` -> ``features`` (+ leakage-safe ``leakage`` rates) ->
``cv`` walk-forward splits -> ``model`` -> ``evaluation`` -> ``betting``.
See ``SUMMARY.md`` for the project overview.
"""

__version__ = "0.1.0"
