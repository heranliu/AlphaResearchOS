# Third-party notices

AlphaResearchOS is distributed under the [MIT License](LICENSE).

The files `types.py`, `parser.py`, and `expression_tree.py` in
`src/alpharesearchos/vendor/factorminer/` incorporate code from
[FactorMiner](https://github.com/minihellboy/factorminer/tree/e21171dbd28b0986597f708816fe354750eb7497),
copyright (c) 2026 FactorMiner Team, licensed under MIT. The complete original
[license](src/alpharesearchos/vendor/factorminer/LICENSE) is included alongside
the code. Local adaptations cover imports, expression validation, missing-value
handling, rolling windows, tied ranks, and numeric serialization. See the
[component notes](docs/UPSTREAM.md) for file provenance and changes.

Python dependencies are distributed by their respective authors under their own
licenses. Benchmark files contain calculated strategy results and evaluation
settings; market data is obtained through the user's chosen data provider.
