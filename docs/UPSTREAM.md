# Numerical component provenance

AlphaResearchOS includes a patched subset of FactorMiner's NumPy expression engine for independent numerical comparison and formula import. Copyright and license notices are preserved in [NOTICE](../NOTICE.md) and the [vendored MIT license](../src/alpharesearchos/vendor/factorminer/LICENSE).

## Included files

Source: [FactorMiner numerical core](https://github.com/minihellboy/factorminer/tree/e21171dbd28b0986597f708816fe354750eb7497), under the MIT License.

- `factorminer/core/types.py`
- `factorminer/core/parser.py`
- `factorminer/core/expression_tree.py`

The files live in `src/alpharesearchos/vendor/factorminer/`. The adapter invokes their `parse(...).evaluate(...)` path on asset-by-time NumPy matrices and compares the result with the local date-by-asset pandas evaluator.

## Local patches

1. Use relative imports within the vendored package.
2. Validate argument counts, finite constants, integral window bounds, and Clip intervals.
3. Bound input length, token count, nesting, and expression-tree size.
4. Preserve NaN for invalid division and exactly zero denominators; retain finite nonzero denominators.
5. Use average ranks for cross-sectional ties.
6. Require complete finite windows for Mean and Std, including nested warmup history.
7. Serialize floating-point constants with round-trip precision.

Patch behavior and import semantics are exercised by `tests/test_factorminer_adapter.py`.

## Supported comparison and import subset

The adapter supports OHLCV fields, finite constants, arithmetic, unary signs, `rank/abs/sign`, `ret/delay/delta` with 1–60 periods, and `mean/std` with 2–250 periods. Standard deviation conversion preserves the respective engines' degrees of freedom. Other valid local DSL expressions retain an explicit skipped-comparison record when they fall outside this contract.

Differential records include the translated formula, compared cells, missing-value agreement, maximum error, and status. Numerical tolerances are `rtol=1e-9, atol=1e-10`.

```python
from alpharesearchos.factorminer_adapter import from_factorminer

expression = from_factorminer("Neg(CsRank(Return($close, 5)))")
# (-rank(ret(close, 5)))
```

The CLI example at [examples/factorminer_candidates.json](../examples/factorminer_candidates.json) exercises formula exchange. Imported strategies are evaluated under AlphaResearchOS's recorded data and trading protocol.
