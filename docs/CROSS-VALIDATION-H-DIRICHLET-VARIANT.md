# Cross-validation H — Dirichlet variant (DUPLICATE of E)

*Generated 2026-05-17 by W20-2. Closes operator check H.*

## Verdict

✅ **DUPLICATE of operator check E** — H asks the verbatim same question as E.

Per the operator's directive list (W18 thread):

> E) Are the Dirichlet distributions resulting from Dirichlet filter
>    application computed correctly? (Check the C++ implementation...)
>    Are the Dirichlet distributions resulting from Dirichlet filter
>    application scaled correctly?
>
> H) Are the Dirichlet distributions resulting from Dirichlet filter
>    application computed correctly? (Check the C++ implementation...)
>    Are the Dirichlet distributions resulting from Dirichlet filter
>    application being scaled correctly?

The two prompts are verbatim identical. Closing H by reference to the
W19-3 + W19/v2-reassessment closure of E.

## E closure recap (cite W19-3 receipt)

`docs/CROSS-VALIDATION-V2-REASSESSMENT.md` § "W19-3 receipt":

> The Python `DirichletPredictor.dirichlet_mean_prediction_vec` and
> `dirichlet_mean_map_update` are **verbatim ports** of v2's
> `dirichlet.cpp` functions. Cross-execution on identical fixture
> (5 portfolios × 5 assets × 5 cases) shows agreement to 1e-16
> absolute (machine precision).

| Function | abs_max diff | rel_max diff |
|---|---|---|
| `dirichlet_mean_prediction_vec` | 1.11e-16 | 3.96e-16 |
| `dirichlet_mean_map_update` | 1.11e-16 | 5.07e-16 |

No bugs in either side. The W19-3 cross-check covered both
sub-checks (prediction + MAP update) of the Dirichlet filter and
both AGREE machine precision against `legacy-cpp-v2/source/dirichlet.cpp`.

## Scaling check (also covered by W19-3)

The compare framework's scale_ratio column in the W19-3 receipt was
1.0000 for both functions → no scaling drift.

## Verdict per W18 matrix

✅ **AGREE machine precision** (inherited from W19-3 closure).

## Bug count after W20-2

Unchanged from W19. No new findings.

## Reproducing

See `docs/CROSS-VALIDATION-V2-REASSESSMENT.md` § "W19-3 receipt" and
`docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md` for the broader
v2-substrate context.
