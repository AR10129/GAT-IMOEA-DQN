# Experiment `inv_g2000`

One shared normalization per workflow across all policies below. IGD ↓ · HV_jin ↑ · Imprecision ↓ · PIBI ↓ · HV_hi ↑. Agent policies: each eval seed is averaged over training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *final* — `+` better, `−` worse, `=` no significant difference.

## Overall (mean over workflows; rank by IGD)

| Policy | mean rank | IGD | HV_jin | Imprecision | PIBI | HV_hi | +/−/= vs final (IGD, HV_jin, Imprecision) |
|---|---|---|---|---|---|---|---|
| final | 1.20 | 0.008651 | 0.877 | 75.63 | 1.752 | 0.6009 | (reference) |
| imoead_orig | 1.80 | 0.01327 | 0.8684 | 67.66 | 1.769 | 0.5799 | 0/0/3 · 0/0/3 · 0/0/3 |

## cybershake_1000@x1

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.007786 ± 0.0038 | 0.9336 ± 0.0049 | 68.09 ± 1.8 | 1.596 ± 0.0033 | 0.6951 ± 0.0038 | 99 | 1.20 |
| imoead_orig | 0.01307 ± 0.0017 = | 0.9165 ± 0.0029 = | 60.48 ± 1.3 = | 1.621 ± 0.0049 = | 0.6646 ± 0.0046 = | 92 | 1.80 |

## epigenomics_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.01016 ± 0.0023 | 0.8866 ± 0.0036 | 76.13 ± 2.9 | 1.716 ± 0.0051 | 0.6156 ± 0.0037 | 100 | 1.00 |
| imoead_orig | 0.01739 ± 0.00074 = | 0.8762 ± 0.0014 = | 68.84 ± 1 = | 1.741 ± 0.0049 = | 0.5903 ± 0.0017 = | 95 | 2.00 |

## sipht_1000@x20

| Policy | IGD | HV_jin | Imprecision | PIBI | HV_hi | front | rank |
|---|---|---|---|---|---|---|---|
| final | 0.008007 ± 0.0037 | 0.8109 ± 0.0056 | 82.69 ± 2.4 | 1.943 ± 0.0072 | 0.492 ± 0.0044 | 99 | 1.40 |
| imoead_orig | 0.009353 ± 0.00042 = | 0.8126 ± 0.0012 = | 73.67 ± 2.9 = | 1.946 ± 0.0022 = | 0.4849 ± 0.00084 = | 93 | 1.60 |
