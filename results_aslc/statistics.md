# Statistische Auswertung (Top-1-Accuracy)

Alle Tests einseitig ('greater'), gepaart. H1a-H1c Holm-korrigiert; H2 separat. Bootstrap B=10,000, Permutation B=10,000.

| Vergleich | n | Diff. > 0 | mittl. Diff. | 95%-KI | dz | t(df) | p | p(Holm) | p(Wilcoxon) | p(Perm) | p(Perm, Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1a (rot05 vs baseline) | 45 | 32 | +0.0062 | [+0.0024, +0.0099] | +0.480 | 3.218 (44) | 0.0012 | 0.0012 | 0.0018 | 0.0011 | 0.0011 |
| H1b (rot10 vs baseline) | 45 | 41 | +0.0156 | [+0.0123, +0.0189] | +1.387 | 9.304 (44) | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0003 |
| H1c (rot20 vs baseline) | 45 | 41 | +0.0192 | [+0.0154, +0.0229] | +1.480 | 9.930 (44) | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0003 |
| H2 (rot10 vs rot20) | 45 | 17 | -0.0036 | [-0.0084, +0.0011] | -0.215 | -1.444 (44) | 0.9221 | -- | 0.9158 | 0.9216 | -- |

Mittlere Differenz in zusaetzlich korrekt klassifizierten Testsequenzen je Lauf: H1a +7.7, H1b +19.4, H1c +23.8, H2 -4.4

TOST H2 (nicht signifikant): Grenze dz=+-0.50 = +-0.0082, p(TOST) = 0.0313

alpha = 0.05; Interpretation primaer ueber Effektgroessen und KIs.

Sensitivitaet (n=45, einseitig, alpha=0.05): Power fuer dz=0.50 betraegt 0.95 (unkorrigiert) bzw. 0.87 (Holm-strengster Vergleich, alpha=0.0167); der kleinste mit 80% Power detektierbare Effekt liegt bei dz~0.38.