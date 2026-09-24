"""Vorab festgelegte Seeds und Offsets der getrennten RNG-Streams.

Alle Bedingungen verwenden dieselben 45 Seeds (gepaartes Design).
"""

SEEDS = list(range(1, 46))

AUG_STREAM_OFFSET = 10_000          # Rotationswinkel
SHUFFLE_STREAM_OFFSET = 20_000      # Reihenfolge der Mini-Batches
