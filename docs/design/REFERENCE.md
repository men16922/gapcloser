# Design reference (Overview, Studio, Benchmark)

Extracted 2026-10-05 from the live CSS of nvidia.com, build.nvidia.com and developer.nvidia.com/isaac/sim.

| Token | Value | Source |
|---|---|---|
| Brand green | `#76B900` (`--brand-nvidia-green`, `--color-brand`) — most frequent hex on the pages (111×) | developer.nvidia.com CSS |
| Brand green hover | `#91C733` (`--brand-nvidia-green-1`, `--color-brand-hover`) | same |
| Gray scale | `#000` · `#111` (950) · `#1A1A1A` (900, `--brand-nvidia-black-1`) · `#222` (800) · `#333` (700) · `#666` (600) · `#808080` (500) · `#999` (400) · `#CCC` (300) · `#EEE` (100) · `#F7F7F7` (50) | `--color-gray-*` |
| Corners | square: `border-radius:0` dominates (89×) over 4px (14×) | developer CSS |
| Typeface | NVIDIA Sans (proprietary, not embeddable) → substitute a Google font with a similar tight industrial feel | `@font-face NVIDIA Sans` |

## Rules applied in the Tether pages

- Dark, near-black surfaces (`#000` page, `#111`/`#1A1A1A` panels, `#333` hairlines), green used sparingly for the single primary accent: active step, success, primary CTA.
- Square corners, thin 1px dividers, uppercase small-caps labels with letter spacing (the Omniverse/Isaac Sim property-panel look).
- Numbers in a monospace face; status colors beyond green are muted (amber for warnings, red for failures) so green stays the signal.
- **No NVIDIA logo or wordmark.** Brand guidelines: assets "may not be used in any manner that isn't expressly authorized in writing by NVIDIA" and no implied endorsement. Credit technologies in plain text only ("Built with NVIDIA Newton · Nemotron via Nebius Token Factory").
