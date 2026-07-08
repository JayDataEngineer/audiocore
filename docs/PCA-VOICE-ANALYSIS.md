# PCA Voice Space Analysis

## Overview

The PCA (Principal Component Analysis) tool decomposes the 2048-dimensional
voice embedding space into its principal axes of variation.  This lets you:

1. **Understand what each axis controls** — e.g. PC1 might separate male
   ↔ female voices, PC2 might separate young ↔ old.
2. **Cluster similar voices** — K-means groups voices that are close in
   embedding space.
3. **Generate steering vectors** — Principal components are saved as
   `.dir` files that can be used in Voice Maker → "Apply steering vector"
   to push any voice along a specific axis.

## .voice / .dir File Format

All voice and direction files use the `QWEN3VOICE` binary format:

| Offset | Size  | Field         | Description                     |
|--------|-------|---------------|---------------------------------|
| 0      | 10 B  | Magic         | `QWEN3VOICE` (ASCII)            |
| 10     | 6 B   | Reserved      | Zeros                           |
| 16     | 4 B   | Version       | `uint32` LE, currently `1`      |
| 20     | 4 B   | Dimension     | `uint32` LE, currently `2048`   |
| 24     | 8 B   | Reserved      | Zeros                           |
| 32     | dim×4 | Embedding     | `float32` LE, little-endian     |

- `.voice` files contain a **speaker embedding** (point in voice space).
- `.dir` files contain a **direction vector** (axis in voice space).

Both use the same format — the difference is semantic.

## How to Use

### From the Webapp (Voices tab → Voice Space Analysis)

1. Navigate to the **Voices** tab.
2. Scroll to the **Voice Space Analysis** card.
3. Set the number of PCs (2–30) and clusters (2–10).
4. Click **📊 Analyze**.

Results:
- **Scatter plot**: Each voice is a dot in PC1 × PC2 space, coloured by
  cluster.  Voices close together sound similar.
- **PC extremes table**: For each PC, shows which voices are at the
  positive and negative ends — this identifies what characteristic that
  PC represents.
- **Cluster breakdown**: Lists which voices belong to each cluster.
- **Saved .dir files**: Principal component directions are saved as
  `pca_pc1.dir`, `pca_pc2.dir`, etc. in the voices directory.

### From the Command Line

```bash
python3 scripts/pca_voices.py \
  --voices-dir webapp/clips/voices \
  --components 10 \
  --clusters 5 \
  --save-dirs
```

Add `--json` for machine-readable output.

### Via the API

```bash
curl -X POST http://localhost:39517/v1/voices/analyze \
  -H "Content-Type: application/json" \
  -d '{"components": 10, "clusters": 5, "save_dirs": true}'
```

**Parameters:**
| Field        | Type | Default | Description                          |
|--------------|------|---------|--------------------------------------|
| `components` | int  | 10      | Number of principal components       |
| `clusters`   | int  | 5       | Number of K-means clusters           |
| `save_dirs`  | bool | false   | Save PC directions as `.dir` files   |

**Response:**
```json
{
  "n_voices": 32,
  "dim": 2048,
  "n_components": 10,
  "n_clusters": 5,
  "explained_variance_ratio": [0.36, 0.25, ...],
  "cumulative_variance": [0.36, 0.61, ...],
  "pc_extremes": [
    {"positive_end": "ryan_feminized.voice", "negative_end": "_designed_male.voice"},
    ...
  ],
  "voices": [
    {"name": "preset_vivian.voice", "cluster": 0, "pc": [-0.59, 1.31, ...]},
    ...
  ],
  "saved_dirs": ["pca_pc1.dir", "pca_pc2.dir", ...]
}
```

## Using PC Directions as Steering Vectors

After running analysis with `save_dirs: true`, the PC direction files
appear in Voice Maker → "Apply steering vector" → Direction dropdown.

- **Positive scale** → pushes the voice toward the positive end of that PC.
- **Negative scale** → pushes toward the negative end.

For example, if PC1 separates male (−) from female (+) voices:
- Apply `pca_pc1.dir` with scale +0.5 to feminize any voice.
- Apply `pca_pc1.dir` with scale −0.5 to masculinize.

## Interpretation Guide

The meaning of each PC depends on the voices in your library.  Check the
**PC extremes table** to identify what each axis represents:

- If male presets are at one end and female presets at the other → **gender axis**.
- If child voices are at one end and adult at the other → **age axis**.
- If whispered voices are at one end and loud at the other → **intensity axis**.

The explained variance ratio tells you how much of the total variation
each PC captures.  The first 3–5 PCs typically explain >80% of variance.

## Implementation

- **Script**: `scripts/pca_voices.py` (uses scikit-learn PCA + KMeans)
- **Endpoint**: `POST /v1/voices/analyze` in `src/server/server.cpp`
- **UI**: Voices tab → Voice Space Analysis card in `webapp/public/`
- **Dependencies**: numpy, scikit-learn (Python runtime required)
