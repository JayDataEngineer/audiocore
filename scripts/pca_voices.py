#!/usr/bin/env python3
"""
PCA + cluster analysis for Qwen3-TTS voice embeddings.

Loads all .voice files from a directory, performs PCA to identify the
principal axes of variation, clusters voices with K-means, and outputs
results as JSON.  Principal component directions can optionally be saved
as .dir files for use in the Voice Maker's steering-vector knob.

Usage:
  python3 pca_voices.py --voices-dir /path/to/voices [--save-dirs] [--json]
"""
import argparse, json, os, struct, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

HEADER_SIZE = 32
MAGIC = b"QWEN3VOICE"

def load_voice(path):
    """Load a .voice or .dir file → 1D float32 numpy array."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HEADER_SIZE + 4:
        raise ValueError(f"{path}: too small ({len(data)} bytes)")
    magic = data[:len(MAGIC)]
    if magic == MAGIC:
        dim = struct.unpack_from("<I", data, 20)[0]
        emb = np.frombuffer(data, dtype=np.float32, count=dim, offset=HEADER_SIZE)
    else:
        # Raw float32 (no header)
        emb = np.frombuffer(data, dtype=np.float32)
    return np.array(emb, dtype=np.float32)

def analyze(voices_dir, n_components=10, n_clusters=5, save_dirs=False):
    """Run PCA + K-means on all .voice files in voices_dir."""
    # Collect voices
    names, vectors = [], []
    for fname in sorted(os.listdir(voices_dir)):
        if not fname.endswith(".voice"):
            continue
        path = os.path.join(voices_dir, fname)
        try:
            v = load_voice(path)
        except Exception:
            continue
        names.append(fname)
        vectors.append(v)

    if len(vectors) < 2:
        return {"error": f"Need at least 2 voice files, found {len(vectors)}"}

    X = np.array(vectors, dtype=np.float32)  # (n_voices, dim)
    n_voices, dim = X.shape

    # PCA
    n_pc = min(n_components, n_voices - 1, dim)
    pca = PCA(n_components=n_pc)
    projections = pca.fit_transform(X)  # (n_voices, n_pc)

    # K-means clustering
    n_cl = min(n_clusters, n_voices)
    km = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
    labels = km.fit_predict(projections)

    # For each PC, find the voices at the extreme ends — this helps
    # identify what perceptual characteristic that PC captures.
    pc_extremes = []
    for i in range(n_pc):
        proj = projections[:, i]
        pos_idx = int(np.argmax(proj))
        neg_idx = int(np.argmin(proj))
        pc_extremes.append({
            "positive_end": names[pos_idx],
            "negative_end": names[neg_idx],
        })

    # Save PC directions as .dir files (for use as steering vectors)
    saved_dirs = []
    if save_dirs:
        for i in range(n_pc):
            pc_name = f"pca_pc{i+1}.dir"
            pc_path = os.path.join(voices_dir, pc_name)
            # Normalize to unit L2 norm for consistent scaling
            direction = pca.components_[i].astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            # Write in QWEN3VOICE format
            with open(pc_path, "wb") as f:
                f.write(MAGIC)
                f.write(b"\x00" * (HEADER_SIZE - len(MAGIC) - 8))
                f.write(struct.pack("<I", 1))      # version
                f.write(struct.pack("<I", dim))     # dimension
                f.write(b"\x00" * 8)                # padding
                f.write(direction.tobytes())
            saved_dirs.append(pc_name)

    # Build result
    voices_out = []
    for idx in range(n_voices):
        voices_out.append({
            "name": names[idx],
            "cluster": int(labels[idx]),
            "pc": [round(float(projections[idx, j]), 4) for j in range(n_pc)],
        })

    # Sort voices by cluster for readability
    voices_out.sort(key=lambda v: (v["cluster"], v["name"]))

    result = {
        "n_voices": n_voices,
        "dim": dim,
        "n_components": n_pc,
        "n_clusters": n_cl,
        "explained_variance_ratio": [round(float(r), 4) for r in pca.explained_variance_ratio_],
        "cumulative_variance": [round(float(r), 4) for r in np.cumsum(pca.explained_variance_ratio_)],
        "pc_extremes": pc_extremes,
        "voices": voices_out,
        "saved_dirs": saved_dirs,
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--components", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--save-dirs", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = analyze(args.voices_dir, args.components, args.clusters, args.save_dirs)

    if args.json:
        print(json.dumps(result))
    else:
        if "error" in result:
            print(result["error"])
            return
        print(f"PCA Analysis: {result['n_voices']} voices, dim={result['dim']}")
        print(f"Top {result['n_components']} components explain "
              f"{result['cumulative_variance'][-1]*100:.1f}% of variance")
        print()
        for i in range(result["n_components"]):
            pct = result["explained_variance_ratio"][i] * 100
            cum = result["cumulative_variance"][i] * 100
            ext = result["pc_extremes"][i]
            print(f"  PC{i+1}: {pct:5.1f}% var (cum {cum:5.1f}%)  "
                  f"|  −  {ext['negative_end']:<30s}  ←→  "
                  f"{ext['positive_end']:<30s}  +")
        print()
        print(f"K-means clusters (k={result['n_clusters']}):")
        for v in result["voices"]:
            print(f"  [C{v['cluster']}]  {v['name']:<35s}  "
                  f"PC1={v['pc'][0]:+.3f}  PC2={v['pc'][1]:+.3f}")
        if result.get("saved_dirs"):
            print(f"\nSaved {len(result['saved_dirs'])} PC direction files: "
                  f"{', '.join(result['saved_dirs'])}")

if __name__ == "__main__":
    main()
