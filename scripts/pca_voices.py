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

    # ── Semantic axis labeling ───────────────────────────────────────────
    # Each PC is an abstract eigenvector. We name it by correlating the PC
    # projection against binary attributes parsed from filenames using
    # point-biserial correlation (Pearson r between binary 0/1 and continuous
    # projection). Each attribute is assigned to at most ONE PC — the one
    # where it correlates most strongly — so labels never repeat.
    def _short(name):
        return name.replace(".voice", "").replace("_", " ").strip()

    FEMALE_HINTS = ("female", "girl", "femin", "anna", "serena", "vivian",
                    "sohee", "chelsie")
    MALE_HINTS = ("male", "ogre", "aiden", "dylan", "eric", "ryan",
                  "ethan", "uncle_fu")

    def _gender_vec():
        """+1 = female, -1 = male, nan = unknown."""
        v = np.full(n_voices, np.nan)
        for i, nm in enumerate(names):
            n = nm.lower()
            if any(h in n for h in FEMALE_HINTS):   v[i] = 1
            elif any(h in n for h in MALE_HINTS):   v[i] = -1
        return v

    def _derived_vec():
        """+1 = shifted/blended/feminized (derived from another voice),
        -1 = original preset/designed/named voice, nan = ambiguous."""
        DERIVED = ("shift", "blend", "slerp", "feminized", "_to_")
        ORIGINAL = ("preset", "designed")
        v = np.full(n_voices, np.nan)
        for i, nm in enumerate(names):
            n = nm.lower()
            if any(k in n for k in DERIVED):    v[i] = 1
            elif any(k in n for k in ORIGINAL): v[i] = -1
        return v

    def _preset_vec():
        """+1 = preset_*, -1 = everything else with a clear origin."""
        v = np.full(n_voices, np.nan)
        for i, nm in enumerate(names):
            n = nm.lower()
            if "preset" in n:               v[i] = 1
            elif "designed" in n:            v[i] = -1
            elif nm.replace(".voice","") in (
                "chelsie","ethan","serena","vivian","youth_voice"):
                v[i] = -1  # named originals
        return v

    def _designed_vec():
        """+1 = _designed_*, -1 = preset_*."""
        v = np.full(n_voices, np.nan)
        for i, nm in enumerate(names):
            n = nm.lower()
            if "designed" in n:  v[i] = 1
            elif "preset" in n:  v[i] = -1
        return v

    # (display_name, vec, pos_label, neg_label)
    ATTRS = [
        ("Gender",   _gender_vec(),   "feminine",  "masculine"),
        ("Derived",  _derived_vec(),  "modified",  "original"),
        ("Preset",   _preset_vec(),   "preset",    "custom"),
        ("Designed", _designed_vec(), "designed",  "preset"),
    ]
    MIN_N = 4  # minimum known values to compute a correlation

    def pb_corr(proj, vec):
        """Point-biserial correlation: Pearson r between binary vec and proj.
        Only uses voices where vec is not nan."""
        mask = ~np.isnan(vec)
        if mask.sum() < MIN_N:
            return None
        x, y = vec[mask], proj[mask]
        # Need at least 2 distinct values for correlation to be meaningful.
        if len(set(x)) < 2:
            return None
        r = float(np.corrcoef(x, y)[0, 1])
        hi_label = 1 if np.mean(proj[vec > 0]) > np.mean(proj[vec < 0]) else -1
        return {"r": r, "hi_sign": hi_label}

    # Score every (PC, attribute) pair.
    # Key: (pc_idx, attr_name) → {r, hi_sign, pos_label, neg_label}
    scores = {}
    for i in range(n_pc):
        proj = projections[:, i]
        for label_name, vec, pos_lab, neg_lab in ATTRS:
            r = pb_corr(proj, vec)
            if r is not None:
                scores[(i, label_name)] = {**r,
                    "pos_label": pos_lab if r["hi_sign"] > 0 else neg_lab,
                    "neg_label": neg_lab if r["hi_sign"] > 0 else pos_lab}

    # Two-pass assignment:
    # Pass 1 — variance order, no duplicates, threshold 0.30. Ensures the
    #   highest-variance PCs get first pick of attributes.
    # Pass 2 — remaining PCs may reuse an attribute if its |r| ≥ 0.50 (i.e.
    #   the correlation is strong enough to be meaningful even on a shared
    #   label). This lets e.g. Gender label both a weak-PC and strong-PC axis.
    used_attrs = set()
    assignments = {}
    for pc_i in range(n_pc):
        best_attr, best_r = None, 0.0
        for attr_name, _vec, _pos, _neg in ATTRS:
            if attr_name in used_attrs:
                continue
            key = (pc_i, attr_name)
            if key not in scores:
                continue
            ar = abs(scores[key]["r"])
            if ar >= 0.30 and ar > best_r:
                best_attr, best_r = attr_name, ar
        if best_attr is not None:
            r = scores[(pc_i, best_attr)]
            assignments[pc_i] = {
                "label": best_attr,
                "pos": r["pos_label"],
                "neg": r["neg_label"],
                "score": round(best_r, 3),
            }
            used_attrs.add(best_attr)

    # Pass 2: allow reuse for strong correlations on unlabeled PCs.
    for pc_i in range(n_pc):
        if pc_i in assignments:
            continue
        best_attr, best_r = None, 0.0
        for attr_name, _vec, _pos, _neg in ATTRS:
            key = (pc_i, attr_name)
            if key not in scores:
                continue
            ar = abs(scores[key]["r"])
            if ar >= 0.45 and ar > best_r:
                best_attr, best_r = attr_name, ar
        if best_attr is not None:
            r = scores[(pc_i, best_attr)]
            assignments[pc_i] = {
                "label": best_attr,
                "pos": r["pos_label"],
                "neg": r["neg_label"],
                "score": round(best_r, 3),
            }

    axis_labels = []
    for i in range(n_pc):
        if i in assignments:
            axis_labels.append(assignments[i])
        else:
            ex = pc_extremes[i]
            axis_labels.append({
                "label": f"PC{i+1}",
                "pos": _short(ex["positive_end"]),
                "neg": _short(ex["negative_end"]),
                "score": 0.0,
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

    # ── Reconstruction data ──
    # Expose the PCA model parameters so a client can reconstruct an embedding
    # from an arbitrary point in PC space:
    #     x ≈ mean + Σ_i  w_i · components_[i]
    # This powers the draggable "Voice Space" explorer in the webapp: drag a
    # point on the PC1×PC2 plane, reconstruct the embedding, and synthesize.
    # `components` is (n_pc, dim) and `mean` is (dim,).
    components = pca.components_.astype(np.float32)  # (n_pc, dim)
    mean = pca.mean_.astype(np.float32)              # (dim,)
    components_list = [[round(float(v), 6) for v in row] for row in components]
    mean_list = [round(float(v), 6) for v in mean]

    result = {
        "n_voices": n_voices,
        "dim": dim,
        "n_components": n_pc,
        "n_clusters": n_cl,
        "explained_variance_ratio": [round(float(r), 4) for r in pca.explained_variance_ratio_],
        "cumulative_variance": [round(float(r), 4) for r in np.cumsum(pca.explained_variance_ratio_)],
        "pc_extremes": pc_extremes,
        "axis_labels": axis_labels,
        "voices": voices_out,
        "saved_dirs": saved_dirs,
        "mean": mean_list,
        "components": components_list,
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
            lab = result["axis_labels"][i]
            ext = result["pc_extremes"][i]
            if lab["score"] > 0:
                print(f"  PC{i+1}: {pct:5.1f}% var (cum {cum:5.1f}%)  "
                      f"[{lab['label']}  {lab['neg']} ←→ {lab['pos']}  r={lab['score']}]")
            else:
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
