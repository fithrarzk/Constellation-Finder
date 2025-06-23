import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import math
from constellations import CONSTELLATIONS
from sklearn.cluster import DBSCAN

# Hyperparameters
BASE_TOLERANCE = 0.05
EPS_CLUSTER = 10
ANCHOR_TOLERANCE = 0.1

# Load sky data
sky = np.load('sky.npy', allow_pickle=True)
sky = list(sky)

# Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Normalized distance matrix (shape descriptor)
def compute_normalized_distances(points):
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dists.append(euclidean_distance(points[i], points[j]))
    avg_dist = sum(dists) / len(dists)
    normalized = [d / avg_dist for d in dists]
    return sorted(normalized)

# Full shape matching
def is_full_match(candidate, pattern):
    candidate_norm = compute_normalized_distances(candidate)
    pattern_norm = compute_normalized_distances(pattern)
    diff = [abs(c - p) for c, p in zip(candidate_norm, pattern_norm)]
    error = np.sqrt(sum(d**2 for d in diff) / len(diff))
    return error < BASE_TOLERANCE

# Precompute anchor distance for quick filtering
def get_anchor_distance(points):
    p1, p2 = points[0], points[1]
    return euclidean_distance(p1, p2)

# Cluster the sky
def cluster_sky(sky, eps=EPS_CLUSTER, min_samples=3):
    coords = np.array(sky)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(tuple(coords[idx]))
    return clusters

# Search inside cluster
def search_in_cluster(cluster_points, pattern):
    matches = []
    pattern_anchor = get_anchor_distance(pattern)

    for candidate in combinations(cluster_points, len(pattern)):
        candidate_anchor = get_anchor_distance(candidate)
        if abs(candidate_anchor - pattern_anchor) > ANCHOR_TOLERANCE * pattern_anchor:
            continue
        if is_full_match(candidate, pattern):
            matches.append(candidate)
    return matches

# Full search function
def search_constellation(sky, pattern):
    matches = []
    clusters = cluster_sky(sky, eps=EPS_CLUSTER, min_samples=len(pattern))
    print(f"Found {len(clusters)} clusters for pattern size {len(pattern)}")
    for cluster_points in clusters.values():
        if len(cluster_points) < len(pattern):
            continue
        result = search_in_cluster(cluster_points, pattern)
        matches.extend(result)
    return matches

# MAIN
start_time = time.time()

results = {}
for name, pattern in CONSTELLATIONS.items():
    print(f"\nSearching for {name} ({len(pattern)} points)...")
    matches = search_constellation(sky, pattern)
    results[name] = matches
    print(f"{name} found: {len(matches)} matches")

end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")

# Visualization
xs, ys = zip(*sky)
plt.scatter(xs, ys, color='white')
plt.gca().set_facecolor('black')

colors = ['red', 'yellow', 'cyan', 'green', 'magenta', 'blue', 'orange', 'lime', 'purple', 'pink', 'gold', 'lightblue']
color_idx = 0

for constellation_name, matches in results.items():
    for match in matches:
        mx, my = zip(*match)
        plt.plot(mx + (mx[0],), my + (my[0],), color=colors[color_idx % len(colors)], linewidth=1)
        plt.text(np.mean(mx), np.mean(my), constellation_name, color=colors[color_idx % len(colors)], fontsize=7)
    color_idx += 1

plt.title("Detected Constellations (Final Perfect Version)")
plt.show()
