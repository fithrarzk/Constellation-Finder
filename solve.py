import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import random
import math
from constellations import CONSTELLATIONS
from sklearn.cluster import DBSCAN

# Hyperparameters
BASE_TOLERANCE_PER_PAIR = 0.015
MAX_SAMPLES = 100000  # Sampling maksimal per pattern
EPS_CLUSTER = 10  # DBSCAN radius

# Load data sky
sky = np.load('sky.npy', allow_pickle=True)
sky = list(sky)

# Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Compute normalized distance matrix (relative shape)
def compute_normalized_distance_matrix(points):
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dists.append(euclidean_distance(points[i], points[j]))
    avg_dist = sum(dists) / len(dists)
    normalized = [d / avg_dist for d in dists]
    return sorted(normalized)

# Matching using adaptive score-based comparison
def is_match(candidate, pattern):
    pattern_norm = compute_normalized_distance_matrix(pattern)
    candidate_norm = compute_normalized_distance_matrix(candidate)
    diff = [abs(c - p) for c, p in zip(candidate_norm, pattern_norm)]
    total_error = sum(diff) / len(diff)

    n_points = len(pattern)
    n_pairs = (n_points * (n_points - 1)) // 2
    adaptive_tolerance = BASE_TOLERANCE_PER_PAIR 

    return total_error < adaptive_tolerance

# Cluster sky to reduce candidate space
def cluster_sky(sky, eps=EPS_CLUSTER, min_samples=3):
    coords = np.array(sky)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tuple(coords[idx]))
    return clusters

# Search within cluster
def search_in_cluster(cluster_points, pattern, max_samples=MAX_SAMPLES):
    matches = []
    total_generated = 0
    total_checked = 0

    for candidate in combinations(cluster_points, len(pattern)):
        total_generated += 1
        if random.random() > (max_samples / total_generated):
            continue
        total_checked += 1
        if is_match(candidate, pattern):
            matches.append(candidate)
        if total_checked >= max_samples:
            break

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

# ================== MAIN ====================

start_time = time.time()

results = {}
for name, pattern in CONSTELLATIONS.items():
    print(f"\nSearching for {name} (pattern size {len(pattern)}) ...")
    matches = search_constellation(sky, pattern)
    results[name] = matches
    print(f"{name} found: {len(matches)} matches")

end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")

# Visualize
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

plt.title("Detected Constellations (Final Clean Superb Full)")
plt.show()
