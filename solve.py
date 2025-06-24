import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random
import time
import math
from constellations import CONSTELLATIONS
from sklearn.cluster import DBSCAN

BASE_TOLERANCE = 0.05   # RMS error tolerance (scale-invariant)
EPS_CLUSTER = 15        # DBSCAN clustering radius
MIN_SAMPLES = 2
MAX_SAMPLE_PER_CLUSTER = 40000

# Load sky data
sky = np.load('sky.npy', allow_pickle=True)
sky = list(sky)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_normalized_distances(points):
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dists.append(euclidean_distance(points[i], points[j]))
    avg_dist = sum(dists) / len(dists)
    normalized = [d / avg_dist for d in dists]
    return sorted(normalized)

def rms_shape_error(candidate, pattern):
    candidate_norm = compute_normalized_distances(candidate)
    pattern_norm = compute_normalized_distances(pattern)
    diff = [(c - p) for c, p in zip(candidate_norm, pattern_norm)]
    rms_error = np.sqrt(sum(d**2 for d in diff) / len(diff))
    return rms_error

def is_match(candidate, pattern):
    return rms_shape_error(candidate, pattern) < BASE_TOLERANCE

def cluster_sky(sky, eps=EPS_CLUSTER, min_samples=MIN_SAMPLES):
    coords = np.array(sky)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(tuple(coords[idx]))
    return clusters

def random_sample_combinations(cluster_points, pattern_size, sample_size):
    all_points = list(cluster_points)
    total_possible = math.comb(len(all_points), pattern_size)
    if total_possible <= sample_size:
        return list(combinations(all_points, pattern_size))
    selected = set()
    while len(selected) < sample_size:
        candidate = tuple(sorted(random.sample(all_points, pattern_size)))
        selected.add(candidate)
    return list(selected)

def search_in_cluster(cluster_points, pattern):
    matches = []
    sampled_candidates = random_sample_combinations(cluster_points, len(pattern), MAX_SAMPLE_PER_CLUSTER)
    for candidate in sampled_candidates:
        if is_match(candidate, pattern):
            matches.append(candidate)
    return matches

def search_constellation(sky, pattern):
    matches = []
    clusters = cluster_sky(sky)
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

# Visualization: Only best match for each constellation
xs, ys = zip(*sky)
plt.scatter(xs, ys, color='white', s=10)
plt.gca().set_facecolor('black')

for constellation_name, matches in results.items():
    if not matches:
        continue
    best = min(matches, key=lambda m: rms_shape_error(m, CONSTELLATIONS[constellation_name]))
    mx, my = zip(*best)
    plt.scatter(mx, my, s=40, label=constellation_name)
    plt.plot(mx + (mx[0],), my + (my[0],), linewidth=2)
    plt.text(np.mean(mx), np.mean(my), constellation_name, fontsize=8, color='cyan')

plt.title("Detected Constellations (FINAL Ultimate Brute Force Matching)")
plt.legend()
plt.show()
