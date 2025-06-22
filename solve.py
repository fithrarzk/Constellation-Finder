import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import random
from constellations import CONSTELLATIONS

THRESHOLD = 0.7
MAX_SAMPLES = 500000  # Limit kombinasi yang dicek per pattern

# Load data sky
sky = np.load('sky.npy', allow_pickle=True)
sky = list(sky)

# Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Precompute pattern scale
def compute_pattern_scale(pattern):
    dists = []
    for i in range(len(pattern)):
        for j in range(i+1, len(pattern)):
            dists.append(euclidean_distance(pattern[i], pattern[j]))
    return np.mean(dists), sorted(dists)

# Matching logic
def is_match(candidate, pattern_dists_sorted, pattern_scale, tolerance=0.7):
    candidate_dists = []
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            candidate_dists.append(euclidean_distance(candidate[i], candidate[j]))
    candidate_dists_sorted = sorted(candidate_dists)
    candidate_scale = np.mean(candidate_dists_sorted)
    
    # Early filtering: scale check
    if abs(candidate_scale - pattern_scale) > tolerance * pattern_scale:
        return False

    # Full check
    return all(abs(c - p) < tolerance for c, p in zip(candidate_dists_sorted, pattern_dists_sorted))

# Generator sampling: do not load all combinations into memory!
def search_constellation(sky, pattern, tolerance=0.7, max_samples=100000):
    matches = []
    total_checked = 0
    pattern_scale, pattern_dists_sorted = compute_pattern_scale(pattern)

    total_generated = 0
    for candidate in combinations(sky, len(pattern)):
        total_generated += 1
        if random.random() > (max_samples / total_generated):
            continue  # skip based on sampling prob
        total_checked += 1
        if is_match(candidate, pattern_dists_sorted, pattern_scale, tolerance):
            matches.append(candidate)
        if total_checked >= max_samples:
            break

    print(f"Total combinations generated: {total_generated}")
    print(f"Total checked (sampled): {total_checked}")
    return matches

# ================== MAIN ====================

start_time = time.time()

results = {}
for name, pattern in CONSTELLATIONS.items():
    print(f"\nSearching for {name} (pattern size {len(pattern)}) ...")
    matches = search_constellation(sky, pattern, tolerance=THRESHOLD, max_samples=MAX_SAMPLES)
    results[name] = matches
    print(f"{name} found: {len(matches)} matches")

end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")
print(f"Threshold tolerance: {THRESHOLD}")

# Visualize
xs, ys = zip(*sky)
plt.scatter(xs, ys, color='white')
plt.gca().set_facecolor('black')

colors = ['red', 'yellow', 'cyan', 'green', 'magenta', 'blue', 'orange', 'lime', 'purple', 'pink', 'gold', 'lightblue']
color_idx = 0

for constellation_name, matches in results.items():
    for match in matches:
        mx, my = zip(*match)
        plt.plot(mx + (mx[0],), my + (my[0],), color=colors[color_idx % len(colors)], linewidth=2)
        plt.text(np.mean(mx), np.mean(my), constellation_name, color=colors[color_idx % len(colors)], fontsize=8)
    color_idx += 1

plt.title("Detected Constellations (Optimized Sampling)")
plt.show()
