# solve.py

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
from constellations import CONSTELLATIONS
THRESHOLD = 0.7

# Load data sky
sky = np.load('sky.npy', allow_pickle=True)
sky = list(sky)

# Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Matching logic
def is_match(candidate, pattern, tolerance=0.7):
    candidate_dists = sorted([
        euclidean_distance(candidate[0], candidate[1]),
        euclidean_distance(candidate[1], candidate[2]),
        euclidean_distance(candidate[0], candidate[2]),
    ])
    pattern_dists = sorted([
        euclidean_distance(pattern[0], pattern[1]),
        euclidean_distance(pattern[1], pattern[2]),
        euclidean_distance(pattern[0], pattern[2]),
    ])
    return all(abs(c - p) < tolerance for c, p in zip(candidate_dists, pattern_dists))

# Search function
def search_constellation(sky, pattern, tolerance=0.7):
    matches = []
    total_checked = 0
    for candidate in combinations(sky, len(pattern)):
        total_checked += 1
        if is_match(candidate, pattern, tolerance):
            matches.append(candidate)
    print(f"Total combinations checked: {total_checked}")
    return matches

start_time = time.time()

results = {}
for name, pattern in CONSTELLATIONS.items():
    matches = search_constellation(sky, pattern, tolerance=THRESHOLD)
    results[name] = matches
    print(f"{name} found: {len(matches)} matches")

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Threshold tolerance: {THRESHOLD}")

# Visualize
xs, ys = zip(*sky)
plt.scatter(xs, ys, color='white')
plt.gca().set_facecolor('black')

colors = ['red', 'yellow', 'cyan', 'green']
color_idx = 0

for constellation_name, matches in results.items():
    for match in matches:
        mx, my = zip(*match)
        plt.plot(mx + (mx[0],), my + (my[0],), color=colors[color_idx % len(colors)], linewidth=2)
        plt.text(np.mean(mx), np.mean(my), constellation_name, color=colors[color_idx % len(colors)])
    color_idx += 1

plt.title("Detected Constellations")
plt.show()
