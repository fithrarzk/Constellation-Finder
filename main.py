# main.py

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from constellations import CONSTELLATIONS

# Generate random sky
def generate_sky(num_stars, xrange=(0, 100), yrange=(0, 100)):
    xs = np.random.uniform(xrange[0], xrange[1], num_stars)
    ys = np.random.uniform(yrange[0], yrange[1], num_stars)
    return list(zip(xs, ys))

# Insert constellation into sky
def insert_constellation(sky, pattern, insert_pos=(50, 50), scale=1.0):
    translated_pattern = [(insert_pos[0] + x*scale, insert_pos[1] + y*scale) for (x, y) in pattern]
    return sky + translated_pattern

# Hitung jarak Euclidean
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Cek apakah pola kandidat mirip dengan pattern
def is_match(candidate, pattern, tolerance=0.5):
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

# Brute Force Search
def search_constellation(sky, pattern, tolerance=0.5):
    matches = []
    for candidate in combinations(sky, len(pattern)):
        if is_match(candidate, pattern, tolerance):
            matches.append(candidate)
    return matches

# Visualisasi
def visualize_sky(sky, results):
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

    plt.show()


# Main program
if __name__ == "__main__":
    # Generate sky
    sky = generate_sky(200)

    # Sisipkan Orion ke langit
    sky = insert_constellation(sky, CONSTELLATIONS["Orion"], insert_pos=(30, 60))

    # Proses pencarian semua rasi
    results = {}
    for name, pattern in CONSTELLATIONS.items():
        matches = search_constellation(sky, pattern, tolerance=0.7)
        results[name] = matches

    # Tampilkan hasil
    visualize_sky(sky, results)
