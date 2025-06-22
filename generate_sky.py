# generate_sky.py

import numpy as np
import matplotlib.pyplot as plt
import random
from constellations import CONSTELLATIONS

def generate_sky(num_stars, xrange=(0, 100), yrange=(0, 100)):
    xs = np.random.uniform(xrange[0], xrange[1], num_stars)
    ys = np.random.uniform(yrange[0], yrange[1], num_stars)
    sky = list(zip(xs, ys))
    return sky

def insert_constellation(sky, pattern, insert_pos, scale=1.0):
    translated_pattern = [
        (insert_pos[0] + x * scale, insert_pos[1] + y * scale)
        for (x, y) in pattern
    ]
    return sky + translated_pattern, translated_pattern

def generate_full_sky(num_stars, num_injected_constellations):
    sky = generate_sky(num_stars)
    injected_info = []

    available_constellations = list(CONSTELLATIONS.items())
    random.shuffle(available_constellations)
    selected_constellations = available_constellations[:num_injected_constellations]

    for name, pattern in selected_constellations:
        insert_x = random.uniform(10, 90)
        insert_y = random.uniform(10, 90)
        scale = random.uniform(3, 8)  # Random scale biar variasi ukuran

        sky, translated_pattern = insert_constellation(sky, pattern, (insert_x, insert_y), scale)
        injected_info.append({
            'name': name,
            'position': (insert_x, insert_y),
            'scale': scale,
            'pattern': translated_pattern
        })

    np.save('sky.npy', sky)
    np.save('ground_truth.npy', injected_info)
    print(f"Generated {num_stars} random stars and injected {num_injected_constellations} constellations.")
    return sky, injected_info

# Visualization helper
def visualize(sky, injected_info):
    xs, ys = zip(*sky)
    plt.scatter(xs, ys, color='white')
    plt.gca().set_facecolor('black')

    for info in injected_info:
        px, py = zip(*info['pattern'])
        plt.plot(px, py, color='red', linewidth=2)
        plt.scatter(px, py, color='yellow')
        plt.text(np.mean(px), np.mean(py), info['name'], color='cyan', fontsize=8)

    plt.title("Generated Sky with Injected Constellations")
    plt.show()

# Main
if __name__ == "__main__":
    NUM_RANDOM_STARS = 200
    NUM_INJECTED_CONSTELLATIONS = 4

    sky, injected_info = generate_full_sky(NUM_RANDOM_STARS, NUM_INJECTED_CONSTELLATIONS)
    visualize(sky, injected_info)
