# generate_sky.py

import numpy as np
import matplotlib.pyplot as plt

# Generator bintang acak
def generate_sky(num_stars, xrange=(0, 100), yrange=(0, 100)):
    xs = np.random.uniform(xrange[0], xrange[1], num_stars)
    ys = np.random.uniform(yrange[0], yrange[1], num_stars)
    sky = list(zip(xs, ys))
    np.save('sky.npy', sky)  # Simpan ke file numpy
    print(f"Generated {num_stars} stars and saved to sky.npy")

    # Visualisasi
    plt.scatter(xs, ys, color='white')
    plt.gca().set_facecolor('black')
    plt.title("Generated Sky")
    plt.show()

if __name__ == "__main__":
    generate_sky(200)
