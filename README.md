# Searching for Constellations: A Brute Force Approach to Geometric Pattern Matching in 2D Star Maps

## Description

This program aims to detect constellations in a two-dimensional star map using the **Brute Force Geometric Pattern Matching** method. The program performs shape matching by comparing normalized pairwise distances between stars and applies **DBSCAN** clustering to narrow the search space.

This program is part of the implementation for the paper topic:
**"Searching for Constellations: A Brute Force Approach to Geometric Pattern Matching in 2D Star Maps"**

## Features

* Generate random star maps with a certain number of stars and injected constellations.
* Search for constellation patterns based on predefined templates.
* Perform brute force search with random sampling of candidate combinations.
* Utilize DBSCAN clustering to reduce the number of matching candidates.
* Visualize the best detected constellation results.

## File Structure

* `generate_sky.py`

  * Script to generate random star maps and inject constellations into them.

* `solve.py`

  * Main script to run the constellation search algorithm.

* `constellations.py`

  * Contains the template definitions of each constellation.

* `sky.npy`

  * Generated star map data (automatically saved by `generate_sky.py`).

* `ground_truth.npy`

  * Ground truth data from injections for validation purposes.

## Dependencies

This program is developed using Python 3 and requires the following libraries:

* `numpy`
* `matplotlib`
* `scikit-learn`

Install dependencies by running:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

### 1. Generate Star Map Data:

```bash
python3 generate_sky.py
```

* The program will generate `sky.npy` and `ground_truth.npy` files.
* The `ground_truth.npy` file can be used for manual validation.

### 2. Run Constellation Search:

```bash
python3 solve.py
```

* The program will search for constellations based on the data in `sky.npy`.
* The search results will be visualized automatically.

## Important Parameters

Several parameters affect the results:

* `BASE_TOLERANCE` : RMS error tolerance for distance comparison (matching sensitivity scale).
* `EPS_CLUSTER` : DBSCAN clustering radius to group stars.
* `MAX_SAMPLE_PER_CLUSTER` : Maximum number of candidate combinations randomly sampled per cluster.

These parameters can be modified directly in the `solve.py` file for experimentation.

## Notes

* Due to the brute force nature, execution time increases exponentially with the number of stars.
* DBSCAN is used to reduce search complexity without sacrificing exhaustive search within relevant clusters.
* This program is suitable for small to medium-scale data, aligned with the research design.

## Author

This program was developed as part of an Algorithm Strategy Paper Project by \[Your Name].
