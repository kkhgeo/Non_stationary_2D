Non-stationary & Anisotropic Spatial Data Generator
📝 Overview
This project provides a Python script to generate 2D spatial data that exhibits both non-stationarity and anisotropy. The generated data is based on the Matérn covariance function, allowing for the simulation of complex spatial patterns where statistical properties—such as variance, correlation length, and directionality—change progressively across the field.

This type of synthetic data is invaluable for research in spatial statistics and machine learning. It can be used to develop and validate advanced models like sophisticated Kriging variants or deep learning-based spatial prediction models, which are designed to handle complex real-world phenomena (e.g., terrain elevation, soil contamination, or climate patterns).

✨ Key Features
Non-stationarity: Generates data where variance and correlation range change linearly across the spatial domain.

Anisotropy: Simulates direction-dependent spatial correlation. The angle and ratio of this anisotropy can also vary spatially.

Matérn Covariance: Utilizes the flexible Matérn covariance model to control the smoothness of the generated field.

GeoTIFF Export: Saves the generated 2D grid data as a .tif file, which is readily usable in GIS software.

Easy Customization: Allows for easy generation of data with different characteristics by modifying the params dictionary within the script.

📊 Generated Data Example
Below is a visualization of a generated field with the following parameters: variance_range=(0.5, 3.0), range_param_range=(0.05, 0.5), angle_range_deg=(-30, 60), and ratio_range=(1.5, 3.0). You can observe how the texture, scale, and orientation of the patterns gradually evolve from left to right.

⚙️ Code Explanation
matern_covariance(...): A function that computes the Matérn covariance value for a given distance.

generate_anisotropic_nonstationary_data(...): The core function containing the main logic.

It generates a grid of spatial coordinates.

It defines "parameter fields" for variance, range, angle, and ratio that vary linearly with the x-coordinate.

When calculating the covariance between any two points (i, j), it uses the "local" average parameters of these points to compute an anisotropic distance, which is then fed into the Matérn function.

It produces the final random field via Cholesky decomposition of the full covariance matrix.

main(): Sets the simulation parameters and calls the functions for data generation, GeoTIFF saving, and visualization in sequence.

⚖️ License
This project is licensed under the MIT License.
