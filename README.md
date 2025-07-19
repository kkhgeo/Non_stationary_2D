````markdown
# Non-stationary & Anisotropic Spatial Data Generator

## 📝 Overview

This project provides a Python script to generate 2D spatial data that exhibits both **non-stationarity** and **anisotropy**. The generated data is based on the Matérn covariance function, allowing for the simulation of complex spatial patterns where statistical properties—such as variance, correlation length, and directionality—change progressively across the field.

Such synthetic data is invaluable for research in spatial statistics and machine learning. It can be used to develop and validate advanced models like:
* Sophisticated Kriging variants
* Deep learning-based spatial prediction models (e.g., terrain elevation, soil contamination, or climate patterns)

## ✨ Key Features

* **Non-stationarity:** Generates data where variance and correlation range change linearly across the spatial domain.
* **Anisotropy:** Simulates direction-dependent spatial correlation. The angle and ratio of anisotropy can also vary spatially.
* **Matérn Covariance:** Utilizes the flexible Matérn covariance model to control the smoothness of the generated field.
* **GeoTIFF Export:** Saves the generated 2D grid data as a `.tif` file, readily usable in GIS software.
* **Easy Customization:** Modify the `params` dictionary in the script to easily generate data with different characteristics.

## 📊 Generated Data Example

The following parameter set was used to produce a spatial field with gradually changing spatial properties:

```python
params = {
    "variance_range": (0.5, 3.0),
    "range_param_range": (0.05, 0.5),
    "angle_range_deg": (-30, 60),
    "ratio_range": (1.5, 3.0)
}
````

This produces a texture where the scale, smoothness, and orientation of spatial variation evolve smoothly from left to right. Ideal for benchmarking non-stationary Gaussian process models or spatial deep learning architectures.

## ⚙️ Code Explanation

### 1\. `matern_covariance(...)`

Computes the Matérn covariance between two points using the formula:
$$C(h) = \sigma^2 \cdot \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h\sqrt{2\nu}}{\rho} \right)^\nu K_\nu\left( \frac{h\sqrt{2\nu}}{\rho} \right)$$

  * **h**: distance between points
  * **σ²**: local variance
  * **ρ**: range parameter
  * **ν**: smoothness
  * **K\_ν**: modified Bessel function of the second kind

### 2\. `generate_anisotropic_nonstationary_data(...)`

The core data generation logic:

  * Constructs a grid of spatial coordinates.
  * Defines spatially varying fields for:
      * variance
      * range
      * angle
      * anisotropy ratio
  * For each pair of points (i, j):
      * Computes an anisotropic distance based on average local parameters.
      * Applies the Matérn covariance function.
  * Assembles the full covariance matrix.
  * Applies Cholesky decomposition to sample from the multivariate Gaussian field.

### 3\. `main()`

Orchestrates the entire workflow:

  * Sets simulation parameters
  * Calls data generation routine
  * Exports result as:
      * GeoTIFF for GIS
      * Matplotlib visualization

## 📂 Output Example

  * **File:** `output/matern_field.tif`
  * **Size:** 100 × 100 pixels (default)
  * **Format:** GeoTIFF
  * **Visualization:** Matplotlib heatmap with color gradient

## 🔧 Customization Tips

Change spatial characteristics by modifying these parameters in `main()`:

| Parameter | Meaning | Recommended Range |
| :--- | :--- | :--- |
| `variance_range` | Local variance | (0.1, 3.0) |
| `range_param_range` | Spatial correlation length| (0.01, 1.0) |
| `angle_range_deg` | Direction of anisotropy | (-90, 90) |
| `ratio_range` | Strength of anisotropy | (1.0, 5.0) |
| `smoothness` | Smoothness of the Matérn function | 1.0 \~ 2.5 |

## 📦 Dependencies

  * numpy
  * scipy
  * matplotlib
  * rasterio

Install with:

```bash
pip install numpy scipy matplotlib rasterio
```

## ⚖️ License

This project is licensed under the **MIT License**. See `LICENSE` for more details.

## 📬 Contact

For questions or contributions, feel free to open an **Issue** or pull request.

```

