# Multiresolution Laplacian Mesh Deformer

## Project Overview

This project implements a **Multiresolution Laplacian Mesh Deformer**, a technique used to manipulate 3D models interactively while preserving their high-frequency geometric details.

This implementation is based on course content from _Professor Daniele Panozzo (NYU)_, who provided the initial setup, including the iPyWidget used in the `DeformExample.ipynb` notebook for interactive visualization and handle manipulation.

## Multiresolution Mesh Editing Pipeline

The overall deformation process is divided into three main phases to achieve detail-preserving manipulation, carried out by methods of the initialized deformer_ instance. The instance is initialized with the original mesh vertices ($V$), faces ($F$), and the indices of the control handles.

1. _Remove High-Frequency Details (Smoothing):_ The original mesh vertices ($V$) are smoothed to create a low-frequency base mesh ($B$) by solving a Bilaplacian minimization problem.

`B = deformer_.smooth_mesh(v)`

2. _Compute Detail Encoding:_ The high-frequency details ($\text{details} = V - B$) are calculated and encoded in a local, orthogonal reference frame for each vertex. This frame is defined by the per-vertex normal ($\mathbf{n}$) and two tangent vectors ($\mathbf{x}$ and $\mathbf{y}$).

```
coeffs, tangents = deformer_.compute_detail_encoding(B)
```


3. _Deform the Smooth Mesh and Transfer Details:_ The low-frequency base mesh ($B$) is deformed based on the user-defined handle movement (`new_pos`) to create the new base mesh ($B'$). The stored detail coefficients are then applied to the new reference frames in $B'$ and added back to compute the final, detail-preserved deformed surface ($S'$).

```
B_prime = deformer_.smooth_mesh(new_pos) # Deformation of the smooth mesh
B_prime_dets = deformer_.apply_detail_encoding(B_prime, coeffs, tangents) # Detail transfer
S_prime = B_prime + B\_prime_dets # Final result
```

## Code Structure and Deformer Utilities

The core deformation logic is encapsulated within the `Deformer` class, which handles the necessary linear algebra for the Bilaplacian formulation. This class is provided in two separate utility files:

`deform_ops.py`: Contains the standard, unoptimized implementation of the deformation operations. This version uses the general sparse solver `spsolve` for each smoothing step.

`deform_ops_chol.py`: Contains the performance-optimized version. This class uses Sparse Cholesky decomposition (`sksparse.cholmod.cholesky`) to pre-factorize the constant part of the system matrix ($\mathbf{A}_{ff}$) during initialization. This significantly speeds up the repeated smoothing steps (`smooth_mesh`), making the interactive deformation much faster.

The main notebook (DeformExample.ipynb) is structured to function with either file imported. To switch between optimized and non-optimized performance, you only need to comment out the corresponding import statement at the beginning of the notebook.

## Usage (Getting Started)

Setup: Ensure all dependencies (e.g., NumPy, SciPy, sksparse.cholmod, and igl) are installed.

Run: Open and run the cells in the `DeformExample.ipynb` notebook.

Data: Mesh files and their corresponding handles
