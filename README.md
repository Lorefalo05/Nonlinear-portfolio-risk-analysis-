# Nonlinear Portfolio Risk Surface Analysis
## Multi-Order Sensitivity Mapping via Taylor Expansion & Spectral Decomposition

A rigorous quantitative finance framework designed to model, approximate, and visualize the risk surfaces of financial portfolios containing complex nonlinear interactions (e.g., structured products, exotic options, or multi-asset derivatives combinations). 

By leveraging **First-Order (Delta)** and **Second-Order (Gamma)** sensitivities, this repository demonstrates how to perform local risk approximations around a reference market state, identify directions of maximum exposure, and execute spectral decomposition on the portfolio's Hessian matrix to extract principal risk factors.

---

## 1. Theoretical Background

Real-world exotic portfolios rarely display linear sensitivity to underlying asset prices. When position values interact nonlinearly (due to cross-gamma effects, structural correlations, or correlation-dependent payoffs), a simple linear risk model fails. 

This framework defines a portfolio value function V(S1, S2) governed by linear position allocations combined with a high-frequency nonlinear interaction surface:

$$V(S_1, S_2) = w_1 S_1 + w_2 S_2 + k \sin\left(\frac{S_1}{20}\right)\cos\left(\frac{S_2}{20}\right)$$

Where:
* w1, w2 represent the nominal position sizes (linear exposures).
* k denotes the nonlinear interaction strength.
* The trigonometric arguments simulate periodic, non-monotonic mark-to-market fluctuations.

### Multi-Variable Taylor Expansion
To assess portfolio stability under localized market shocks, we construct a multi-variable Taylor series approximation around a specified reference state S* = (S1*, S2*):

$$V(S^* + \Delta S) \approx V(S^*) + \nabla V(S^*)^T \Delta S + \frac{1}{2} \Delta S^T \mathcal{H}(S^*) \Delta S$$

#### First-Order Sensitivity (The Delta Gradient)
The Gradient vector maps the local linear directional sensitivities:

$$\nabla V(S_1, S_2) = \begin{bmatrix} \frac{\partial V}{\partial S_1} \\ \frac{\partial V}{\partial S_2} \end{bmatrix} = \begin{bmatrix} w_1 + \frac{k}{20}\cos\left(\frac{S_1}{20}\right)\cos\left(\frac{S_2}{20}\right) \\ w_2 - \frac{k}{20}\sin\left(\frac{S_1}{20}\right)\sin\left(\frac{S_2}{20}\right) \end{bmatrix}$$

The magnitude and normalized direction vector define the path of maximum portfolio appreciation (and conversely, maximum short exposure).

#### Second-Order Sensitivity (The Gamma Hessian)
The Hessian matrix captures the curvature (Gamma and Cross-Gamma risk) of the surface:

$$\mathcal{H}(S_1, S_2) = \begin{bmatrix} \frac{\partial^2 V}{\partial S_1^2} & \frac{\partial^2 V}{\partial S_1 \partial S_2} \\ \frac{\partial^2 V}{\partial S_2 \partial S_1} & \frac{\partial^2 V}{\partial S_2^2} \end{bmatrix}$$

Where the explicit analytical partial derivatives are:

$$\frac{\partial^2 V}{\partial S_1^2} = -\frac{k}{400}\sin\left(\frac{S_1}{20}\right)\cos\left(\frac{S_2}{20}\right)$$
$$\frac{\partial^2 V}{\partial S_2^2} = -\frac{k}{400}\sin\left(\frac{S_1}{20}\right)\cos\left(\frac{S_2}{20}\right)$$
$$\frac{\partial^2 V}{\partial S_1 \partial S_2} = -\frac{k}{400}\cos\left(\frac{S_1}{20}\right)\sin\left(\frac{S_2}{20}\right)$$

### Principal Risk Factors (Spectral Decomposition)
By executing an eigendecomposition on the symmetric Hessian matrix:

$$\mathcal{H} = Q \Lambda Q^T$$

We extract eigenvalues and their corresponding orthogonal eigenvectors. In a risk management context:
* **Eigenvalues:** quantify the magnitude of the portfolio's acceleration/deceleration under stress along specific market vectors. A negative eigenvalue implies a locally concave surface (short Gamma exposure), representing tail-risk under large shifts.
* **Eigenvectors:** define the precise synthetic combinations of underlying asset movements that represent the Principal Risk Axes.

---

## 2. Key Features

* **Analytical Exactness:** Computes exact symbolic First and Second-order derivatives rather than relying on unstable finite-difference approximations.
* **Multi-Order Comparison:** Evaluates the empirical breakdown of linear Delta-normal models versus Quadratic Gamma-adjusted models under stressed asset paths.
* **Spectral Analysis:** Isolates orthogonal risk channels via eigen-decomposition of the Hessian matrix.
* **Production-Ready Visualizations:** Generates comprehensive subplots mapping the risk surface contour, cross-sectional profiles, and diagonal Taylor convergence bounds.

---

## 3. Project Structure & Core Logic
