import numpy as np 

import matplotlib.pyplot as plt 

# ============================================================ 

# PORTFOLIO PARAMETERS 

# ============================================================ 

w1, w2 = 100, 80      # Position sizes (shares) 

k = 50                # Nonlinear interaction strength 

s1_star, s2_star = 100, 100  # Reference point 

# ============================================================ 

# CORE FUNCTIONS (from earlier sections) 

# ============================================================ 

def portfolio_value(s1, s2): 

  """V(S1, S2) = w1*S1 + w2*S2 + k*sin(S1/20)*cos(S2/20)""" 

  return w1 * s1 + w2 * s2 + k * np.sin(s1 / 20) * np.cos(s2 / 20) 

def compute_gradient(s1, s2): 

  """Compute partial derivatives (gradient) at a point.""" 

  dV_dS1 = w1 + (k / 20) * np.cos(s1 / 20) * np.cos(s2 / 20) 

  dV_dS2 = w2 - (k / 20) * np.sin(s1 / 20) * np.sin(s2 / 20) 

  return np.array([dV_dS1, dV_dS2]) 

def compute_hessian(s1, s2): 

  """Compute Hessian matrix at a point.""" 

  h11 = -(k / 400) * np.sin(s1 / 20) * np.cos(s2 / 20) 

  h22 = -(k / 400) * np.sin(s1 / 20) * np.cos(s2 / 20) 

  h12 = -(k / 400) * np.cos(s1 / 20) * np.sin(s2 / 20) 

  return np.array([[h11, h12], [h12, h22]]) 

def taylor_approximation(s1, s2, order=2): 

  """Taylor approximation of V around (s1_star, s2_star).""" 

  V_star = portfolio_value(s1_star, s2_star) 

  grad = compute_gradient(s1_star, s2_star) 

  delta = np.array([s1 - s1_star, s2 - s2_star]) 

  if order == 1: 

      return V_star + np.dot(grad, delta) 

  else: 

      H = compute_hessian(s1_star, s2_star) 

      return V_star + np.dot(grad, delta) + 0.5 * np.dot(delta, H @ delta) 

# ============================================================ 

# COMPUTE ALL QUANTITIES AT REFERENCE POINT 

# ============================================================ 
V_star = portfolio_value(s1_star, s2_star) 

grad = compute_gradient(s1_star, s2_star) 

H = compute_hessian(s1_star, s2_star) 

eigenvalues, eigenvectors = np.linalg.eigh(H) 

# Sort by magnitude 

idx = np.argsort(np.abs(eigenvalues))[::-1] 

eigenvalues = eigenvalues[idx] 

eigenvectors = eigenvectors[:, idx] 

grad_magnitude = np.linalg.norm(grad) 

grad_direction = grad / grad_magnitude 

# ============================================================ 

# PRINT SUMMARY 

# ============================================================ 

print("=" * 60) 

print("PORTFOLIO RISK SURFACE ANALYSIS") 

print("=" * 60) 

print(f"\nReference Point: (S₁, S₂) = ({s1_star}, {s2_star})") 

print(f"Portfolio Value: V = ${V_star:,.2f}") 

print(f"\n--- FIRST-ORDER (Delta) ---") 

print(f"∂V/∂S₁ = {grad[0]:.4f}  (delta to asset 1)") 

print(f"∂V/∂S₂ = {grad[1]:.4f}  (delta to asset 2)") 

print(f"Gradient magnitude: {grad_magnitude:.4f}") 

print(f"Steepest ascent direction: ({grad_direction[0]:.3f}, {grad_direction[1]:.3f})") 

print(f"\n--- SECOND-ORDER (Gamma) ---") 

print(f"Hessian matrix:") 

print(f"  [{H[0,0]:+.4f}  {H[0,1]:+.4f}]") 

print(f"  [{H[1,0]:+.4f}  {H[1,1]:+.4f}]") 

print(f"Eigenvalues: λ₁ = {eigenvalues[0]:.4f}, λ₂ = {eigenvalues[1]:.4f}") 

print(f"Principal direction 1: ({eigenvectors[0,0]:.3f}, {eigenvectors[1,0]:.3f})") 

print(f"\n--- APPROXIMATION TEST ---") 

test_point = (110, 95) 

V_true = portfolio_value(*test_point) 

V_linear = taylor_approximation(*test_point, order=1) 

V_quad = taylor_approximation(*test_point, order=2) 

print(f"At ({test_point[0]}, {test_point[1]}):") 

print(f"  True value:      ${V_true:,.2f}") 

print(f"  Linear approx:   ${V_linear:,.2f}  (error: ${abs(V_true-V_linear):.2f})") 

print(f"  Quadratic approx:${V_quad:,.2f}  (error: ${abs(V_true-V_quad):.2f})") 

print("=" * 60) 

# ============================================================ 

# VISUALIZATION 

# ============================================================ 

fig, axes = plt.subplots(1, 3, figsize=(14, 4)) 

# Plot 1: Surface heatmap with gradient arrow 

s1_range = np.linspace(60, 140, 100) 

s2_range = np.linspace(60, 140, 100) 

S1, S2 = np.meshgrid(s1_range, s2_range) 

V = portfolio_value(S1, S2) 

im = axes[0].contourf(S1, S2, V, levels=20, cmap='viridis') 

axes[0].contour(S1, S2, V, levels=10, colors='white', alpha=0.3, linewidths=0.5) 

axes[0].quiver(s1_star, s2_star, grad[0]*0.3, grad[1]*0.3, color='red', 

             scale=50, width=0.015, label='Gradient') 

axes[0].scatter([s1_star], [s2_star], color='red', s=100, zorder=5, edgecolors='white') 

axes[0].set_xlabel('S₁') 

axes[0].set_ylabel('S₂') 

axes[0].set_title('Portfolio Surface & Gradient') 

plt.colorbar(im, ax=axes[0], label='V($)') 

# Plot 2: Cross-sections 

s1_slice = np.linspace(60, 140, 100) 

V_s1_slice = portfolio_value(s1_slice, s2_star) 

V_s2_slice = portfolio_value(s1_star, s1_slice) 

axes[1].plot(s1_slice, V_s1_slice, 'b-', linewidth=2, label=f'V(S₁, {s2_star})') 

axes[1].plot(s1_slice, V_s2_slice, 'g-', linewidth=2, label=f'V({s1_star}, S₂)') 

axes[1].axvline(s1_star, color='gray', linestyle='--', alpha=0.5) 

axes[1].scatter([s1_star], [V_star], color='red', s=100, zorder=5) 

axes[1].set_xlabel('Price') 

axes[1].set_ylabel('Portfolio Value ($)') 

axes[1].set_title('Cross-Sections at (100, 100)') 

axes[1].legend() 

axes[1].grid(True, alpha=0.3) 

# Plot 3: Approximation comparison along diagonal 

t = np.linspace(-15, 15, 100) 

s1_diag = s1_star + t 

s2_diag = s2_star + t 

V_true_diag = portfolio_value(s1_diag, s2_diag) 

V_lin_diag = np.array([taylor_approximation(s1, s2, order=1) for s1, s2 in zip(s1_diag, s2_diag)]) 

V_quad_diag = np.array([taylor_approximation(s1, s2, order=2) for s1, s2 in zip(s1_diag, s2_diag)]) 

axes[2].plot(t, V_true_diag, 'b-', linewidth=2, label='True V') 

axes[2].plot(t, V_lin_diag, 'r--', linewidth=2, label='Linear approx') 

axes[2].plot(t, V_quad_diag, 'g:', linewidth=2, label='Quadratic approx') 

axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5) 

axes[2].scatter([0], [V_star], color='red', s=100, zorder=5) 

axes[2].set_xlabel('Distance along (1,1) diagonal') 

axes[2].set_ylabel('Portfolio Value ($)') 

axes[2].set_title('Taylor Approximation Accuracy') 

axes[2].legend() 

axes[2].grid(True, alpha=0.3) 

plt.tight_layout() 

plt.show() 

print("\n" + "=" * 60) 

print("KEY INSIGHT: First-order (delta) captures linear sensitivity.") 

print("Second-order (gamma) captures how sensitivity changes.") 

print("The gradient points toward maximum value increase.") 

print("=" * 60) 
