import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Import your existing helper functions
# Ensure tools.py is in the same directory!
try:
    from tools import solve_angles_trig, get_joint_positions_trig, calculate_jacobian, calculate_transmission_angle
except ImportError:
    st.error("Could not import 'tools.py'. Please make sure the file exists in the same directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Linkage Simulator")

st.title("Four-Bar Linkage Simulator")
st.markdown("""
Adjust the link lengths in the sidebar to visualize the mechanism and its performance.
The graphs update automatically.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Linkage Parameters")

L0 = st.sidebar.slider("L₀ (Ground)", 20.0, 120.0, 39.0, 0.1)
L1 = st.sidebar.slider("L₁ (Input Crank)", 20.0, 120.0, 40.0, 0.1)
L2 = st.sidebar.slider("L₂ (Coupler)", 20.0, 120.0, 40.0, 0.1)
L3 = st.sidebar.slider("L₃ (Output Rocker)", 20.0, 120.0, 40.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("State Variables")
theta1_deg = st.sidebar.slider("Input Angle θ₁ (°)", 0.0, 360.0, 45.0, 1.0)
torque = st.sidebar.number_input("Input Torque (Nm)", value=1.0)
config_select = st.sidebar.radio("Configuration", ["Open", "Crossed"], index=0)
config = 1 if config_select == "Open" else -1

# --- Calculations ---
theta1_rad = np.deg2rad(theta1_deg)

# 1. Full Range Analysis (for Graphs)
theta1_range_rad = np.linspace(0, 2 * np.pi, 360)
forces = []
transmission_angles = []
reachable_angles_deg = []

for t1 in theta1_range_rad:
    t2, t3 = solve_angles_trig(t1, L0, L1, L2, L3, config=config)
    if t2 is not None:
        jacobian = calculate_jacobian(t1, t2, t3, L1)
        if np.isclose(jacobian, 0):
            force = np.nan # Use NaN to break the line in plot instead of inf
        else:
            force = abs(torque / jacobian)
        
        forces.append(force)
        transmission_angles.append(calculate_transmission_angle(t2, t3))
        reachable_angles_deg.append(np.rad2deg(t1))

forces_arr = np.array(forces, dtype=float)

# 2. Current Position Analysis (for Diagram)
theta2_rad, theta3_rad = solve_angles_trig(theta1_rad, L0, L1, L2, L3, config=config)

# --- Layout: Columns for Plots ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Mechanism Visualizer")
    
    fig_linkage, ax_linkage = plt.subplots(figsize=(8, 6))
    ax_linkage.set_aspect('equal', adjustable='box')
    ax_linkage.grid(True, linestyle='--', alpha=0.6)

    title_text = "Linkage Cannot Assemble"
    
    if theta2_rad is not None:
        # Calculate joint positions
        A, B, C, D = get_joint_positions_trig(theta1_rad, theta3_rad, L0, L1, L3)
        
        # Plot links
        ax_linkage.plot([D[0], A[0]], [D[1], A[1]], 's-', label=r'$L_0$ (Ground)', lw=4, color='black')
        ax_linkage.plot([A[0], B[0]], [A[1], B[1]], 'o-', label=r'$L_1$ (Crank)', lw=5, color='darkorange')
        ax_linkage.plot([B[0], C[0]], [B[1], C[1]], 'o-', label=r'$L_2$ (Coupler)', lw=5, color='steelblue')
        ax_linkage.plot([C[0], D[0]], [C[1], D[1]], 'o-', label=r'$L_3$ (Rocker)', lw=5, color='seagreen')

        # Current Stats Calculation
        jacobian_current = calculate_jacobian(theta1_rad, theta2_rad, theta3_rad, L1)
        force_current = float('inf') if np.isclose(jacobian_current, 0) else abs(torque / jacobian_current)
        trans_current = calculate_transmission_angle(theta2_rad, theta3_rad)

        title_text = f"Output Force: {force_current:.2f} N | Trans. Angle: {trans_current:.1f}°"
        
        # Visual Wedges (Optional, simplified for Streamlit speed)
        wedge_radius = min(L1, L2, L3) * 0.4
        ax_linkage.add_patch(Wedge(A, wedge_radius, 0, theta1_deg, color='darkorange', alpha=0.3))

    # Set dynamic limits based on link lengths
    max_range = L0 + L1 + L2 + L3
    ax_linkage.set_xlim(-L1 * 1.5, L0 + L3 * 1.5)
    ax_linkage.set_ylim(-max_range * 0.5, max_range * 0.8)
    ax_linkage.set_title(title_text)
    ax_linkage.legend(loc='upper right')
    
    st.pyplot(fig_linkage)


with col2:
    # --- Force Plot ---
    st.subheader("Mechanical Advantage")
    fig_force, ax_force = plt.subplots(figsize=(6, 3))
    
    if len(forces) > 0:
        ax_force.plot(reachable_angles_deg, forces_arr, color='cornflowerblue')
        ax_force.set_xlabel("Input Angle θ₁ (°)")
        ax_force.set_ylabel("Force Magnitude")
        ax_force.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Dynamic Y-Limit (similar to your logic)
        finite_forces = forces_arr[np.isfinite(forces_arr)]
        if len(finite_forces) > 0:
            upper_limit = np.nanpercentile(finite_forces, 98)
            ax_force.set_ylim(0, upper_limit * 1.2)
        
        # Current Position Marker
        if theta2_rad is not None:
             # Clamp for plotting
            plot_force = min(force_current, ax_force.get_ylim()[1])
            ax_force.plot(theta1_deg, plot_force, 'ro', markersize=8)

    st.pyplot(fig_force)

    # --- Transmission Angle Plot ---
    st.subheader("Transmission Angle")
    fig_trans, ax_trans = plt.subplots(figsize=(6, 3))
    
    if len(transmission_angles) > 0:
        ax_trans.plot(reachable_angles_deg, transmission_angles, color='purple')
        ax_trans.axhline(90, color='green', linestyle='--', alpha=0.5)
        ax_trans.set_xlabel("Input Angle θ₁ (°)")
        ax_trans.set_ylabel("Angle μ (°)")
        ax_trans.set_ylim(0, 185)
        ax_trans.grid(True, linestyle='--', alpha=0.6)

        # Current Position Marker
        if theta2_rad is not None:
             ax_trans.plot(theta1_deg, trans_current, 'ro', markersize=8)

    st.pyplot(fig_trans)