import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 1. Title and Description
st.title("Gripper Kinematics Test")
st.write("Adjust the sliders to see the live graph update.")

# 2. Setup Inputs (Sidebar)
st.sidebar.header("Mechanism Parameters")
# Slider for Link Length (L1)
L1 = st.sidebar.slider("Link 1 Length (mm)", min_value=10, max_value=100, value=50)
# Slider for Input Angle (theta)
theta_start = st.sidebar.slider("Input Angle Offset (deg)", 0, 360, 45)

# 3. Simple Calculation (Dummy Math for Testing)
# Generates x values (0 to 360 degrees)
x = np.linspace(0, 360, 100)
# y represents a dummy 'Mechanical Advantage' curve that shifts based on your sliders
y = L1 * np.sin(np.radians(x + theta_start))

# 4. Plotting
fig, ax = plt.subplots()
ax.plot(x, y, color='blue', label='Mechanical Advantage')
ax.set_xlabel("Input Rotation (deg)")
ax.set_ylabel("Force Output (N)")
ax.grid(True)
ax.legend()

# 5. Display the Plot in Streamlit
st.pyplot(fig)