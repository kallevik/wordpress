import numpy as np
from scipy.optimize import fsolve
import warnings


def solve_angles_trig(theta1_rad, L0, L1, L2, L3, config=+1):
    """
    Solves for the angles theta2 and theta3 of a four-bar linkage using the Law of Cosines.
    This method is more geometrically intuitive and robust for special cases.
    Returns the angles in radians or (None, None) if the position is unreachable.
    'config' determines the assembly mode (+1 for open, -1 for crossed).
    """
    # A is at (0,0), D is at (L0, 0)
    # Calculate position of joint B
    B = np.array([L1 * np.cos(theta1_rad), L1 * np.sin(theta1_rad)])
    
    # Calculate the length of the diagonal from B to D
    dist_BD_sq = (B[0] - L0)**2 + B[1]**2
    if dist_BD_sq <= 0: return None, None
    dist_BD = np.sqrt(dist_BD_sq)

    # Check if the linkage can be assembled using the triangle inequality for triangle BCD
    if (L2 + L3 < dist_BD) or (abs(L2 - L3) > dist_BD):
        return None, None

    # --- Solve for theta2 ---
    # Angle of the diagonal line BD
    angle_BD = np.arctan2(B[1], B[0] - L0)
    # Internal angle DBC from Law of Cosines on triangle BCD
    cos_angle_DBC = (L2**2 + dist_BD_sq - L3**2) / (2 * L2 * dist_BD)
    cos_angle_DBC = np.clip(cos_angle_DBC, -1.0, 1.0) # Avoid domain errors
    angle_DBC = np.arccos(cos_angle_DBC)
    # Absolute angle of L2
    theta2_rad = angle_BD + config * angle_DBC + np.pi
    
    # --- Solve for theta3 using the same geometric principles ---
    # Internal angle BDC from Law of Cosines on triangle BCD
    cos_angle_BDC = (L3**2 + dist_BD_sq - L2**2) / (2 * L3 * dist_BD)
    cos_angle_BDC = np.clip(cos_angle_BDC, -1.0, 1.0)
    angle_BDC = np.arccos(cos_angle_BDC)
    # Absolute angle of L3. Note the sign of config is flipped for the angle at the other end of the diagonal.
    theta3_rad = angle_BD - config * angle_BDC
    
    return theta2_rad, theta3_rad

def solve_angles_from_mu(mu_rad, L0, L1, L2, L3, config=+1, initial_guess_t1=np.pi/2):
    """
    Solves for (theta1, theta2, theta3) given a fixed transmission angle mu.
    Uses a numerical root-finding approach.
    mu is the angle between L2 and L3.
    """
    def equations(vars):
        t1, t2 = vars
        # From mu = t3 - t2, we get t3 = t2 + mu
        t3 = t2 + mu_rad
        
        # Vector loop closure equation: L1 + L2 = L0 + L3
        # L1*cos(t1) + L2*cos(t2) - L3*cos(t3) - L0 = 0
        # L1*sin(t1) + L2*sin(t2) - L3*sin(t3)      = 0
        eq1 = L1 * np.cos(t1) + L2 * np.cos(t2) - L3 * np.cos(t3) - L0
        eq2 = L1 * np.sin(t1) + L2 * np.sin(t2) - L3 * np.sin(t3)
        return (eq1, eq2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solution, _, ier, _ = fsolve(equations, (initial_guess_t1, initial_guess_t1), full_output=True)

    if ier == 1: # Solution found
        t1, t2 = solution
        t3 = t2 + mu_rad
        # Check if the solution matches the desired configuration (e.g., open vs crossed)
        # This is a simplified check. A more robust check would compare against a known good point.
        # For now, we return the first valid solution found.
        return t1, t2, t3
    return None, None, None

def solve_angles_from_nu(nu_rad, L0, L1, L2, L3, config=+1, initial_guess_t1=np.pi/2):
    """
    Solves for (theta1, theta2, theta3) given a fixed angle nu between L1 and L2.
    Uses a numerical root-finding approach.
    nu is the angle between L1 and L2.
    """
    def equations(vars):
        t1 = vars[0]
        # From nu = t2 - t1, we get t2 = t1 + nu
        t2 = t1 + nu_rad

        # We need to solve for t3. We can use the vector loop closure equation,
        # but this time to find the intersection of two circles.
        # Circle 1: center B, radius L2. Circle 2: center D, radius L3.
        # This is the same logic as solve_angles_trig, but we need to solve for t3.
        B = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
        C = B + np.array([L2 * np.cos(t2), L2 * np.sin(t2)])
        
        # The error is the difference between the calculated distance |C-D| and L3.
        error = np.linalg.norm(C - np.array([L0, 0])) - L3
        return error

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solution, _, ier, _ = fsolve(equations, (initial_guess_t1,), full_output=True)

    if ier == 1:
        t1 = solution[0]
        t2, t3 = solve_angles_trig(t1, L0, L1, L2, L3, config=config)
        return t1, t2, t3
    return None, None, None

def solve_angles_from_l3(theta3_rad, L0, L1, L2, L3, config=+1):
    """
    Solves for the angles theta1 and theta2 of a four-bar linkage when the output link L3 is the driver.
    This is the inverse kinematic problem solved using the Law of Cosines.
    Returns (theta1_rad, theta2_rad) or (None, None) if unreachable.
    """
    # A is at (0,0), D is at (L0, 0)
    # Calculate position of joint C based on the driver L3
    C = np.array([L0 + L3 * np.cos(theta3_rad), L3 * np.sin(theta3_rad)])

    # Calculate the length of the diagonal from C to A
    dist_AC_sq = C[0]**2 + C[1]**2
    if dist_AC_sq <= 0: return None, None
    dist_AC = np.sqrt(dist_AC_sq)

    # Check if the linkage can be assembled using the triangle inequality for triangle ABC
    if (L1 + L2 < dist_AC) or (abs(L1 - L2) > dist_AC):
        return None, None

    # --- Solve for theta1 ---
    # Angle of the diagonal line AC
    angle_AC = np.arctan2(C[1], C[0])
    # Internal angle BAC from Law of Cosines on triangle ABC
    cos_angle_BAC = (L1**2 + dist_AC_sq - L2**2) / (2 * L1 * dist_AC)
    cos_angle_BAC = np.clip(cos_angle_BAC, -1.0, 1.0)
    angle_BAC = np.arccos(cos_angle_BAC)
    # Absolute angle of L1 (theta1)
    theta1_rad = angle_AC - config * angle_BAC # Note the sign flip for this geometry

    # --- Solve for theta2 ---
    # Position of B is now known
    B = np.array([L1 * np.cos(theta1_rad), L1 * np.sin(theta1_rad)])
    # theta2 is the angle of the vector from B to C
    theta2_rad = np.arctan2(C[1] - B[1], C[0] - B[0])

    return theta1_rad, theta2_rad

def solve_angles(theta1_rad, L0, L1, L2, L3, config=+1):
    """
    Solves for the angles theta2 and theta3 of a four-bar linkage using Freudenstein's equation.
    Returns the angles in radians or (None, None) if the position is unreachable.
    'config' determines the assembly mode (+1 for open, -1 for crossed).
    theta1_rad: angle of L1 relative to the horizontal, vertex at pivot A.
    theta2_rad: angle of L2 relative to the horizontal, vertex at pivot B.
    theta3_rad: angle of L3 relative to the horizontal, vertex at pivot D.
    """
    # Freudenstein's equation coefficients for theta3
    K1 = L0**2 + L1**2 - L2**2 + L3**2
    K2 = 2 * L0 * L3
    K3 = 2 * L1 * L3

    A = K1 - 2 * L0 * L1 * np.cos(theta1_rad)
    B = -K3 * np.sin(theta1_rad)
    C = K2 - K3 * np.cos(theta1_rad)

    # Check if a solution exists
    sqrt_B2_C2 = np.sqrt(B**2 + C**2)
    if sqrt_B2_C2 == 0 or abs(A / sqrt_B2_C2) > 1.0:
        return None, None
        
    # Solve for theta3
    theta3_rad = np.arctan2(B, C) + config * np.arccos(A / sqrt_B2_C2)
    
    # Freudenstein's equation coefficients for theta2
    K4 = L0**2 + L1**2 + L2**2 - L3**2
    K5 = 2 * L0 * L2

    D = K4 - 2 * L0 * L1 * np.cos(theta1_rad)
    E = -2 * L1 * L2 * np.sin(theta1_rad)
    F = K5 - 2 * L1 * L2 * np.cos(theta1_rad)

    # Check if a solution exists
    sqrt_E2_F2 = np.sqrt(E**2 + F**2)
    if sqrt_E2_F2 == 0 or abs(D / sqrt_E2_F2) > 1.0:
        return None, None
        
    # Solve for theta2
    theta2_rad = np.arctan2(E, F) + config * np.arccos(D / sqrt_E2_F2)
    
    return theta2_rad, theta3_rad

def get_joint_positions_trig(theta1_rad, theta3_rad, L0, L1, L3):
    """Calculates the (x, y) coordinates of all linkage joints."""
    A = np.array([0, 0])
    B = np.array([L1 * np.cos(theta1_rad), L1 * np.sin(theta1_rad)])
    D = np.array([L0, 0])
    # C's position is defined by L3 rotating around D at angle theta3.
    C = D + np.array([L3 * np.cos(theta3_rad), L3 * np.sin(theta3_rad)])
    return A, B, C, D

def get_joint_positions(theta1_rad, theta2_rad, L0, L1, L2):
    """Calculates the (x, y) coordinates of all linkage joints."""
    A = np.array([0, 0])
    B = np.array([L1 * np.cos(theta1_rad), L1 * np.sin(theta1_rad)])
    D = np.array([L0, 0])
    # C's position is defined by L2 rotating around B at angle theta2 - Freudenstein
    C = B + np.array([L2 * np.cos(theta2_rad), L2 * np.sin(theta2_rad)]) 
    # C's position is defined by L3 rotating around D at angle theta3 - Law of Cosines
    # C = D + np.array([L3 * np.cos(theta3_rad), L3 * np.sin(theta3_rad)])
    return A, B, C, D


def calculate_jacobian(theta1_rad, theta2_rad, theta3_rad, L1):
    """
    Calculates the Jacobian (dX_C / dTheta_1) for the horizontal motion of point C.
    This is used to determine the mechanical advantage for a horizontal gripper.
    Avoids division by zero, returning infinity if singularity is approached.
    """
    denominator = np.sin(theta2_rad - theta3_rad)
    if np.isclose(denominator, 0):
        # At a singularity (toggle point), output velocity is zero, so Jacobian is zero.
        return 0
    
    numerator = L1 * np.sin(theta1_rad - theta2_rad) * np.sin(theta3_rad)
    return numerator / denominator



def calculate_transmission_angle(theta2_rad, theta3_rad):
    """Calculates the transmission angle (mu) between L2 and L3."""
    angle_diff = np.degrees(theta3_rad - theta2_rad)
    # Normalize to the smallest angle (0 to 180)
    # Normalize to the acute angle for consistent visualization
    mu = abs(angle_diff) % 360
    return min(mu, 360 - mu)