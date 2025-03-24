def calculate_uniform_height(Q, K, pipe_info, tol):
    slope = pipe_info[3] * 10**(-2)
    
    if abs(slope) < 1e-10:
        return np.inf
    
    def uniform_depth_equation(h):
        A = calculate_wetted_area(h, pipe_info, tol)
        P = calculate_wetted_perimeter_from_A(A, pipe_info, tol)
        Rh = A / P if P > 0 else 0
        # Use the absolute value of the slope because it can be negative
        return Q - K * A * Rh**(2/3) * np.sqrt(abs(slope))
    
    # Initial estimate using the absolute value of the slope
    if pipe_info[0] == 'rectangular':
        h_init = (Q / (K * pipe_info[2] * np.sqrt(abs(slope))))**(3/5)
    else:
        h_init = pipe_info[2] / 2  # Radius for circular section
    
    h_uniform = fsolve(uniform_depth_equation, h_init)[0]
    return max(h_uniform, 0)  # Avoid negative heights

def calculate_critical_depth(Q, G, pipe_info, tol):
    # Use absolute value of Q to handle negative flows
    Q_abs = abs(Q)
    
    # Protection against zero flow
    if Q_abs < 1e-10:
        return 0.0
    
    def critical_depth_equation(h):
        # Ensure h is positive
        h_pos = max(h, 1e-6)
        
        A = calculate_wetted_area(h_pos, pipe_info, tol)
        T = calculate_top_width_from_A(A, pipe_info, tol)
        
        # Protection against division by zero
        if A <= 0 or T <= 0:
            return float('inf')
            
        return Q_abs**2 * T - G * A**3
    
    # Better initial estimate based on pipe geometry
    if pipe_info[0] == 'rectangular':
        width = pipe_info[2]
        hc_init = (Q_abs**2/(G * width**2))**(1/3)
    elif pipe_info[0] == 'circular':
        R = pipe_info[2]  # Radius
        # For circular pipes, a good initial estimate is around 0.3-0.4 times the diameter
        hc_init = 0.4 * (2*R)
    else:
        # Default initial estimate
        hc_init = 0.1
    
    # Bounded optimization to ensure positive result
    from scipy.optimize import minimize_scalar
    
    # Define a bounded objective function that finds where critical_depth_equation = 0
    def objective(h):
        return abs(critical_depth_equation(h))
    
    # Use bounded optimization with reasonable bounds
    if pipe_info[0] == 'circular':
        max_depth = 2 * pipe_info[2]  # Maximum depth is diameter
    else:
        # For rectangular or other sections, use a large value
        max_depth = 10.0  # Some large value
    
    try:
        # First attempt with minimize_scalar which is more robust
        result = minimize_scalar(objective, 
                               bounds=(1e-6, max_depth),
                               method='bounded',
                               options={'xatol': 1e-8})
        
        if result.success:
            hc = result.x
        else:
            # Fallback to fsolve with safeguards
            hc = fsolve(critical_depth_equation, hc_init)[0]
            
        # Final sanity check
        hc = max(0, min(hc, max_depth))
        
        # Check if solution is reasonable
        if hc < 1e-6 or hc > max_depth:
            # Use simplified formula as last resort
            if pipe_info[0] == 'rectangular':
                hc = (Q_abs**2/(G * pipe_info[2]**2))**(1/3)
            else:
                # Approximate circular as rectangular with width = 0.8*diameter at critical depth
                effective_width = 0.8 * (2 * pipe_info[2])
                hc = (Q_abs**2/(G * effective_width**2))**(1/3)
            
            hc = max(0, min(hc, max_depth))
    
    except Exception as e:
        print(f"Error in critical depth calculation: {e}")
        # Fallback calculation
        if pipe_info[0] == 'rectangular':
            hc = max(0, (Q_abs**2/(G * pipe_info[2]**2))**(1/3))
        else:
            # Approximate circular as rectangular with width = 0.8*diameter
            effective_width = 0.8 * (2 * pipe_info[2])
            hc = max(0, (Q_abs**2/(G * effective_width**2))**(1/3))
    
    print(f"Critical depth calculation: Q={Q_abs}, hc={hc}")
    return hc

def calculate_step_height(h, slope, J, Fr, DeltaX, direction=1, pipe_info=None):
    if abs(1 - Fr**2) > 1e-6:
        dh_dx = (slope - J) / (1 - Fr**2)
        h_next = h + dh_dx * (direction * DeltaX)
        
        if pipe_info is not None:
            if pipe_info[0] == 'circular':
                h_next = min(h_next, 2 * pipe_info[2])  # Limit to 2*radius for circular section
            h_next = max(h_next, 0)  # Avoid negative heights
        
        return h_next
    return h

def calculate_waterline(DeltaX, NMesh, Q, G, K, h_upstream, h_downstream, pipe_info, tol):
    print(f"Input Parameters:")
    print(f"DeltaX: {DeltaX}, NMesh: {NMesh}, Q: {Q}, G: {G}, K: {K}")
    print(f"h_upstream: {h_upstream}, h_downstream: {h_downstream}")
    print(f"pipe_info: {pipe_info}")
    
    slope = pipe_info[3] * 10**(-2)
    
    # For negative flow, swap the upstream/downstream conditions and invert the slope sign
    if Q < 0:
        # Swap upstream/downstream conditions for negative flow
        h_temp = h_upstream
        h_upstream = h_downstream
        h_downstream = h_temp
        
        slope = -slope  # Invert slope for negative flow
        Q_calc = abs(Q)  # Work with the absolute value of the flow rate
    else:
        Q_calc = Q
    
    # Calculate characteristic heights
    hcr = calculate_critical_depth(Q_calc, G, pipe_info, tol)
    
    try:
        hu = calculate_uniform_height(Q_calc, K, pipe_info, tol)
    except:
        # If the uniform height calculation fails, use a default value
        print("Warning: Unable to calculate the uniform height, using a default value")
        hu = hcr * 1.5  # Arbitrary value, to be adjusted based on context
    
    print(f"Characteristic heights: hcr = {hcr}, hu = {hu}")

    # Calculate the Froude numbers
    A_up, _, T_up, _, _, _ = calculate_flow_parameters(h_upstream, Q_calc, G, K, pipe_info, tol)
    A_down, _, T_down, _, _, _ = calculate_flow_parameters(h_downstream, Q_calc, G, K, pipe_info, tol)
    if abs(Q_calc) < 1e-6:
        Fr_upstream = 0
        Fr_downstream = 0

    denom_up = G * A_up**3 / T_up
    if denom_up < 1e-6:
        denom_up = 1e-6
    Fr_upstream = abs(Q_calc) / np.sqrt(denom_up)
    denom_down = G * A_down**3 / T_down
    if denom_down < 1e-6:
        denom_down = 1e-6
    Fr_downstream = abs(Q_calc) / np.sqrt(denom_down)
    
    print(f"Upstream Froude number: {Fr_upstream}")
    print(f"Downstream Froude number: {Fr_downstream}")
    
    # Determine the flow regime without a special prefix for negative flows
    # The flow direction has already been accounted for with the inversion of conditions
    if Fr_upstream > 1 and Fr_downstream > 1:
        regime_initial = "Supercritical"
    elif Fr_upstream < 1 and Fr_downstream < 1:
        regime_initial = "Subcritical"
    else:
        regime_initial = "Supercritical and Subcritical"
    
    print(f"Initial regime: {regime_initial}")

    # Process based on the flow regime
    h = np.zeros(NMesh)
    
    if regime_initial == "Subcritical":
        # For the river regime, calculate from downstream
        direction = 1
        h = calculate_subcritical_profile(h_downstream, NMesh, Q_calc, G, K, 
                                         DeltaX, slope, pipe_info, direction, tol)
    
    elif regime_initial == "Supercritical":
        # For the torrent regime, calculate from upstream
        direction = 1
        h = calculate_supercritical_profile(h_upstream, NMesh, Q_calc, G, K, 
                                          DeltaX, slope, pipe_info, direction, tol)
    
    elif regime_initial == "Supercritical and Subcritical":
        # For the mixed regime, determine the shock position
        h = calculate_mixed_flow(hcr, hu, NMesh, Q_calc, G, K, DeltaX, 
                               slope, h_upstream, h_downstream, pipe_info, tol)
    
    # For negative flows, invert the calculated profile
    if Q < 0:
        h = h[::-1]  # Invert the profile for negative flow
    
    return h


def calculate_subcritical_profile(h_known, NMesh, Q_calc, G, K, DeltaX, slope, pipe_info, direction, tol):
    # Calculates the subcritical flow profile
    # direction: 1 for calculation from downstream to upstream (normal), -1 for upstream to downstream (reverse)

    h = np.zeros(NMesh)
    
    # Starting point for the calculation based on the direction
    if direction == 1:  # Standard calculation (downstream to upstream)
        h[NMesh-2] = h_known  # Downstream boundary condition
        
        # Calculation moving upstream
        for i in range(NMesh-2, 1, -1):
            # Calculate parameters for the current section
            A, P, T, Fr, Rh, J = calculate_flow_parameters(h[i], Q_calc, G, K, pipe_info, tol)
            
            # Protection against problematic values
            if abs(1 - Fr**2) < 1e-6:  # If Fr is close to 1 (critical)
                h[i-1] = h[i]  # Keep the same height
                continue
            
            # Calculate the derivative of h with respect to x
            dh_dx = (slope - J) / (1 - Fr**2)
            
            # Update the height for the previous section
            h[i-1] = h[i] - dh_dx * DeltaX
            
            # Check physical limits
            if pipe_info[0] == 'circular':
                h[i-1] = min(h[i-1], 2 * pipe_info[2])  # Limit to 2*radius for circular section
            h[i-1] = max(h[i-1], 0)  # Avoid negative heights
    else:  # Reverse calculation (upstream to downstream) - rarely used
        h[1] = h_known  # Upstream boundary condition
        
        # Calculation moving downstream
        for i in range(1, NMesh-1):
            A, P, T, Fr, Rh, J = calculate_flow_parameters(h[i], Q_calc, G, K, pipe_info, tol)
            
            # Protection against problematic values
            if abs(1 - Fr**2) < 1e-6:
                h[i+1] = h[i]
                continue
            
            dh_dx = (slope - J) / (1 - Fr**2)
            h[i+1] = h[i] + dh_dx * DeltaX
            
            if pipe_info[0] == 'circular':
                h[i+1] = min(h[i+1], 2 * pipe_info[2])
            h[i+1] = max(h[i+1], 0)
    
    return h

def calculate_supercritical_profile(h_known, NMesh, Q_calc, G, K, DeltaX, slope, pipe_info, direction, tol):
    # Calculates the supercritical flow profile
    # direction: 1 for calculation from upstream to downstream (normal), -1 for downstream to upstream (reverse)

    h = np.zeros(NMesh)
    
    if direction == 1:  # Standard calculation (upstream to downstream)
        h[1] = h_known  # Upstream boundary condition
    
        # Calculation moving downstream
        for i in range(1, NMesh-1):
            A, P, T, Fr, Rh, J = calculate_flow_parameters(h[i], Q_calc, G, K, pipe_info, tol)
            
            # Protection against problematic values
            if abs(1 - Fr**2) < 1e-6:
                h[i+1] = h[i]
                continue
            
            dh_dx = (slope - J) / (1 - Fr**2)
            h[i+1] = h[i] + dh_dx * DeltaX
            
            if pipe_info[0] == 'circular':
                h[i+1] = min(h[i+1], 2 * pipe_info[2])
            h[i+1] = max(h[i+1], 0)
    else:  # Reverse calculation (downstream to upstream) - rarely used
        h[NMesh-2] = h_known  # Downstream boundary condition
        
        # Calculation moving upstream
        for i in range(NMesh-2, 1, -1):
            A, P, T, Fr, Rh, J = calculate_flow_parameters(h[i], Q_calc, G, K, pipe_info, tol)
            
            if abs(1 - Fr**2) < 1e-6:
                h[i-1] = h[i]
                continue
            
            dh_dx = (slope - J) / (1 - Fr**2)
            h[i-1] = h[i] - dh_dx * DeltaX
            
            if pipe_info[0] == 'circular':
                h[i-1] = min(h[i-1], 2 * pipe_info[2])
            h[i-1] = max(h[i-1], 0)
    
    return h

def calculate_flow_parameters(h_local, Q_calc, G, K, pipe_info, tol):
    # Calculates hydraulic parameters for a given height
    # Protection against zero or negative heights
    h_local = max(h_local, 1e-6)  # Small positive value to avoid division by zero
    
    A = calculate_wetted_area(h_local, pipe_info, tol)
    P = calculate_wetted_perimeter_from_A(A, pipe_info, tol)
    T = calculate_top_width_from_A(A, pipe_info, tol)
    
    A_safe = max(A, tol)
    P_safe = max(P, tol)
    T_safe = max(T, tol)
    
    denom = G * A_safe**3 / T_safe
    if denom < 1e-6:
        denom = 1e-6
    
    Fr = abs(Q_calc) / np.sqrt(denom)
    
    # Calculation of hydraulic radius
    Rh = A / P_safe
    Rh_safe = max(Rh, tol)
    
    # Calculation of friction slope (with protection)
    if abs(Q_calc) < 1e-6:
        J = 0
    else:
        J = (abs(Q_calc) / max(K * A_safe * Rh_safe**(2/3), 1e-6))**2
    
    return A, P, T, Fr, Rh, J

def calculate_mixed_flow(hcr, hu, NMesh, Q_calc, G, K, DeltaX, slope, h_upstream, h_downstream, pipe_info, tol):
    # Calculates a mixed flow with a hydraulic jump based on the conjugate heights

    print(f"Calculating a mixed regime with a hydraulic jump")
    print(f"Upstream: h={h_upstream}, Downstream: h={h_downstream}, hcr={hcr}")
    
    # 1. Calculate the supercritical profile from upstream (upstream profile)
    direction = 1
    h_super = calculate_supercritical_profile(h_upstream, NMesh, Q_calc, G, K, DeltaX, slope, pipe_info, direction, tol)
    
    # 2. Calculate the subcritical profile from downstream (downstream profile)
    direction = 1
    h_sub = calculate_subcritical_profile(h_downstream, NMesh, Q_calc, G, K, DeltaX, slope, pipe_info, direction, tol)
    
    # 3. Determine the position of the hydraulic jump
    # Calculate the Froude numbers along the supercritical profile
    Fr_super = np.zeros(NMesh)
    for i in range(1, NMesh-1):
        if h_super[i] > 1e-6:
            A, _, T, Fr, _, _ = calculate_flow_parameters(h_super[i], Q_calc, G, K, pipe_info, tol)
            Fr_super[i] = Fr
    
    # 4. Calculate the conjugate heights along the supercritical profile
    h_conj = np.zeros(NMesh)
    for i in range(1, NMesh-1):
        if Fr_super[i] > 1:  # Only in supercritical regime
            h_conj[i] = h_super[i] * 0.5 * (-1 + np.sqrt(1 + 8 * Fr_super[i]**2))
        else:
            h_conj[i] = h_super[i]
    
    # 5. Find the position of the jump where the conjugate height corresponds to the subcritical height
    ressaut_pos = None
    min_diff = float('inf')
    
    for i in range(1, NMesh-10):  # Avoid domain edges
        if Fr_super[i] > 1:  # Only if in supercritical regime
            # Difference between conjugate height and subcritical profile
            diff = abs(h_conj[i] - h_sub[i])
            
            if diff < min_diff:
                min_diff = diff
                ressaut_pos = i
    
    # If no valid position is found, place the jump where the Froude number drops below 1
    if ressaut_pos is None:
        for i in range(1, NMesh-2):
            if Fr_super[i] > 1 and Fr_super[i+1] < 1:
                ressaut_pos = i
                break
    
    # If still no position found, place the jump in the middle
    if ressaut_pos is None:
        ressaut_pos = NMesh // 2
        print("Warning: Jump position not found, placed in the middle of the domain")
    else:
        print(f"Hydraulic jump found at position x = {ressaut_pos * DeltaX:.2f} m (mesh {ressaut_pos})")
        print(f"Height before jump: {h_super[ressaut_pos]:.5f} m")
        print(f"Froude number: {Fr_super[ressaut_pos]:.5f}")
        print(f"Conjugate height: {h_conj[ressaut_pos]:.5f} m")
    
    # 6. Build the final profile with the jump
    h_result = np.zeros(NMesh)
    
    # Supercritical part before the jump
    for i in range(1, ressaut_pos + 1):
        h_result[i] = h_super[i]
    
    # Transition to the jump (conjugate height)
    h_result[ressaut_pos + 1] = h_conj[ressaut_pos]
    
    # Gradual smoothing after the jump to match the subcritical profile
    # Transition length after the jump
    trans_length = min(10, NMesh - ressaut_pos - 3)
    
    for i in range(1, trans_length):
        idx = ressaut_pos + 1 + i
        if idx < NMesh - 1:
            # Progressive weighting for transition to subcritical profile
            alpha = i / trans_length
            h_result[idx] = (1 - alpha) * h_conj[ressaut_pos] + alpha * h_sub[idx]
    
    # Subcritical part after the transition
    for i in range(ressaut_pos + 1 + trans_length, NMesh):
        h_result[i] = h_sub[i]
    
    # 7. Check and correct potential instabilities
    for i in range(2, NMesh-1):
        # Avoid abrupt non-physical jumps
        if abs(h_result[i] - h_result[i-1]) > 0.5 * max(h_result[i], h_result[i-1]):
            h_result[i] = 0.5 * (h_result[i] + h_result[i-1])
        
        # Ensure positive heights
        h_result[i] = max(h_result[i], 1e-6)
    
    return h_result

def calculate_conjugate_depth(h1, Fr1, pipe_info):
    # Calculates the conjugate height in a hydraulic jump
    if pipe_info[0] == 'rectangular':
        # Exact formula for rectangular section
        h2 = h1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr1**2))
        return h2
    elif pipe_info[0] == 'circular':
        # Approximation for circular section
        # This approximation is based on the conservation of momentum
        # but adapted for a circular section
        if h1 > pipe_info[2]:  # If height exceeds the radius
            # Transition to rectangular formula when near full
            ratio = min(h1 / (2 * pipe_info[2]), 1.0)  # 0 to 1
            rect_h2 = h1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr1**2))
            return ratio * rect_h2 + (1 - ratio) * min(2 * pipe_info[2], rect_h2)
        else:
            # Approximation for partially filled section
            rect_h2 = h1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr1**2))
            return min(2 * pipe_info[2], rect_h2)  # Limit to diameter
    else:
        # Default case: rectangular approximation
        h2 = h1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr1**2))
        return h2