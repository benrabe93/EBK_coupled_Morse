import numpy as np
from aux_functions import leap_frog, compute_energy
from numba import njit
from numba.typed import List



@njit
def sort_vertices(vertices, mean=None):
    """Sort vertices of a 2D polygon in counter-clockwise order.

    Args:
        vertices (np.ndarray): Coordinates of the vertices of a polygon.
        mean (tuple, optional): Custom location of the mean of the vertices. Defaults to None.

    Returns:
        np.ndarray: Sorted vertices in counter-clockwise order.
    """
    
    if mean is None:
        x_mean = np.mean(vertices[:,0])
        y_mean = np.mean(vertices[:,1])
    else:
        x_mean, y_mean = mean
        
    angles = np.arctan2(vertices[:,1] - y_mean, vertices[:,0] - x_mean) % (2*np.pi) # compute angles w.r.t. mean
    ini_ind = np.argmin(angles) # Choose smallest angle as starting point
    sorted_vertices = np.zeros(vertices.shape)
    sorted_vertices[0] = vertices[ini_ind]
    angle_ = angles[ini_ind]
    for i in range(1, len(vertices)):
        diff_angles = angles - angle_
        diff_angles_ind = (diff_angles > 0)
        new_ind = np.argmin(diff_angles[diff_angles_ind])
        sorted_vertices[i] = vertices[diff_angles_ind][new_ind]
        angle_ = angles[diff_angles_ind][new_ind]
    return sorted_vertices


@njit
def polygon_area(vertices):
    """Compute the area of a 2D polygon using the shoelace formula.

    Args:
        vertices (np.ndarray): Coordinates of the sorted vertices of a polygon.

    Returns:
        float: Area of the polygon.
    """
    
    N = len(vertices)
    area = 0.0
    for i in range(N):
        j = (i+1) % N
        area += vertices[i,0] * vertices[j,1] - vertices[i,1] * vertices[j,0]
    return 0.5*abs(area)


@njit
def check_coherence_and_return_area(sos, sos_tol=0.01, sos_tol_num=6):
    """Check surfaces of section for ergodicity and large gaps.

    Args:
        sos (np.ndarray): Unsorted coordinates of the SOS points.
        sos_tol (float, optional): Tolerance for large gaps between SOS points. Defaults to 0.01.
        sos_tol_num (int, optional): Tolerance for number of fluctuating points. Defaults to 6.

    Returns:
        float: area of SOS polygon; nan for ergodicity or large gaps
    """
    
    sorted_vertices = sort_vertices(sos) # Sort vertices in counter-clockwise order
    len_vertices = len(sorted_vertices)
    sos_area = polygon_area(sorted_vertices) # Compute area of sorted vertices
    next_points = np.vstack((sorted_vertices, sorted_vertices[:sos_tol_num+1]))

    # Check for large gaps
    x1 = np.mean(sos[:,0]); y1 = np.mean(sos[:,1])
    x2, y2 = next_points[:len_vertices].T; x3, y3 = next_points[1:len_vertices+1].T
    triangle_area = 0.5*np.abs(x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)
    if np.max(triangle_area) > sos_tol*sos_area:
        print('Large gap in sos detected!')
        return np.nan, False
    
    # Check for ergocity: is sorted vertices distance greater than distance to next *sos_tol_num* points?
    dist_sorted_vertices = np.sqrt((next_points[1:len_vertices+1, 0] - sorted_vertices[:,0])**2 + (next_points[1:len_vertices+1, 1] - sorted_vertices[:,1])**2)
    next_n_points_x, next_n_points_y = np.zeros((2, sos_tol_num, len_vertices))
    for i in range(sos_tol_num):
        next_n_points_x[i] = next_points[i+2:len_vertices+i+2, 0]
        next_n_points_y[i] = next_points[i+2:len_vertices+i+2, 1]
    next_n_dist = np.sqrt((next_n_points_x - sorted_vertices[:,0])**2 + (next_n_points_y - sorted_vertices[:,1])**2)
    min_next_n_dist = np.zeros((len_vertices, 2))
    for i in range(len_vertices):
        min_next_n_dist[i] = np.partition(next_n_dist[:,i], 1)[:2] # Two closest points
    # min_next_n_dist = np.array([np.min(col) for col in np.sqrt((next_n_points_x - sorted_vertices[:,0])**2 + (next_n_points_y - sorted_vertices[:,1])**2).T]) # One closest point
    # min_next_n_dist = np.min(np.sqrt((next_n_points_x - sorted_vertices[:,0])**2 + (next_n_points_y - sorted_vertices[:,1])**2), axis=0)
    if np.sum((dist_sorted_vertices > min_next_n_dist[:,0]) & (dist_sorted_vertices > min_next_n_dist[:,1])) > sos_tol*len_vertices:
        print('Ergodicity detected!')
        return np.nan, True
    
    return sos_area, False


@njit
def trajectory_close_method(x, y, px, py, sos, S1, slope, intsec, n_c1=+1):
    """_summary_

    Args:
        n_c1 (int, optional): If positive, then S_close_corr < 0, if negative, then S_close_corr > 0.
    Choose   n_c1 = -k   for   k - 0.5 < period_c1 < k
    and      n_c1 = +k   for         k < period_c1 < k + 0.5. Defaults to +1.

    Returns:
        float: Total action S_tot along closed path
        float: Action correction along phase space torus
    """
    
    if np.isnan(S1):
        return np.nan, np.nan
    
    ind_ini = 0
    ind_fin = ind_ini + 1
    
    # Compute total action from one sos point to next: S_close
    S_close = 0.0
    sos_count_ = -1
    trigger = 0
    for i in range(1, len(x)):
        if np.isnan(slope):
            if y[i-1] >= intsec and y[i] < intsec:
                sos_count_ += 1
                
                x1 = x[i-1] + (intsec - y[i-1])*(x[i] - x[i-1])/(y[i] - y[i-1])
                y1 = intsec
                px1 = px[i-1] + (intsec - y[i-1])*(px[i] - px[i-1])/(y[i] - y[i-1])
                py1 = py[i-1] + (intsec - y[i-1])*(py[i] - py[i-1])/(y[i] - y[i-1])
        else:
            if y[i-1] >= slope*x[i-1] + intsec and y[i] < slope*x[i] + intsec:
                sos_count_ += 1
                
                x1 = (y[i-1] - x[i-1]*(y[i] - y[i-1])/(x[i] - x[i-1]) - intsec)/(slope - (y[i] - y[i-1])/(x[i] - x[i-1]))
                y1 = slope*x1 + intsec
                px1 = px[i-1] + (px[i] - px[i-1])*(x1 - x[i-1])/(x[i] - x[i-1])
                py1 = py[i-1] + (py[i] - py[i-1])*(x1 - x[i-1])/(x[i] - x[i-1])
        
        if sos_count_ == ind_ini and not trigger:
            trigger = 1
            S_close += 0.5*(px[i] + px1)*(x[i] - x1) + 0.5*(py[i] + py1)*(y[i] - y1)
            # x1_start = x1; px1_start = px1; y1_start = y1; py1_start = py1
        elif ind_ini <= sos_count_ and sos_count_ < ind_fin and trigger:
            S_close += 0.5*(px[i] + px[i-1])*(x[i] - x[i-1]) + 0.5*(py[i] + py[i-1])*(y[i] - y[i-1])
        elif sos_count_ == ind_fin:
            S_close += 0.5*(px1 + px[i-1])*(x1 - x[i-1]) + 0.5*(py1 + py[i-1])*(y1 - y[i-1])
            # S_close_corr = 0.5*(px1_start + px1)*(x1_start - x1) + 0.5*(py1_start + py1)*(y1_start - y1)
            break
        
    # Compute action correction along sos path
    sos = np.asarray(sos)
    sorted_vertices = sort_vertices(sos)
    sorted_vertices_double = np.concatenate((sorted_vertices, sorted_vertices))
    sorted_ind_ini = np.where((sorted_vertices[:,0] == sos[ind_ini,0]) & (sorted_vertices[:,1] == sos[ind_ini,1]))[0][0]
    sorted_ind_fin = np.where((sorted_vertices[:,0] == sos[ind_fin,0]) & (sorted_vertices[:,1] == sos[ind_fin,1]))[0][0]
    x_mean = np.mean(sos[:,0])
    y_mean = np.mean(sos[:,1])
    angle_ini = np.arctan2(sos[ind_ini,1] - y_mean, sos[ind_ini,0] - x_mean) % (2*np.pi)
    angle_fin = np.arctan2(sos[ind_fin,1] - y_mean, sos[ind_fin,0] - x_mean) % (2*np.pi)
    angle_diff = (angle_fin - angle_ini) % (2*np.pi)
    if angle_diff == 0:
        S_close_corr = 0.0
    else:
        if angle_diff <= np.pi:
            if sorted_ind_ini < sorted_ind_fin:
                range_ = (sorted_ind_ini, sorted_ind_fin)
            else:
                range_ = (sorted_ind_ini, len(sorted_vertices) + sorted_ind_fin)
        else:
            if sorted_ind_fin < sorted_ind_ini:
                range_ = (sorted_ind_fin, sorted_ind_ini)
            else:
                range_ = (sorted_ind_fin, len(sorted_vertices) + sorted_ind_ini)
        S_close_corr = (-np.sign(n_c1)*0.5*abs(np.sum((sorted_vertices_double[range_[0]+1:range_[1]+1,1] + sorted_vertices_double[range_[0]:range_[1],1]) *
                                                      (sorted_vertices_double[range_[0]+1:range_[1]+1,0] - sorted_vertices_double[range_[0]:range_[1],0]))))
    # print(S_close_corr_, angle_diff, S_close_corr)

    # Return action S_2
    n_c2 = ind_fin - ind_ini # = 1
    S2 = ((S_close + S_close_corr) - abs(n_c1)*S1)/n_c2
    return S_close, abs(S_close_corr)


@njit
def run_trajectory_and_get_sections(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling, n_t=1e6, symmetric_only=0):
    """Run trajectory and locate proper surfaces of section (sos) for diagonally symmetric or non-symmetric trajectories.

    Args:
        n_t (scalar, optional): Number of time steps. Defaults to 1e6.
        symmetric_only (int, optional): 1 - Treat trajectory as diagonally symmetric.
                                        0 - Treat arbitrary trajectories. Defaults to 0.

    Returns:
        x, y: Coordinates of the trajectory.
        slopes: Slopes and intersection points of the proper surfaces of section.
        corners: Coordinates of the corner points of the trajectory path.
    """
    
    n_t = round(n_t)
    x, y, px, py, energy = np.zeros((5, n_t+1))
    x[0], y[0], px[0], py[0], energy[0] = np.array([x_ini, y_ini, px_ini, py_ini, energy_ini])
    closest_point_xy = np.inf
    
    # Compute classical trajectory
    for t_ in range(n_t):
        x_, y_, px_, py_ = leap_frog(x[t_], y[t_], px[t_], py[t_], dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling)
        x[t_+1], y[t_+1], px[t_+1], py[t_+1] = np.array([x_, y_, px_, py_])
        energy[t_+1] = compute_energy(x_, y_, px_, py_, k, omega1, omega2, D1, a1, D2, a2, system, coupling)
        
        # Find index of closest point along x = y
        if y[t_] >= x[t_] and y_ < x_:
            new_closest_point = 0.5*(x_ + y_)
            if new_closest_point < closest_point_xy:
                closest_point_xy = new_closest_point

    if symmetric_only: # Find proper surfaces of section (sos) for diagonally symmetric trajectories
        ind_x_max = np.argmax(x)
        ind_y_max = np.argmax(y)
        intsec2_min = 2*closest_point_xy
        intsec2_max = min(x[ind_x_max] + y[ind_x_max], x[ind_y_max] + y[ind_y_max])
        intsec2 = 0.5*(intsec2_min + intsec2_max)
        return x, y, np.array([1.0, 0.0, -1.0, intsec2]), np.nan*np.ones((2,4))
    
    else: # Find proper surfaces of section (sos) by first locating corner points of trajectory path
        n_p = round(0.02*n_t)
        n_ = 500
        p_total = px**2 + py**2
        p_total_argsort = np.argsort(p_total)[:n_p]

        ind_corner_far_start = p_total_argsort[np.argmax((x[p_total_argsort] - x_ini)**2 + (y[p_total_argsort] - y_ini)**2)]
        while True:
            x_corner_far_start = x[ind_corner_far_start]
            y_corner_far_start = y[ind_corner_far_start]
            dist_argsort = np.argsort((x - x_corner_far_start)**2 + (y - y_corner_far_start)**2)[:n_]
            ind_corner_far = dist_argsort[np.argmin(p_total[dist_argsort])]
            if ind_corner_far == ind_corner_far_start:
                break
            ind_corner_far_start = ind_corner_far
        
        distances_3 = (np.stack((np.sqrt((x[p_total_argsort] - x_ini)**2 + (y[p_total_argsort] - y_ini)**2), 
                                    np.sqrt((x[p_total_argsort] - x[ind_corner_far])**2 + (y[p_total_argsort] - y[ind_corner_far])**2))))
        ind_corner_3_start = p_total_argsort[np.argmax(np.array([np.min(col) for col in distances_3.T]))]
        # ind_corner_3_start = p_total_argsort[np.argmax(np.min(distances_3, axis=0))]
        while True:
            x_corner_3_start = x[ind_corner_3_start]
            y_corner_3_start = y[ind_corner_3_start]
            dist_argsort = np.argsort((x - x_corner_3_start)**2 + (y - y_corner_3_start)**2)[:n_]
            ind_corner_3 = dist_argsort[np.argmin(p_total[dist_argsort])]
            if ind_corner_3 == ind_corner_3_start:
                break
            ind_corner_3_start = ind_corner_3

        if (x[ind_corner_3] == x_ini and y[ind_corner_3] == y_ini) or (x[ind_corner_3] == x[ind_corner_far] and y[ind_corner_3] == y[ind_corner_far]):
            print('No four distinctive vertices found!')
            return x, y, np.nan*np.ones(4), np.nan*np.ones((2,4))
        distances_4 = (np.stack((np.sqrt((x[p_total_argsort] - x_ini)**2 + (y[p_total_argsort] - y_ini)**2), 
                                    np.sqrt((x[p_total_argsort] - x[ind_corner_far])**2 + (y[p_total_argsort] - y[ind_corner_far])**2), 
                                    np.sqrt((x[p_total_argsort] - x[ind_corner_3])**2 + (y[p_total_argsort] - y[ind_corner_3])**2))))
        weights_4 = np.ones(len(p_total_argsort))
        rel_dist = np.sqrt((x[ind_corner_3] - x_ini)**2 + (y[ind_corner_3] - y_ini)**2) / np.sqrt((x[ind_corner_3] - x[ind_corner_far])**2 + (y[ind_corner_3] - y[ind_corner_far])**2)
        if rel_dist < 1:
            weights_4[np.array([np.argmin(col) for col in distances_4.T]) == 1] = 1/rel_dist
            # weights_4[np.argmin(distances_4, axis=0) == 1] = 1/rel_dist
        else:
            weights_4[np.array([np.argmin(col) for col in distances_4.T]) == 0] = rel_dist
            # weights_4[np.argmin(distances_4, axis=0) == 0] = rel_dist
        ind_corner_4_start = p_total_argsort[np.argmax(np.array([np.min(col) for col in distances_4.T])*weights_4)]
        # ind_corner_4_start = p_total_argsort[np.argmax(np.min(distances_4, axis=0)*weights_4)]

        while True:
            x_corner_4_start = x[ind_corner_4_start]
            y_corner_4_start = y[ind_corner_4_start]
            dist_argsort = np.argsort((x - x_corner_4_start)**2 + (y - y_corner_4_start)**2)[:n_]
            ind_corner_4 = dist_argsort[np.argmin(p_total[dist_argsort])]
            if ind_corner_4 == ind_corner_4_start:
                break
            ind_corner_4_start = ind_corner_4

        delta_x_min = 1e-3
        vertices = np.array([(x_ini, y_ini), (x[ind_corner_far], y[ind_corner_far]), (x[ind_corner_3], y[ind_corner_3]), (x[ind_corner_4], y[ind_corner_4])])

        # from matplotlib import pyplot as plt
        # print(vertices)
        # plt.scatter(x[p_total_argsort], y[p_total_argsort], color='C1')
        # plt.plot(x, y, lw=0.5)
        # plt.scatter(vertices[0,0], vertices[0,1], color='C2', label='$x_{ini}$')
        # plt.scatter(vertices[1,0], vertices[1,1], color='C3', label='$x_{far}$')
        # plt.scatter(vertices[2,0], vertices[2,1], color='C4', label='$x_3$')
        # plt.scatter(vertices[3,0], vertices[3,1], color='C5', label='$x_4$')
        # plt.scatter(x_corner_3_start, y_corner_3_start, marker='x', color='C4')
        # plt.scatter(x_corner_4_start, y_corner_4_start, marker='x', color='C5')
        # plt.legend()
        # plt.show()

        distances = np.concatenate((np.sqrt((vertices[0,0] - vertices[1:,0])**2 + (vertices[0,1] - vertices[1:,1])**2), 
                                    np.sqrt((vertices[1,0] - vertices[2:,0])**2 + (vertices[1,1] - vertices[2:,1])**2), 
                                    np.sqrt((vertices[2,0] - vertices[3:,0])**2 + (vertices[2,1] - vertices[3:,1])**2)))
        if np.min(distances) < delta_x_min:
            print('No four distinctive vertices found!')
            return x, y, np.nan*np.ones(4), np.nan*np.ones((2,4))
        else:
            sorted_vertices = sort_vertices(vertices)
            x_corners, y_corners = sorted_vertices.T

            midpoint1_x = 0.5*(x_corners[0] + x_corners[1])
            midpoint1_y = 0.5*(y_corners[0] + y_corners[1])
            midpoint2_x = 0.5*(x_corners[1] + x_corners[2])
            midpoint2_y = 0.5*(y_corners[1] + y_corners[2])
            midpoint3_x = 0.5*(x_corners[2] + x_corners[3])
            midpoint3_y = 0.5*(y_corners[2] + y_corners[3])
            midpoint4_x = 0.5*(x_corners[3] + x_corners[0])
            midpoint4_y = 0.5*(y_corners[3] + y_corners[0])
            if (abs(midpoint3_x - midpoint1_x) < delta_x_min and abs(midpoint4_y - midpoint2_y) < delta_x_min) or (abs(midpoint3_y - midpoint1_y) < delta_x_min and abs(midpoint4_x - midpoint2_x) < delta_x_min):
                slope1, intsec1, slope2, intsec2 = np.array([np.nan, 0.0, np.nan, 0.0])
            else:
                slope1_ = (midpoint3_y - midpoint1_y)/(midpoint3_x - midpoint1_x)
                intsec1_ = midpoint1_y - slope1_*midpoint1_x
                slope2_ = (midpoint4_y - midpoint2_y)/(midpoint4_x - midpoint2_x)
                intsec2_ = midpoint2_y - slope2_*midpoint2_x
                if slope1_ > 0:
                    slope1, intsec1, slope2, intsec2 = np.array([slope1_, intsec1_, slope2_, intsec2_])
                else:
                    slope1, intsec1, slope2, intsec2 = np.array([slope2_, intsec2_, slope1_, intsec1_])

            return x, y, np.array([slope1, intsec1, slope2, intsec2]), np.stack((x_corners, y_corners))


@njit
def run_trajectory_and_converge_sos(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling, slopes, n_sos=100, sos_tol=1e-6, max_n_t=1e7):
    """Run trajectory and converge the areas inside the surfaces of section (sos).

    Args:
        n_sos (int, optional): Batch number of sos points. Defaults to 100.
        sos_tol (_type_, optional): Tolerance for area increase inside the sos. Defaults to 1e-6.
        max_n_t (_type_, optional): Maximum number of iterations. Defaults to 1e7.

    Returns:
        x, y, px, py: Coordinates of the trajectory.
        energy: Energy of the trajectory.
        sos1, sos2: Coordinates of the surfaces of section points.
        sos1_area, sos2_area: Area inside the surfaces of section.
        S_close, S_close_corr: Total action and action correction along closed path.
    """
    
    max_n_t = round(max_n_t)

    # Compute classical trajectory and surfaces of section ; use linear intersection point with SOS
    slope1, intsec1, slope2, intsec2 = slopes
    x = List([x_ini]); y = List([y_ini]); px = List([px_ini]); py = List([py_ini]); energy = List([energy_ini])
    x0_, y0_, px0_, py0_ = np.array([x_ini, y_ini, px_ini, py_ini])
    sos1 = []; sos2 = []; sos1_area = 0.0; sos2_area = 0.0
    # sos1_x = []; sos1_y = []; sos1_x_area = 0.0; sos1_y_area = 0.0
    # sos2_x = []; sos2_y = []; sos2_x_area = 0.0; sos2_y_area = 0.0
    
    while True:
        n_sos1 = len(sos1); n_sos2 = len(sos2)
        val_sos1_area = sos1_area; val_sos2_area = sos2_area

        while min(len(sos1) - n_sos1, len(sos2) - n_sos2) < n_sos:
            x_, y_, px_, py_ = leap_frog(x0_, y0_, px0_, py0_, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling)

            if np.isnan(slope1):
                if y0_ >= intsec1 and y_ < intsec1:
                    x1 = x0_ + (intsec1 - y0_)*(x_ - x0_)/(y_ - y0_)
                    y1 = intsec1
                    px1 = px0_ + (intsec1 - y0_)*(px_ - px0_)/(y_ - y0_)
                    py1 = py0_ + (intsec1 - y0_)*(py_ - py0_)/(y_ - y0_)
                    sos1.append([x1, px1])
            else:
                if y0_ >= slope1*x0_ + intsec1 and y_ < slope1*x_ + intsec1:
                    x1 = (y0_ - x0_*(y_ - y0_)/(x_ - x0_) - intsec1)/(slope1 - (y_ - y0_)/(x_ - x0_))
                    y1 = slope1*x1 + intsec1
                    if x1 + intsec1/slope1 >= 0:
                        z1 = np.sqrt((x1 + intsec1/slope1)**2 + (slope1*x1 + intsec1)**2)
                    else:
                        z1 = -np.sqrt((x1 + intsec1/slope1)**2 + (slope1*x1 + intsec1)**2)
                    px1 = px0_ + (px_ - px0_)*(x1 - x0_)/(x_ - x0_)
                    py1 = py0_ + (py_ - py0_)*(x1 - x0_)/(x_ - x0_)
                    pz1 = (px1 + slope1*py1)/np.sqrt(1.0 + slope1**2)
                    sos1.append([z1, pz1])
                    # sos1_x.append([x1, px1]); sos1_y.append([y1, py1])
                    
            if np.isnan(slope2):
                if x0_ >= intsec2 and x_ < intsec2:
                    x2 = intsec2
                    y2 = y0_ + (intsec2 - x0_)*(y_ - y0_)/(x_ - x0_)
                    px2 = px0_ + (intsec2 - x0_)*(px_ - px0_)/(x_ - x0_)
                    py2 = py0_ + (intsec2 - x0_)*(py_ - py0_)/(x_ - x0_)
                    sos2.append([y2, py2])
            else:
                if y0_ >= slope2*x0_ + intsec2 and y_ < slope2*x_ + intsec2:
                    x2 = (y0_ - x0_*(y_ - y0_)/(x_ - x0_) - intsec2)/(slope2 - (y_ - y0_)/(x_ - x0_))
                    y2 = slope2*x2 + intsec2
                    if x2 + intsec2/slope2 >= 0:
                        z2 = np.sqrt((x2 + intsec2/slope2)**2 + (slope2*x2 + intsec2)**2)
                    else:
                        z2 = -np.sqrt((x2 + intsec2/slope2)**2 + (slope2*x2 + intsec2)**2)
                    px2 = px0_ + (px_ - px0_)*(x2 - x0_)/(x_ - x0_)
                    py2 = py0_ + (py_ - py0_)*(x2 - x0_)/(x_ - x0_)
                    pz2 = (px2 + slope2*py2)/np.sqrt(1.0 + slope2**2)
                    sos2.append([z2, pz2])
                    # sos2_x.append([x2, px2]); sos2_y.append([y2, py2])
            
            x.append(x_), y.append(y_), px.append(px_), py.append(py_)
            energy.append(compute_energy(x_, y_, px_, py_, k, omega1, omega2, D1, a1, D2, a2, system, coupling))
            
            if len(x) >= max_n_t:
                if len(sos1) < 3:
                    print('Not enough points on sos1!')
                    sos1_area = np.nan#; sos1_x_area = np.nan; sos1_y_area = np.nan
                else:
                    sos1_area, ergodicity1 = check_coherence_and_return_area(np.asarray(sos1))
                    # if len(sos1_x) > 0: sos1_x_area = polygon_area(sort_vertices(np.asarray(sos1_x)))
                    # if len(sos1_y) > 0: sos1_y_area = polygon_area(sort_vertices(np.asarray(sos1_y)))
                    
                if len(sos2) < 3:
                    print('Not enough points on sos2!')
                    sos2_area = np.nan#; sos2_x_area = np.nan; sos2_y_area = np.nan
                else:
                    sos2_area, ergodicity2 = check_coherence_and_return_area(np.asarray(sos2))
                    # if len(sos2_x) > 0: sos2_x_area = polygon_area(sort_vertices(np.asarray(sos2_x)))
                    # if len(sos2_y) > 0: sos2_y_area = polygon_area(sort_vertices(np.asarray(sos2_y)))
                    
                S_close, S_close_corr = trajectory_close_method(x, y, px, py, sos1, sos1_area, slope1, intsec1)
                return np.asarray(x), np.asarray(y), np.asarray(px), np.asarray(py), np.asarray(energy), sos1, sos2, sos1_area, sos2_area, S_close, S_close_corr#, sos1_x_area, sos1_y_area, sos2_x_area, sos2_y_area
            
            else:
                x0_, y0_, px0_, py0_ = np.array([x_, y_, px_, py_])
        
        # print(energy_ini)
        # # D = 30; a = 0.08; omega = a*np.sqrt(2*D)
        # # x_plot = np.linspace(np.min(x), np.max(x), 100)
        # # X, Y = np.meshgrid(x_plot, x_plot)
        # # Z = V1(X) + V2(Y)
        # # CS = plt.contour(X, Y, Z)
        # # plt.clabel(CS, inline=True, fontsize=10)
        # plt.plot(x, y, lw=0.5)
        # plt.show()

        sos1_area = polygon_area(sort_vertices(np.asarray(sos1)))
        sos2_area = polygon_area(sort_vertices(np.asarray(sos2)))
        
        if abs(val_sos1_area - sos1_area) < sos_tol*val_sos1_area and abs(val_sos2_area - sos2_area) < sos_tol*val_sos2_area:
            sos1_area, ergodicity1 = check_coherence_and_return_area(np.asarray(sos1))
            sos2_area, ergodicity2 = check_coherence_and_return_area(np.asarray(sos2))
            # if len(sos1_x) > 0: sos1_x_area = polygon_area(sort_vertices(np.asarray(sos1_x)))
            # if len(sos1_y) > 0: sos1_y_area = polygon_area(sort_vertices(np.asarray(sos1_y)))
            # if len(sos2_x) > 0: sos2_x_area = polygon_area(sort_vertices(np.asarray(sos2_x)))
            # if len(sos2_y) > 0: sos2_y_area = polygon_area(sort_vertices(np.asarray(sos2_y)))
            S_close, S_close_corr = trajectory_close_method(x, y, px, py, sos1, sos1_area, slope1, intsec1)
            return np.asarray(x), np.asarray(y), np.asarray(px), np.asarray(py), np.asarray(energy), sos1, sos2, sos1_area, sos2_area, S_close, S_close_corr#, sos1_x_area, sos1_y_area, sos2_x_area, sos2_y_area


@njit
def run_trajectory_and_get_sos(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling, slopes, max_n_t=1e7):
    """Compute surface of section points.

    Args:
        max_n_t (_type_, optional): Maximum number of iterations. Defaults to 1e7.

    Returns:
        np.ndarray: surface of section points
    """
    
    max_n_t = round(max_n_t)

    # Compute classical trajectory and surfaces of section ; use linear intersection point with SOS
    slope1, intsec1, slope2, intsec2 = slopes
    x = List([x_ini]); y = List([y_ini]); px = List([px_ini]); py = List([py_ini]); energy = List([energy_ini])
    x0_, y0_, px0_, py0_ = np.array([x_ini, y_ini, px_ini, py_ini])
    sos1 = []
    
    for i in range(max_n_t):
        x_, y_, px_, py_ = leap_frog(x0_, y0_, px0_, py0_, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling)

        if np.isnan(slope1):
            if y0_ >= intsec1 and y_ < intsec1:
                x1 = x0_ + (intsec1 - y0_)*(x_ - x0_)/(y_ - y0_)
                y1 = intsec1
                px1 = px0_ + (intsec1 - y0_)*(px_ - px0_)/(y_ - y0_)
                py1 = py0_ + (intsec1 - y0_)*(py_ - py0_)/(y_ - y0_)
                sos1.append((x1, px1))
        else:
            if y0_ >= slope1*x0_ + intsec1 and y_ < slope1*x_ + intsec1:
                x1 = (y0_ - x0_*(y_ - y0_)/(x_ - x0_) - intsec1)/(slope1 - (y_ - y0_)/(x_ - x0_))
                y1 = slope1*x1 + intsec1
                if x1 + intsec1/slope1 >= 0:
                    z1 = np.sqrt((x1 + intsec1/slope1)**2 + (slope1*x1 + intsec1)**2)
                else:
                    z1 = -np.sqrt((x1 + intsec1/slope1)**2 + (slope1*x1 + intsec1)**2)
                px1 = px0_ + (px_ - px0_)*(x1 - x0_)/(x_ - x0_)
                py1 = py0_ + (py_ - py0_)*(x1 - x0_)/(x_ - x0_)
                pz1 = (px1 + slope1*py1)/np.sqrt(1.0 + slope1**2)
                sos1.append((z1, pz1))
                # sos1_x.append((x1, px1)); sos1_y.append((y1, py1))
                
        x.append(x_), y.append(y_), px.append(px_), py.append(py_)
        energy.append(compute_energy(x_, y_, px_, py_, k, omega1, omega2, D1, a1, D2, a2, system, coupling))
        
        x0_, y0_, px0_, py0_ = np.array([x_, y_, px_, py_])
    
    return np.asarray(x), np.asarray(y), np.asarray(px), np.asarray(py), np.asarray(energy), sos1

