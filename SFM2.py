import math

import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_rot, corresponding_p_ind = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize_single_vector(pt, focal, pp):
    return [(pt[0] - pp[0]) / focal, (pt[1] - pp[1]) / focal, 1]


def unnormalize_single_vector(pt, focal, pp):
    return [pt[0] * focal + pp[0], pt[1] * focal + pp[1], 1]


def rotate_single_point(pt, R):
    v = R.dot(pt)
    return np.array([v[0] / v[2], v[1] / v[2], 1])


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array([normalize_single_vector(pt, focal, pp) for pt in pts])


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array([unnormalize_single_vector(pt, focal, pp) for pt in pts])



def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, 3]
    foe = (T[0] / T[2], T[1] / T[2])
    return R, foe, T[2]


def rotate(pts, R):
    # rotate the points - pts using R
    return np.array([rotate_single_point(pt, R) for pt in pts])


def distance(m, b, p):
    return abs((m * p[0] - p[1] + b) / math.sqrt(m ** 2 + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    # m = (foe[1] - p[1]) / (foe[0] - p[0])
    # n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])
    # a = np.array([1, m + n])
    # b = np.array([0, n])
    # p, idx = min([(np.cross(b - a, pt[:2] - a) / np.linalg.norm(b - a), i) for i, pt in enumerate(norm_pts_rot)])
    # return norm_pts_rot[idx], idx

    # def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    b = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    sol_min = (0, norm_pts_rot[0])
    min_dist = distance(m, b, norm_pts_rot[0])
    for index, point in enumerate(norm_pts_rot[1:]):
        cur_dist = distance(m, b, point)
        if cur_dist < min_dist:
            sol_min = (index + 1, point)
            min_dist = cur_dist
    return sol_min[1], sol_min[0]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    zx = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    zy = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])
    return (zx + zy) / 2
