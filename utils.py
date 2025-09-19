import torch
import numpy as np

import torch

def subdivide_by_step(route, step=2.0):
    route = route.clone()
    while True:
        new_route = [route[0]]
        needs_more_split = False

        for i in range(1, len(route)):
            p1 = route[i - 1]
            p2 = route[i]
            dist = torch.norm(p2 - p1)
            if dist > step:
                mid = 0.5 * (p1 + p2)
                new_route.append(mid)
                needs_more_split = True
            new_route.append(p2)

        route = torch.stack(new_route)
        if not needs_more_split:
            break
    return route

def smooth_route(in_route, iterations=1):
    route = in_route.clone()
    for _ in range(iterations):
        smoothed = [route[0]]  # 保留第一點
        for i in range(1, len(route) - 1):
            avg = 0.5 * (route[i - 1] + route[i + 1])
            smoothed.append(avg)
        smoothed.append(route[-1])  # 保留最後一點
        route = torch.stack(smoothed)
    return route

def subdivide_route(route, step=4.0, smooth_iters=4):
    smoothed_route = route.clone()
    for idx in range(smooth_iters):
        curr_step = step * (smooth_iters - idx)
        divided_route = subdivide_by_step(smoothed_route, step=curr_step)
        smoothed_route = smooth_route(divided_route, iterations=1)
    return smoothed_route


def map_en_route_to_veh_route(en_route, en_pose):
    # 1. Translation
    origin = torch.stack([en_pose.position_x, en_pose.position_y])  # (2,)
    shifted = en_route - origin                                     # (N,2)

    # 2. Rotation (counter-clockwise) 
    theta = en_pose.heading                 # 0-D tensor
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    R = torch.stack([
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta,  cos_theta])
        ])                                   # (2,2) tensor

    veh_route = shifted @ R.T               # (N,2)
    return veh_route

def compute_walk_point(route: torch.Tensor, dist: float):
    """
    Walk a specified distance along the route and return the interpolated point
    and the index of the segment where the point lies.

    Parameters:
        route: torch.Tensor of shape (N, 2)
            Sequence of 2D points representing the path.
        dist: float
            Distance to walk from the beginning of the route.

    Returns:
        point: torch.Tensor of shape (2,)
            The interpolated position after walking the specified distance.
        idx: int
            The index of the segment where the point lies (i.e., between route[idx] and route[idx+1]).
    """
    traveled = 0.0
    for idx in range(len(route) - 1):
        p1 = route[idx]
        p2 = route[idx + 1]
        segment_vector = p2 - p1
        segment_length = torch.norm(segment_vector)

        if traveled + segment_length >= dist:
            ratio = (dist - traveled) / segment_length
            point = p1 + ratio * segment_vector
            return point, idx

        traveled += segment_length

    # If the specified distance exceeds the total route length, return the last point
    return route[-1], len(route) - 1

def clip_en_route(en_route, gps, prev_index, subroute_length):
    matching_index, _ = match_route(en_route, prev_index, gps)

    # Output a subroute = en_route[matching_idx:], where matching_idx is the first point index of the matching segment 
    if matching_index == en_route.shape[0] - 1: # Exceed the route, return empty
        return torch.empty((0, en_route.shape[1])), matching_index
    
    # Clip route by subroute_length
    remain_route = en_route[matching_index:].clone()
    end_point, end_index = compute_walk_point(remain_route, subroute_length)

    # Clip subroute by end_point
    subroute = None
    if end_index < remain_route.shape[0]-1:
        subroute = torch.vstack([remain_route[:end_index+1], end_point.unsqueeze(0)])
    else:
        subroute = remain_route
    return subroute, matching_index


def project_point_to_segment(point, segment):
    A, B = segment
    AB = B - A
    if torch.allclose(AB, torch.zeros_like(AB)):
        return A, torch.norm(point - A)
    t = torch.dot(point - A, AB) / torch.dot(AB, AB)
    t = torch.clamp(t, 0, 1)
    proj = A + t * AB
    dist = torch.norm(point - proj)
    return proj, dist

def match_route(en_route, prev_index, gps, search_range=40):
    # Match current gps into EN route
    segments = torch.stack([en_route[prev_index:-1], en_route[prev_index+1:]], dim=1)
    segments.requires_grad = False

    if segments.shape[0] == 0:
        return prev_index, en_route[prev_index]
    
    matching_index = None
    min_dist = search_range
    for i in range(len(segments)):
        mid_curr = (segments[i][0] + segments[i][1]) / 2.0
        dist_curr = torch.norm(gps - mid_curr)
        if dist_curr < min_dist:
            matching_index = i
            min_dist = dist_curr


    if matching_index is None: # Distances to all segments are exceed the search range: GPS are not already on the route
        matching_index = segments.shape[0] # GPS exceed the route

    proj_point = None
    if matching_index >= 0:
        proj_idx = matching_index if matching_index < segments.shape[0] else segments.shape[0]-1 # Choose last segment if exceed the route
        proj_point, proj_dist = project_point_to_segment(gps, segments[proj_idx])

    # Add starting index (prev_index) back.
    route_point_index = matching_index + prev_index
    return route_point_index, proj_point

