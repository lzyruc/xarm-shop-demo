"""All translations/depths are mm; xArm Euler angles are degrees, Rz Ry Rx."""
import math
import numpy as np


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return a


def transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Calibration must be a finite 4x4 matrix")
    r = t[:3, :3]
    if not (np.allclose(t[3], [0, 0, 0, 1]) and
            np.allclose(r.T @ r, np.eye(3), atol=1e-5) and
            np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError("Invalid rigid transform")
    return t


def pose_matrix(pose):
    x, y, z, roll, pitch, yaw = vector(pose, 6, "TCP pose")
    r, p, w = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cw, sw = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(w), math.sin(w)
    t = np.eye(4)
    t[:3, :3] = [[cw*cp, cw*sp*sr-sw*cr, cw*sp*cr+sw*sr],
                 [sw*cp, sw*sp*sr+cw*cr, sw*sp*cr-cw*sr],
                 [-sp, cp*sr, cp*cr]]
    t[:3, 3] = [x, y, z]
    return t


def depth_at(depth, uv, radius=3, min_mm=100, max_mm=2000, max_spread_mm=20):
    u, v = (int(round(x)) for x in vector(uv, 2, "pixel"))
    h, w = depth.shape
    if not (radius <= u < w-radius and radius <= v < h-radius):
        raise ValueError("Target pixel too close to image border")
    patch = depth[v-radius:v+radius+1, u-radius:u+radius+1]
    values = patch[np.isfinite(patch) & (patch >= min_mm) & (patch <= max_mm)]
    if values.size < max(5, math.ceil(patch.size * 0.6)):
        raise ValueError("Insufficient valid depth near target")
    if np.percentile(values, 90) - np.percentile(values, 10) > max_spread_mm:
        raise ValueError("Unstable depth / object edge near target")
    return float(np.median(values))


def deproject(uv, depth_mm, intrinsics, distortion=None):
    fx, fy, cx, cy = vector(intrinsics, 4, "intrinsics")
    if fx <= 0 or fy <= 0 or not np.isfinite(depth_mm) or depth_mm <= 0:
        raise ValueError("Invalid focal length or depth")
    u, v = vector(uv, 2, "pixel")
    if distortion is not None and np.any(distortion):
        import cv2
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        ray = cv2.undistortPoints(np.array([[[u, v]]]), k, np.asarray(distortion, float))[0, 0]
    else:
        ray = np.array([(u-cx)/fx, (v-cy)/fy])
    return np.array([ray[0]*depth_mm, ray[1]*depth_mm, depth_mm])


def base_point(camera_point, tcp_pose, tcp_offset, flange_from_camera):
    # get_position returns configured TCP, while hand-eye calibration uses flange.
    base_from_flange = pose_matrix(tcp_pose) @ np.linalg.inv(pose_matrix(tcp_offset))
    return (base_from_flange @ transform(flange_from_camera) @
            np.r_[vector(camera_point, 3, "camera point"), 1])[:3]


def check_workspace(pose, bounds):
    xyz = vector(pose, 6, "pose")[:3]
    b = np.asarray(bounds, float)
    if b.shape != (3, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("workspace_mm must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]]")
    if np.any(xyz < b[:, 0]) or np.any(xyz > b[:, 1]):
        raise ValueError(f"TCP outside workspace: {xyz.tolist()}")


def make_tool_axis_plan(point_base, tcp_pose, grasp_depth_offset_mm, motion, bounds):
    """Plan fixed-orientation tool-axis translations using configured axis roles."""
    point = vector(point_base, 3, "target point")
    pose = vector(tcp_pose, 6, "TCP pose")
    offset = float(grasp_depth_offset_mm)
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("grasp_depth_offset_mm must be a non-negative measured value")

    axis_index = {"x": 0, "y": 1, "z": 2}
    roles = [motion["lateral_axis"], motion["vertical_axis"], motion["depth_axis"]]
    if set(roles) != set(axis_index):
        raise ValueError("lateral_axis, vertical_axis and depth_axis must use x/y/z exactly once")
    lateral_i, vertical_i, depth_i = (axis_index[name] for name in roles)

    rotation = pose_matrix(pose)[:3, :3]
    raw = rotation.T @ (point - pose[:3])
    lateral = float(raw[lateral_i])
    vertical = float(raw[vertical_i])
    depth = float(raw[depth_i])
    if abs(lateral) > motion["max_alignment_mm"] or abs(vertical) > motion["max_alignment_mm"]:
        raise ValueError(f"Required lateral/vertical alignment exceeds limit: {[lateral, vertical]}")
    if not motion["min_approach_mm"] <= abs(depth) <= motion["max_approach_mm"]:
        raise ValueError(f"Required depth approach outside limits: {depth:.3f} mm")
    approach = depth + math.copysign(offset, depth)
    deadband = motion["axis_deadband_mm"]
    if abs(lateral) <= deadband:
        lateral = 0.0
    if abs(vertical) <= deadband:
        vertical = 0.0

    up_component = float(rotation[:, vertical_i] @ np.array([0.0, 0.0, 1.0]))
    if abs(up_component) < motion["vertical_axis_min_up_component"]:
        raise ValueError(f"Tool {roles[1].upper()} axis is not sufficiently vertical at the observation pose")
    lift = math.copysign(float(motion["lift_mm"]), up_component)

    def axis_delta(index, distance):
        delta = np.zeros(3)
        delta[index] = distance
        return delta

    commands = [
        ("align_lateral", roles[0], axis_delta(lateral_i, lateral)),
        ("align_vertical", roles[1], axis_delta(vertical_i, vertical)),
        ("approach_depth", roles[2], axis_delta(depth_i, approach)),
        ("lift", roles[1], axis_delta(vertical_i, lift)),
        ("pullout", roles[2], axis_delta(depth_i, -approach)),
    ]
    xyz = pose[:3].copy()
    steps = []
    for name, axis, delta in commands:
        xyz = xyz + rotation @ delta
        target_pose = np.r_[xyz, pose[3:]].tolist()
        check_workspace(target_pose, bounds)
        steps.append({"name": name, "axis": axis, "delta_tool_mm": delta.tolist(),
                      "target_pose": target_pose})
    desired = raw.copy()
    desired[depth_i] = approach
    return {"axis_roles": {"lateral": roles[0], "vertical": roles[1], "depth": roles[2]},
            "target_delta_tool_mm": desired.tolist(), "steps": steps}
