import math

import pytest

from conftest import CAM
from worldlabs_atlas import ArgsError
from worldlabs_atlas import camera as cam


def approx(a, b, tol=1e-5):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def base():
    return cam.normalize_camera(CAM)


def test_quaternion_roundtrip():
    R = cam.mm(cam.rot_y(33), cam.rot_x(-12))
    R2 = cam.quat_to_mat(cam.mat_to_quat(R))
    assert all(approx(R[i], R2[i]) for i in range(3))


def test_look_at_looks_down_minus_z_and_up_is_y():
    R = cam.look_at_matrix([0, 0, 0], [0, 0, -5])
    assert approx(cam.mv(R, [0, 0, -1]), [0, 0, -1])
    assert approx(cam.mv(R, [0, 1, 0]), [0, 1, 0])
    R = cam.look_at_matrix([0, 0, 0], [5, 0, 0])
    assert approx(cam.mv(R, [0, 0, -1]), [1, 0, 0])


def test_look_at_straight_down_is_not_degenerate():
    R = cam.look_at_matrix([0, 5, 0], [0, 0, 0])
    assert approx(cam.mv(R, [0, 0, -1]), [0, -1, 0])


def test_relative_move_directions():
    R, t = cam.relative_pose({"forward": 2, "right": 1, "up": 0.5})
    assert t == [1.0, 0.5, -2.0]
    R, t = cam.relative_pose({"yaw": 90})               # turns left: looks along -X
    assert approx(cam.mv(R, [0, 0, -1]), [-1, 0, 0])
    R, t = cam.relative_pose({"pitch": 90})             # looks up
    assert approx(cam.mv(R, [0, 0, -1]), [0, 1, 0])


def test_mesh_pose_api_roundtrip():
    b = base()
    K = cam.grid_intrinsics(b)
    R, t = cam.look_at_matrix([1, 0.2, 0.5], [0, 0, -3]), [1, 0.2, 0.5]
    api = cam.mesh_pose_to_camera(b, K, R, t)
    R2, t2 = cam.camera_to_mesh_pose(b, api)
    assert approx(t, t2, 1e-4) and all(approx(R[i], R2[i], 1e-4) for i in range(3))


def test_base_offset_is_applied():
    # the first view sits at y=1: a mesh-frame origin camera maps to world position y=1
    api = cam.mesh_pose_to_camera(base(), cam.grid_intrinsics(base()), cam.relative_pose({})[0], [0, 0, 0])
    assert api["extrinsics"]["position"] == [0.0, 1.0, 0.0]
    assert api["intrinsics"]["width"] == 1280 and api["intrinsics"]["height"] == 720


def test_fov_sets_focal_length_and_is_clamped():
    api, _pose, used = cam.resolve_camera(base(), {"position": [0, 0, 0], "look_at": [0, 0, -1]}, 60)
    assert used == 60
    assert api["intrinsics"]["fy"] == pytest.approx(360 / math.tan(math.radians(30)))
    _a, _p, used = cam.resolve_camera(base(), {"position": [0, 0, 0], "look_at": [0, 0, -1], "fov": 500})
    assert used == cam.MAX_FOV
    _a, _p, used = cam.resolve_camera(base(), {"forward": 1})
    assert 20 < used < 80                                 # taken from the first picture's lens


def test_matrix_spec_row_major_and_flat():
    m = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    R, t = cam.mesh_pose_of_spec({"matrix": m})
    assert t == [1, 2, 3]
    R2, t2 = cam.mesh_pose_of_spec({"matrix": [v for r in m for v in r]})
    assert t2 == t


@pytest.mark.parametrize("spec", [{}, None, {"position": [0, 0, 0]}, {"position": [1, 1, 1], "look_at": [1, 1, 1]},
                                  {"matrix": [1, 2, 3]}])
def test_bad_specs_raise(spec):
    with pytest.raises(ArgsError):
        cam.mesh_pose_of_spec(spec)


def test_orbit_geometry():
    c = cam.orbit_camera([0, 0, -3], 2, 0, 0)
    assert c["position"] == [0, 0, -1] and c["look_at"] == [0, 0, -3]
    c = cam.orbit_camera([0, 0, -3], 2, 90, 0)
    assert approx(c["position"], [2, 0, -3])
    c = cam.orbit_camera([0, 0, -3], 2, 0, 90)
    assert approx(c["position"], [0, 2, -3])


def test_orbit_sweep_endpoints_and_count():
    s = cam.orbit_sweep([0, 0, 0], 1, -30, 30, 5)
    assert len(s) == 5
    assert approx(s[0]["position"], cam.orbit_camera([0, 0, 0], 1, -30)["position"])
    assert approx(s[-1]["position"], cam.orbit_camera([0, 0, 0], 1, 30)["position"])
    assert len(cam.orbit_sweep([0, 0, 0], 1, 0, 10, 1)) == 1


def test_spec_hash_view_rounds_noise():
    assert cam.spec_hash_view({"position": [1.0000001, 2, 3]}) == cam.spec_hash_view({"position": [1.0, 2, 3]})
