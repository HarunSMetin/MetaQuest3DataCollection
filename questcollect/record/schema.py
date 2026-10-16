"""Raw episode layout (one HDF5 file per episode). Keys are HDF5 dataset paths, time is axis 0.

meta (file attrs): schema_version, task, instruction, operator, input_mode, success, fps, seed,
                   created_at, duration_s, num_steps, model_xml, config_yaml, camera_names, objects

/time                         (T,)        sim time [s]
/wall_time                    (T,)        host monotonic time [s]
/obs/joint_pos                (T, 14)     [left 7, right 7] rad
/obs/joint_vel                (T, 14)
/obs/gripper_width            (T, 2)      finger opening [m] (0 .. 0.08)
/obs/ee_pose_left             (T, 7)      tcp [x y z qw qx qy qz] world
/obs/ee_pose_right            (T, 7)
/obs/objects                  (T, k, 7)   task object poses (privileged, sim only)
/action/joint_pos             (T, 16)     [7 L joints, grip L, 7 R joints, grip R]; grip in [0 closed, 1 open]
/action/ee_pose               (T, 16)     [L xyz qwxyz, R xyz qwxyz, grip L, grip R] IK targets
/action/ctrl                  (T, nu)     raw actuator command
/sim/state                    (T, S)      mj_getState(mjSTATE_INTEGRATION) before the step
/sim/initial_state            (S,)
/success                      (T,)        task success predicate per step
/teleop/head                  (T, 7)      XR head pose [x y z qx qy qz qw]
/teleop/{left,right}/source   (T,)        0 none, 1 controller, 2 hand
/teleop/{left,right}/pose     (T, 7)      raw XR control pose
/teleop/{left,right}/buttons  (T, 8)
/teleop/{left,right}/joints   (T, 25, 7)  WebXR hand joints (zeros for controllers)
/teleop/{left,right}/pinch    (T, 2)
/teleop/engaged               (T, 2)
/teleop/tracked               (T, 2)
/teleop/rtt_ms                (T,)
/images/<camera>              (T, H, W, 3) uint8   only if record.live_images
"""

SCHEMA_VERSION = 1

ACTION_NAMES = (
    [f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"]
)
