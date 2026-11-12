"""Test setup shared by all test files."""

import os
import sys

# Offscreen rendering without a window, like the CLI (cli._headless_gl). Without it MuJoCo picks GLFW, which aborts
# the interpreter in a shell that has no usable display (seen with tests/test_learn.py: "Fatal Python error").
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
