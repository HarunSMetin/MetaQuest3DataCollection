"""Quest Touch controller -> clutch / gripper / commands."""

from __future__ import annotations

from questcollect.server.protocol import (
    B_PRIMARY, B_SECONDARY, B_SQUEEZE, B_STICK_PRESS, B_TRIGGER, HandInput,
)

# Rising-edge button -> command, per arm. Left "Y" (secondary) toggles the wrist menu client-side.
COMMANDS = {
    ("r", B_PRIMARY): "record_toggle",   # A
    ("r", B_SECONDARY): "discard",       # B
    ("l", B_PRIMARY): "save",            # X
    ("l", B_STICK_PRESS): "recenter",    # left stick click
}


class ControllerMapper:
    ENGAGE_ON, ENGAGE_OFF = 0.6, 0.4

    def __init__(self, arm: str):
        self.arm = arm
        self.clutch_down = False
        self.prev_buttons: dict[int, bool] = {}

    def reset(self) -> None:
        self.clutch_down = False

    def update(self, h: HandInput) -> tuple[bool, float, list[str]]:
        """Returns (clutch held, gripper open fraction, commands)."""
        sq = h.buttons[B_SQUEEZE]
        if self.clutch_down:
            self.clutch_down = sq > self.ENGAGE_OFF
        else:
            self.clutch_down = sq > self.ENGAGE_ON
        gripper = 1.0 - float(h.buttons[B_TRIGGER])
        commands = []
        for (arm, b), cmd in COMMANDS.items():
            if arm != self.arm:
                continue
            down = h.buttons[b] > 0.5
            if down and not self.prev_buttons.get(b, False):
                commands.append(cmd)
            self.prev_buttons[b] = down
        return self.clutch_down, gripper, commands
