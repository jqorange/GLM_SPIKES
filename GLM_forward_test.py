# -*- coding: utf-8 -*-
"""
Batch GLM forward selection entry point (Poisson, residual feature test).

Uses residualized kinematic features (x_res = x - E[x|pos]) in place of
speed/roll/yaw/pitch while generating the same outputs as the standard
GLM_Poisson_Forward pipeline.
"""

from pathlib import Path

from glm_poisson_forward.pipeline import main


if __name__ == "__main__":
    main(
        use_residual_speed=True,
        use_residual_roll=True,
        use_residual_yaw=True,
        use_residual_pitch=True,
        weights_base=Path("weights_Poisson_forward_residual"),
    )
