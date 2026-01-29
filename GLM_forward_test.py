# -*- coding: utf-8 -*-
"""
Batch GLM forward selection entry point (Poisson, residual speed test).

Uses residual speed (s_res = s - E[s|pos]) in place of speed while generating
the same outputs as the standard GLM_Poisson_Forward pipeline.
"""

from pathlib import Path

from glm_poisson_forward.pipeline import main


if __name__ == "__main__":
    main(use_residual_speed=True, weights_base=Path("weights_Poisson_forward_residual"))
