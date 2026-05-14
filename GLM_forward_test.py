# -*- coding: utf-8 -*-
"""
Batch GLM forward selection entry point (Poisson).
"""

from pathlib import Path

from glm_poisson_forward.pipeline import main


if __name__ == "__main__":
    main(weights_base=Path("weights_Poisson_forward"))
