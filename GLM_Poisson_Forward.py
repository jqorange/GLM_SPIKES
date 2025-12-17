# -*- coding: utf-8 -*-
"""
Batch GLM forward selection entry point (Poisson, 50 Hz).

The implementation is split across the ``glm_poisson_forward`` package:
- ``config``: 全局配置 (paths、超参数等)
- ``io_utils``: IO 与数据预处理
- ``forward_search``: forward search 流程（单 session）
- ``metrics`` / ``plotting_utils`` / ``training``: 其他函数（llhi、Wilcoxon、画图等）
"""

from glm_poisson_forward.pipeline import main


if __name__ == "__main__":
    main()

