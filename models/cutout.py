import random
from typing import Optional, Union

import numpy as np
import torch


class Cutout:

    def __init__(
        self,
        min_n_holes: int = 2,
        max_n_holes: int = 8,
        max_hole_ratio: float = 0.1,
        fill_value: Optional[Union[int, float]] = None,
        p: float = 0.5,
    ):
        self.min_n_holes = min_n_holes
        self.max_n_holes = max_n_holes
        self.max_hole_ratio = max_hole_ratio
        self.fill_value = fill_value
        self.p = p

    def __call__(self, xs: torch.Tensor) -> torch.Tensor:
        xs = xs.clone()
        b, _, h, w = xs.shape

        for i in range(b):
            if random.random() > self.p:
                n_holes = random.randint(self.min_n_holes, self.max_n_holes)
                for _ in range(n_holes):
                    y = random.randint(0, h)
                    x = random.randint(0, w)
                    h_size = random.randint(2, int(h * self.max_hole_ratio) + 1)
                    w_size = random.randint(2, int(w * self.max_hole_ratio) + 1)
                    y1 = np.clip(y - h_size // 2, 0, h)
                    y2 = np.clip(y1 + h_size, 0, h)
                    x1 = np.clip(x - w_size // 2, 0, w)
                    x2 = np.clip(x1 + w_size, 0, w)
                    fill_value = self.fill_value if self.fill_value is not None else random.randint(0, 255)
                    xs[i, :, y1:y2, x1:x2] = fill_value
        return xs
