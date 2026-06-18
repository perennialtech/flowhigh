from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]


class ODESampler:
    def __init__(
        self,
        *,
        method: str = "midpoint",
        backend: str = "fixed",
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.method = method
        self.backend = backend
        self.atol = atol
        self.rtol = rtol

    def sample(self, y0: Tensor, steps: int, fn: VelocityFn) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be greater than 0")

        if self.backend == "fixed":
            return self._fixed_step(y0, steps, fn)

        if self.backend == "torchdiffeq":
            return self._torchdiffeq(y0, steps, fn)

        if self.backend == "torchode":
            return self._torchode(y0, steps, fn)

        raise ValueError(f"Unsupported ODE backend: {self.backend}")

    def _fixed_step(self, y0: Tensor, steps: int, fn: VelocityFn) -> Tensor:
        y = y0
        dt = 1.0 / steps

        for step in range(steps):
            t0 = y.new_tensor(step * dt)

            if self.method == "euler":
                y = y + dt * fn(t0, y)
                continue

            if self.method == "midpoint":
                k1 = fn(t0, y)
                tm = y.new_tensor((step + 0.5) * dt)
                y = y + dt * fn(tm, y + 0.5 * dt * k1)
                continue

            raise ValueError(f"Unsupported fixed-step ODE method: {self.method}")

        return y

    def _torchdiffeq(self, y0: Tensor, steps: int, fn: VelocityFn) -> Tensor:
        from torchdiffeq import odeint

        t = torch.linspace(0, 1, steps + 1, device=y0.device, dtype=y0.dtype)
        trajectory = odeint(
            fn,
            y0,
            t,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
        )
        return trajectory[-1]

    def _torchode(self, y0: Tensor, steps: int, fn: VelocityFn) -> Tensor:
        import torchode as to
        from einops import pack, rearrange, repeat, unpack

        batch = y0.shape[0]
        t_eval = torch.linspace(0, 1, steps + 1, device=y0.device, dtype=y0.dtype)
        t_eval = repeat(t_eval, "n -> b n", b=batch)

        y0_flat, packed_shape = pack([y0], "b *")

        def flat_fn(t: Tensor, y_flat: Tensor) -> Tensor:
            y = unpack(y_flat, packed_shape, "b *")[0]
            return rearrange(fn(t, y), "b ... -> b (...)")

        term = to.ODETerm(flat_fn)
        step_method = to.Tsit5(term=term)
        controller = to.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
        solver = to.AutoDiffAdjoint(step_method, controller)
        problem = to.InitialValueProblem(y0=y0_flat, t_eval=t_eval)

        sol = solver.solve(problem)
        return unpack(sol.ys[:, -1], packed_shape, "b *")[0]
