from dataclasses import dataclass

import torch

from internal.optimizers import OptimizerConfig


@dataclass
class SelectiveAdam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs):
        from diff_gaussian_rasterization import SparseGaussianAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SparseGaussianAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                num_hit_pixels = outputs.get("num_hit_pixels", None)
                if num_hit_pixels is None:
                    return

                self.visibility = (num_hit_pixels > 0).contiguous()

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility, self.visibility.shape[0])

                return loss

        return Adapter(params, lr, *args, **kwargs)


@dataclass
class SelectiveOccupancyAdam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs):
        from diff_gaussian_rasterization import SparseGaussianAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SparseGaussianAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                delaunay_ids = getattr(gaussian_model, "get_delaunay_gaussian_ids", None)
                if delaunay_ids is None:
                    return

                self.visibility = outputs["visibility_filter"][delaunay_ids].contiguous()

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility, self.visibility.shape[0])

                return loss

        return Adapter(params, lr, *args, **kwargs)
