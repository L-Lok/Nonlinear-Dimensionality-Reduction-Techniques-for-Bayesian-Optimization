from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch


def tensor_to_list(value: torch.Tensor | None) -> list[Any] | None:
    if value is None:
        return None
    return value.detach().cpu().tolist()


@dataclass
class BoxTransform:
    """Affine map between a fixed box and the global unit box.

    The transform owns one immutable full-domain box.  BO-VAE uses this for
    ambient VAE inputs and for latent SDR coordinates so no component silently
    re-normalizes per iteration, per trust region, or per retraining stage.
    """

    bounds: torch.Tensor
    name: str = "box"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        bounds = torch.as_tensor(self.bounds)
        if not bounds.is_floating_point():
            bounds = bounds.to(dtype=torch.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"{self.name} bounds must have shape d x 2")
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        if torch.any(~torch.isfinite(bounds)):
            raise ValueError(f"{self.name} bounds contain non-finite values")
        if torch.any(upper <= lower + self.eps):
            bad = torch.where(upper <= lower + self.eps)[0].detach().cpu().tolist()
            raise ValueError(f"{self.name} bounds have zero-width dimensions: {bad}")
        self.bounds = bounds

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def lower(self) -> torch.Tensor:
        return self.bounds[:, 0]

    @property
    def upper(self) -> torch.Tensor:
        return self.bounds[:, 1]

    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BoxTransform":
        bounds = self.bounds
        if device is not None or dtype is not None:
            bounds = bounds.to(device=device, dtype=dtype)
        return BoxTransform(bounds=bounds, name=self.name, eps=self.eps)

    def to_unit(self, x: torch.Tensor, clip: bool = False) -> torch.Tensor:
        x = x.to(self.bounds)
        unit = (x - self.lower) / self.width
        if clip:
            unit = unit.clamp(0.0, 1.0)
        return unit

    def from_unit(self, unit_x: torch.Tensor, clip: bool = False) -> torch.Tensor:
        unit_x = unit_x.to(self.bounds)
        if clip:
            unit_x = unit_x.clamp(0.0, 1.0)
        return self.lower + unit_x * self.width

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.bounds)
        return torch.minimum(torch.maximum(x, self.lower), self.upper)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.bounds)
        low = torch.clamp(self.lower - x, min=0.0)
        high = torch.clamp(x - self.upper, min=0.0)
        return torch.maximum(low, high)

    def round_trip_error(self, x: torch.Tensor) -> float:
        x = x.to(self.bounds)
        restored = self.from_unit(self.to_unit(x))
        return float(torch.max(torch.abs(restored - x)).item())

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "bounds": tensor_to_list(self.bounds),
            "eps": self.eps,
        }


class ObjectiveTransform(BoxTransform):
    """Named objective-domain box transform."""


class VAEInputTransform(BoxTransform):
    """Named VAE-input-domain box transform."""


def unit_bounds(dim: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros(dim, dtype=dtype, device=device),
            torch.ones(dim, dtype=dtype, device=device),
        ],
        dim=1,
    )


def assert_unit_box(bounds: torch.Tensor, *, name: str = "bounds", tol: float = 1e-10) -> None:
    bounds = torch.as_tensor(bounds)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"{name} must have shape d x 2")
    if torch.any(bounds[:, 0] < -tol) or torch.any(bounds[:, 1] > 1.0 + tol):
        raise ValueError(f"{name} must live in the global unit box")
    if torch.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"{name} has invalid lower/upper ordering")


def assert_finite_tensor(x: torch.Tensor, *, name: str = "tensor") -> None:
    if torch.any(~torch.isfinite(torch.as_tensor(x))):
        raise ValueError(f"{name} contains non-finite values")


def assert_unit_tensor(x: torch.Tensor, *, name: str = "unit tensor", tol: float = 1e-10) -> None:
    x = torch.as_tensor(x)
    assert_finite_tensor(x, name=name)
    if torch.any(x < -tol) or torch.any(x > 1.0 + tol):
        raise ValueError(f"{name} must live in the global unit box")


def map_between_boxes(x: torch.Tensor, source: BoxTransform, target: BoxTransform, *, clip_source_unit: bool = False) -> torch.Tensor:
    if source.dim != target.dim:
        raise ValueError("source and target boxes must have the same dimension")
    return target.from_unit(source.to_unit(x, clip=clip_source_unit))


class LatentTransform:
    """Versioned transform between decoder latent z, optional calibrated w, and unit u."""

    version = "bovae_latent_transform_v1"
    kind = "base"

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def fit(self, z_anchor: torch.Tensor) -> "LatentTransform":
        raise NotImplementedError

    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def w_to_z(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def w_to_u(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def u_to_w(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def z_to_u(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_to_u(self.z_to_w(z))

    def u_to_z(self, u: torch.Tensor) -> torch.Tensor:
        return self.w_to_z(self.u_to_w(u))

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LatentTransform":
        """Return an equivalent transform on the requested device and dtype."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        state = self.state_dict()
        return {
            "kind": self.kind,
            "version": self.version,
            "dim": self.dim,
            "state": state,
        }

    def diagnostics(self, z: torch.Tensor) -> dict[str, Any]:
        z = z.detach()
        w = self.z_to_w(z)
        u = self.w_to_u(w)
        outside = (u < 0.0) | (u > 1.0)
        return {
            "kind": self.kind,
            "version": self.version,
            "z_min": tensor_to_list(z.min(dim=0).values),
            "z_max": tensor_to_list(z.max(dim=0).values),
            "w_min": tensor_to_list(w.min(dim=0).values),
            "w_max": tensor_to_list(w.max(dim=0).values),
            "u_min": tensor_to_list(u.min(dim=0).values),
            "u_max": tensor_to_list(u.max(dim=0).values),
            "coordinate_outside_unit_fraction": float(outside.to(dtype=z.dtype).mean().item()),
            "point_outside_unit_fraction": float(outside.any(dim=1).to(dtype=z.dtype).mean().item()),
        }


class IdentityLatentBoxTransform(LatentTransform):
    """Manuscript baseline: calibrated coordinate w is exactly decoder latent z."""

    kind = "identity_latent_box"

    def __init__(self, bounds: torch.Tensor, *, name: str = "latent_manuscript_box") -> None:
        self.box = BoxTransform(bounds, name=name)

    @property
    def dim(self) -> int:
        return self.box.dim

    def fit(self, z_anchor: torch.Tensor) -> "IdentityLatentBoxTransform":
        if z_anchor.ndim != 2 or z_anchor.shape[1] != self.dim:
            raise ValueError("z_anchor has incompatible shape")
        assert_finite_tensor(z_anchor, name="z_anchor")
        return self

    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(z, name="z")
        return z.to(self.box.bounds)

    def w_to_z(self, w: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(w, name="w")
        return w.to(self.box.bounds)

    def w_to_u(self, w: torch.Tensor) -> torch.Tensor:
        return self.box.to_unit(w, clip=False)

    def u_to_w(self, u: torch.Tensor) -> torch.Tensor:
        assert_unit_tensor(u, name="latent unit coordinate")
        return self.box.from_unit(u, clip=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "bounds": tensor_to_list(self.box.bounds),
            "box_name": self.box.name,
        }

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "IdentityLatentBoxTransform":
        bounds = self.box.bounds.to(device=device, dtype=dtype)
        return self.__class__(bounds, name=self.box.name)

    @classmethod
    def load_state_dict(cls, state: dict[str, Any]) -> "IdentityLatentBoxTransform":
        if state.get("kind") != cls.kind:
            raise ValueError(f"state kind {state.get('kind')} does not match {cls.kind}")
        return cls(torch.tensor(state["bounds"], dtype=torch.float64), name=state.get("box_name", "latent_box"))


class DiagonalLatentTransform(LatentTransform):
    """Diagonal calibration w_i = (z_i - center_i) / scale_i with a fixed w box."""

    kind = "diagonal_latent_calibration"

    def __init__(self, center: torch.Tensor, scale: torch.Tensor, w_bounds: torch.Tensor, *, estimator: str = "mean_std") -> None:
        center = torch.as_tensor(center)
        scale = torch.as_tensor(scale)
        if not center.is_floating_point():
            center = center.to(dtype=torch.float64)
        if not scale.is_floating_point():
            scale = scale.to(dtype=torch.float64)
        if center.ndim != 1 or scale.ndim != 1 or center.shape != scale.shape:
            raise ValueError("center and scale must be same-shaped rank-1 tensors")
        if torch.any(scale <= 0.0) or torch.any(~torch.isfinite(scale)):
            raise ValueError("scale must be finite and positive")
        self.center = center
        self.scale = scale
        self.w_box = BoxTransform(w_bounds, name="diagonal_w_box")
        if self.w_box.dim != int(center.numel()):
            raise ValueError("w_bounds dimension must match center")
        self.estimator = estimator

    @property
    def dim(self) -> int:
        return int(self.center.numel())

    @classmethod
    def fit_anchor(
        cls,
        z_anchor: torch.Tensor,
        *,
        estimator: str = "mean_std",
        coverage_quantile: float = 0.995,
        min_scale: float = 1e-6,
    ) -> "DiagonalLatentTransform":
        assert_finite_tensor(z_anchor, name="z_anchor")
        if z_anchor.ndim != 2:
            raise ValueError("z_anchor must be rank-2")
        if estimator == "mean_std":
            center = z_anchor.mean(dim=0)
            scale = z_anchor.std(dim=0, unbiased=False)
        elif estimator == "median_mad":
            center = z_anchor.median(dim=0).values
            scale = torch.median(torch.abs(z_anchor - center), dim=0).values * 1.4826
        else:
            raise ValueError(f"unsupported diagonal estimator {estimator}")
        scale = torch.clamp(scale, min=float(min_scale))
        w = (z_anchor - center) / scale
        radius = torch.quantile(torch.max(torch.abs(w), dim=1).values, float(coverage_quantile))
        radius = torch.clamp(radius, min=torch.tensor(1.0, dtype=z_anchor.dtype, device=z_anchor.device))
        w_bounds = torch.stack([-torch.ones(z_anchor.shape[1], dtype=z_anchor.dtype, device=z_anchor.device) * radius, torch.ones(z_anchor.shape[1], dtype=z_anchor.dtype, device=z_anchor.device) * radius], dim=1)
        return cls(center.detach(), scale.detach(), w_bounds.detach(), estimator=estimator)

    def fit(self, z_anchor: torch.Tensor) -> "DiagonalLatentTransform":
        if z_anchor.ndim != 2 or z_anchor.shape[1] != self.dim:
            raise ValueError("z_anchor has incompatible shape")
        assert_finite_tensor(z_anchor, name="z_anchor")
        return self

    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(z, name="z")
        return (z.to(self.center) - self.center) / self.scale

    def w_to_z(self, w: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(w, name="w")
        return self.center + w.to(self.center) * self.scale

    def w_to_u(self, w: torch.Tensor) -> torch.Tensor:
        return self.w_box.to_unit(w, clip=False)

    def u_to_w(self, u: torch.Tensor) -> torch.Tensor:
        assert_unit_tensor(u, name="latent unit coordinate")
        return self.w_box.from_unit(u, clip=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "center": tensor_to_list(self.center),
            "scale": tensor_to_list(self.scale),
            "w_bounds": tensor_to_list(self.w_box.bounds),
            "estimator": self.estimator,
        }

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DiagonalLatentTransform":
        return self.__class__(
            self.center.to(device=device, dtype=dtype),
            self.scale.to(device=device, dtype=dtype),
            self.w_box.bounds.to(device=device, dtype=dtype),
            estimator=self.estimator,
        )

    @classmethod
    def load_state_dict(cls, state: dict[str, Any]) -> "DiagonalLatentTransform":
        if state.get("kind") != cls.kind:
            raise ValueError(f"state kind {state.get('kind')} does not match {cls.kind}")
        return cls(
            torch.tensor(state["center"], dtype=torch.float64),
            torch.tensor(state["scale"], dtype=torch.float64),
            torch.tensor(state["w_bounds"], dtype=torch.float64),
            estimator=state.get("estimator", "mean_std"),
        )


class WhitenedLatentTransform(LatentTransform):
    """Full-covariance calibration using z = center + Lw."""

    kind = "whitened_latent_calibration"

    def __init__(self, center: torch.Tensor, cholesky: torch.Tensor, w_bounds: torch.Tensor, *, epsilon: float) -> None:
        center = torch.as_tensor(center)
        cholesky = torch.as_tensor(cholesky)
        if not center.is_floating_point():
            center = center.to(dtype=torch.float64)
        if not cholesky.is_floating_point():
            cholesky = cholesky.to(dtype=torch.float64)
        if center.ndim != 1 or cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1] or cholesky.shape[0] != center.numel():
            raise ValueError("center and cholesky dimensions are inconsistent")
        if torch.any(torch.diagonal(cholesky) <= 0.0):
            raise ValueError("cholesky diagonal must be positive")
        self.center = center
        self.cholesky = cholesky
        self.w_box = BoxTransform(w_bounds, name="whitened_w_box")
        if self.w_box.dim != int(center.numel()):
            raise ValueError("w_bounds dimension must match center")
        self.epsilon = float(epsilon)

    @property
    def dim(self) -> int:
        return int(self.center.numel())

    @classmethod
    def fit_anchor(
        cls,
        z_anchor: torch.Tensor,
        *,
        coverage_quantile: float = 0.995,
        epsilon: float = 1e-6,
    ) -> "WhitenedLatentTransform":
        assert_finite_tensor(z_anchor, name="z_anchor")
        if z_anchor.ndim != 2:
            raise ValueError("z_anchor must be rank-2")
        center = z_anchor.mean(dim=0)
        centered = z_anchor - center
        denom = max(int(z_anchor.shape[0]) - 1, 1)
        cov = centered.T @ centered / denom
        cov = cov + float(epsilon) * torch.eye(z_anchor.shape[1], dtype=z_anchor.dtype, device=z_anchor.device)
        cholesky = torch.linalg.cholesky(cov)
        w = torch.linalg.solve_triangular(cholesky, centered.T, upper=False).T
        radius = torch.quantile(torch.max(torch.abs(w), dim=1).values, float(coverage_quantile))
        radius = torch.clamp(radius, min=torch.tensor(1.0, dtype=z_anchor.dtype, device=z_anchor.device))
        w_bounds = torch.stack([-torch.ones(z_anchor.shape[1], dtype=z_anchor.dtype, device=z_anchor.device) * radius, torch.ones(z_anchor.shape[1], dtype=z_anchor.dtype, device=z_anchor.device) * radius], dim=1)
        return cls(center.detach(), cholesky.detach(), w_bounds.detach(), epsilon=epsilon)

    def fit(self, z_anchor: torch.Tensor) -> "WhitenedLatentTransform":
        if z_anchor.ndim != 2 or z_anchor.shape[1] != self.dim:
            raise ValueError("z_anchor has incompatible shape")
        assert_finite_tensor(z_anchor, name="z_anchor")
        return self

    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(z, name="z")
        centered = z.to(self.center) - self.center
        return torch.linalg.solve_triangular(self.cholesky, centered.T, upper=False).T

    def w_to_z(self, w: torch.Tensor) -> torch.Tensor:
        assert_finite_tensor(w, name="w")
        return self.center + w.to(self.center) @ self.cholesky.T

    def w_to_u(self, w: torch.Tensor) -> torch.Tensor:
        return self.w_box.to_unit(w, clip=False)

    def u_to_w(self, u: torch.Tensor) -> torch.Tensor:
        assert_unit_tensor(u, name="latent unit coordinate")
        return self.w_box.from_unit(u, clip=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "center": tensor_to_list(self.center),
            "cholesky": tensor_to_list(self.cholesky),
            "w_bounds": tensor_to_list(self.w_box.bounds),
            "epsilon": self.epsilon,
        }

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WhitenedLatentTransform":
        return self.__class__(
            self.center.to(device=device, dtype=dtype),
            self.cholesky.to(device=device, dtype=dtype),
            self.w_box.bounds.to(device=device, dtype=dtype),
            epsilon=self.epsilon,
        )

    @classmethod
    def load_state_dict(cls, state: dict[str, Any]) -> "WhitenedLatentTransform":
        if state.get("kind") != cls.kind:
            raise ValueError(f"state kind {state.get('kind')} does not match {cls.kind}")
        return cls(
            torch.tensor(state["center"], dtype=torch.float64),
            torch.tensor(state["cholesky"], dtype=torch.float64),
            torch.tensor(state["w_bounds"], dtype=torch.float64),
            epsilon=float(state.get("epsilon", 1e-6)),
        )


def load_latent_transform_state(state: dict[str, Any]) -> LatentTransform:
    kind = state.get("kind")
    if kind == IdentityLatentBoxTransform.kind:
        return IdentityLatentBoxTransform.load_state_dict(state)
    if kind == DiagonalLatentTransform.kind:
        return DiagonalLatentTransform.load_state_dict(state)
    if kind == WhitenedLatentTransform.kind:
        return WhitenedLatentTransform.load_state_dict(state)
    raise ValueError(f"Unsupported latent transform kind {kind}")
