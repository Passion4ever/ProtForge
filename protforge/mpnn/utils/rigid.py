"""
Rigid body transformations for protein structures.

This module provides classes for representing 3D rotations and rigid
transformations (rotation + translation), used in sidechain packing.

Original code from OpenFold (AlQuraishi Laboratory) and AlphaFold (DeepMind).
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable, Any, Sequence

import numpy as np
import torch


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication of two rotation matrix tensors."""
    def row_mul(i):
        return torch.stack([
            a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0],
            a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1],
            a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2],
        ], dim=-1)
    return torch.stack([row_mul(0), row_mul(1), row_mul(2)], dim=-2)


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Applies a rotation to a vector."""
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack([
        r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
        r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
        r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
    ], dim=-1)


def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    return rots


def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    return torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


class Rotation:
    """A 3D rotation represented by a rotation matrix."""

    def __init__(self, rot_mats: torch.Tensor):
        if rot_mats.shape[-2:] != (3, 3):
            raise ValueError("Incorrectly shaped rotation matrix")
        self._rot_mats = rot_mats.to(dtype=torch.float32)

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
    ) -> "Rotation":
        rot_mats = identity_rot_mats(shape, dtype, device, requires_grad)
        return Rotation(rot_mats=rot_mats)

    def __getitem__(self, index: Any) -> "Rotation":
        if type(index) != tuple:
            index = (index,)
        rot_mats = self._rot_mats[index + (slice(None), slice(None))]
        return Rotation(rot_mats=rot_mats)

    def __mul__(self, right: torch.Tensor) -> "Rotation":
        rot_mats = self._rot_mats * right[..., None, None]
        return Rotation(rot_mats=rot_mats)

    def __rmul__(self, left: torch.Tensor) -> "Rotation":
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        return self._rot_mats.shape[:-2]

    @property
    def dtype(self) -> torch.dtype:
        return self._rot_mats.dtype

    @property
    def device(self) -> torch.device:
        return self._rot_mats.device

    def get_rot_mats(self) -> torch.Tensor:
        return self._rot_mats

    def compose_r(self, r: "Rotation") -> "Rotation":
        new_rot_mats = rot_matmul(self._rot_mats, r._rot_mats)
        return Rotation(rot_mats=new_rot_mats)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        return rot_vec_mul(self._rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        inv_rot_mats = invert_rot_mat(self._rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self) -> "Rotation":
        return Rotation(rot_mats=invert_rot_mat(self._rot_mats))

    def unsqueeze(self, dim: int) -> "Rotation":
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
        return Rotation(rot_mats=rot_mats)

    @staticmethod
    def cat(rs: Sequence["Rotation"], dim: int) -> "Rotation":
        rot_mats = torch.cat([r._rot_mats for r in rs], dim=dim if dim >= 0 else dim - 2)
        return Rotation(rot_mats=rot_mats)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Rotation":
        rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
        rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
        rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
        return Rotation(rot_mats=rot_mats)

    def detach(self) -> "Rotation":
        return Rotation(rot_mats=self._rot_mats.detach())


class Rigid:
    """A rigid transformation (rotation + translation)."""

    def __init__(self, rots: Optional[Rotation], trans: Optional[torch.Tensor]):
        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(batch_dims, dtype, device, requires_grad)
        elif trans is None:
            trans = identity_trans(batch_dims, dtype, device, requires_grad)

        trans = trans.to(dtype=torch.float32)
        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
    ) -> "Rigid":
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> "Rigid":
        if type(index) != tuple:
            index = (index,)
        return Rigid(self._rots[index], self._trans[index + (slice(None),)])

    def __mul__(self, right: torch.Tensor) -> "Rigid":
        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]
        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> "Rigid":
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self._trans.device

    def get_rots(self) -> Rotation:
        return self._rots

    def get_trans(self) -> torch.Tensor:
        return self._trans

    def compose(self, r: "Rigid") -> "Rigid":
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> "Rigid":
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)
        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Rigid":
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)
        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> "Rigid":
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")
        rots = Rotation(rot_mats=t[..., :3, :3])
        trans = t[..., :3, 3]
        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8
    ) -> "Rigid":
        """Constructs transformations from sets of 3 points using Gram-Schmidt."""
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))
        rot_obj = Rotation(rot_mats=rots)
        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """
        Returns a transformation object from reference coordinates.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)
        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots)
        return Rigid(rot_obj, translation)

    def unsqueeze(self, dim: int) -> "Rigid":
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)
        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: Sequence["Rigid"], dim: int) -> "Rigid":
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)
        return Rigid(rots, trans)
