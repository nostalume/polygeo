"""Numerical problem admission, preparation, workspaces, and results."""

from ._polygeo_native import solve as _native

StorageLimit = _native.StorageLimit
WorkLimit = _native.WorkLimit
Executor = _native.Executor
Policy = _native.Policy
CancellationToken = _native.CancellationToken
Problem = _native.Problem
Prepared = _native.Prepared
Workspace = _native.Workspace
ProblemError = _native.ProblemError
SolveError = _native.SolveError
PoissonResult = _native.PoissonResult
DirichletResult = _native.DirichletResult
HeatResult = _native.HeatResult

__all__ = [
    "StorageLimit",
    "WorkLimit",
    "Executor",
    "Policy",
    "CancellationToken",
    "Problem",
    "Prepared",
    "Workspace",
    "ProblemError",
    "SolveError",
    "PoissonResult",
    "DirichletResult",
    "HeatResult",
]
