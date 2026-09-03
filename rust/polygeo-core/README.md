# polygeo-core

`polygeo-core` is PolyGeo's proof-bearing Rust library for discrete topology,
exact chain algebra, discrete exterior calculus, and triangle-surface geometry.
It is independent of Python and PyO3.

## Scope

The crate provides:

- immutable topology owners with admitted capability evidence;
- exact integer and rational chains, cochains, maps, and integral homology;
- binary64 forms, matrix-free operators, and circumcentric metrics;
- bounded numerical problems, reusable preparations, and certified results;
- triangle-surface geometry, connections, holonomy, and direction fields.

Values retain their topology or realization owner. Operations reject foreign
owners, unsupported domains, exhausted resource policies, and uncertified
numerical results instead of silently identifying equal-shaped data.

Version 0.1.x is experimental. The implemented paths are tested, but the public
API and numerical policies are not yet stable.

## Documentation

API documentation: <https://docs.rs/polygeo-core>. The
[repository architecture](https://github.com/nostalume/polygeo/blob/main/docs/architecture.md)
describes ownership, invariants, and the Python adapter boundary. Python users
should use the separately packaged `polygeo` project.

The minimum supported Rust version is 1.97.1.

## License

Licensed under the MIT License.
