# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2024-05-18

### Fixed

- Bug in `MatCreateMPIAIJWithSplitArrays_args`.

## [0.1.1] - 2024-02-28

### Fixed

- Bugs in KSP solver wrapper.

### Added

- More methods to the low-level API.

## [0.1.0] - 2024-01-27

### Added

- Configuration of PETSc installation using Preferences.jl.
- High-level wrapper of KSP solvers.
- Support for sequential runs.
- Support for parallel parallel runs with PartitionedArrays.jl.
- Commonly used low-level API for KSP solvers.
