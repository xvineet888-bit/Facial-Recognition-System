# Changelog

## 2026-10-28 — Working tree enhancements

Completed in this maintenance session:

- Added and verified a project-local Python 3.10 environment with a compiled dlib 20.0.1 Windows wheel and compatible setuptools pin.
- Repaired missing imports, invalid image handling, model path handling, and array/label validation.
- Reused the installed 68-point dlib predictor package asset for optional landmarks instead of requiring a duplicate local model file.
- Added labeled-folder encoding preparation with ambiguous multi-face images skipped, and kept distance-based webcam recognition separate from the linear SVC training workflow.
- Fixed direct webcam script execution and selected DirectShow on Windows after the default MSMF backend failed to read a frame.
- Replaced dataset-dependent tests with synthetic tests and verified 16 passing tests covering persistence, recognition, detection, training, and missing optional assets.
- Updated data exclusions, setup instructions, attribution, project limitations, and portfolio documentation.

## Planned

- Validate matching thresholds on a consented, representative local dataset.
- Test live webcam behavior on the target device.
- Add configurable command-line options for camera index and tolerance.
