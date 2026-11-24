Project Attribution and Third-Party Notices
Project
Face Identification System

This repository contains a substantially modified version of a face-identification project. The current implementation includes changes to the application structure, input validation, dataset preparation, recognition workflow, evaluation utilities, testing, and documentation.
Developer of this adapted version
Vineet Yadav

The modifications and experiments in this version should be understood separately from the upstream project and from the third-party libraries listed below.
Upstream project
This work was originally based on:
- Project: Facial Recognition System
- Original repository: https://github.com/revanthchristober/Facial-Recognition-System
- Original author: Revanth Christober M
- License: MIT
The upstream MIT license and its required copyright and permission notices are retained in the accompanying LICENSE file. This attribution does not imply that the upstream author developed the modifications in this repository.
Main modifications in this version
The adapted version includes work such as:
- Robust input validation and repository-relative paths.
- Labeled-dataset encoding preparation.
- Explicit distance-threshold face recognition.
- A separate SVC training and evaluation workflow.
- Camera cleanup and error handling.
- Synthetic unit tests.
- Local-data exclusions and updated setup and usage documentation.
Separate research experiments have also evaluated dlib, FaceNet, and ArcFace using an open-set identification protocol on the Labeled Faces in the Wild (LFW) dataset. These experiments are research workflows and should not be interpreted as FaceNet or ArcFace being integrated into the Flask application unless the application code explicitly implements that integration.
Third-party technologies
The project uses or may use the following third-party technologies. Refer to each project's own license and notices for the applicable terms:
- OpenCV — BSD 3-Clause.
- dlib — Boost Software License 1.0; separately downloaded model files may have their own terms.
- face_recognition — MIT.
- face_recognition_models — check the license and provenance of the specific model files distributed with or downloaded for the project.
- NumPy — BSD 3-Clause.
- scikit-learn — BSD 3-Clause.
- Joblib — BSD 3-Clause.
- pytest — MIT.
- Jupyter — BSD 3-Clause for core Jupyter components; individual extensions and dependencies may have separate terms.
- FaceNet / facenet-pytorch — consult the repository and model-weight terms.
- InsightFace — consult the applicable code and pretrained-model licensing terms before redistribution.
Dataset notice
The Labeled Faces in the Wild (LFW) dataset is distributed under its own terms. Obtain it from its official source and review those terms before sharing the dataset or any derived files.
Disclaimer
This file is an attribution and dependency-notice summary, not legal advice. Retain applicable upstream copyright notices, license texts, and third-party notices when redistributing the project. Verify the license of every bundled model, dataset, and dependency version.