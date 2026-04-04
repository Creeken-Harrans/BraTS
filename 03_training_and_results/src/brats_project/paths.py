#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

from brats_project.project_layout import get_default_environment_paths

_DEFAULT_PATHS = get_default_environment_paths()

PROJECT_RAW = (
    os.environ.get("PROJECT_RAW")
    or _DEFAULT_PATHS["PROJECT_RAW"]
)
PROJECT_PREPROCESSED = (
    os.environ.get("PROJECT_PREPROCESSED")
    or _DEFAULT_PATHS["PROJECT_PREPROCESSED"]
)
PROJECT_RESULTS = (
    os.environ.get("PROJECT_RESULTS")
    or _DEFAULT_PATHS["PROJECT_RESULTS"]
)

if PROJECT_RAW is None:
    print(
        "PROJECT_RAW is not defined. Dataset preparation can still be inspected, "
        "but raw-dataset driven processing will not work."
    )

if PROJECT_PREPROCESSED is None:
    print(
        "PROJECT_PREPROCESSED is not defined. Preprocessing metadata and training "
        "cannot be loaded."
    )

if PROJECT_RESULTS is None:
    print(
        "PROJECT_RESULTS is not defined. Training outputs and inference outputs "
        "cannot be written."
    )
