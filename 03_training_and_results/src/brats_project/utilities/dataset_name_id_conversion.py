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
from typing import Union

from brats_project.paths import PROJECT_PREPROCESSED, PROJECT_RAW, PROJECT_RESULTS
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def find_candidate_datasets(dataset_id: int):
    startswith = "Dataset%03.0d" % dataset_id
    if PROJECT_PREPROCESSED is not None and isdir(PROJECT_PREPROCESSED):
        candidates_preprocessed = subdirs(PROJECT_PREPROCESSED, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if PROJECT_RAW is not None and isdir(PROJECT_RAW):
        candidates_raw = subdirs(PROJECT_RAW, prefix=startswith, join=False)
    else:
        candidates_raw = []

    candidates_trained_models = []
    if PROJECT_RESULTS is not None and isdir(PROJECT_RESULTS):
        candidates_trained_models += subdirs(PROJECT_RESULTS, prefix=startswith, join=False)

    all_candidates = candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    return unique_candidates


def convert_id_to_dataset_name(dataset_id: int):
    unique_candidates = find_candidate_datasets(dataset_id)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (dataset_id, PROJECT_RAW, PROJECT_PREPROCESSED, PROJECT_RESULTS))
    if len(unique_candidates) == 0:
        raise RuntimeError(f"Could not find a dataset with the ID {dataset_id}. Make sure the requested dataset ID "
                           f"exists and that nnU-Net knows where raw and preprocessed data are located "
                           f"(see Documentation - Installation). Here are your currently defined folders:\n"
                           f"PROJECT_PREPROCESSED={os.environ.get('PROJECT_PREPROCESSED') if os.environ.get('PROJECT_PREPROCESSED') is not None else 'None'}\n"
                           f"PROJECT_RESULTS={os.environ.get('PROJECT_RESULTS') if os.environ.get('PROJECT_RESULTS') is not None else 'None'}\n"
                           f"PROJECT_RAW={os.environ.get('PROJECT_RAW') if os.environ.get('PROJECT_RAW') is not None else 'None'}\n"
                           f"If something is not right, adapt your environment variables.")
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:10])
    return dataset_id


def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct dataset name. Your input: %s" % dataset_name_or_id)
    return convert_id_to_dataset_name(dataset_name_or_id)
