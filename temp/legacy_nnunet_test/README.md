Legacy nnUNet_test leftovers that were moved into the BraTS project during workspace consolidation.

- The active dataset preparation entrypoint is `01_data_preparation/scripts/prepare_brats2020_for_project.py`.
- The old duplicate converter script was removed after consolidation so there is only one source of truth.
- `tools/torch_test.py` is kept only as an environment check utility migrated from `nnUNet_test`.
