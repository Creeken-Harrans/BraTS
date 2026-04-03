import os

from brats_project.utilities.default_n_proc_DA import get_allowed_n_proc_DA

default_num_processes = 8 if 'BRATS_DEF_N_PROC' not in os.environ else int(os.environ['BRATS_DEF_N_PROC'])

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()
