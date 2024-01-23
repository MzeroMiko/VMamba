# Modified by Mzero #20240123
# simplified version of mamba 
# kHasZ = False # delete implementation for z
# kIsVariableC = True; kIsVariableB = True # delete implementation for B, C not variable
# kIsComplex = False # delete implementation for complex_t

import torch
from .selective_scan_interface import SelectiveScanFn, selective_scan_fn, selective_scan_ref
