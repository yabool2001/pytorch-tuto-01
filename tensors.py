import torch

import numpy as np
from numpy.typing import NDArray

barker13_filename = "barker13_samples_clipped.npy"
n : NDArray [ np.complex128 ] = np.load ( barker13_filename ).astype ( np.complex128 , copy = False )

x = torch.rand ( 2 , 2 , dtype = torch.float16 , requires_grad = True )
y = torch.from_numpy ( n ).to ( torch.complex128 )

print ( x )
print ( y )
