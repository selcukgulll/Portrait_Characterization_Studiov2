__version__ = '0.27.1+cu130'
git_version = 'df56172e4d5a8d0cd51384273bc6c5747f5ab931'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
