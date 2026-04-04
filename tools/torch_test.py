import torch
import torchvision
import torchaudio

print(torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
