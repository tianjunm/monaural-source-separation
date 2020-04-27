import torch
import os
import os
print(torch.version.cuda)
print(torch.cuda.device_count())
print(os.environ.get('CUDA_VISIBLE_DEVICES'))
print(torch._C._cuda_getDriverVersion())
print(torch.cuda.is_available())
a = torch.tensor([1, 2])
print(a.cuda())
print(torch.backends.cudnn.enabled)
