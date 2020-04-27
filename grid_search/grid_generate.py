import standard_grid
import pickle
import time
import os

WORK_ROOT="/work/sbali/monaural-source-separation/experiments/hyperparameter"
if __name__=="__main__":
    data_spec1 = os.path.join(WORK_ROOT, "dataset/t0-2s-10c/4s-waveunet.json")
    model_spec = {}
    model_spec[0] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/000.json")
    model_spec[1] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/001.json")
    model_spec[2] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/002.json")
    model_spec[3] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/003.json")
    model_spec[4] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/004.json")
    model_spec[5] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/005.json")
    model_spec[6] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/006.json")
    model_spec[7] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/007.json")
    model_spec[8] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/008.json")
    model_spec[9] = os.path.join(WORK_ROOT, "model/Wave-U-Net/t0-2s-10c/009.json")



    grid = standard_grid.Grid("../train.py","../results/")
    
    grid.register('model_spec', [model_spec[i] for i in model_spec.keys()])
    grid.register('dataset_spec', [data_spec1])
    grid.register('checkpoint_freq', [50])
    grid.generate_grid()
    grid.generate_shell_instances(prefix="python3 ")
    
    #grid.create_runner(num_runners=4,runners_prefix=["CUDA_VISIBLE_DEVIDES=%d sh"%i for i in range(4)],parallel=2)
