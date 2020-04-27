import pickle
import standard_grid

name1 = '.06ff838cfebe84bee87c51c4064d9d4e59701fe7220a7445021581d35d8c1dc1e558e6c6212d4965fe98b1c758e3b2efdc7410dd9d542f9e9edefbfe839630e1.pkl'
name2 = '.7f5a2aaac2855baf99c732a8753e00040318f18c066834c1358533fd438ef05b301164990434966ff92cc038efd4c2f9dd23e8879b0e0550351bc912c6a542c9.pkl'
name3 = '.e10e0e3bd536e63f4bdb7032fb6c88c1b1c3f936b13478f6ce7c3d10b722a75eac298aed9844bbc209df0c58ce1592033daee985a563e41e16b5690f12647de2.pkl'
name4 = '.f2f1d3c50f0f407ce9e646050977e96f4c59575cc53a938277058b38fef8fea164a163a731cb2b5f89343d4dc59c44baf170148bef43feae63ae5c6b33bc570e.pkl'
name5 = '.7854f23d3d4d4183b5b2c0d582563d574a0e513bc36cf0af9e58aeee81c270c62e66828fff830a2dea79c1d46f7117b2bf5c27d223ff884b387f276b5febf64f.pkl'
name6 = '.66c9f5414f86d91903173997de18f54adc5922ac8fd5cf03cfb10b19121d2ed6e14a1fc16f0802e874a0e67afb7458edeef6435d98a562cd0be3fd4cad4201fe.pkl'
name7 = '.99fdabdaf03bab237be2e2348fe6d027f4cf2c5256b04556963082a98ee30f402d00de2b6f3907c7f85d3d83b267f515acfda72f5acd7641c588017b65e0bcbb.pkl'
name8 = '.925e7b9c66b953b85259148dd8db6c2de261f99a0e792f196c20984de06191d4c8c293a3d25bd5898b2626cf713dc1583c0795240cc5667e83b8b3b6de7197ee.pkl'
if __name__=='__main__':
    hash_val = name8
    #".06ff838cfebe84bee87c51c4064d9d4e59701fe7220a7445021581d35d8c1dc1e558e6c6212d4965fe98b1c758e3b2efdc7410dd9d542f9e9edefbfe839630e1.pkl"
    with open(hash_val, 'rb') as pickle_file:
        grid = pickle.load(pickle_file)
    total_at_a_time=5
    #grid.get_status()
    #grid.create_runner(num_runners=total_at_a_time,runners_prefix=["sbatch -p gpu_low -c 1 --gres=gpu:1 -W"]*total_at_a_time)
    grid.create_runner(num_runners=4,runners_prefix=["CUDA_VISIBLE_DEVIDES=%d sh"%i for i in range(4)])
    #grid.create_runner(num_runners=4,runners_prefix=["CUDA_VISIBLE_DEVIDES=%d sh"%i for i in range(4)],parallel=2)
    #grid.__del__()
