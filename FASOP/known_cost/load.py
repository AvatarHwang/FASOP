import numpy as np

tp_arr = [1,2,4]
model = ['gpt2XL']
gpu_type = ['A10','A100']


for m in model:
    for g in gpu_type:
                
        for tp in tp_arr:
            filename = f"{m}_{g}_{tp}.npy"
            data_1 = np.load(filename)
            print(f"{filename} len: {len(data_1)}")
            print(f"{data_1[0]:.4f} {data_1[1]:.4f} {data_1[-1]:.4f}")