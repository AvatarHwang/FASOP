import numpy as np

def fix_data(lst):
    embedding_layer = lst[0]
    post_process = lst[-1]
    transformer = np.mean(lst[1:-1])
    data_fixed_1 = []
    data_fixed_1.append(embedding_layer)
    for i in range(48):
        data_fixed_1.append(transformer)
    data_fixed_1.append(post_process)
    return data_fixed_1

# load gpt2XL_A10_1.npy
data_1 = np.load('gpt2XL_A10_1.npy')
fixed_data_1 = fix_data(data_1)
#np.save('gpt2XL_A10_1.npy',fixed_data_1)
print(fixed_data_1)
assert False
data_2 = np.load('gpt2XL_A10_2.npy')
fixed_data_2 = fix_data(data_2)
#np.save('gpt2XL_A10_2.npy',fixed_data_2)
print(fixed_data_2)
data_4 = np.load('gpt2XL_A10_4.npy')
fixed_data_4 = fix_data(data_4)
#np.save('gpt2XL_A10_4.npy',fixed_data_4)
print(fixed_data_4)

data_1_a100 = np.load('gpt2XL_A100_1.npy')
fixed_data_1 = fix_data(data_1_a100)
#np.save('gpt2XL_A100_1.npy',fixed_data_1)
print(fixed_data_1)
data_2_a100 = np.load('gpt2XL_A100_2.npy')
fixed_data_1 = fix_data(data_2_a100)
#np.save('gpt2XL_A100_2.npy',fixed_data_1)
print(fixed_data_1)
data_4_a100 = np.load('gpt2XL_A100_4.npy')
fixed_data_1 = fix_data(data_4_a100)
#np.save('gpt2XL_A100_4.npy',fixed_data_1)
print(fixed_data_1)
