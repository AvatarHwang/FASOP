import numpy as np

input_embedding_time=0.000212169
encoder_time=0.001450825
post_process_time=1.67131E-05
decoder_embedding_time=0.000187683
decoder_time=0.002590394
decoder_post_process_time=1.64032E-05

data_1 = []

data_1.append(input_embedding_time)
for i in range(24):
    data_1.append(encoder_time)
data_1[-1] += post_process_time
data_1.append(decoder_embedding_time)
for i in range(24):
    data_1.append(decoder_time)
data_1[-1] += decoder_post_process_time

input_embedding_time=0.00040257
encoder_time=0.001484366
post_process_time=2.10285E-05
decoder_embedding_time=0.00058794
decoder_time=0.002575711
decoder_post_process_time=2.00748E-05

data_2 = []
data_2.append(input_embedding_time)
for i in range(24):
    data_2.append(encoder_time)
data_2[-1] += post_process_time
data_2.append(decoder_embedding_time)
for i in range(24):
    data_2.append(decoder_time)
data_2[-1] += decoder_post_process_time

input_embedding_time=0.000471425
encoder_time=0.001431004
post_process_time=2.10762E-05
decoder_embedding_time=0.000588989
decoder_time=0.002607838
decoder_post_process_time=1.98603E-05

data_4 = []

data_4.append(input_embedding_time)
for i in range(24):
    data_4.append(encoder_time)
data_4[-1] += post_process_time
data_4.append(decoder_embedding_time)
for i in range(24):
    data_4.append(decoder_time)
data_4[-1] += decoder_post_process_time


np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A10_1.npy', data_1)
np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A10_2.npy', data_2)
np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A10_4.npy', data_4)

input_embedding_time=0.000190997
encoder_time=0.001170739
post_process_time=1.73807E-05
decoder_embedding_time=0.000220585
decoder_time=0.002099025
decoder_post_process_time=1.58787E-05

data_1 = []

data_1.append(input_embedding_time)
for i in range(24):
    data_1.append(encoder_time)
data_1[-1] += post_process_time
data_1.append(decoder_embedding_time)
for i in range(24):
    data_1.append(decoder_time)
data_1[-1] += decoder_post_process_time

input_embedding_time=0.000410318
encoder_time=0.001364652
post_process_time=1.9002E-05
decoder_embedding_time=0.000517321
decoder_time=0.00242337
decoder_post_process_time=1.65701E-05

data_2 = []

data_2.append(input_embedding_time)
for i in range(24):
    data_2.append(encoder_time)
data_2[-1] += post_process_time
data_2.append(decoder_embedding_time)
for i in range(24):
    data_2.append(decoder_time)
data_2[-1] += decoder_post_process_time


input_embedding_time=0.000441623
encoder_time=0.001504668
post_process_time=1.96934E-05
decoder_embedding_time=0.000487876
decoder_time=0.002515952
decoder_post_process_time=1.63078E-05


data_4 = []

data_4.append(input_embedding_time)
for i in range(24):
    data_4.append(encoder_time)
data_4[-1] += post_process_time
data_4.append(decoder_embedding_time)
for i in range(24):
    data_4.append(decoder_time)
data_4[-1] += decoder_post_process_time


np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A100_1.npy', data_1)
np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A100_2.npy', data_2)
np.save('/home1/soonyear/tdpp/FASOP/known_cost/T5_A100_4.npy', data_4)

