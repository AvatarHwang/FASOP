import numpy as np
import os

class Profile():
    def __init__(self, model_name, gpu_type, 
                 input_embedding_time=[], encoder_time=[], post_process_time=[], 
                 decoder_embedding_time=[], decoder_time=[], decoder_post_process_time=[], 
                 en_layer_num=0, de_layer_num=0, transformer_type=None, add_name=None, dir_path=None):
        self.model_name = model_name
        self.gpu_type = gpu_type
        
        # self.xxx_time = array[TP=1, TP=2, TP=4]
        
        self.input_embedding_time = input_embedding_time
        # print(type(self.input_embedding_time))
        self.encoder_time = encoder_time
        self.post_process_time = post_process_time
        self.decoder_embedding_time = decoder_embedding_time
        self.decoder_time = decoder_time
        self.decoder_post_process_time = decoder_post_process_time
        self.en_layer_num = en_layer_num
        self.de_layer_num = de_layer_num
        self.profile_array = []
        self.tp_array = ['1','2','4']
        self.transformer_type = transformer_type
        self.add_name = add_name
        
        if self.add_name is not None: 
            self.add_name = "_" + add_name 
        else :
            self.add_name = ""
        self.dir_path = dir_path
        # is_model()
    
    def is_model(self):
        
        if self.model_name == 'T5':
            assert self.en_layer_num > 0 and self.de_layer_num > 0
            return "both"
        elif self.model_name == 'gpt2XL':
            assert self.de_layer_num > 0
            return "de"
        elif self.model_name == 'bert':
            assert self.en_layer_num > 0
            return "en"
        else:
            assert False
    
    def make_nparray(self, iet, et, ppt, det, dt, dppt):
        
        # iet: input_embedding_time
        # et: encoder_time
        # pt: post_process_time
        # det: decoder_embedding_time
        # dt: decoder_time
        # dppt: decoder_post_process_time

        if self.transformer_type in ["en", "both"]:
            # pre process    
            self.profile_array.append(iet)
            # encoder
            for i in range(self.en_layer_num):
                self.profile_array.append(et)
            # post process    
            self.profile_array[-1] += ppt
        
        if self.transformer_type in ["de", "both"]:
            # pre process(=embedding)
            self.profile_array.append(det)
            # decoder
            for i in range(self.de_layer_num):
                self.profile_array.append(dt)
            # post process    
            if self.model_name == "T5":
                self.profile_array[-1] += dppt
            elif self.model_name == "gpt2XL":
                self.profile_array.append(dppt)
        return self.profile_array
        
    def save_nparray(self):
        self.transformer_type = self.is_model()
        
        for i in range(len(self.tp_array)):
            self.profile_array = []
            if self.transformer_type in ["en", "both"]:
                iet = self.input_embedding_time[i]
                et = self.encoder_time[i]
                ppt = self.post_process_time[i]
                det = 0
                dt = 0
                dppt = 0
            if self.transformer_type in ["de", "both"]:
                iet = 0
                et = 0
                ppt = 0
                det = self.decoder_embedding_time[i]
                dt = self.decoder_time[i]
                dppt = self.decoder_post_process_time[i]
            
            nparray = self.make_nparray(iet, et, ppt, det, dt, dppt)
            filename = self.model_name +"_"+ self.gpu_type + "_" + self.tp_array[i] + self.add_name + ".npy"
            np.save(f'./{filename}', nparray)
    
    def print_nparray(self):
        for i in range(len(self.tp_array)):
            ar = np.load(f"./{self.model_name}_{self.gpu_type}_{self.tp_array[i]}" + self.add_name + ".npy")
            print(f"./{self.model_name}_{self.gpu_type}_{self.tp_array[i]}" + self.add_name + ".npy")
            print(f"size: {len(ar)}")
            print(ar)
def main():
    
    # example: GPT2XL A10/A100
    
    gpt2xl_A10 = Profile("gpt2XL", "A10",
                         decoder_embedding_time=[0.000207661, 0.000544237, 0.001509664],
                         decoder_time=[0.002076069, 0.001571831, 0.001878196],
                         decoder_post_process_time=[0.000077414, 0.000083302, 0.000083302],de_layer_num=48, transformer_type="de", add_name="test")
    gpt2xl_A10.save_nparray()
    gpt2xl_A10.print_nparray()
    
if __name__ == "__main__":
    main()


