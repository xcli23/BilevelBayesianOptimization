import time
import tensorflow as tf
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def get_strs(word_cache, x):
    strs = []
    for j in range(x.size(0)):
        row_elements = [str(word_cache[i][x[j][i]]) for i in range(x.size(1))]
        merged_row = ' '.join(row_elements)
        strs.append(merged_row)
    return strs

def process_row(j, word_cache, x):
    row_elements = [word_cache[i][x[j][i]] for i in range(x.size(1))]
    merged_row = ' '.join(row_elements)
    return merged_row

def multi_get_strs(word_cache, x, max_workers=8):
    x_size0 = x.size(0)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        partial_process_row = partial(process_row, word_cache=word_cache, x=x)
        strs = list(executor.map(partial_process_row, range(x_size0)))
    return strs

class KernelABC:
    
    def get_tensors(self, x_strs):
        # x_embed = self.embed(x_strs)     # V5
        x_embed = self.embed(x_strs)['outputs']     # V3
        x_tensor = torch.from_numpy(x_embed.numpy()).to('cuda')
        return x_tensor
    def convert_to_embedding(self, x1, x2,  word_filtered=2):   
        assert word_filtered in [0, 1, 2], "word_filtered is not valid. It should be 0, 1, or 2."
        x1, x2 = x1.int(), x2.int()
        if word_filtered == 2:
            word_cache = [row for row in self.BBM.word_substitution_cache if len(row) > 1] 
            # start_time = time.time()
            x1_strs = get_strs(word_cache, x1)
            x2_strs = get_strs(word_cache, x2)
            # x1_strs = multi_get_strs(word_cache, x1)
            # x2_strs = multi_get_strs(word_cache, x2)
            x1_tensor = self.get_tensors(x1_strs)
            x2_tensor = self.get_tensors(x2_strs)
            # end_time = time.time()
            # print(f'run_time: {end_time - start_time}')
            # total_time += end_time - start_time
        return x1_tensor, x2_tensor
    
    def old_convert_to_embedding(self, x1, x2,  word_filtered=2):
        assert word_filtered in [0, 1, 2], "word_filtered is not valid. It should be 0, 1, or 2."
        start_time = time.time()
        GPU = tf.device('/GPU:0')
        # n_vertices = torch.tensor(self.BBM.n_vertices).to('cuda:0')  
        n_vertices = torch.tensor(self.BBM.n_vertices, device='cuda')
        indices = torch.tensor(n_vertices) == 1
        indices = torch.nonzero(indices).view(-1)
        
        if word_filtered == 0: 
            word_cache = list(self.BBM.word_substitution_cache)
            ones_column_1 = torch.zeros(x1.size(0), 1).to('cuda:0')
            for i in indices:
                x1 = torch.cat([x1[:, :i], ones_column_1, x1[:, i:]], dim=1)
            ones_column_2 = torch.zeros(x2.size(0), 1).to('cuda:0')  
            for i in indices:
                x2 = torch.cat([x2[:, :i], ones_column_2, x2[:, i:]], dim=1)
            sentence = self.BBM.x0.text
            punctuation_pattern = f"[{re.escape(string.punctuation)}]"
            words_and_symbols = re.findall(r"[\w]+|\s|" + punctuation_pattern, sentence)
            word_indices = [index for index, item in enumerate(words_and_symbols) if re.match(r'\w+', item)]
            strs = []
            start_time_1 = time.time()
            for i in range(x1.size(0)): 
                for j in range(x1.size(1)):
                    words_and_symbols[word_indices[j]] = word_cache[j][x1[i][j].long()]
                merged_row = ''.join(words_and_symbols)
                strs.append(merged_row)
            for i in range(x2.size(0)): 
                for j in range(x2.size(1)):
                    words_and_symbols[word_indices[j]] = word_cache[j][x2[i][j].long()]
                merged_row = ''.join(words_and_symbols)
                strs.append(merged_row)
        elif word_filtered == 1:
            word_cache = list(self.BBM.word_substitution_cache)
            ones_column_1 = torch.zeros(x1.size(0), 1).to('cuda:0')  
            for i in indices:
                x1 = torch.cat([x1[:, :i], ones_column_1, x1[:, i:]], dim=1)
            ones_column_2 = torch.zeros(x2.size(0), 1).to('cuda:0')  
            for i in indices:
                x2 = torch.cat([x2[:, :i], ones_column_2, x2[:, i:]], dim=1)
            strs = []
            start_time_1 = time.time()
            for j in range(x1.size(0)):
                row_elements = [word_cache[i][x1[j][i].long()] for i in range(x1.size(1))]
                merged_row = ' '.join(row_elements)
                strs.append(merged_row)
            for j in range(x2.size(0)):
                row_elements = [word_cache[i][x2[j][i].long()] for i in range(x2.size(1))]
                merged_row = ' '.join(row_elements)
                strs.append(merged_row)
        elif word_filtered == 2:
            word_cache = list(self.BBM.word_substitution_cache)    
            word_cache_filtered = [row for row in word_cache if len(row) > 1]  
            word_cache = word_cache_filtered
            strs = []
            start_time_1 = time.time()
            for j in range(x1.size(0)):
                row_elements = [word_cache[i][x1[j][i].long()] for i in range(x1.size(1))]
                merged_row = ' '.join(row_elements)
                strs.append(merged_row)
            for j in range(x2.size(0)):
                row_elements = [word_cache[i][x2[j][i].long()] for i in range(x2.size(1))]
                merged_row = ' '.join(row_elements)
                strs.append(merged_row)
            end_time_1 = time.time()
            d_time = end_time_1 - start_time_1
        with GPU:
            rets = self.embed(strs)   
        ret_1 = rets[:x1.size(0)]   
        ret_2 = rets[x1.size(0):]
        ret_1 = tf.concat(ret_1, axis=0)
        ret_1 = torch.from_numpy(ret_1.numpy()).to('cuda:0')
        ret_2 = tf.concat(ret_2, axis=0)
        ret_2 = torch.from_numpy(ret_2.numpy()).to('cuda:0')

        return ret_1, ret_2