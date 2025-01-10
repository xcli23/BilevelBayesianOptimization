'''code version consistent with version v1 of the fastai library
follow installation instructions at https://github.com/fastai/fastai
'''
import fire
from warnings import filterwarnings
from  copy import deepcopy
filterwarnings("ignore")
import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from model_utils import *
from attack_codes.attack_util import get_data_and_learner, load_pretrained_model, fix_seed
from fastai.basic_train import loss_batch
import time
from attack_codes.gen_synonym import get_synonym
from attack_codes.greedy_attack import greedy_attack
from attack_codes.bayesian_attack import bayesian_attack
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import algorithms.glo as glo
import time
import contextlib
glo._init()
glo.set_value('bo_weight', 0)

kwargs_defaults = {
"working_folder":"datasets/clas_ec/clas_ec_ec50_level1", # folder with preprocessed data 
"model_filename_prefix":"model", # filename for saved model
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":400, # embedding size
"nh":1150, # number of hidden units
"nl":3, # number of layers

"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":128, # batch size

"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)

############ For attack ##############
"method": "bayesian",
"block_size": 20,
"max_patience": 20,
"fit_iter": 3,
"eval": 0,
"seed": 0,
"sidx": 0,
"num_seqs":50,
"save_key":"",
"Dataset": "IPS",
"Model_Type": "Normal"
}
debug = True

class Model(object):
    def generic_model(self, **kwargs):
        return generic_model(**kwargs)

    def languagemodel(self, **kwargs):
        return self.generic_model(clas=False, **kwargs)
    
    def classification(self, **kwargs):
        return self.generic_model(clas=True, **kwargs)

def generic_model(clas=True, **kwargs):
    kwargs["clas"]=clas
    for k in kwargs_defaults.keys():
        if(not( k in kwargs.keys()) or kwargs[k] is None):
            kwargs[k]=kwargs_defaults[k]
    
    WORKING_FOLDER = Path(kwargs["Dataset"])    
    ADV_DIR = get_adv_dir(WORKING_FOLDER,kwargs)
    #ADV_DIR.mkdir(exist_ok=True)
    if os.path.isdir(ADV_DIR):
        pass
    else:
        os.makedirs(ADV_DIR)
    fix_seed(seed=kwargs['seed'])

    target_seqs, vocab = get_data_and_learner(WORKING_FOLDER, kwargs, num_seqs=500) # get shuffled data
    model = load_pretrained_model(kwargs)
    model.eval()

    syndict = get_synonym(vocab, WORKING_FOLDER)

    BBM = BlackBoxModel(model, syndict)

    results = []

    kernellist = kwargs["kernel"].split('-')
    glo.set_value('kernellist',kernellist)
    kernel_num = len(kernellist)
    max_iter=kwargs["max_iter"]
    count_array = [0] * (max_iter + kernel_num - 1)
    threshold = kwargs["threshold"]
    filename = fr'{ADV_DIR}.txt'
    if not os.path.exists(filename):
        file_path = os.path.dirname(filename)
        os.makedirs(file_path, exist_ok=True)
        with open(filename, 'w') as file:
            pass
    failed_ids = []
    successes_idx = []
    skipped_idx = []
    start_time = time.time()
        
    if kwargs['eval']==1:
        orig_acc = get_accuracy(target_seqs, BBM) # accuracy of whole test dataset

        for key, l_ in syndict.items(): print(key, l_)
        avg_len = 0
        for i in range(kwargs['sidx'],kwargs['sidx'] + kwargs['num_seqs']):
            ADV_PATH = ADV_DIR/'{}.npy'.format(int(i))
            result = np.load(ADV_PATH,allow_pickle=True)
            results.append(result)
            xb_att = result[0]
            avg_len += xb_att.shape[1]
            print_att_info(kwargs['sidx']+i, result[3], result[4], result[5], result[6])
        avg_len = avg_len / kwargs['num_seqs']
        print(f"orig acc : {orig_acc}, avg len : {avg_len}")
    else:
        for idx, (xb,yb) in enumerate(target_seqs[int(kwargs['sidx']):int(kwargs['sidx']+kwargs['num_seqs'])]):
            ADV_PATH = ADV_DIR/'{}.npy'.format(int(kwargs['sidx'] + idx))
            try:
                result = np.load(ADV_PATH,allow_pickle=True)
                if result != None:
                    xadv, xb, yb, nq, modif, succ_count, elapsed_time, attack_logs = result
                    if succ_count == 1:
                        successes_idx.append(idx)
                    if succ_count == 0:
                        failed_ids.append(idx)
                    if succ_count == -1:
                        skipped_idx.append(idx)
                results.append(result)
            except:
                if kwargs['method'] == 'greedy': 
                    xb_att, num_queries, modif_rate, succ, elapsed_time, attack_logs = greedy_attack(xb, yb, syndict, BBM)
                elif kwargs['method'] == 'bayesian':
                    glo.set_value('model_params',None)
                    if kernel_num == 2:
                        bounds = [{'name': 'bo_weight', 'type': 'continuous', 'domain': (0, 1)}]
                        last_BO_result = None
                        BO_query = 0
                        iteration = 0
                        succ_count = None
                        initial_point = np.array([[0.5],[1],[0]])
                        score_array = [0] * 3
                        def attack_function(params):
                            nonlocal last_BO_result
                            nonlocal BO_query
                            nonlocal iteration
                            nonlocal succ_count
                            bo_weight = params[0]
                            iteration = iteration + 1
                            glo.set_value('bo_weight', bo_weight)
                            xb_att, inner_num_queries, modif_rate, succ, elapsed_time, attack_logs, y_prob = bayesian_attack(xb, yb, syndict, BBM, block_size=kwargs['block_size'], max_patience=kwargs['max_patience'], fit_iter=kwargs['fit_iter'], Dataset=kwargs['Dataset'])
                            BO_query += inner_num_queries
                            result = [xb_att.cpu().detach().numpy(), xb.cpu().detach().numpy(), yb.cpu().detach().numpy(), BO_query, modif_rate, succ, elapsed_time, attack_logs]
                            last_BO_result = result
                            succ_count = succ
                            with open(filename, 'a') as file:
                                if iteration < (kernel_num + 2):
                                    score_array[iteration - 1] = y_prob
                                    print(f'score_array:{score_array}')
                                    if all(score > threshold for score in score_array):
                                        print(f"attac-->{idx}, fail",file=file)
                                        raise TypeError
                                if succ == -1:
                                    print(f"attac-->{idx}, skip",file=file)
                                    raise TypeError
                                if succ == 1:
                                    print(f"attac-->{idx}, succ",file=file)
                                    count_array [iteration - 1] += 1
                                    raise TypeError
                                if succ == 0:
                                    print(f"attac-->{idx}, fail",file=file)
                            return y_prob
                        try:
                            model = BayesianOptimization(f=attack_function, domain=bounds, model_type='GP',X=initial_point, Y=None)
                            model.run_optimization(max_iter = max_iter-1)
                        except TypeError:
                            print('EarlyStop')
                        result = last_BO_result
                        num_queries = BO_query
                    if kernel_num >= 3: 
                        n = kernel_num - 1
                        bounds = [{'name': f'weight_{i}', 'type': 'continuous', 'domain': (0, 1)} for i in range(n)]
                        last_BO_result = None   
                        BO_query = 0
                        iteration = 0
                        succ_count = None
                        score_array = [0] * (kernel_num + 1)
                        def generate_lists(n):
                            first_list =np.array([[1 / (n- i + 1) for i in range(n)]])
                            first_n_lists = np.array([[1 if j == i else 0 for j in range(n)] for i in range(n)])
                            last_list = np.array([[0] * n])
                            arrays_to_concatenate = [first_list, first_n_lists, last_list]
                            combined_array = np.concatenate(arrays_to_concatenate, axis=0)
                            return combined_array
                        initial_point = generate_lists(n)
                        # initial_point= np.array([[1 / (n- i + 1) for i in range(n)]])
                        def convert_to_weights(alpha):
                            alpha = np.squeeze(alpha)
                            n = len(alpha) + 1
                            weights = np.zeros(n)
                            prod = 1.0
                            for i in range(n-1):
                                weights[i] = alpha[i] * prod
                                prod *= (1 - alpha[i])
                            weights[-1] = prod
                            return weights
                        def attack_function(params):
                            nonlocal last_BO_result
                            nonlocal BO_query
                            nonlocal iteration
                            nonlocal succ_count
                            bo_weight = convert_to_weights(params)
                            iteration = iteration + 1
                            glo.set_value('bo_weight', bo_weight)
                            xb_att, inner_num_queries, modif_rate, succ, elapsed_time, attack_logs, y_prob = bayesian_attack(xb, yb, syndict, BBM, block_size=kwargs['block_size'], max_patience=kwargs['max_patience'], fit_iter=kwargs['fit_iter'], Dataset=kwargs['Dataset'])
                            BO_query += inner_num_queries
                            result = [xb_att.cpu().detach().numpy(), xb.cpu().detach().numpy(), yb.cpu().detach().numpy(), BO_query, modif_rate, succ, elapsed_time, attack_logs]
                            last_BO_result = result
                            succ_count = succ
                            with open(filename, 'a') as file:
                                if iteration < (kernel_num + 2):
                                    score_array[iteration - 1] = y_prob
                                    if all(score > threshold for score in score_array):
                                        print(f"attac-->{idx}, fail",file=file)
                                        raise TypeError
                                if succ == -1:
                                    print(f"attac-->{idx}, skip",file=file)
                                    raise TypeError
                                if succ == 1:
                                    print(f"attac-->{idx}, succ",file=file)
                                    count_array [iteration - 1] += 1
                                    raise TypeError
                                if succ == 0:
                                    print(f"attac-->{idx}, fail",file=file)
                            return y_prob
                        try:
                            model = BayesianOptimization(f=attack_function, domain=bounds,model_type='GP',X=initial_point, Y=None )
                            model.run_optimization(max_iter = max_iter-1)
                        except TypeError:
                            print('EarlyStop')
                        result = last_BO_result
                        num_queries = BO_query
                    if kernel_num ==1 :
                        xb_att, num_queries, modif_rate, succ, elapsed_time, attack_logs, y_prob = bayesian_attack(xb, yb, syndict, BBM, block_size=kwargs['block_size'], max_patience=kwargs['max_patience'], fit_iter=kwargs['fit_iter'], Dataset=kwargs['Dataset'])
                        succ_count = succ
                        result = [xb_att.cpu().detach().numpy(), xb.cpu().detach().numpy(), yb.cpu().detach().numpy(), num_queries, modif_rate, succ, elapsed_time, attack_logs]
                    if succ_count == 1:
                        successes_idx.append(idx)
                    if succ_count == 0:
                        failed_ids.append(idx)
                    if succ_count == -1:
                        skipped_idx.append(idx)
                np.save(ADV_PATH, result, allow_pickle=True)
                results.append(result)
    asr, am, anq, at = evaluate_adv(results, BBM)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    with open(filename, 'a') as file:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print(f'count_array:{count_array}')
            print(f'attack_success_rate : {asr}%')
            print(f'avg_word_perturbed_perc : {am}%')
            print(f'avg_num_queries : {anq}')
            print(f"time : {hours}h_{minutes}m_{seconds}s")
        log_output = output.getvalue()
        
        file.write(log_output)
        print(log_output)
    print(f"Avg. Att. Succ. Rate : {asr}")
    print(f"Avg. Modif. Rate : {am}")
    print(f"Avg. Num Queries : {anq}")
    print(f"Avg. Elapsed Time : {at}")
    return [asr, am, anq, at]    

def get_accuracy(dataset, BBM):
    correct = 0
    t0 = time.time()
    # Original Accuracy
    score_ct = 0
    with torch.no_grad():
        val_losses, ybs = [], []
        for xb,yb in dataset:
            xb = xb.cpu().detach()
            yb = yb.cpu().detach()
            BBM.set_xy(xb,yb)
            _, y_pred = BBM.get_pred(xb)
            score = BBM.get_score(xb,require_transform=False)
            val_losses.append(y_pred)
            ybs.append(yb)
            if score > 0: score_ct += 1
    outs = torch.cat(val_losses, dim=0).cpu().detach()
    ybs = torch.cat(ybs).cpu().detach()
    correct = torch.sum((outs == ybs))
    Orig_Acc = (correct/outs.shape[0]).item()
    print(f"Original Accuracy via get_pred function : {100*Orig_Acc:.2f}%")
    print(f"Original Accuracy via get_score function : {100-score_ct/outs.shape[0]*100:.2f}")
    return Orig_Acc

def evaluate_adv(results, BBM):
    nql, modifl, succl, etl = [], [], [], []
    BBM.set_query_budget(float('inf'))
    filtered_results = []

    for i in range(len(results)):
        if results[i] != None:
            filtered_results.append(results[i])
    for xadv, xb, yb, nq, modif, succ, elapsed_time, attack_logs in filtered_results:
        BBM.set_xy(xb, yb)
        if BBM.get_score(xb, require_transform=False) >= 0:
            assert succ == -1, "something wrong"
        else:
            if BBM.get_score(xadv, require_transform=False) >= 0:
                assert succ == 1, "something wrong"
                modifl.append(modif)
            else:
                assert succ == 0, "something wrong"
            nql.append(nq)
            succl.append(succ)
            etl.append(elapsed_time)            
            
    asr = '{:.2f}'.format(sum(succl)/len(succl)*100)
    anq = '{:.1f}'.format(sum(nql)/len(nql))
    am = '{:.2f}'.format(sum(modifl)/len(modifl)*100)
    at = '{:.2f}'.format(sum(etl)/len(etl))
    return asr, am, anq, at

def get_adv_dir(working_folder, kwargs):
    if kwargs['method'] == 'bayesian':
        adv_dir = working_folder/kwargs['pkl_dir']/'BAYES{}_{}_{}_{}{}'.format(kwargs['seed'],kwargs['block_size'],kwargs['max_patience'],kwargs['fit_iter'],kwargs['save_key'])
    elif kwargs['method'] == 'greedy':
        adv_dir = working_folder/'GREEDY{}{}'.format(kwargs['seed'],kwargs['save_key'])
    else:
        raise ValueError
    return adv_dir

def print_att_info(index, qrs, mr, succ, elapsed_time):
    str_ = f'{index}th sample '
    if succ == -1:
        str_ += 'SKIPPED'
    elif succ == 0:
        str_ += 'FAILED\n'
        str_ += f"num queries : {qrs:.1f}"
    elif succ == 1:
        str_ += 'SUCCESSED\n'
        str_ += f"num queries : {qrs:.1f}, modif rate : {100*mr:.1f}"
    str_ += f"\nelapsed_time : {elapsed_time:.2f}"
    print(str_)
    
class BlackBoxModel():
    def __init__(self, model, syndict):
        self.eval_cache = dict()
        self.initialize_num_queries()
        self.model = model
        #self.cb_handler = cb_handler
        self.syndict = syndict
        self.query_budget = float('inf')

    def initialize_num_queries(self):
        self.clean_cache()
        self.num_queries = 0
    
    def clean_cache(self):
        del self.eval_cache
        self.eval_cache = dict()
    
    def set_query_budget(self, query_budget):
        self.query_budget=query_budget
    
    def set_xy(self, x0, y):
        if type(x0) == np.ndarray: x0 = torch.LongTensor(x0)
        if type(y) == np.ndarray: y = torch.LongTensor(y)
        self.x0 = x0
        self.y = y
        self.len_seq = x0.numel()
        self.word_substitution_cache = [[] for _ in range(self.len_seq)]
        for ind in range(self.len_seq):
            self.word_substitution_cache[ind] = deepcopy(self.syndict[x0[0][ind].cpu().item()])
        self.n_vertices = [len(w_candids) for w_candids in self.word_substitution_cache]
        self.target_indices = [ind for ind in range(self.len_seq) if self.n_vertices[ind]>1]
    
    def seq2input(self, seq):
        assert type(seq) == torch.Tensor, f"type(seq) is {type(seq)}"
        if len(seq.shape) == 1:
            assert seq.shape[0] == self.len_seq, "indices length should be seq length"
            seq_ = seq
        elif len(seq.shape) == 2:
            assert seq.shape[0] == 1 and seq.shape[1] == self.len_seq, "indices length should be seq length"
            seq_ = seq.view(-1)
        cur_seq = self.x0
        modified_indices = [ct for ct, ind in enumerate(seq_) if ind > 0 and ct in self.target_indices]
        words = [self.word_substitution_cache[ct][int(ind)] for ct, ind in enumerate(seq_) if ind > 0 and ct in self.target_indices]
        new_seq = self.replace_words_at_indices(cur_seq,modified_indices, words)
        return new_seq
    
    def replace_words_at_indices(self, seq, modified_indices, words):
        new_seq = deepcopy(seq)
        for ind, word in zip(modified_indices, words):
            new_seq[0][ind] = word
        return new_seq    

    def get_initial_block_order(self, inds_list):
        index_scores = []
        for inds in inds_list:
            start, end = self.target_indices[inds[0]], self.target_indices[inds[-1]]
            del_seq = deepcopy(self.x0)
            del_seq = torch.cat([del_seq[:,:start],del_seq[:,end+1:]],dim=-1)
            score = self.get_score(del_seq,require_transform=False)
            index_scores.append(score)
        index_scores = np.array(index_scores)
        index_order = (-index_scores).argsort()
        return index_order

    def seq2str(self, seq):
        assert seq.type() == 'torch.LongTensor', f"{seq.type()} should be 'torch.LongTensor'"
        seq_ = seq.view(-1).cpu().detach()
        str_ = ''
        for i in seq_:
            str_ += f'{int(i)},'
        return str_
        
    def get_pred(self, x):
        '''
            output:
                pred : 1 x nlabel tensor
                y_pred : 1 tensor
        '''
        if type(x) == np.ndarray: x = torch.LongTensor(x)
        if x.type() != torch.LongTensor: x = x.long()
        with torch.no_grad():
            x_str = self.seq2str(x)
            if x_str in self.eval_cache:
                val_loss = self.eval_cache[x_str]
            else:
                if self.num_queries >= self.query_budget: return None, None
                val_loss = self.model(x.cuda())
                #val_loss = loss_batch(self.model, x.cuda(), self.y.cuda(), cb_handler=self.cb_handler)
                self.eval_cache[x_str] = val_loss
                self.num_queries += 1
        return val_loss, torch.argmax(val_loss).view(1)
    
    def get_score(self,x,require_transform=True):
        if require_transform:
            x_ = self.seq2input(x.view(-1))
        else:
            x_ = x
        with torch.no_grad():
            pred, _ = self.get_pred(x_)
            if type(pred) == type(None): return None
            prob = torch.nn.functional.softmax(pred)
            y_ = self.y.cpu().detach().item()
            prob_del = torch.cat([prob[:,:y_],prob[:,y_+1:]],dim=-1)
            score = (torch.max(prob_del) - prob[0][self.y]).cpu().detach().item()
        return score
    
    def get_score_prob(self,x,require_transform=True):
        if require_transform:
            x_ = self.seq2input(x.view(-1))
        else:
            x_ = x
        with torch.no_grad():
            pred, _ = self.get_pred(x_)
            if type(pred) == type(None): return None
            prob = torch.nn.functional.softmax(pred)
            y_ = self.y.cpu().detach().item()
            prob_del = torch.cat([prob[:,:y_],prob[:,y_+1:]],dim=-1)
            score = (torch.max(prob_del) - prob[0][self.y]).cpu().detach().item()
        return score, prob[0][self.y]
    
    def get_scores(self, xs, require_transform=True):
        s = []
        for x in xs:
            s.append(self.get_score(x,require_transform=require_transform))
        return s

if __name__ == '__main__':
    fire.Fire(Model)
