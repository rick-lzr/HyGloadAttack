import argparse
import os
import numpy as np
from pathlib import Path
from scipy.special import softmax
np.random.seed(1234)
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
import random
random.seed(0)
import csv
import time

import math
import json

import joblib
import sys
csv.field_size_limit(sys.maxsize)

import tensorflow_hub as hub

import tensorflow.compat.v1 as tf
import copy
import tensorflow as my_tf2
from tqdm import trange
tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from sklearn.cluster import KMeans
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig, BertForSequenceClassification_embed

tf.compat.v1.disable_eager_execution()

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        print("embed")
        # self.embed = hub.Module(module_url)
        # 原语句加载模型失效。解决加载失效问题。 这里加载的v5版本。原语句加载的v4版本。
        self.embed = my_tf2.saved_model.load("[path to HyGloadAttack dir]/dependencies/others/USE_cache/usel5")
        print("embed ok")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
        # 下面是OpenAttack的相似度计算方式，所有baseline采用上方的cosine_similarity计算相似度
        # self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores



class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        if torch.cuda.is_available():
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)

        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):

        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)
    
class NLI_infer_embed_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_embed_BERT, self).__init__()
        if torch.cuda.is_available():
            self.model = BertForSequenceClassification_embed.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification_embed.from_pretrained(pretrained_dir, num_labels=nclasses)

        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):

        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()

            with torch.no_grad():
                pooled_output,logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return pooled_output,torch.cat(probs_all, dim=0)



class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):


    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)


            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):

    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text
    batch_size = 16
    total_semantic_sims = np.array([])
    for i in range(0, len(new_texts), batch_size):
        batch = new_texts[i:i+batch_size]
        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), batch)))[0]
        total_semantic_sims = np.concatenate((total_semantic_sims, semantic_sims))
    return total_semantic_sims

def get_attack_result(new_text, predictor, orig_label, batch_size):
    '''
        查看attack是否成功
        return: true 攻击成功
                false 攻击失败
    '''
    new_probs = predictor(new_text, batch_size=batch_size)
    pr=(orig_label!= torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr


def soft_threshold(alpha, beta):
    '''
    lammda l1
    gamma[i][0] γ
    软阈值 lammda l1 ,gamma[i][0] γ 让gradient_2 小于 -0.1则+0.1 大于0.1则-0.1
    '''
    if beta > alpha:
        return beta - alpha
    elif beta < -alpha:
        return beta + alpha
    else:
        return 0
def get_word_embed(words_perturb_doc_idx,text,word_idx_dict,embed_content):
    text_embed = []
    for idx in words_perturb_doc_idx:
        text_embed.append(
            [float(num) for num in embed_content[word_idx_dict[text[idx]]].strip().split()[1:]])
    text_embed_matrix = np.asarray(text_embed)
    return text_embed_matrix

def normalize_min_max(distance, min_value, max_value):
    return (distance - min_value) / (max_value - min_value)

def combined_similarity(vector, vectors, k=3):
    vector_reshape = vector.reshape(-1)
    vectors_reshape = vectors.reshape(vectors.shape[0],vector_reshape.shape[0])
    cosine_similarities = np.dot(vectors_reshape, vector_reshape) / (np.linalg.norm(vector_reshape) * np.linalg.norm(vectors_reshape, axis=1))
    cosine_similarities = 0.5 * (cosine_similarities + 1)
    # 耗时特短
    combined_scores = cosine_similarities
    sorted_score = np.argsort(combined_scores)
    if len(sorted_score) == 1:
        most_similar_index = sorted_score[-1]
    else:
        # most_similar_index = sorted_score[-1-k:-1]
        most_similar_index = sorted_score[-1-k:-1]
    
    return most_similar_index, combined_scores

def remove_duplicate_lists(input_list):
    # 使用集合来存储唯一的子列表
    unique_list_set = set(tuple(sublist) for sublist in input_list)
    # 将集合转换回列表，并返回结果
    unique_list = [list(sublist_tuple) for sublist_tuple in unique_list_set]
    return unique_list

def get_shortest_text(predictor,text,orig_label,batch_size,best_changed_num,text_ls):
    replaced_text = []
    for i in range(len(text_ls)):
        tmp_text = copy.deepcopy(text)
        if text_ls[i] != text[i]:
            tmp_text[i] = text_ls[i]
            replaced_text.append(tmp_text)
    prs = get_attack_result(replaced_text, predictor, orig_label, batch_size)
    adv_text = []
    for i in range(len(prs)):
        if np.sum(prs[i]) >= 0:
            adv_text.append(replaced_text[i])
    
    


def HyGload_attack(
            fuzz_val, top_k_words,  sample_index, text_ls,
            true_label, predictor, stop_words_set, word2idx, idx2word,
            cos_sim, sim_predictor=None, import_score_threshold=-1.,
            sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
            batch_size=32,embed_func = '',budget=1000,myargs=None):
    '''
        HyGload_attack(
            top_k_words, 选取topk words
            text_ls, text 真实的需要攻击的text
            true_label, 
            predictor, 
            word2idx,
            idx2word,
            cos_sim, 
            sim_predictor=None, 
            sim_score_window=15,
            batch_size=32,
            embed_func:embed function
            budget:查询预算)

        predictor: 预测器
        @return: 
            ' '.join(best_attack), max_changes, len(changed_indices), \
            orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
            new_text, num_changed, random_changed, \
            orig_label, new_label, num_queries, sim, random_sim
        HyGload_attack(
            args.fuzz, args.top_k_words, idx, text,
            true_label, predictor, stop_words_set, word2idx, idx2word,
            sim_lis , sim_predictor=use, sim_score_threshold=args.sim_score_threshold,
            import_score_threshold=args.import_score_threshold,
            sim_score_window=args.sim_score_window,
            synonym_num=args.synonym_num,
            batch_size=args.batch_size,
            embed_func = args.counter_fitting_embeddings_path,
            budget=args.budget)
    '''
    # print("myargs:",myargs.budget)
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        # word2idx 构建
        word_idx_dict={}
        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index
        # word：[embed -0.022007 -0.05519 0.02872 0.068785 xxxx]
        embed_file=open(embed_func)
        embed_content=embed_file.readlines()
        # 获取单词的词性pos_ls = ['NOUN','VERB','NOUN','VERB']
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1
        
        # find ["ADJ", "ADV", "VERB", "NOUN"] in text
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        # for pos in pos_pref:
        #     for i in range(len(pos_ls)):
        #         if pos_ls[i] == pos and len(text_ls[i]) > 2:
        #             words_perturb.append((i, text_ls[i]))
        for i in range(len(pos_ls)):
            if pos_ls[i] in pos_pref and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]

        # get words perturbed idx embed doc_idx.find synonyms and make a dict of synonyms of each word.
        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ].strip().split()[1:]])

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)

        # 干扰的同义词选取 
        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms
        
        
        qrs = 0
        num_changed = 0
        flag = 0
        th = 0 # thershold 

        # Initialization
        # random initialize result to attack model
        # TODO: 直接拉满pert 而后search reduction。再并集
        # 拉满pert
        random_sample_adv_n = 1 # 设置初始化有几个advsample
        random_adv_sampled_n = 0
        random_adv_samples = []
        for _ in range(2500):
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], predictor, orig_label, batch_size)
            qrs+=1
            if np.sum(pr)>0:
                random_adv_samples.append(random_text)
                flag = 1
                changes = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=random_text[i]:
                        changes+=1
                # print("changes",changes,"|sentence:"," ".join(random_text))
                random_adv_sampled_n+=1
            if random_adv_sampled_n>=random_sample_adv_n:
                break
        # 如果全量没找到
        if not np.sum(pr)>0:
            prepared_flag = 0
            old_qrs = qrs
            while qrs < len(text_ls)+old_qrs:
            # for i in range(len(text_ls)*2):
                prepared_text = text_ls[:]
                for i in range(len(synonyms_all)):
                    idx = synonyms_all[i][0]
                    syn = synonyms_all[i][1]
                    prepared_text[idx] = random.choice(syn[:])
                    if i >= th:
                        break
                pr = get_attack_result([prepared_text], predictor, orig_label, batch_size)
                qrs+=1
                th +=1
                if th >= len_text or th >= len(synonyms_all):
                    th = 0
                if np.sum(pr)>0:
                    prepared_flag = 1
                    break
            if prepared_flag == 1:
                # print("into local search ")
                random_adv_samples.append(prepared_text)
            else:
                return ' '.join(random_text), 0, 0, \
                        orig_label, orig_label, qrs, 0, 0
            return ' '.join(random_text), 0, 0, \
                    orig_label, orig_label, qrs, 0, 0
        # TODO:my search reduction
        random_adv_reducted = []
        for i in random_adv_samples:
            while True:
                choices = []
                pert_index = []
                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        pert_index.append(i)
                # For each word substituted in the original text, change it with its original word and compute
                # the change in semantic similarity.
                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        qrs+=1
                        pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        if np.sum(pr) > 0:
                            choices.append((i,semantic_sims[0]))

                # Sort the relacements by semantic similarity and replace back the words with their original
                # counterparts till text remains adversarial.
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        qrs+=1
                        if pr[0] == 0:
                            continue
                        random_text[choices[i][0]] = text_ls[choices[i][0]]
                # 
                if len(choices) == 0:
                    break
            random_adv_reducted.append(random_text)
        # # TODO:求pert交集
        combined_list = []
        # print(None in combined_list)
        for items in zip(*random_adv_reducted):
            if len(set(items)) == 1:
                combined_list.append(items[0])
            else:
                combined_list.append(items[0])  # Or some marker for different values
        # 重新对交集部分进行随机替换
        random_text = combined_list[:]
        # 
        if flag == 1:
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1

            # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
            while True:
                choices = []
                # For each word substituted in the original text, change it with its original word and compute
                # the change in semantic similarity.
                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        qrs+=1
                        pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        if np.sum(pr) > 0:
                            choices.append((i,semantic_sims[0]))
                # Sort the relacements by semantic similarity and replace back the words with their original
                # counterparts till text remains adversarial.
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        qrs+=1
                        if pr[0] == 0:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]
                # 
                if len(choices) == 0:
                    break
            # 
            changed_indices = []
            num_changed = 0

            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
            # print("changed:",str(num_changed)+"\tqrs"+str(qrs))
            random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]

            # out of budget
            if qrs > budget:
                return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                    orig_label, torch.argmax(predictor([random_text])), qrs, random_sim, random_sim
            
            best_attack = random_text
            best_sim = random_sim
            old_random_sim = copy.deepcopy(random_sim)
            old_random_text = copy.deepcopy(random_text)
            # if num changed == 1
            if np.sum(get_attack_result([random_text], predictor, orig_label, batch_size)) > 0 and (num_changed == 1):
                change_idx = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=random_text[i]:
                        change_idx = i
                        break
                idx = word2idx[text_ls[change_idx]]
                res = list(zip(*(cos_sim[idx])))
                x_ts = []
                for widx in res[1]:
                    w = idx2word[widx]
                    random_text[change_idx] = w
                    x_ts.append(random_text[:])
                prs = get_attack_result(x_ts, predictor, orig_label, batch_size)
                sims = calc_sim(text_ls, x_ts, -1, sim_score_window, sim_predictor)

                is_update_random_attack = False
                for x_t_, pr, sim in zip(x_ts, prs, sims):
                    qrs += 1
                    if np.sum(pr) > 0 and sim >= best_sim:
                        best_attack = x_t_[:]
                        best_sim = sim
                        is_update_random_attack = True
                if is_update_random_attack:
                    return ' '.join(best_attack), 1, 1, \
                        orig_label, torch.argmax(predictor([best_attack])), qrs, best_sim, best_sim
                else:
                    # print("old_random_text:",get_attack_result([old_random_text], predictor, orig_label, batch_size))
                    return ' '.join(old_random_text), 1, 1, \
                        orig_label, torch.argmax(predictor([old_random_text])), qrs, old_random_sim, old_random_sim
            # STEP 3: Optimization
            # Optimization Procedure
            random_adv_embed = []
            for idx in words_perturb_doc_idx:
                random_adv_embed.append([float(num) for num in embed_content[word_idx_dict[random_text[idx]]].strip().split()[1:]])
            random_adv_embed_matrix = np.asarray(random_adv_embed)
            random_2_text_ls_dist = np.sum((random_adv_embed_matrix)**2)
            is_update_best_attack = True
            current_pert_num = None
            for t in trange(100):

                theta_old_text = best_attack
                
                changes = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=best_attack[i]:
                        changes+=1
                # 是否更新了best_attack
                if is_update_best_attack:
                    # print(" ".join(best_attack))
                    # print("changes",changes,"|sentence:"," ".join(best_attack))
                    old_adv_embed = []
                    for idx in words_perturb_doc_idx:
                        old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]].strip().split()[1:]])
                    old_adv_embed_matrix = np.asarray(old_adv_embed)
                    # P0 = E0 − E
                    theta_old = old_adv_embed_matrix-words_perturb_embed_matrix
                    theta_old_2_text_ls_dist = np.sum((theta_old)**2)
                    dont_use_v = []
                    opt_stack = [] # 优化历史栈，记录push过的sentence
                    is_update_best_attack = False
                    # TODO:构建搜索空间，降重了，不会有重复的query 。
                    # TODO:标识每组 nonzero_ele 的 text。当当前text不可用时，当前text产生时前面的text也不可用。
                    # TODO:验证text的按数量计算的可用性。
                    theta_old_neighbor_text_search_space = []
                    theta_old_neighbor_text_search_space_dic = {}
                    for_nums = 300
                    nonzero_ele = np.nonzero(np.linalg.norm(theta_old,axis = -1))[0].tolist()
                    # p_sim = []
                    # for i in nonzero_ele:
                    #     tmp_text = copy.deepcopy(best_attack)
                    #     tmp_text[synonyms_all[i][0]] = text_ls[synonyms_all[i][0]]
                    #     sim = calc_sim(text_ls, [tmp_text], -1, sim_score_window, sim_predictor)[0]
                    #     p_sim.append(sim-best_sim)
                    for _ in range(for_nums):
                        # print(changes)
                        # V = P + βU
                        # random perturb
                        u_vec = np.random.normal(loc=0.0, scale=1,size=theta_old.shape)
                        # 随机生成0.8-1.2之间的单个数字 动态耗时巨大且收益不高
                        # theta_old_neighbor = theta_old+0.5*u_vec/random_2_text_ls_dist*theta_old_2_text_ls_dist*random.uniform(0.9, 1.1)
                        theta_old_neighbor = theta_old+0.5*u_vec*random.uniform(0.9, 1.1)
                        # theta_perturb_dist 距离
                        theta_perturb_dist = np.sum((theta_old_neighbor)**2, axis=1)
                        # nonzero_ele = np.nonzero(np.linalg.norm(theta_perturb_dist, axis = -1))[0].tolist()
                        perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])
                        # Optimizing ρi
                        theta_old_neighbor_text = text_ls[:]
                        # for changed words to get 
                        # print("nonzero1:",len(nonzero_ele),nonzero_ele)
                        pert_num = 1
                        theta_perturb_dist = np.sum((theta_old)**2, axis=1)
                        perturb_word_idx_list = []
                        perturb_word_idx_list = nonzero_ele
                        for perturb_idx in range(len(perturb_word_idx_list)):
                            
                            perturb_word_idx = perturb_word_idx_list[perturb_idx]
                            # find the replaceable words
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_old_neighbor[perturb_word_idx]
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat_set.append(syn_feat)

                            # find the neighbour synonyms words
                            # syn_feat_set - perturb_target is the target
                            perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                            # TODO:这里需要深拷贝才能构建，但是是否真的需要深拷贝呢？我们直接采用最大深度的数据是否更好？
                            # 已证明，深拷贝采用更大深度数据不一定好，但是不深拷贝可以减少 set后的影响。qrs增加了，但是pert和sim都更优化了。
                            theta_old_neighbor_text[synonyms_all[perturb_word_idx][0]] = replacement
                            # 必须要先扰动再判断绕过。
                            # if not (pert_num == changes and pert_num == changes-1):
                            if not (pert_num == changes):
                                pert_num+=1
                                continue
                            # 采用
                            tmp = copy.deepcopy(theta_old_neighbor_text)
                            # 采用最大深度方式
                            # tmp = theta_old_neighbor_text
                            theta_old_neighbor_text_search_space.append(tmp)

                            if pert_num not in theta_old_neighbor_text_search_space_dic.keys():
                                theta_old_neighbor_text_search_space_dic[pert_num] = []
                            if tmp not in theta_old_neighbor_text_search_space_dic[pert_num]:
                                theta_old_neighbor_text_search_space_dic[pert_num].append([tmp,u_vec])
                            pert_num+=1
                    # print("changesd")
                else:
                    pass
                
                if current_pert_num is None:
                    current_pert_num = list(theta_old_neighbor_text_search_space_dic.keys())[-1]
                else:
                    current_pert_num = changes
                if current_pert_num == 0:
                    max_changes = 0
                    for i in range(len(text_ls)):
                        if text_ls[i]!=best_attack[i]:
                            max_changes+=1
                    return ' '.join(best_attack), max_changes, len(changed_indices), \
                            orig_label, torch.argmax(predictor([best_attack])), qrs, best_sim, random_sim
                # for _ in range(int((int(math.sqrt(num_changed))+2)*10)):
                for _ in range(num_changed*3):
                    # 更新搜索pert num 空间
                    # TODO:
                    # 选点
                    if current_pert_num==1:
                        current_pert_num_tmp = current_pert_num
                    else:
                        # current_pert_num_tmp = current_pert_num -1
                        current_pert_num_tmp = current_pert_num
                    # print(search_space_th_tmp)
                    while True:
                        try:
                            item = theta_old_neighbor_text_search_space_dic[current_pert_num_tmp].pop()
                            u_vec = item[1]
                            theta_old_neighbor_text = item[0]
                            break
                        except Exception as e:
                            if len(theta_old_neighbor_text_search_space_dic[current_pert_num_tmp])==0 and \
                                len(theta_old_neighbor_text_search_space_dic[current_pert_num])==0:
                                theta_old_neighbor_text = best_attack
                                # print("in while ")
                                # 解除注释后 性能会更好
                                # if theta_old_neighbor_text in opt_stack:
                                #     sim = best_sim
                                #     max_changes = 0
                                #     for i in range(len(text_ls)):
                                #         if text_ls[i]!=best_attack[i]:
                                #             max_changes+=1
                                #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                                #             orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                                is_update_best_attack = True
                                opt_stack.append(theta_old_neighbor_text)
                                break
                            current_pert_num_tmp = current_pert_num

                    if " ".join(theta_old_neighbor_text) in dont_use_v:
                        continue
                    tmp_v_list = [row[1] for row in theta_old_neighbor_text_search_space_dic[current_pert_num_tmp]]
                    tmp_t_list = [row[0] for row in theta_old_neighbor_text_search_space_dic[current_pert_num_tmp]]

                    pr = get_attack_result([theta_old_neighbor_text], predictor, orig_label, batch_size)
                    
                    qrs+=1

                    if qrs > budget:
                        sim = best_sim
                        max_changes = 0
                        for i in range(len(text_ls)):
                            if text_ls[i]!=best_attack[i]:
                                max_changes+=1

                        return ' '.join(best_attack), max_changes, len(changed_indices), \
                            orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                    if np.sum(pr)>0:
                        break
                    if len(tmp_v_list)>=1:
                        most_similar_index, combined_scores = combined_similarity(u_vec,np.asarray(tmp_v_list))
                        if type(most_similar_index)==type(np.array(0)):
                            for i in most_similar_index:
                                dont_use_v.append(" ".join(tmp_t_list[i]))
                        else:
                            dont_use_v.append(" ".join(tmp_t_list[most_similar_index]))
                if np.sum(pr)<=0:
                    if len(tmp_v_list)>=1:
                        most_similar_index, combined_scores = combined_similarity(u_vec,np.asarray(tmp_v_list))
                        if type(most_similar_index)==type(np.array(0)):
                            for i in most_similar_index:
                                dont_use_v.append(" ".join(tmp_t_list[i]))
                        else:
                            dont_use_v.append(" ".join(tmp_t_list[most_similar_index]))
                    continue
                
                changes = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=best_attack[i]:
                        changes+=1
                # print("--------remove unnecessary words----")
                # -----------remove unnecessary words-------------
                while True:
                    choices = []
                    pert_index = []
                    for i in range(len(text_ls)):
                        if theta_old_neighbor_text[i] != text_ls[i]:
                            pert_index.append(i)
                    # For each word substituted in the original text, change it with its original word and compute
                    # the change in semantic similarity.
                    for i in range(len(text_ls)):
                        if theta_old_neighbor_text[i] != text_ls[i]:
                            new_text = theta_old_neighbor_text[:]
                            new_text[i] = text_ls[i]
                            semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                            qrs+=1
                            pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                            if np.sum(pr) > 0:
                                choices.append((i,semantic_sims[0]))

                    # Sort the relacements by semantic similarity and replace back the words with their original
                    # counterparts till text remains adversarial.
                    if len(choices) > 0:
                        choices.sort(key = lambda x: x[1])
                        choices.reverse()
                        for i in range(len(choices)):
                            new_text = theta_old_neighbor_text[:]
                            new_text[choices[i][0]] = text_ls[choices[i][0]]
                            pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                            qrs+=1
                            if pr[0] == 0:
                                break
                            theta_old_neighbor_text[choices[i][0]] = text_ls[choices[i][0]]
                    # 
                    if len(choices) == 0:
                        break
                # -----------remove unnecessary words-------------

                # -----------if change num == 1 -------------
                x_t = theta_old_neighbor_text
                num_changed = 0
                for i in range(len(text_ls)):
                    if text_ls[i] != x_t[i]:
                        num_changed += 1

                x_t_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                # if attack success and num changed ==1
                if np.sum(get_attack_result([x_t], predictor, orig_label, batch_size)) > 0 and (num_changed == 1):
                    change_idx = 0
                    for i in range(len(text_ls)):
                        if text_ls[i]!=x_t[i]:
                            change_idx = i
                            break
                    idx = word2idx[text_ls[change_idx]]
                    res = list(zip(*(cos_sim[idx])))
                    x_ts = []
                    for widx in res[1]:
                        w = idx2word[widx]
                        x_t[change_idx] = w
                        x_ts.append(x_t[:])
                    prs = get_attack_result(x_ts, predictor, orig_label, batch_size)
                    sims = calc_sim(text_ls, x_ts, -1, sim_score_window, sim_predictor)
                    for x_t_, pr, sim in zip(x_ts, prs, sims):
                        qrs += 1
                        if np.sum(pr) > 0 and sim >= best_sim:
                            best_attack = x_t_[:]
                            is_update_best_attack = True
                            best_sim = sim
                    return ' '.join(best_attack), 1, len(changed_indices), \
                        orig_label, torch.argmax(predictor([best_attack])), qrs, best_sim, random_sim
                # -----------if change num == 1 -------------

                x_t = theta_old_neighbor_text
                x_t_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                # if attack success and is best sim
                if np.sum(get_attack_result([x_t], predictor, orig_label, batch_size)) > 0 and x_t_sim >= best_sim:
                    best_attack = x_t[:]
                    is_update_best_attack = True
                    best_sim = x_t_sim
                # get adv embed
                x_t_adv_embed = []
                for idx in words_perturb_doc_idx:
                    x_t_adv_embed.append(
                        [float(num) for num in embed_content[word_idx_dict[x_t[idx]]].strip().split()[1:]])
                x_t_adv_embed_matrix = np.asarray(x_t_adv_embed)
                x_t_pert = x_t_adv_embed_matrix - words_perturb_embed_matrix
                x_t_perturb_dist = np.sum((x_t_pert) ** 2, axis=1)
                nonzero_ele = np.nonzero(np.linalg.norm(x_t_pert, axis=-1))[0].tolist()
                perturb_word_idx_list = nonzero_ele
                for i in range(1):
                    # 生成可替换的sample
                    replaced_txt = []
                    for i in perturb_word_idx_list:
                        for j in synonyms_all[i][1]:
                            tmp_txt = copy.deepcopy(x_t)
                            tmp_txt[synonyms_all[i][0]] = j
                            replaced_txt.append(tmp_txt)
                    # 计算sim
                    sims = calc_sim(text_ls, replaced_txt, -1, sim_score_window, sim_predictor)
                    # 过滤低sim
                    candi_samples_filter = []
                    for i in range(len(replaced_txt)):
                        if sims[i]>=best_sim:
                            candi_samples_filter.append(replaced_txt[i])
                    # 倒排sim找最大sim
                    filtered_sorted_sim = np.argsort(-sims[sims>=best_sim])
                    for i in filtered_sorted_sim:
                        pr = get_attack_result([candi_samples_filter[i]], predictor, orig_label, batch_size)
                        qrs+=1
                        if np.sum(pr) > 0:
                            best_attack = candi_samples_filter[i]
                            is_update_best_attack = True
                            best_sim = sims[sims>=best_sim][i]
                            break
                        if qrs >= budget:
                            max_changes = 0
                            for i in range(len(text_ls)):
                                if text_ls[i]!=best_attack[i]:
                                    max_changes+=1
                            return ' '.join(best_attack), 1, max_changes, \
                                orig_label, torch.argmax(predictor([best_attack])), qrs, best_sim, random_sim
                
            sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
            max_changes = 0

            for i in range(len(text_ls)):
                if text_ls[i]!=best_attack[i]:
                    max_changes+=1
            
            return ' '.join(best_attack), max_changes, len(changed_indices), \
                  orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

        else:
            print("Not Found")
            return '', 0,0, orig_label, orig_label, 0, 0, 0


def main():
    # if True 方便看代码
    if True:
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset_path",
                            type=str,
                            # required=True,
                            default="[path to HyGloadAttack dir]/data/mr",
                            help="Which dataset to attack.")
        parser.add_argument("--nclasses",
                            type=int,
                            default=2,
                            help="How many classes for classification.")
        parser.add_argument("--target_model",
                            type=str,
                            # required=True,
                            default="wordCNN",
                            choices=['wordLSTM', 'bert', 'wordCNN'],
                            help="Target models for text classification: fasttext, charcnn, word level lstm "
                                "For NLI: InferSent, ESIM, bert-base-uncased")
        parser.add_argument("--target_model_path",
                            type=str,
                            # required=True,
                            default="[path to HyGloadAttack dir]/dependencies/models/cnn/mr",
                            help="pre-trained target model path")
        parser.add_argument("--word_embeddings_path",
                            type=str,
                            default='[path to HyGloadAttack dir]/dependencies/others/glove.6B.200d.txt',
                            help="path to the word embeddings for the target model")
        parser.add_argument("--counter_fitting_embeddings_path",
                            type=str,
                            default="[path to HyGloadAttack dir]/dependencies/others/counter-fitted-vectors.txt",
                            help="path to the counter-fitting embeddings we used to find synonyms")
        parser.add_argument("--counter_fitting_cos_sim_path",
                            type=str,
                            default='[path to HyGloadAttack dir]/dependencies/others/mat.txt',
                            help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
        parser.add_argument("--USE_cache_path",
                            type=str,
                            # required=True,
                            default="[path to HyGloadAttack dir]/dependencies/others/USE_cache",
                            help="Path to the USE encoder cache.")
        parser.add_argument("--output_dir",
                            type=str,
                            default='kmeans_adv_results',
                            help="The output directory where the attack results will be written.")

        parser.add_argument("--sim_score_window",
                            default=40,
                            type=int,
                            help="Text length or token number to compute the semantic similarity score")
        parser.add_argument("--import_score_threshold",
                            default=-1.,
                            type=float,
                            help="Required mininum importance score.")
        parser.add_argument("--sim_score_threshold",
                            default=0.7,
                            type=float,
                            help="Required minimum semantic similarity score.")
        parser.add_argument("--synonym_num",
                            default=50,
                            type=int,
                            help="Number of synonyms to extract")
        parser.add_argument("--batch_size",
                            default=32,
                            type=int,
                            help="Batch size to get prediction")
        parser.add_argument("--data_size",
                            default=1000,
                            type=int,
                            help="Data size to create adversaries")
        parser.add_argument("--perturb_ratio",
                            default=0.,
                            type=float,
                            help="Whether use random perturbation for ablation study")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="max sequence length for BERT target model")
        parser.add_argument("--target_dataset",
                            default="imdb_test",
                            type=str,
                            help="Dataset Name")
        parser.add_argument("--fuzz",
                            default=0,
                            type=int,
                            help="Word Pruning Value")
        parser.add_argument("--top_k_words",
                            default=1000000,
                            type=int,
                            help="Top K Words")
        
        parser.add_argument("--budget",
                            type=int,
                            # required=True,
                            default=15000,
                            help="Number of Budget Limit")
        
        parser.add_argument("--test_dataset",
                            default="ag",
                            type=str,
                            help="所测试数据集的名称")
        parser.add_argument("--test_len",
                            default=1000,
                            type=int,
                            help="数据集测试长度")


    args = parser.parse_args()
    #TODO: 这里根据不同的数据集选择不同的参数 不同的模型 不同的数据输入 不同的n_class 这里需要配置自己的路径
    model_dic = {
        "wordLSTM":"lstm",
        "bert":"bert",
        "wordCNN":"cnn"
    }
    dataset_info_dic = {
        "imdb":{
            "dataset_path":"[path to HyGloadAttack dir]/data/imdb",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/imdb".format(model_dic[args.target_model]),
            "n_classes":2
        },
        "yelp":{
            "dataset_path":"[path to HyGloadAttack dir]/data/yelp",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/yelp".format(model_dic[args.target_model]),
            "n_classes":2
        },
        "yahoo":{
            "dataset_path":"[path to HyGloadAttack dir]/data/yahoo",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/yahoo".format(model_dic[args.target_model]),
            "n_classes":10
        },
        "ag":{
            "dataset_path":"[path to HyGloadAttack dir]/data/ag",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/ag".format(model_dic[args.target_model]),
            "n_classes":4
        },
        "mr":{
            "dataset_path":"[path to HyGloadAttack dir]/data/mr",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/mr".format(model_dic[args.target_model]),
            "n_classes":2
        },
        "imdb_test":{
            "dataset_path":"[path to HyGloadAttack dir]/data/mr",
            "target_model_path":"[path to HyGloadAttack dir]/dependencies/models/{}/mr".format(model_dic[args.target_model]),
            "n_classes":2
        }
    }
    args.target_model_path = dataset_info_dic[args.test_dataset]["target_model_path"]
    args.dataset_path = dataset_info_dic[args.test_dataset]["dataset_path"]
    args.nclasses = dataset_info_dic[args.test_dataset]["n_classes"]


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    texts, labels = dataloader.read_corpus(args.dataset_path,csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size]
    print("Data import finished!")

    print("Building Model...")

    if args.target_model == 'wordLSTM':
        if torch.cuda.is_available():
            model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
            model.load_state_dict(checkpoint)
        else:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses)
            checkpoint = torch.load(args.target_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        if torch.cuda.is_available():
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
            checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
            model.load_state_dict(checkpoint)
        else:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True)
            checkpoint = torch.load(args.target_model_path,map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    idx2word = {}
    word2idx = {}
    sim_lis=[]

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]

                embeddings.append(embedding)

        embeddings = np.array(embeddings,dtype='float64')
        embeddings = embeddings[:30000]


        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)

    print("Cos sim import finished!")
    # if True 方便看代码
    if True:
        use = USE(args.USE_cache_path)

        orig_failures = 0.
        adv_failures = 0.
        avg=0.
        tot = 0
        elapsed_times = []
        changed_rates = []
        nums_queries = []
        orig_texts = []
        adv_texts = []
        true_labels = []
        new_labels = []
        wrds=[]
        s_queries=[]
        f_queries=[]
        success=[]
        results=[]
        fails=[]
        final_sims = []
        random_sims = []
        random_changed_rates = []
        log_dir = "kmeans_results_hard_label/"+args.target_model+"/"+args.test_dataset
        res_dir = "kmeans_results_hard_label/"+args.target_model+"/"+args.test_dataset
        log_file = "kmeans_results_hard_label/"+args.target_model+"/"+args.test_dataset+"/log.txt"
        result_file = "kmeans_results_hard_label/"+args.target_model+"/"+args.test_dataset+"/results_final.csv"
        process_file = os.path.join(log_dir,"sampled_process_log.txt")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        stop_words_set = criteria.get_stopwords()
        print('Start attacking!')
    res = []
    for idx, (text, true_label) in enumerate(data[:args.test_len]):
        

        '''
        text = ["i'm", 'convinced', 'i', 'could', 'keep', 'a', 'family', 
        'of', 'five', 'blind', ',', 'crippled', ',', 'amish', 'people',
        'alive', 'in', 'this', 'situation', 'better', 'than', 'these', 
        'british', 'soldiers', 'do', 'at', 'keeping', 'themselves', 'kicking']
        true_label = 0
        '''
        if idx % 20 == 0:
            print(str(idx)+" Samples Done. success num:{} . changed_rates:{}. final_sims:{}. qrs:{}. elapsed_time:{}.".format(len(success),np.mean(changed_rates),
                                                                                                            np.mean(final_sims),np.mean(nums_queries),
                                                                                                            np.mean(elapsed_times)))
            # print(final_sims)
            # with open(os.path.join(log_dir, 'sampled_processd.txt'), 'a') as ofile:
            #     ofile.write('{} Samples Done \t final_sims :{}\tnums_queries :{}\t avg changed rate: {}%\n'.format(str(idx),np.mean(final_sims),
            #                                                                                                       np.mean(nums_queries),np.mean(changed_rates)*100))
        # print(" ".join(text))
        start_time = time.time()
        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim = HyGload_attack(args.fuzz,args.top_k_words,
                                            idx,text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, sim_lis , sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,myargs=args,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size,embed_func = args.counter_fitting_embeddings_path,budget=args.budget,
                                            )
        end_time = time.time()
        elapsed_time = end_time - start_time

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1
        changed_rate = 1.0 * num_changed / len(text)
        random_changed_rate = 1.0 * random_changed / len(text)
        if true_label == orig_label and true_label != new_label:
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            random_changed_rates.append(random_changed_rate)
            random_sims.append(random_sim)
            if type(sim) == type([]):
                sim = sim[0]
            final_sims.append(sim)
            temp.append(idx)
            temp.append(orig_label)
            temp.append(new_label)
            temp.append(' '.join(text))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(random_sim)
            temp.append(sim)
            temp.append(changed_rate * 100)
            temp.append(random_changed_rate * 100)
            results.append(temp)
            elapsed_times.append(elapsed_time)
            print("Attacked: "+str(idx),"\tqrs",num_queries,"\tsim: ",sim,"\tnum_changed:",num_changed,"\telapsed_time:",elapsed_time)
        if true_label == orig_label and true_label == new_label:
            f_queries.append(num_queries)
            temp1=[]
            temp1.append(idx)
            temp1.append(' '.join(text))
            temp1.append(new_text)
            temp1.append(num_queries)
            fails.append(temp1)

    # joblib.dump(res, "kmeans_first_"+args.target_model+".pkl")
    message =  'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}, final_sims : {:.3f} \n'.format(
                                                                     (1-orig_failures/args.data_size)*100,
                                                                     (1-adv_failures/args.data_size)*100,
                                                                     np.mean(random_changed_rates)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries),
                                                                     np.mean(random_sims),
                                                                     np.mean(final_sims))
    print(message)
    

    log=open(log_file,'a')
    log.write(message)
    with open(result_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    with open(os.path.join(args.output_dir, args.target_model+'_adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()
