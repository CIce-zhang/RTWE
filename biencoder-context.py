import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *
from collections import defaultdict
import random
import numpy as np

from wsd_models.util import *
from wsd_models.loss import *
from wsd_models.models import BiEncoderModel
from copy import deepcopy

import json
import random
import time
parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--weight', type=float, default=0.1, choices=[0.1, 0.2, 0.3, 0.4])
parser.add_argument('--train_mode', type=str, default='mean')
parser.add_argument('--train_data', type=str, default='semcor', choices=['semcor', 'semcor-wngt'])
parser.add_argument('--context_len', type=int, default=2, choices=[1, 2, 3, 4])
parser.add_argument('--current_epoch', type=int, default=0)
parser.add_argument('--context_mode', type=str, default='all', choices=['nonselect', 'nonwindow', 'all'])
parser.add_argument('--word', type=str, default='word', choices=['word', 'non'])
parser.add_argument('--same', action='store_true', help='whether the gloss and context encoder use the same context')
parser.add_argument('--gloss_mode', type=str, default='sense-pred', choices=['non', 'sense', 'sense-pred'])
parser.add_argument('--num_head', type=int, default=6)
parser.add_argument('--train_sent', type=int, default=1000000)
parser.add_argument('--dev_sent', type=int, default=1000000)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true', help='Flag to supress training progress bar for each epoch')
parser.add_argument('--lr', type=float, default=0.00001, choices=[1e-5, 5e-5, 1e-6, 5e-6])
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context_max_length', type=int, default=128)
parser.add_argument('--gloss_max_length', type=int, default=32)
parser.add_argument('--step_mul', type=int, default=50, help='to slow down learning rate dropping after warmed up')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--context-bsz', type=int, default=4)
parser.add_argument('--gloss-bsz', type=int, default=400, choices=[150, 200, 300, 400],
    help='Maybe： the maxium number of candidate sense in same batch')
parser.add_argument('--encoder-name', type=str, default='roberta-base',
    choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large', 'xlmroberta-base', 'xlmroberta-large'])
parser.add_argument('--ckpt', type=str, default='./data',
    help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--data-path', type=str, default='../WSD_Evaluation_Framework', #./data/WSD_Evaluation_Framework
    help='Location of top-level directory for the Unified WSD Framework')

#sets which parts of the model to freeze during training for ablation
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--sec_wsd', action='store_true')
parser.add_argument('--freeze_gloss', action='store_true')
parser.add_argument('--freeze_context', action='store_true')
parser.add_argument('--tie_encoders', action='store_true')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
    help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
    choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL'],
    help='Which evaluation split on which to evaluate probe')

#despatched
parser.add_argument('--mixr', type=float, default=-1.0, help='mix r for cross entropy loss and contrastive learning loss,-1 for no mixing')
parser.add_argument('--tau', type=float, default=0.5, help='mix r for cross entropy loss and contrastive learning loss,-1 for no mixing')
parser.add_argument('--stackcl', action='store_true')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--duplabel', action='store_true')
parser.add_argument('--example_aug', type=int, default=-1)
parser.add_argument('--context_example_aug', type=int, default=-1)
parser.add_argument('--example_maxlen', type=int, default=64,
    help='Train examples : Mean 5.9291 Median 6.0 Max 32; Test examples : Mean 6.1297 Median 6.0 Max 23. orignal examples may not contain the lemma')
parser.add_argument('--enhanced_cl', action='store_true')
parser.add_argument('--enhanced_tau', type=float, default=0.5, help='mix r for cross entropy loss and contrastive learning loss,-1 for no mixing')

parser.add_argument('--base_normalize', action='store_true')
parser.add_argument('--base_gloss', action='store_true')
parser.add_argument('--encoder_contrastive', action='store_true') #用encoder做contrastive learning
parser.add_argument('--base_test', action='store_true') #测试时不走sense encoder
parser.add_argument('--sense_aug', type=int, default=-1)

parser.add_argument('--base_model', action='store_true')
parser.add_argument('--rtwe', action='store_true')
parser.add_argument('--fews', type=str, default='non', choices=['non', 'few', 'zero', 'all'])

parser.add_argument('--datasample', type=str, default='base', choices=['base', 'v1', 'v2'])
parser.add_argument('--test_freq', type=int, default=-1)


def preprocess_resample_base(sent_senses, gloss_bsz=1):
    st = time.time()
    sent_index, current_list = [0], []
    for index, i in enumerate(sent_senses):
        current_list.append(i)
        if sum(current_list) > gloss_bsz:
            sent_index.append(index)
            current_list = current_list[-1:]
    sent_index.append(len(sent_senses))
    sent_list = []
    for i in range(len(sent_index)-1):
        sent_list.append([j for j in range(sent_index[i],sent_index[i+1])])
    print("take time:",time.time()-st)
    return sent_list


def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        if 'xlm' in args.encoder_name:
            g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(gloss_text)]
        else:
            g_ids = [torch.tensor([[x]]) for x in
                 tokenizer.encode(tokenizer.cls_token) + tokenizer.encode(gloss_text) + tokenizer.encode(
                     tokenizer.sep_token)]
        g_attn_mask = [1]*len(g_ids)
        g_fake_mask = [-1]*len(g_ids)
        if 'xlm' in args.encoder_name:
            g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len,
                                                     pad_id=tokenizer.encode(tokenizer.pad_token)[1])
        else:
            g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len,
                                                 pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)

    return glosses, masks

#creates a sense label/ gloss dictionary for training/using the gloss encoder
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1, sense_aug=-1):
    sense_glosses = {}
    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label == -1:
                continue  # ignore unlabeled words
            else:
                key = generate_key(lemma, pos)
                if key not in sense_glosses:
                    # get all sensekeys for the lemma/pos pair
                    sensekey_arr = wn_senses[key]
                    if max_len <= 32:
                        gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]
                        # s = 'love%1:23:00::' ; wn.lemma_from_key(s) = Lemma('love.n.05.love')
                    else:
                        gloss_arr = [wn.lemma_from_key(s).synset().definition() + ' ' + '. '.join(
                         wn.lemma_from_key(s).synset().examples()) for s in sensekey_arr]
                    # preprocess glosses into tensors
                    gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                    gloss_ids = torch.cat(gloss_ids, dim=0)
                    gloss_masks = torch.stack(gloss_masks, dim=0)
                    sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr, gloss_arr)
                if label not in sense_glosses[key][2]:
                    print(label,sense_glosses[key][2])
                assert label in sense_glosses[key][2]
    #print(sensekey_arr)
    return sense_glosses


def generate_example(tokenizer, label, max_len=-1):
    exp_ids,exp_mask,exp_o_mask = [],[],[]
    lemma = label[:label.index("%")]
    sents = wn.lemma_from_key(label).synset().examples()
    if len(sents) == 0:
        return []
    for sent in sents:
        sent = sent.split(" ")
        lemma_count = 0
        if lemma not in sent:
            sent.insert(0,lemma)
        if 'xlm' in args.encoder_name:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)[1:-1]])]
        else:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])]
        o_masks = [-1]
        #For each word in sentence...
        for idx, word in enumerate(sent):
            #tensorize word for context ids
            if 'xlm' in args.encoder_name:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())[1:-1]]
            else:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
            c_ids.extend(word_ids)

            if word == lemma and lemma_count == 0: # note that only a few examples(less than 85) have more than one lemma in the sentence. just ignore
                lemma_count+=1
                #add word to bert output mask to be labeled
                o_masks.extend([idx]*len(word_ids))
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))
            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                #assert lemma_count != 0
                #if lemma_count == 0:
                #    print("examples are too long!",lemma,sent)
                print("examples are too long!",lemma,sent)
                break
        #if lemma_count == 0:
        #    print("No lemma in examples!",lemma,sent)
        if 'xlm' in args.encoder_name:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)[1:-1]])) #aka eos token
        else:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)]))  # aka eos token
        c_attn_mask = [1]*len(c_ids)
        o_masks.append(-1)
        assert len(c_ids) == len(o_masks)
        assert len(c_ids) <= max_len
        #m_len = max(m_len,len(c_ids))
        assert lemma_count != 0
        if 'xlm' in args.encoder_name:
            c_ids,c_attn_mask,o_masks = normalize_length(c_ids,c_attn_mask,o_masks, max_len, tokenizer.encode(tokenizer.pad_token)[1])
        else:
            c_ids,c_attn_mask,o_masks = normalize_length(c_ids,c_attn_mask,o_masks, max_len, tokenizer.encode(tokenizer.pad_token)[0])
        exp_ids.append(c_ids)
        exp_mask.append(c_attn_mask)
        exp_o_mask.append(o_masks)
    return list(zip(exp_ids,exp_mask,exp_o_mask))


def preprocess_context(tokenizer, text_data, gloss_dict=None, bsz=1, max_len=-1, sample_mode='base', base_model=False):
    if max_len == -1: assert bsz==1 #otherwise need max_length for padding

    context_ids = []
    context_attn_masks = []

    example_keys = []

    context_output_masks = []
    instances = []
    labels = []
    example_dict = {}

    #tensorize data
    # print(tokenizer.encode(tokenizer.cls_token), tokenizer.encode(tokenizer.sep_token))
    for sent in (text_data):
        if 'xlm' in args.encoder_name:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)[1:-1]])]
        else:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])]
        o_masks = [-1]
        sent_insts = []
        sent_keys = []
        sent_labels = []
        key_len = []
        for idx, (word, lemma, pos, inst, label) in enumerate(sent):
            if 'xlm' in args.encoder_name:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())[1:-1]]
            else:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
            c_ids.extend(word_ids)

            #if word is labeled with WSD sense...
            if label != -1:
                #add word to bert output mask to be labeled
                o_masks.extend([idx]*len(word_ids))
                #track example instance id
                sent_insts.append(inst)
                #track example instance keys to get glosses
                ex_key = generate_key(lemma, pos)
                sent_keys.append(ex_key)
                key_len.append(len(gloss_dict[ex_key][2]))
                sent_labels.append(label)
                if label not in example_dict:
                    example_dict[label] = generate_example(tokenizer,label,args.example_maxlen)
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))

            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                break

        if 'xlm' in args.encoder_name:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)[1:-1]])) #aka eos token
        else:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)]))  # aka eos token
        c_attn_mask = [1]*len(c_ids)
        o_masks.append(-1)
        assert len(c_ids) == len(o_masks)
        if len(sent_insts) > 0:
            context_ids.append(c_ids) #[tensor([[0]]), tensor([[141]]),..., tensor([[2]])]
            context_attn_masks.append(c_attn_mask) #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            context_output_masks.append(o_masks) #[-1, -1, 1, -1, -1, 4, -1, -1, 7, -1, 9, -1, -1, 12, -1, 14, 15, -1, -1]
            example_keys.append(sent_keys) #['long+a', 'be+v', 'review+v', 'objective+n', 'benefit+n', 'service+n', 'program+n']
            instances.append(sent_insts) #['d000.s000.t000', 'd000.s000.t001', 'd000.s000.t002', 'd000.s000.t003', 'd000.s000.t004', 'd000.s000.t005', 'd000.s000.t006']
            labels.append(sent_labels) #['long%3:00:02::', 'be%2:42:03::', 'review%2:31:00::', 'objective%1:09:00::', 'benefit%1:21:00::', 'service%1:04:07::', 'program%1:09:01::']

    context_dict = dict()
    if base_model:
        data = [list(i) for i in
                list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))]
    else:
        doc_id, doc_seg = [], []
        for index, x in enumerate(instances):
            inst = '.'.join(x[0].split('.')[:-2])
            if inst not in doc_id:
                doc_id.append(inst)
                doc_seg.append(index)
        doc_seg.append(len(instances))
        new_context, new_attn_mask, new_out_mask = [], [], []

        from sklearn.feature_extraction.text import TfidfVectorizer

        for seg_index, seg_id in enumerate((doc_seg[:-1])):
            ids_c = context_ids[seg_id: doc_seg[seg_index + 1]]
            attn_masks_c = context_attn_masks[seg_id: doc_seg[seg_index + 1]]
            output_masks_c = context_output_masks[seg_id: doc_seg[seg_index + 1]]
            example_keys_c = example_keys[seg_id: doc_seg[seg_index + 1]]
            instances_c = instances[seg_id: doc_seg[seg_index + 1]]
            valid_instance = [i for i in instances_c[0] if i != -1][0]
            sent_ids = ['.'.join(i[0].split('.')[:-1]) for i in instances_c]
            if len(valid_instance.split('.')[0]) > 2:
                doc = [' '.join([i.split('+')[0] for i in examp if i.split('+')[1] in 'nvar']) for examp in example_keys_c]
                vectorizer = TfidfVectorizer()
                doc_mat = vectorizer.fit_transform(doc).toarray()
                for sent_id, vec in enumerate(doc_mat):
                    scores = doc_mat[:, doc_mat[sent_id].nonzero()[0]].sum(1)
                    id_score = [j for j in
                                sorted(zip([i for i in range(len(doc_mat))], scores), key=lambda x: x[1], reverse=True) if
                                j[0] != sent_id][:args.context_len]

                    selected = [i[0] for i in id_score]
                    window_id = [i for i in range(len(doc_mat))][
                                max(sent_id - args.context_len, 0):sent_id + args.context_len + 1]
                    pure_neighbor = [i for i in window_id if i != sent_id]
                    #
                    if args.context_mode == 'all':
                        ids = sorted(set(selected + [sent_id] + pure_neighbor))
                    elif args.context_mode == 'nonselect':
                        ids = sorted(set([sent_id] + pure_neighbor))
                    elif args.context_mode == 'nonwindow':
                        ids = sorted(set(selected + [sent_id]))
                    else:
                        ids = [sent_id]

                    total_len = len(sum([ids_c[i]for i in ids], []))
                    while total_len > 512:
                        distance_index = sorted([(abs(s_id-sent_id), s_id) for s_id in ids], reverse=True)
                        ids.remove(distance_index[0][1])
                        total_len = len(sum([ids_c[i] for i in ids], []))
                    if args.context_len > 0:
                        new_context.append(sum([ids_c[i]for i in ids], []))
                        new_attn_mask.append(sum([attn_masks_c[i] for i in ids], []))
                        new_out_mask.append(
                            sum([[-1] * len(output_masks_c[i]) if i != sent_id else output_masks_c[i] for i in ids], []))
                        assert len(new_context[-1]) == len(new_attn_mask[-1]) == len(new_out_mask[-1])
                    else:
                        new_context.append(ids_c[sent_id])
                        new_attn_mask.append(attn_masks_c[sent_id])
                        new_out_mask.append(output_masks_c[sent_id])
                    context_dict[sent_ids[sent_id]] = [sent_ids[i] for i in ids]
            else:
                new_context.extend(ids_c)
                new_attn_mask.extend(attn_masks_c)
                new_out_mask.extend(output_masks_c)

                for sent_id in sent_ids:
                    context_dict[sent_id] = [sent_id]
        assert len(context_ids) == len(new_context)
        data = [list(i) for i in
                list(zip(new_context, new_attn_mask, new_out_mask, example_keys, instances, labels))]
    batched_data = []
    sent_senses = [sum([len(gloss_dict[ex_key][2]) for ex_key in sent[3]]) for sent in data]
    sent_index = preprocess_resample_base(sent_senses,args.gloss_bsz) 
    for index, data_index in enumerate(sent_index):
        b = [data[i] for i in data_index] #if data_index is a list
        max_len_b = max([len(x[1]) for x in b]) #x[1] = new_attn_mask
        if args.context_len > 0: #context_len in 1/2/3/4
            max_len = max(max_len_b, max_len)
        for b_index, sent in enumerate(b):
            if 'xlm' in args.encoder_name:
                b[b_index][0], b[b_index][1], b[b_index][2] = normalize_length(sent[0], sent[1], sent[2], max_len,
                                                                           tokenizer.encode(tokenizer.pad_token)[1])
            else:
                b[b_index][0], b[b_index][1], b[b_index][2] = normalize_length(sent[0], sent[1], sent[2], max_len,
                                                                           tokenizer.encode(tokenizer.pad_token)[0])

        context_ids = torch.cat([torch.cat(x, dim=-1) for x, _, _, _, _, _ in b], dim=0)[:, :max_len_b]
        context_attn_mask = torch.cat([torch.tensor(x).unsqueeze(dim=0) for _, x, _, _, _, _ in b], dim=0)[:,
                            :max_len_b]
        context_output_mask = torch.cat([torch.tensor(x).unsqueeze(dim=0) for _, _, x, _, _, _ in b], dim=0)[:,
                              :max_len_b]
        example_keys = []
        for _, _, _, x, _, _ in b: example_keys.extend(x)
        instances = []
        for _, _, _, _, x, _ in b: instances.extend(x)
        labels = []
        for _, _, _, _, _, x in b: labels.extend(x)
        #print(context_ids.size())
        #labels = ['long%3:00:02::','review%2:31:00::','objective%1:09:00::']
        exp_label,exp_ids,exp_mask,exp_o_mask = [],[],[],[]
        for label in labels:
            if len(example_dict[label]) > 0:
                for exp in example_dict[label]:
                    exp_ids.append(torch.cat(exp[0],dim=-1))
                    exp_mask.append(torch.tensor(exp[1]).unsqueeze(0))
                    exp_o_mask.append(torch.tensor(exp[2]).unsqueeze(0))
                    exp_label.append(label)
        if len(exp_ids)==0:
            exp_ids = torch.tensor(exp_ids)
            exp_mask = torch.tensor(exp_mask)
            exp_o_mask = torch.tensor(exp_o_mask)
        else:
            exp_ids = torch.cat(exp_ids,dim=0)
            exp_mask = torch.cat(exp_mask,dim=0)
            exp_o_mask = torch.cat(exp_o_mask,dim=0)
        assert len(exp_label) == exp_ids.size(0)

        batched_data.append(
            (context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels, exp_ids, exp_mask, exp_o_mask, exp_label))
    return batched_data, context_dict, example_dict


def generate_context_example(tokenizer, text_data, max_len=-1):
    context_ids = []
    context_attn_masks = []
    context_output_masks = []
    instances = []
    labels = []
    example_dict = {}
    context_example_dict = dict()
    lem2lab = dict()
    #tensorize data
    # print(tokenizer.encode(tokenizer.cls_token), tokenizer.encode(tokenizer.sep_token))
    for sent in (text_data):
        #cls token aka sos token, returns a list with index
        #print(sent)
        #('How', 'how', 'ADV', -1, -1), ('long', 'long', 'ADJ', 'd000.s000.t000', 'long%3:00:02::')
        #(word, lemma, pos, inst, label)
        if 'xlm' in args.encoder_name:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)[1:-1]])]
        else:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])]
        o_masks = [-1]
        sent_insts = []
        sent_labels = []
        o_mask_list = []

        #For each word in sentence...
        key_len = []
        for idx, (word, lemma, pos, inst, label) in enumerate(sent):
            if 'xlm' in args.encoder_name:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())[1:-1]]
            else:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
            c_ids.extend(word_ids)
            if label != -1:
                #add word to bert output mask to be labeled
                if max_len != -1 and len(c_ids) >= max_len:
                    o_mask_list.append([-1]*len(o_masks) + [idx]*(max_len-len(o_masks)))
                else:
                    o_mask_list.append([-1]*len(o_masks) + [idx]*len(word_ids) + [-1]*(max_len-len(o_masks)-len(word_ids)))
                o_masks.extend([idx]*len(word_ids))
                #track example instance id
                sent_insts.append(inst)
                #track example instance keys to get glosses
                sent_labels.append(label)
                assert len(o_mask_list[-1]) == max_len
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))

            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                break
        if 'xlm' in args.encoder_name:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)[1:-1]])) #aka eos token
        else:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)]))  # aka eos token
        c_attn_mask = [1]*len(c_ids)
        o_masks.append(-1)
        assert len(c_ids) == len(o_masks)

        if 'xlm' in args.encoder_name:
            c_ids,c_attn_mask,o_masks = normalize_length(c_ids,c_attn_mask,o_masks, max_len, tokenizer.encode(tokenizer.pad_token)[1])
        else:
            c_ids,c_attn_mask,o_masks = normalize_length(c_ids,c_attn_mask,o_masks, max_len, tokenizer.encode(tokenizer.pad_token)[0])
        if len(sent_insts) > 0:
            context_ids.append(torch.cat(c_ids,dim=-1))
            context_attn_masks.append(torch.tensor(c_attn_mask).unsqueeze(0))
            #context_ids.append(c_ids)
            #context_attn_masks.append(c_attn_mask)
        assert len(o_mask_list) == len(sent_labels)
        for i in range(len(sent_labels)):
            lem = label_2_word(sent_labels[i])
            if lem not in lem2lab:
                lem2lab[lem] = set()
            lem2lab[lem].add(sent_labels[i])
            if sent_labels[i] not in context_example_dict.keys():
                context_example_dict[sent_labels[i]] = []
            o_mask_list[i] = torch.tensor(o_mask_list[i]).unsqueeze(0)
            context_example_dict[sent_labels[i]].append([sent_labels[i],len(context_ids)-1,o_mask_list[i]])
    
    return context_example_dict, list(zip(context_ids,context_attn_masks)), lem2lab



def _rtwe_train(train_data, model, gloss_dict, optim, schedule, train_index, train_dict, key_mat, cl_loss_func, train_example, args, ecl_loss_func, context_example, context_data, lem2lab, all_data,all_gloss_dict,dev_pred,dev_dict):
    model.train()
    mixr = args.mixr
    train_data = enumerate(train_data)
    train_data = tqdm(list(train_data))
    model.zero_grad()
    loss = 0.
    gloss_sz = 0
    context_sz = 0

    all_instance, pre_instance, mfs_instance, last_instance = 0, 0, 0, 0
    count = 0

    total_loss = 0.
    ce_total_loss = 0.
    cl_total_loss = 0.
    ccl_total_loss = 0.
    cl_loss_analy = [0.,0.,0.]

    for b_index, (context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels, exp_ids, exp_mask, exp_o_mask, exp_label) in train_data:
        for label in exp_label:
            if label not in labels:
                print(label)

        sent_id, sent_seg = [], []
        key_len_list = []
        for in_index, inst in enumerate(instances):
            s_id = '.'.join(inst.split('.')[:-1])
            if s_id not in sent_id:
                sent_id.append(s_id)
                sent_seg.append(in_index)
        sent_seg.append(len(instances))

        for seg_index, seg in enumerate(sent_seg[:-1]):
            key_len_list.append([len(gloss_dict[key][2]) for key in example_keys[seg:sent_seg[seg_index + 1]]])

        total_sense = sum(sum(key_len_list, []))
        if 'large' in args.encoder_name:
            if total_sense > args.gloss_bsz:
                count += 1
                continue
        #run example sentence(s) through context encoder
        context_ids = context_ids.cuda()
        context_attn_mask = context_attn_mask.cuda()
        context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
        #assert len(context_output.size())==2
        if args.base_normalize:
            context_output = F.normalize(context_output)
            print("normalized")

        max_len_gloss = max(
            sum([[torch.sum(mask_list).item() for mask_list in gloss_dict[key][1]] for key in example_keys],
                []))
        gloss_ids_all = torch.cat([gloss_dict[key][0][:, :max_len_gloss] for key in example_keys])
        gloss_attn_mask_all = torch.cat([gloss_dict[key][1][:, :max_len_gloss] for key in example_keys])

        gloss_ids = gloss_ids_all.cuda()
        gloss_attn_mask = gloss_attn_mask_all.cuda()
        gat_out_all, base_out_all = model.gat_forward(gloss_ids, gloss_attn_mask, args, key_len_list, instances, train_index, train_dict)
        #extract example embedding       
        #exp_ids, exp_mask, exp_o_mask, exp_label
        
        re_label = {}
        #
        for seg_index, seg in enumerate(sent_seg[:-1]):
            current_example_keys = example_keys[seg: sent_seg[seg_index + 1]]
            current_key_len = key_len_list[seg_index]
            current_context_output = context_output[:, seg: sent_seg[seg_index + 1], :]
            current_insts = instances[seg: sent_seg[seg_index + 1]]
            current_labels = labels[seg: sent_seg[seg_index + 1]]

            gloss_output_pad = gat_out_all[sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),:]
            curren_gloss_mask = torch.sum(gloss_attn_mask[sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),:], dim=-1).unsqueeze(-1)
            
            c_senses = sum([gloss_dict[key][2] for key in current_example_keys], [])

            gloss_sz += gloss_output_pad.size(0)
            
            gat_cpu = gloss_output_pad.cpu()
            assert len(c_senses) == len(gloss_output_pad)
            for k_index, key in enumerate(c_senses):
                key_mat[key] = gat_cpu[k_index:k_index + 1]
            gloss_output_pad = torch.cat([F.pad(
                gloss_output_pad[sum(current_key_len[:i]): sum(current_key_len[:i+1]), :],
                pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)], dim=0)

            gloss_output_pad_base = base_out_all[
                    sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),
                    :]
            gloss_output_pad_base = torch.cat([F.pad(
                gloss_output_pad_base[sum(current_key_len[:i]): sum(current_key_len[:i+1]), :],
                pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)], dim=0)

            gloss_mask_pad = torch.cat([F.pad(
                curren_gloss_mask[sum(current_key_len[:i]): sum(current_key_len[:i+1]), :],
                pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)], dim=0)
            out = model.att_forward(gloss_output_pad, current_context_output, gloss_mask_pad.squeeze(-1))

            context_sz += 1
            cand_size = out.size(1)
            for j, (key, label) in enumerate(zip(current_example_keys, current_labels)):
                idx = gloss_dict[key][2].index(label)
                label_tensor = torch.tensor([idx]).cuda()
                assert current_key_len[j] >= idx
                if seg_index == 0 and j == 0:
                    gloss_op = gloss_output_pad_base[j][idx].unsqueeze(0)
                    if args.sense_aug > 0:
                        gloss_op_aug = gloss_output_pad_aug[j][idx].unsqueeze(0)
                    
                    temp_label = [0]
                    temp_gloss = [gloss_dict[key][3][idx]]
                    re_label[label] = 0
                else:
                    gloss_op = torch.cat((gloss_op,gloss_output_pad_base[j][idx].unsqueeze(0)))
                    if args.sense_aug > 0:
                        gloss_op_aug = torch.cat((gloss_op_aug,gloss_output_pad_aug[j][idx].unsqueeze(0)))
                    if gloss_dict[key][3][idx] in temp_gloss:
                        re_label[label] = temp_gloss.index(gloss_dict[key][3][idx])
                        temp_label.append(re_label[label])
                    else:
                        temp_gloss.append(gloss_dict[key][3][idx])
                        re_label[label] = temp_gloss.index(gloss_dict[key][3][idx])
                        temp_label.append(re_label[label])
                        
                if j == 0:
                    out_mask = F.pad(torch.ones(current_key_len[j]),(0,cand_size-current_key_len[j])).unsqueeze(0).cuda()
                else:
                    out_mask = torch.cat((out_mask,F.pad(torch.ones(current_key_len[j]),(0,cand_size-current_key_len[j])).unsqueeze(0).cuda()))
                ####Cross Entropy loss
                train_index[current_insts[j]] = out[j:j + 1, :current_key_len[j]].argmax(dim=1).item()
                loss += F.cross_entropy(out[j:j + 1, :current_key_len[j]], label_tensor)
                #取出output中的某一行，由于每行candidate gloss数量不定，故交叉熵部分取第j行的前current_key_len[j]个元素
                if out[j:j + 1, :current_key_len[j]].argmax(dim=1).item() == idx:
                    pre_instance += 1
                all_instance += 1
                if idx == 0:
                    mfs_instance += 1
        loss = loss / gloss_sz
        print("batch",b_index,"CE loss:",loss.item())
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optim.step()
        schedule.step() # Update learning rate schedule
        #reset loss and gloss_sz
        loss = 0.
        gloss_sz = 0
        context_sz = 0
        #reset model
        model.zero_grad()
        torch.cuda.empty_cache()
        if args.test_freq > 0 and b_index > 0 and b_index%args.test_freq == 0:
            model.eval()
            temp_sec_wsd = args.sec_wsd
            args.sec_wsd = False
            eval_preds, dev_pred, key_mat = _eval(all_data, model, all_gloss_dict, dev_pred, dev_dict, None,'ALL',args.base_test) #all_data,all_gloss_dict,dev_pred,dev_dict
            pred_all_filepath = os.path.join(args.ckpt, 'all_tmp_predictions_%s_during_epoch.txt' % args.train_mode)
            with open(pred_all_filepath, 'w') as f:
                for inst, prediction in eval_preds:
                        f.write('{} {}\n'.format(inst, prediction))
            scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
            gold_all_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/ALL/ALL.gold.key.txt')
            _, _, test_f1 = evaluate_output(scorer_path, gold_all_filepath, pred_all_filepath)
            print("Batch:%d test F1:%.4f"%(b_index,test_f1))
            args.sec_wsd = temp_sec_wsd
            model.train()
    print(count)
    print(pre_instance / all_instance, mfs_instance / all_instance)
    print("Epoch total loss:%.6f"%(total_loss/b_index))
    return model, optim, schedule, key_mat


def _eval(eval_data, model, gloss_dict, dev_index, dev_dict, key_mat=None, eval_file='ALL', base_test=False, base_model=False, sense_aug=-1):
    model.eval()
    csi_data = pickle.load(open('./data/csi_data', 'rb'))
    tag_lemma, tag_sense = pickle.load(open('./data/tag_semcor.txt', 'rb'))
    zsl, zss = [], []
    eval_preds = []
    gold_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/{}.gold.key.txt'.format(eval_file, eval_file))
    gold_labels = {i.split()[0]: i.split()[1:] for i in open(gold_path, 'r').readlines()}
    name = locals()
    dataset_name = sorted(set([i.split('.')[0] for i in gold_labels]))
    pos_tran = {'a': 'ADJ', 'n': 'NOUN', 'r': 'ADV', 'v': 'VERB'}
    for i in dataset_name:
        name['pred_c_%s' % i], name['pred_all_%s' % i] = 0, 0
    for pos in pos_tran.values():
        name['pred_c_%s' % pos], name['pred_all_%s' % pos] = 0, 0
    mfs_list, lfs_list = [], []
    correct_id = []
    if key_mat:
        key_dict, vec_list = {}, []
        count = 0
        for key, vec in key_mat.items():
            if '%' in key:
                synset = wn.lemma_from_key(key).synset().name()
            else:
                synset = wn._synset_from_pos_and_offset(key[-1], int(key[3:-1])).name()
            if synset not in key_dict:
                key_dict[synset] = count
                vec_list.append([vec])
                count += 1
            else:
                vec_list[key_dict[synset]].append(vec)
        vec_list = [torch.mean(torch.cat(i), dim=0).unsqueeze(0) for i in vec_list]
        vec_mat = torch.cat(vec_list).cuda()

    related_synset = {}
    for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _, _, _, _, _ in tqdm(eval_data):
        with torch.no_grad():

            context_ids = context_ids.cuda()
            context_attn_mask = context_attn_mask.cuda()
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            max_len_gloss = max(
                sum([[torch.sum(mask_list).item() for mask_list in gloss_dict[key][1]] for key in example_keys],
                    []))
            gloss_ids_all = torch.cat([gloss_dict[key][0][:, :max_len_gloss] for key in example_keys])
            gloss_attn_mask_all = torch.cat([gloss_dict[key][1][:, :max_len_gloss] for key in example_keys])

            gloss_ids = gloss_ids_all.cuda()
            gloss_attn_mask = gloss_attn_mask_all.cuda()

            sent_id, sent_seg = [], []
            key_len_list = []
            for in_index, inst in enumerate(insts):
                s_id = '.'.join(inst.split('.')[:-1])
                if s_id not in sent_id:
                    sent_id.append(s_id)
                    sent_seg.append(in_index)
            sent_seg.append(len(insts))

            for seg_index, seg in enumerate(sent_seg[:-1]):
                key_len_list.append([len(gloss_dict[key][2]) for key in example_keys[seg:sent_seg[seg_index + 1]]])

            senses = [gloss_dict[key][2] for key in example_keys]
            
            gat_out_all,_ = model.gat_forward(gloss_ids, gloss_attn_mask, args, key_len_list, insts, dev_index, dev_dict)

            for seg_index, seg in enumerate(sent_seg[:-1]):
                current_example_keys = example_keys[seg: sent_seg[seg_index + 1]]
                current_key_len = key_len_list[seg_index]
                current_context_output = context_output[:, seg: sent_seg[seg_index + 1], :]
                current_insts = insts[seg: sent_seg[seg_index + 1]]

                gat_out = gat_out_all[
                          sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),
                          :]
                curren_gloss_mask = torch.sum(gloss_attn_mask[sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),:], dim=-1).unsqueeze(-1)

                
                if key_mat:
                    c_senses = sum([gloss_dict[key][2] for key in current_example_keys], [])
                    gat_cpu = gat_out.cpu()
                    assert len(c_senses) == len(gat_out)
                    for k_index, key in enumerate(c_senses):
                        key_mat[key] = gat_cpu[k_index:k_index+1]

                    #context_key = torch.mm(current_context_output, vec_mat.T)
                    if args.rtwe:
                        context_key = torch.mm(torch.mean(current_context_output[-4:,:,:], dim=0), vec_mat.T)
                    else:
                        context_key = torch.mm(current_context_output, vec_mat.T)


                gloss_output_pad = torch.cat([F.pad(
                    gat_out[sum(current_key_len[:i]): sum(current_key_len[:i + 1]), :],
                    pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)],
                                             dim=0)
                
                gloss_mask_pad = torch.cat([F.pad(
                    curren_gloss_mask[sum(current_key_len[:i]): sum(current_key_len[:i+1]), :],
                    pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)], dim=0)
                out = model.att_forward(gloss_output_pad, current_context_output, gloss_mask_pad.squeeze(-1)).float().cpu()
                
                
                for j, key in enumerate(current_example_keys):
                    pred_idx = out[j:j + 1, :current_key_len[j]].topk(1, dim=-1)[1].squeeze().item()
                    if args.sec_wsd and len(gloss_dict[key][2]) >= 2:
                        if '%' in gloss_dict[key][2][0]:
                            synsets = [wn.lemma_from_key(i).synset().name() for i in gloss_dict[key][2]]
                        else:
                            synsets = [wn._synset_from_pos_and_offset(s[-1], int(s[3:-1])).name() for s in
                                       gloss_dict[key][2]]
                        key_sim = [(index, (k, sim)) for index, (k, sim) in
                                   enumerate(zip(synsets, out[j, :current_key_len[j]].tolist()))]
                        key_sim = sorted(key_sim, key=lambda x: x[1][1], reverse=True)
                        sec_sim = [0] * len(synsets)
                        for n_index, (index, (synset, _)) in enumerate(key_sim):
                            if n_index <= 2:
                                if synset in csi_data[0]:
                                    sec_synsets = sum([csi_data[1][q] for q in csi_data[0][synset]], [])
                                else:
                                    sec_synsets = []
                                if sec_synsets:
                                    sim = np.mean(sorted(
                                        [context_key[j, key_dict[k]].item() for k in sec_synsets if
                                         k in key_dict], reverse=True)[:1])
                                else:
                                    sim = out[j, index]
                            else:
                                sim = 0
                            sec_sim[index] = sim
                        sec_sim = torch.tensor(sec_sim)
                        pred_idx = ((1 - args.weight) *out[j, :current_key_len[j]] +
                                    args.weight *sec_sim).topk(1, dim=-1)[1].squeeze().item()

                    # if current_insts[j] not in dev_index:
                    dev_index[current_insts[j]] = pred_idx
                    pred_label = gloss_dict[key][2][pred_idx]
                    eval_preds.append((current_insts[j], pred_label))
                    if key not in tag_lemma:
                        if len(gloss_dict[key][2]) > 1:
                            if pred_label in gold_labels[current_insts[j]]:
                                zsl.append(1)
                            else:
                                zsl.append(0)
                    if not tag_sense.intersection(gold_labels[current_insts[j]]):
                        if len(gloss_dict[key][2]) > 1:
                            if pred_label in gold_labels[current_insts[j]]:
                                zss.append(1)
                            else:
                                zss.append(0)
                    if set(gloss_dict[key][2][:1]).intersection(gold_labels[current_insts[j]]):
                        if pred_label in gold_labels[current_insts[j]]:
                            mfs_list.append(1)
                        else:
                            mfs_list.append(0)
                    else:
                        if pred_label in gold_labels[current_insts[j]]:
                            lfs_list.append(1)
                        else:
                            lfs_list.append(0)
                    for i in dataset_name:
                        if i in current_insts[j]:
                            name['pred_all_%s' % i] += 1
                            if pred_label in gold_labels[current_insts[j]]:
                                name['pred_c_%s' % i] += 1
                                correct_id.append(current_insts[j])
                    for pos in pos_tran.values():
                        if pos in pos_tran[key.split('+')[1]]:
                            name['pred_all_%s' % pos] += 1
                            if pred_label in gold_labels[current_insts[j]]:
                                name['pred_c_%s' % pos] += 1

    correct_pred, all_pred = 0, 0
    for i in dataset_name:
        correct_pred += name['pred_c_%s' % i]
        all_pred += name['pred_all_%s' % i]
        if '2007' in i:
            print(i, name['pred_c_%s' % i]/name['pred_all_%s' % i], end='\t')
    for pos in pos_tran.values():
        if name['pred_all_%s' % pos] != 0:
            print(pos, name['pred_c_%s' % pos] / name['pred_all_%s' % pos], end='\t')

    print('ALL', correct_pred/all_pred, all_pred)
    print('zss %d, zsl %d' % (len(zss), len(zsl)), 'zss %f, zsl %f' % (sum(zss) / len(zss), sum(zsl) / len(zsl)))
    print(sum(mfs_list) / len(mfs_list), sum(lfs_list) / len(lfs_list), len(mfs_list), len(lfs_list),
          len(mfs_list + lfs_list))

    return eval_preds, dev_index, key_mat


def train_model(args):
    print('Training WSD bi-encoder model...')
    if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

    '''
    LOAD PRETRAINED TOKENIZER, TRAIN AND DEV DATA
    '''
    print('Loading data + preprocessing...')
    sys.stdout.flush()

    tokenizer = load_tokenizer(args.encoder_name)

    #loading WSD (semcor) data
    train_path = os.path.join(args.data_path, 'FEWS/train/')
    train_data = load_data(train_path, 'train', args.train_sent)

    test_path = os.path.join(args.data_path, 'Evaluation_Datasets/ALL/')
    all_data = load_data(test_path, eval_file)[:args.dev_sent]

    #load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
    wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
    wn_senses = load_wn_senses(wn_path)

    train_gloss_dict = load_and_preprocess_glosses(train_data, tokenizer, wn_senses, max_len=args.gloss_max_length,sense_aug = args.sense_aug)
    all_gloss_dict = load_and_preprocess_glosses(all_data, tokenizer, wn_senses, max_len=args.gloss_max_length,sense_aug = args.sense_aug)

    context_example, context_data, lem2lab = generate_context_example(tokenizer, train_data, max_len=args.example_maxlen)
    train_data, train_dict, train_example = preprocess_context(tokenizer, train_data, train_gloss_dict, bsz=args.context_bsz,
                                                max_len=args.context_max_length,sample_mode=args.datasample, base_model=args.base_model)
    
    all_data, dev_dict, test_example = preprocess_context(tokenizer, all_data, all_gloss_dict, bsz=args.context_bsz,
                                            max_len=args.context_max_length,sample_mode=args.datasample, base_model=args.base_model)
    epochs = args.epochs
    overflow_steps = -1
    t_total = len(train_data)*args.step_mul
    update_count = 0

    if not args.continue_train:
        model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, 
                                rtwe=args.rtwe, tie_encoders=args.tie_encoders, num_heads=args.num_head)
        model = model.cuda()
    else:
        model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, 
                                rtwe=args.rtwe, tie_encoders=args.tie_encoders, num_heads=args.num_head)
        model_path = os.path.join(args.ckpt, 'best_model_%s.ckpt' % args.train_mode)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()

    weight_decay = 0.0 #this could be a parameter
    if args.rtwe:
        weight_decay = 0.001 #this could be a parameter

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]}
        ]
    adam_epsilon = 1e-8
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon, weight_decay=weight_decay)
    schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)
    if args.continue_train:
        print("load pre optimizer from %s"%args.ckpt)
        pre_path = os.path.join(args.ckpt, 'best_optim.ckpt')
        optimizer.load_state_dict(torch.load(pre_path)['optim'])
        schedule.load_state_dict(torch.load(pre_path)['schedule'])
        print("load optim successfully")

    '''
    TRAIN MODEL
    '''
    best_dev_f1 = 0.
    best_test_f1 = 0.
    print('Training probe...')
    sys.stdout.flush()

    train_index = dict()
    dev_pred = dict()
    dev_pred_list = [{}]
    key_mat_list = []
    mdev_pred_list = defaultdict(list)
    mkey_mat_list = defaultdict(list)

    key_mat = dict()
    #change
    cl_loss_func = SupContrastive_Loss(args.tau)
    ecl_loss_func = SupContrastive_Loss(args.enhanced_tau)

    #cant do base test,cause we do not have dev_pred/key_mat
    #print("base test")
    #_,_,_ = _eval(all_data, model, all_gloss_dict, dev_pred, dev_dict, key_mat,'ALL',args.base_test)

    for epoch in range(1, epochs+1):
        args.current_epoch = epoch
        #if last epoch, pass in overflow steps to stop epoch early
        train_steps = -1
        if epoch == epochs and overflow_steps > 0: train_steps = overflow_steps
        #train model for one epoch or given number of training steps
        model, optimizer, schedule, key_mat = _rtwe_train(train_data, model, train_gloss_dict, optimizer, schedule,
                                                    train_index, train_dict, key_mat, cl_loss_func, train_example, args, 
                                                    ecl_loss_func, context_example, context_data, lem2lab,
                                                    all_data, all_gloss_dict, dev_pred, dev_dict)
        key_mat_copy = {i: j for i,j in key_mat.items()}
        key_mat_list.append(key_mat_copy)

        eval_preds, dev_pred, key_mat = _eval(all_data, model, all_gloss_dict, dev_pred, dev_dict, key_mat,'ALL',args.base_test, args.base_model)

        dev_pred_list.append(deepcopy(dev_pred))
        #generate predictions file
        pred_filepath = os.path.join(args.ckpt, 'se07_tmp_predictions_%s.txt' % args.train_mode)
        with open(pred_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                if '2007' in inst:
                    f.write('{} {}\n'.format('.'.join(inst.split('.')[1:]), prediction))
        
        pred_all_filepath = os.path.join(args.ckpt, 'all_tmp_predictions_%s.txt' % args.train_mode)
        with open(pred_all_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                    f.write('{} {}\n'.format(inst, prediction))
        #run predictions through scorer
        eval_file = args.split
        gold_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/%s/%s.gold.key.txt' % (eval_file, eval_file))
        gold_all_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/ALL/ALL.gold.key.txt')
        scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
        _, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
        open('./data/dev_result.txt', 'a+').write('%s-%d-%f\n' % (args.train_mode, epoch, dev_f1))
        _, _, test_f1 = evaluate_output(scorer_path, gold_all_filepath, pred_all_filepath)
        print('Dev f1 after {} epochs = {}, Test f1 = {}'.format(epoch, dev_f1,test_f1))
        sys.stdout.flush()
        
        #if dev_f1 >= best_dev_f1:
        if test_f1 >= best_test_f1:
            print('updating best model at epoch {}...'.format(epoch))
            sys.stdout.flush()
            #best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            #save to file if best probe so far on dev set
            model_fname = os.path.join(args.ckpt, 'best_model_%s.ckpt' % args.train_mode)
            with open(model_fname, 'wb') as f:
                torch.save(model.state_dict(), f)
            #change
            model_optname = os.path.join(args.ckpt, 'best_optim.ckpt')
            train_states = {"optim":optimizer.state_dict(),"schedule":schedule.state_dict()}
            with open(model_optname, 'wb') as f:
                torch.save(train_states, f)
            
            sys.stdout.flush()
            pickle.dump(dev_pred_list[-2], open('%s/dev_pred_%s_%s' % (args.ckpt, args.train_mode, 'ALL'), 'wb'), -1)
            for i, j in mdev_pred_list.items():
                pickle.dump(mdev_pred_list[i][-2], open('%s/dev_pred_%s_%s' % (args.ckpt, args.train_mode, i), 'wb'), -1)
            pickle.dump(key_mat_list[-1], open('%s/key_mat_%s_%s' % (args.ckpt, args.train_mode, 'ALL'), 'wb'), -1)
            update_count = 0
        else:
            update_count += 1
        if update_count >= 5:
            exit()
        # shuffle train set ordering after every epoch
        random.shuffle(train_data)

    return



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    #parse args
    args = parser.parse_args()
    print(args)

    #set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    #evaluate model saved at checkpoint or...
    train_model(args)

#EOF