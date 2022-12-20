'''
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from wsd_models.util import *

def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text)+tokenizer.encode(tokenizer.sep_token)]
        g_attn_mask = [1]*len(g_ids)
        g_fake_mask = [-1]*len(g_ids)
        g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)

    return glosses, masks

def mask_logits(target, mask, logit=-1e30):
    return target * mask + (1 - mask) * (logit)

def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer

class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            #assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []
        for i in range(output.size(0)):
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output

class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):
        #encode gloss text
        if self.is_frozen:
            with torch.no_grad():
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[-1][-4:]

        gloss_output = torch.cat([i.unsqueeze(0) for i in gloss_output], dim=0).mean(0)
        gloss_output = gloss_output[:,:,:].squeeze(dim=1)
        return gloss_output

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context, rtwe):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context
        self.is_rtwe = rtwe

    def forward(self, input_ids, attn_mask, output_mask):
        #encode context
        if self.is_rtwe:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[-1][-12:] 
            context_output_temp = []
            for i in range(len(context_output)):
                example_arr = []
                for j in range(context_output[i].size(0)):
                    example_arr.append(process_encoder_outputs(context_output[i][j], output_mask[j], as_tensor=True))  #len=10
                context_output_temp.append(torch.cat(example_arr, dim=0).unsqueeze(0))  #shape=(67,768)
            context_output = torch.cat(context_output_temp, dim=0)   #shape=(12,67,768)
        else:
            if self.is_frozen:
                with torch.no_grad():
                    context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
            else:
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[-1][-4:]
            context_output = torch.cat([i.unsqueeze(0) for i in context_output], dim=0).mean(0)
            #average representations over target word(s)
            example_arr = []
            #print("Contextoutput :",context_output.size())
            for i in range(context_output.size(0)):
                example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
            context_output = torch.cat(example_arr, dim=0)
        return context_output


class LinearAttention(nn.Module):
    def __init__(self, in_dim=300, mem_dim=300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(1e-2)
        self.linear1 = nn.Linear(in_dim, mem_dim)
        self.linear2 = nn.Linear(in_dim, mem_dim)
        torch.nn.init.xavier_normal_(self.linear.weight.data)
        torch.nn.init.xavier_normal_(self.linear1.weight.data)
        torch.nn.init.xavier_normal_(self.linear2.weight.data)

    def forward(self, feature, aspect_v, dmask, word='word'):
        Q = self.linear(aspect_v.float())
        Q = nn.functional.normalize(Q, dim=1)

        attention_s = torch.mm(Q, Q.T)
        attention_sk = mask_logits(attention_s, dmask, 0)

        if 'word' in word:
            new_feature = self.linear(feature.float())
            new_feature = nn.functional.normalize(new_feature, dim=2)

            feature_reshape = new_feature.reshape(new_feature.shape[0] * new_feature.shape[1], -1)
            attention_ww = torch.mm(feature_reshape, feature_reshape.T)
            attention_w = torch.stack(
                torch.stack(attention_ww.split(new_feature.shape[1]), dim=0).mean(1).squeeze(1).split(new_feature.shape[1],
                                                                                                      dim=1), dim=1).mean(2)
            attention_wk = mask_logits(attention_w, dmask, 0)

            att_weight = attention_sk + attention_wk
        else:
            att_weight = attention_sk

        att_weight[att_weight == 0] = -1e30
        attention = F.softmax(att_weight, dim=1)
        new_out = torch.mm(attention, aspect_v)

        return new_out

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, rtwe=False, tie_encoders=False, num_heads=6):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context, rtwe)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim
        self.gat = [LinearAttention(self.gloss_encoder.gloss_hdim, self.gloss_encoder.gloss_hdim).cuda() for _ in
                    range(num_heads)]
        
        self.att_model = BertSenseClassifier()

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)
    
    def base_gloss_forward(self, gloss_input, gloss_mask, args, key_len_list, instances, pre_index, context_dict, senses=''):
        gloss_out_all = self.gloss_encoder.forward(gloss_input, gloss_mask)
        return gloss_out_all[:, 0, :]

    def gat_forward(self, gloss_input, gloss_mask, args, key_len_list, instances, pre_index, context_dict, senses=''):
        gloss_out_all = self.gloss_encoder.forward(gloss_input, gloss_mask)
        if 'sense' in args.gloss_mode:
            key_len = sum(key_len_list, [])
            adjacency_mat = torch.zeros(sum(key_len), sum(key_len))
            sense_index = [sum(key_len[:i]) for i in range(len(key_len))]
            if 'pred' in args.gloss_mode:
                p_index = [pre_index.get(inst, 0) for inst in instances]
                sense_index = [sense_index[i] + p_index[i] for i in range(len(p_index))]
            doc_sent = [('.'.join(i.split('.')[:-2]), int(i.split('.')[-2][1:]), '.'.join(i.split('.')[:-1])) for i in
                        instances]
            adjacency_mat[:, sense_index] = 1
            for i in range(len(instances)):
                index = []
                for s_index, sense in enumerate(sense_index):
                    if args.same:
                        if doc_sent[s_index][-1] not in context_dict[doc_sent[i][-1]]:
                            index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                    else:
                        if len(doc_sent[s_index][0]) > 2:
                            if doc_sent[s_index][0] != doc_sent[i][0] or abs(doc_sent[s_index][1] - doc_sent[i][1]) > 0:
                                index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                        elif abs(doc_sent[s_index][1] - doc_sent[i][1]) > 0:
                            index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                adjacency_mat[sum(key_len[:i]): sum(key_len[:i + 1]), index] = 0

            for k_index, j in enumerate(key_len):
                start, end = sum(key_len[:k_index]), sum(key_len[:k_index + 1])
                adjacency_mat[start: end, start: end] = 0

            adjacency_mat_f = adjacency_mat + torch.eye(sum(key_len))

            att_out = [att.forward(gloss_out_all[:, 1:-1, :], gloss_out_all[:, 0, :], adjacency_mat_f.cuda(),
                                   args.word).unsqueeze(1) for att in self.gat]
            att_out = torch.cat(att_out, dim=1)
            att_out = att_out.mean(dim=1)  # (N, D)min(31, gloss_out_all.shape[1]-1)
            assert len(gloss_out_all) == len(att_out)
            return att_out,gloss_out_all[:, 0, :]
        else:
            return gloss_out_all[:, 0, :],gloss_out_all[:, 0, :]

    def att_forward(self, gloss_output_pad, current_context_output, gloss_mask):
        return self.att_model(current_context_output.transpose(0, 1), gloss_output_pad, gloss_mask)    #(7,15)






class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, transform_value=True, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.all_head_dim = hidden_dim

        self.query = nn.Linear(hidden_dim, self.all_head_dim)
        self.key = nn.Linear(hidden_dim, self.all_head_dim)
        self.value = nn.Linear(hidden_dim, self.all_head_dim) if transform_value else None
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x, num_heads, attn_head_dim):
        new_x_shape = x.size()[:-1] + (num_heads, attn_head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_in, key_in, target, num_heads, attention_mask):

        self.num_heads = num_heads
        self.attn_head_dim = self.hidden_dim
        query_in = query_in.to(torch.float32)
        # mx_query_layer = self.query(query_in).reshape(query_in.size(0),1,-1)
        mx_query_layer = query_in.reshape(query_in.size(0),1,-1)
        # mx_key_layer = self.key(key_in)
        mx_key_layer = key_in
        # mx_value_layer = self.value(target) if self.value is not None else target
        mx_value_layer = target


        query_layer = self.transpose_for_scores(mx_query_layer, num_heads, query_in.size(-1))
        key_layer = mx_key_layer.unsqueeze(-2).permute(0, 2, 1, 3)
        value_layer = mx_value_layer.unsqueeze(-2).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        ext_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores + (1. - ext_attn_mask) * -10000.
        attn_probs = F.softmax(attn_scores, dim=-1).to(torch.float32)                  #torch.Size([97, 12, 1, gloss_num])


        if self.num_heads == 1:
            context_layer = torch.matmul(attn_probs, value_layer)   #torch.Size([97, 12, 1, 768])
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.squeeze(1)  #torch.Size([97, gloss_num+1, 768])
            return context_layer, attn_probs.squeeze(2)
        attn_probs = self.dropout(attn_probs)  #(bs, 12, 1, glossnum)
        beta = attn_probs.sum(1)/12  #torch.Size(bs, 1, gloss_num])
        context_layer = (beta.unsqueeze(-1).expand(-1,-1,-1,value_layer.size(-1)) * value_layer).sum(-2).squeeze(1) #size(97, 768)
      
        return context_layer, beta

class BertSenseClassifier(nn.Module):
    def __init__(self, mlp_dropout=0.,
            attn_dropout=0., pad_ix=0, unk_ix=0, layer=-1, use_glu=False, residual_glu=False,
            act_fn='gelu', top_attn_head=1, sent_attn_query=False, freeze_bert=False):
        super(BertSenseClassifier, self).__init__()

        # layer-wise attention to weight different layer outputs
        self.layer = layer
        self.hidden_size = 768
        self.layer_attn = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout) if layer < 0 else None
        if self.layer_attn is not None:
            self.layer_attn.apply(self.init_weights)
        self.gloss_attn = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout) if layer < 0 else None
        if self.gloss_attn is not None:
            self.gloss_attn .apply(self.init_weights)
        self.layer_attn2 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.layer_attn2.apply(self.init_weights)
        self.gloss_attn2 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.gloss_attn2.apply(self.init_weights)
        self.layer_attn3 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.layer_attn3.apply(self.init_weights)
        self.gloss_attn3 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.gloss_attn3.apply(self.init_weights)
        self.layer_attn4 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.layer_attn4.apply(self.init_weights)
        self.gloss_attn4 = MultiHeadedAttention(self.hidden_size, transform_value=False, dropout=attn_dropout)
        self.gloss_attn4.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    # def forward(self, sentences, offsets, lexelts, batch_gloss, batch_gloss_num, is_log=True):
    def forward(self, sentence_query, gloss_flat, gloss_mask, is_log=True):

        layer_attn_mask = torch.ones(sentence_query.size(0), sentence_query.size(1))
        layer_attn_mask = layer_attn_mask.to(dtype=gloss_flat.dtype, device=gloss_flat.device)
        g1, beta1 = self.layer_attn(sentence_query, gloss_flat, gloss_flat, sentence_query.size(1), gloss_mask)  #(bs, 768)
        ht1, _ = self.gloss_attn(g1, sentence_query, sentence_query, 1, layer_attn_mask)
        g2, beta2 = self.layer_attn2(ht1, gloss_flat, gloss_flat, 1, gloss_mask)
        ht2, _ = self.gloss_attn2(g2, sentence_query, sentence_query, 1, layer_attn_mask)
        g3, beta3 = self.layer_attn3(ht2, gloss_flat, gloss_flat, 1, gloss_mask)
        ht3, _ = self.gloss_attn3(g3, sentence_query, sentence_query, 1, layer_attn_mask)
        g4, beta4 = self.layer_attn4(ht3, gloss_flat, gloss_flat, 1, gloss_mask)
        slices, _ = self.gloss_attn4(g4, sentence_query, sentence_query, 1, layer_attn_mask)


        logits = (gloss_flat * slices.expand(-1, gloss_flat.size(1), -1)).sum(-1)  # (batch_size*targetword_num, gloss_num)  点乘相似度
        # return logits
        logits = logits + F.relu(1. - gloss_mask) * -10000.

        if is_log:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

#EOF