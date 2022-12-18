from ast import arg
import os
import time
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from utils import set_seed, read_vocab
from parse_args_ConvE import args
from sklearn.model_selection import train_test_split


START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'
PADDING_ENTITIES = [DUMMY_ENTITY, NO_OP_ENTITY]
PADDING_ENTITIES_ID = [0, 1]
PADDING_RELATIONS = [DUMMY_RELATION, START_RELATION, NO_OP_RELATION]

def run_train(args, model, train_data, dev_data, dev_objects, dummy_mask, checkpoint_path):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_dev_metrics = -float("inf")
    iters_not_improved = 0
    best_model = model.state_dict()

    start = time.time()
    for epoch_id in range(0, args.num_epochs):
        model.train()
        random.shuffle(train_data)
        batch_losses = []

        epoch_start = time.time()
        for example_id in range(0, len(train_data), args.batch_size):
            optimizer.zero_grad()
            mini_batch = train_data[example_id:example_id + args.batch_size]
            loss = model.loss(mini_batch)
            loss.backward()

            if args.grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.grad_norm)

            optimizer.step()

            batch_losses.append(loss.item())
        
        epoch_end = time.time()
        print('Epoch {}: average training loss = {}, using {} seconds.'.format(epoch_id, np.mean(batch_losses), round(epoch_end - epoch_start, 2)))
        
        if use_cuda:
            torch.cuda.empty_cache()

        
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            model.eval()

            with torch.no_grad():
                total_mrr = 0.0
                for example_id in range(0, len(dev_data), args.batch_size):
                    mini_batch = dev_data[example_id:example_id + args.batch_size]
                    dev_batch_score = model.forward(mini_batch)
                
                    _, _, _, _, mrr = hits_and_ranks(mini_batch, dev_batch_score, dev_objects, dummy_mask)
                    total_mrr += mrr
                
                total_mrr = total_mrr / len(dev_data)
                metrics = total_mrr 
                if metrics > best_dev_metrics:
                    best_dev_metrics = metrics
                    iters_not_improved = 0
                    best_model = model.state_dict()
                    print('Epoch {}: best vaild MRR = {}.'.format(epoch_id, best_dev_metrics))
                
                elif metrics < best_dev_metrics and iters_not_improved * args.num_wait_epochs < args.early_stop_patience:
                    iters_not_improved += 1
                    print("Vaild MRR decreases to %f from %f, %d more epoch to check"%(metrics, best_dev_metrics, args.early_stop_patience - iters_not_improved * args.num_wait_epochs))
                elif iters_not_improved * args.num_wait_epochs == args.early_stop_patience:
                    end = time.time()
                    print("Model has exceed patience. Saving best model and exiting. Using {} seconds.".format(round(end - start, 2)))
                    torch.save(best_model, checkpoint_path)
                    break

                if epoch_id == args.num_epochs-1:
                    end = time.time()
                    print("Final Epoch has reached. Stopping and saving model. Using {} seconds.".format(round(end - start, 2)))
                    torch.save(best_model, checkpoint_path)
                    break
                    

    return best_dev_metrics

def run_inference(args, model, test_data, test_objects, dummy_mask, checkpoint_path):
    model.eval()
    model.load(checkpoint_path)
    final_hits_1 = 0.0
    final_hits_3 = 0.0
    final_hits_5 = 0.0
    final_hits_10 = 0.0
    final_mrr = 0.0

    for example_id in range(0, len(test_data), args.batch_size):
        mini_batch = test_data[example_id:example_id + args.batch_size]
        test_batch_score = model.forward(mini_batch)
    
        hits_1, hits_3, hits_5, hits_10, mrr = hits_and_ranks(mini_batch, test_batch_score, test_objects, dummy_mask)
        final_hits_1 += hits_1
        final_hits_3 += hits_3
        final_hits_5 += hits_5
        final_hits_10 += hits_10
        final_mrr += mrr
    
    final_hits_1 = round(final_hits_1 / len(test_data), 4)
    final_hits_3 = round(final_hits_3 / len(test_data), 4)
    final_hits_5 = round(final_hits_5 / len(test_data), 4)
    final_hits_10 = round(final_hits_10 / len(test_data), 4)
    final_mrr = round(final_mrr / len(test_data), 4)
    metrics = (final_hits_1, final_hits_3, final_hits_5, final_hits_10, final_mrr)

    return metrics


class ConvE(nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    def forward(self, E1, R, E2):
        E1 = E1.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = R.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))

        S = torch.sigmoid(X)
        return S

    def forward_fact(self, E1, R, E2):
        E1 = E1.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2) 
        R = R.view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)

        S = torch.sigmoid(X)
        return S

class KGE_framework(nn.Module):
    def __init__(self, args, num_entities, num_relations, use_cuda):
        super(KGE_framework, self).__init__()
        self.use_cuda = use_cuda
        self.batch_size = args.batch_size
        self.label_smoothing_epsilon = args.label_smoothing_epsilon
        self.margin = args.margin
        self.model = args.KGE_model
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dropout_rate = args.emb_dropout_rate

        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

        self.initialize_modules() 
        self.loss_fun = nn.BCELoss()
        self.kge_model = ConvE(args)
    
    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
    
    def format_batch(self, batch_data, num_labels=-1):
        def convert_to_binary_multi_object(e2):
            e2_label = torch.zeros([len(e2), num_labels])
            if self.use_cuda:
                e2_label = e2_label.cuda()
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, r, e2 = batch_data[i]
            batch_e1.append(e1)
            batch_r.append(r)
            batch_e2.append(e2)

        batch_e1 = torch.LongTensor(batch_e1)
        batch_r = torch.LongTensor(batch_r)

        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        else:
            batch_e2 = torch.LongTensor(batch_e2)
        
        if self.use_cuda:
            batch_e1 = batch_e1.cuda()
            batch_r = batch_r.cuda()
            batch_e2 = batch_e2.cuda()

        return batch_e1, batch_r, batch_e2

    def loss(self, mini_batch):
        e1, r, e2 = self.format_batch(mini_batch, num_labels = self.num_entities)
        e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))

        E1 = self.EDropout(self.entity_embeddings(e1))
        R = self.RDropout(self.relation_embeddings(r))
        E2 = self.EDropout(self.entity_embeddings.weight)
        
        pred_scores = self.kge_model.forward(E1, R, E2)
        loss = self.loss_fun(pred_scores, e2_label)

        return loss
    
    def predict(self, mini_batch):
        e1, r, e2 = self.format_batch(mini_batch)
        E1 = self.EDropout(self.entity_embeddings(e1))
        R = self.RDropout(self.relation_embeddings(r))
        E2 = self.EDropout(self.entity_embeddings.weight)
        pred_scores = self.kge_model.forward(E1, R, E2)
        
        return pred_scores

    def forward(self, mini_batch):
        pred_score = self.predict(mini_batch)
        return pred_score
    
    def predict_rl(self, batch_entities, batch_relations):
        E1 = self.entity_embeddings(batch_entities)
        R = self.relation_embeddings(batch_relations)
        E2 = self.EDropout(self.entity_embeddings.weight)

        pred_scores = self.kge_model.forward(E1, R, E2)
        return pred_scores


    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)

def get_adjacent_from_train_valid_test(train_triples, valid_triples, test_triples = None):
    triple_dict = defaultdict(defaultdict)
    if test_triples == None:
        merge_triples = train_triples + valid_triples
    else:
        merge_triples = train_triples + valid_triples + test_triples

    for triple in merge_triples:
        s, r, o = triple
        if r not in triple_dict[s]:
            triple_dict[s][r] = set()
        triple_dict[s][r].add(o)
    
    return triple_dict

def load_all_triples_from_txt(data_path, d_entity2id, d_relation2id, add_reverse_relations=False):
    triples = []
    
    def triple2ids(s, r, o):
        return (d_entity2id[s], d_relation2id[r], d_entity2id[o])
    
    with open(data_path) as f:
        for line in f.readlines():
            if args.dataset.startswith("MetaQA"):
                s, r, o = line.lower().strip().split("\t")
            elif args.dataset == "WebQSP":
                s, r, o = line.split("\t")

            s, r, o = s.strip(), r.strip(), o.strip()
            triples.append(triple2ids(s, r, o))
            if add_reverse_relations:
                triples.append(triple2ids(o, r + '_inverse', s))
    
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def load_triples(triples, group_examples_by_query=False):
    new_triples = []
    triple_dict = defaultdict(defaultdict)

    for triple in triples:
        s_id, r_id, o_id = triple
        
        if r_id not in triple_dict[s_id]:
            triple_dict[s_id][r_id] = set()
        triple_dict[s_id][r_id].add(o_id)

        new_triples.append(triple)
    
    if group_examples_by_query:
        new_triples = []
        for e1_id in triple_dict:
            for r_id in triple_dict[e1_id]:
                new_triples.append((e1_id, r_id, list(triple_dict[e1_id][r_id])))

    return new_triples, triple_dict

def hits_and_ranks(examples, scores, dev_objects, dummy_mask): 
    assert (len(examples) == scores.shape[0])
    for i, example in enumerate(examples):
        e1, r, e2 = example
        e2_multi = dummy_mask + list(dev_objects[e1][r]) 
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score
    
    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, scores.size(1))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    
    for i, example in enumerate(examples):
        e1, r, e2 = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1

            mrr += 1.0 / (pos + 1)

    return (hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr)