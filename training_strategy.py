import os
import time
import torch
import random
from math import ceil
import linecache
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
from rollout import rollout_beam
from parse_args import args
from torch.nn.utils import clip_grad_norm_
from environment import Environment
from Reasoner import Reasoner_Network
from dataloader import Dataset_Model, DataLoader_Model
from utils import index2word, load_jsonl, dump_jsonl

def generate_training_dataset(train_path_ori):
    parent_path = os.path.abspath(os.path.join(train_path_ori, ".."))
    new_train_file = "{}_training_percentage_{}_seed_{}.json".format(args.dataset, args.training_data_percentage, args.seed)
    new_train_path = os.path.join(parent_path, new_train_file)

    if not os.path.exists(new_train_path):
        l_training_data = load_jsonl(train_path_ori)

        random.shuffle(l_training_data)
        data_len = len(l_training_data)
        keep_num = ceil(args.training_data_percentage * data_len)
        l_training_data_new = l_training_data[:keep_num]

        dump_jsonl(l_training_data_new, new_train_path)
        print(args.training_data_percentage, len(l_training_data), len(l_training_data_new))

    return new_train_path


def loading_dataloader(train_path_ori, valid_path, d_entity2id, d_relation2id_kb,d_word2id):
    if train_path_ori is not None:
        new_train_path = generate_training_dataset(train_path_ori)
        train_dataset = Dataset_Model(new_train_path, d_entity2id, d_relation2id_kb, d_word2id)
        train_dataloader = DataLoader_Model(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    
    valid_dataset = Dataset_Model(valid_path, d_entity2id, d_relation2id_kb, d_word2id)
    valid_dataloader = DataLoader_Model(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    if train_path_ori is not None:
        return train_dataloader, valid_dataloader
    else:
        return valid_dataloader


def run_train(train_path, valid_path, output_path, d_entity2id, d_relation2id_kb, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, l_relation_names):
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    valid_csv_collaborate = os.path.join(output_path, "train_valid_{}.csv".format(now))

    ckpt_path = os.path.join(output_path, "rl.ckpt")

    best_epoch, best_dev_metrics = 0, -1
    if os.path.exists(ckpt_path):
        return best_epoch, best_dev_metrics

    print("run_train:")

    train_dataloader, valid_dataloader = loading_dataloader(train_path, valid_path, d_entity2id, d_relation2id_kb, d_word2id)

    env_train = Environment(args, len(d_relation2id_kb))
    env_valid = Environment(args, len(d_relation2id_kb))

    reasoner = Reasoner_Network(args, len(d_relation2id_kb), word_embeddings, entity_embeddings, relation_embeddings, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, d_entity2id, d_relation2id_kb, d_word2id, l_relation_names).cuda()

    optimizer_reasoner = optim.Adam(filter(lambda p: p.requires_grad, reasoner.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)

    best_dev_metrics = -float("inf")
    iters_not_improved = 0
    best_epoch = -1
    best_reasoner_model = reasoner.state_dict()

    l_epochs_valid = []
    l_hits1_valid =[]

    start = time.time()
    for epoch_id in range(0, args.total_epoch):
        reasoner.train()
        total_reasoner_loss = 0

        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit = "batches")

        for i_batch, batch_data in enumerate(train_loader):
            batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global = batch_data

            if use_cuda:
                batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_onehot.cuda(), batch_candidates_onehot.cuda()
            
            batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global)
            
            env_train.reset(batch_data, reasoner)

            rollout_beam(env_train, reasoner)

            reasoner_loss = env_train.calculate_beam_loss()

            optimizer_reasoner.zero_grad()
            reasoner_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(reasoner.parameters(), args.grad_norm)
            
            optimizer_reasoner.step()

            total_reasoner_loss += reasoner_loss.item()

            if use_cuda:
                torch.cuda.empty_cache()
            
            linecache.clearcache()
        
        print("epoch = {}, reasoner_loss = {}.".format(epoch_id, total_reasoner_loss))
    
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            reasoner.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches")
            total_hits1 = 0.0
            total_num = 0.0

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global = batch_data

                    if use_cuda:
                        batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_onehot.cuda(), batch_candidates_onehot.cuda()
                    
                    batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global) 
                    env_valid.reset(batch_data, reasoner)

                    rollout_beam(env_valid, reasoner)

                    hits1_item = env_valid.inference_hits1()
                    total_hits1 += hits1_item
                    total_num += env_valid.batch_size

                answer_hits_1 = 1.0 * total_hits1 / total_num

                l_epochs_valid.append(epoch_id + 1)
                l_hits1_valid.append(round(answer_hits_1, 4))

                if answer_hits_1 > best_dev_metrics:
                    best_dev_metrics = answer_hits_1
                    best_epoch = epoch_id
                    best_reasoner_model = reasoner.state_dict()
                    torch.save(best_reasoner_model, ckpt_path)
                    iters_not_improved = 0
                    print('Epoch {}: best valid Hits@1 = {}.'.format(epoch_id, best_dev_metrics))
                
                elif answer_hits_1 < best_dev_metrics and iters_not_improved * args.num_wait_epochs < args.early_stop_patience:
                    iters_not_improved += 1
                    print("Valid Hits@1 decreases to %f from %f, %d more epoch to check"%(answer_hits_1, best_dev_metrics, args.early_stop_patience - iters_not_improved * args.num_wait_epochs))
                
                elif iters_not_improved * args.num_wait_epochs == args.early_stop_patience:
                    end = time.time()
                    print("Model has exceed patience. Saving best model and exiting. Using {} seconds.".format(round(end - start, 2)))
                    break
    
    d_valid_csv = {"epoch": l_epochs_valid, "Hits@1": l_hits1_valid}
    df_valid = pd.DataFrame(d_valid_csv)
    df_valid.to_csv(valid_csv_collaborate, index=False, sep=',')

    return best_epoch, best_dev_metrics


def run_inference(test_path, output_path, d_entity2id, d_relation2id_kb, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, l_relation_names):
    d_id2word = index2word(d_word2id)
    d_id2entity = index2word(d_entity2id)
    d_id2relation_kb = index2word(d_relation2id_kb)

    def relationid2name(rel_id):
        return d_id2relation_kb[rel_id]

    ckpt_path = os.path.join(output_path, "rl.ckpt")

    if not os.path.exists(ckpt_path):
        return -1

    test_dataloader = loading_dataloader(None, test_path, d_entity2id, d_relation2id_kb, d_word2id)

    env_test = Environment(args, len(d_relation2id_kb))

    reasoner = Reasoner_Network(args, len(d_relation2id_kb), word_embeddings, entity_embeddings, relation_embeddings, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, d_entity2id, d_relation2id_kb, d_word2id, l_relation_names).cuda()

    reasoner.load(ckpt_path)
    reasoner.eval()

    test_loader = tqdm(test_dataloader, total=len(test_dataloader), unit="batches")
    total_hits1 = 0.0
    total_num = 0.0
    total_use_dc_rel = 0
    step_use_dc_rel = np.zeros(args.max_hop)

    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global = batch_data

            if use_cuda:
                batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_onehot.cuda(), batch_candidates_onehot.cuda()
            
            batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_onehot, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global) 

            env_test.reset(batch_data, reasoner)

            log_reasoner_action_prob_history = rollout_beam(env_test, reasoner)

            hits1_item = env_test.inference_hits1()
            total_hits1 += hits1_item
            total_num += env_test.batch_size
        
        answer_hits_1 = 1.0 * total_hits1 / total_num

    return answer_hits_1