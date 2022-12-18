import json
import torch
import random
import linecache
from math import ceil
from parse_args import args
from torch.utils.data import Dataset, DataLoader
from utils import token_to_id, get_adjacent, initialize_action_space, index2word, generate_local_map

class Dataset_Model(Dataset):
    def __init__(self, data_path, d_entity2id, d_relation2id_kb, d_word2id):
        self.data_path = data_path
        self.entity_num = len(d_entity2id)
        self.d_entity2id = d_entity2id
        self.d_relation2id_kb = d_relation2id_kb
        self.d_word2id = d_word2id
    
    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        one_hot = torch.LongTensor(self.entity_num)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __len__(self):
        len_num = len(linecache.getlines(self.data_path))
        
        return len_num

    def __getitem__(self, index):
        line = linecache.getline(self.data_path, index + 1)
        qa_data = json.loads(line.strip())

        q_id = qa_data["id"]
        question_pattern = qa_data["question_pattern"]
        topic_entity = qa_data["topic_entity"]
        answer_entities = qa_data["answer_entities"]

        if args.use_candidate_entities_constraint:
            candidate_entities = qa_data["candidate_entities"]
        else:
            candidate_entities = [ent for i, ent in enumerate(self.d_entity2id.keys()) if i > 1]

        candidate_triples = qa_data["candidate_triples"]
        
        topic_entity_id = self.d_entity2id[topic_entity.strip()]
        answer_entities_id = [self.d_entity2id[ent.strip()] for ent in answer_entities]
        candidate_kg_entities_id = [self.d_entity2id[ent.strip()] for ent in candidate_entities]
        candidate_triples_id = [(self.d_entity2id[s.strip()], self.d_relation2id_kb[r.strip()], self.d_entity2id[o.strip()]) for (s, r, o) in candidate_triples]

        question_pattern_id = [token_to_id(word, self.d_word2id) for word in question_pattern.strip().split(" ")]

        answer_entities_onehot = self.toOneHot(answer_entities_id)
        candidate_kg_entities_onehot = self.toOneHot(candidate_kg_entities_id)

        random.shuffle(candidate_triples_id)
        triples_len = len(candidate_triples_id)
        keep_num = ceil(args.kb_triples_percentage * triples_len)
        candidate_triples_id = candidate_triples_id[:keep_num]

        return [q_id, torch.LongTensor(question_pattern_id), topic_entity_id, answer_entities_onehot, answer_entities_id, candidate_kg_entities_onehot, candidate_kg_entities_id, candidate_triples_id]

def _collate_fn(batch):
    d_entity_global2local = {0: 0, 1: 1} 

    sorted_seq = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
    sorted_seq_lengths = [len(i[1]) for i in sorted_seq] 
    if args.use_ecm_tokens_internal_memory:
        longest_sample = args.max_question_len_global
    else:
        longest_sample = sorted_seq_lengths[0]

    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    l_input_lengths = []
    l_head_global = []
    l_tail_onehot = []
    l_candidate_onehot = []
    l_batch_entities_global = []
    l_batch_entities_local = []
    l_batch_kg_triples_global = []
    l_batch_kg_triples_local = []

    for x in range(minibatch_size):
        qid = sorted_seq[x][0]
        question_pattern_id = sorted_seq[x][1]
        seq_len = len(question_pattern_id)

        l_input_lengths.append(seq_len)
        inputs[x].narrow(0,0,seq_len).copy_(question_pattern_id)

        topic_entity_id = sorted_seq[x][2]
        l_head_global.append(topic_entity_id)

        answer_entities_onehot = sorted_seq[x][3]
        l_tail_onehot.append(answer_entities_onehot)

        answer_entities_id = sorted_seq[x][4]

        candidate_kg_entities_onehot = sorted_seq[x][5]
        l_candidate_onehot.append(candidate_kg_entities_onehot)

        candidate_kg_entities_id = sorted_seq[x][6]

        candidate_triples_id = sorted_seq[x][7]
        l_batch_kg_triples_global.extend(candidate_triples_id)

        all_entities = answer_entities_id + candidate_kg_entities_id + [topic_entity_id]
        l_batch_entities_global.extend(all_entities)

    l_batch_entities_global = list(set(l_batch_entities_global))
    l_batch_kg_triples_global = list(set(l_batch_kg_triples_global))

    for ent_id in l_batch_entities_global:
        d_entity_global2local = generate_local_map(d_entity_global2local, ent_id)
    
    d_entity_local2global = index2word(d_entity_global2local)
    
    l_batch_entities_local = list(d_entity_local2global.keys())

    for (s_global, r_global, o_global) in l_batch_kg_triples_global:
        s_local = d_entity_global2local[s_global]
        o_local = d_entity_global2local[o_global]
        l_batch_kg_triples_local.append((s_local, r_global, o_local))

    d_batch_kg_adjacent_local = get_adjacent(l_batch_kg_triples_local)

    tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local = initialize_action_space(l_batch_entities_local, d_batch_kg_adjacent_local, args.bucket_interval)

    del d_batch_kg_adjacent_local

    return inputs, torch.LongTensor(l_input_lengths), torch.LongTensor(l_head_global), torch.stack(l_tail_onehot), torch.stack(l_candidate_onehot), tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global

class DataLoader_Model(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_Model, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn