import re
import os
import json
import copy
import random
import torch
import pickle
import zipfile
import numpy as np
from math import ceil
from collections import Counter
from collections import defaultdict
import operator
from tqdm import tqdm
import torch.nn as nn
from parse_args import args

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def load_jsonl(path):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list

def dump_jsonl(data_list, path):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj) + "\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def index2word(word2id):
    return {i: w for w, i in word2id.items()}

def get_dataset_path(args):
    if args.dataset.endswith("1H"):
        args.max_hop = 1
    
    elif args.dataset.endswith("2H"):
        args.max_hop = 2
    
    elif args.dataset.endswith("3H"):
        args.max_hop = 3

    train_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/qa_train_subgraph_split.json".format(args.dataset)
    valid_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/qa_dev_subgraph_split.json".format(args.dataset)
    test_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/qa_test_subgraph_split.json".format(args.dataset)
    
    entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/entity2id.txt".format(args.dataset)
    kb_relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/relation2id.txt".format(args.dataset)

    entity2types_pkl = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/d_entity2types.pkl".format(args.dataset)
    type2relations_pkl = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/2_subgraph_generate/{}/d_type2relations.pkl".format(args.dataset)

    entity_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/entity_embeddings_ConvE.npy".format(args.dataset)
    relation_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/relation_embeddings_ConvE.npy".format(args.dataset)
    kge_ckpt = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/{}_ConvE_best.ckpt".format(args.dataset, args.dataset)

    retriever_ckpt = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/1_retriever/{}/SimBERT/{}".format(args.dataset, 0)

    word2id_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/word2id.pkl").format(args.dataset)
    word_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/word_embeddings.npy").format(args.dataset)
    output_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/outputs/{}/").format(args.dataset)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return train_path, valid_path, test_path, entity2id_path, kb_relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, output_path, entity2types_pkl, type2relations_pkl, kge_ckpt, retriever_ckpt


def read_vocab(vocab_file):
    d_item2id = {}
    l_items = []

    with open(vocab_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            d_item2id[items[0]] = int(items[1])
            l_items.append(items[0])

    return d_item2id, l_items

def get_id_vocab(entity2id_path, kb_relation2id_path):
    d_entity2id, _ = read_vocab(entity2id_path)
    d_relation2id_kb, l_relation_names = read_vocab(kb_relation2id_path)

    return d_entity2id, d_relation2id_kb, l_relation_names

def build_qa_vocab(train_file, valid_file, word2id_output_file, min_freq):
    flag_words = ['<pad>', '<unk>']
    count = Counter()

    with open(train_file) as f:
        for i, line in enumerate(f):
            if i > 0 and i % 5000 == 0:
                print(i)
            qa_data = json.loads(line.strip())
            question_pattern = qa_data["question_pattern"]

            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)
    
    with open(valid_file) as f:
        for i, line in enumerate(f):
            if i > 0 and i % 5000 == 0:
                print(i)
            qa_data = json.loads(line.strip())
            question_pattern = qa_data["question_pattern"]

            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)

    count = {k: v for k, v in count.items()}
    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    vocab = [w[0] for w in count if w[1] >= min_freq]
    vocab = flag_words + vocab
    word2id = {k: v for k, v in zip(vocab, range(0, len(vocab)))}
    print("word len: ", len(word2id))
    assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"

    with open(word2id_output_file, 'wb') as fw:
        pickle.dump(word2id, fw)

    return word2id

def initialize_word_embedding(word2id, glove_path, word_embedding_file):
    word_embeddings = np.random.uniform(-0.1, 0.1, (len(word2id), 300))
    seen_words = []

    gloves = zipfile.ZipFile(glove_path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in word2id and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        word_embeddings[word2id[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen_words.append(word)

    word_embeddings = word_embeddings.astype(np.float32)
    word_embeddings[0, :] = 0.
    print("pretrained vocab %s among %s" %(len(seen_words), len(word_embeddings)))
    unseen_words = set([k for k in word2id]) - set(seen_words)
    print("unseen words = ", len(unseen_words), unseen_words)
    np.save(word_embedding_file, word_embeddings)
    return word_embeddings

def token_to_id(token, token2id, flag_words = "<unk>"):
    return token2id[token] if token in token2id else token2id[flag_words]

def get_adjacent(triples):
    triple_dict = defaultdict(defaultdict)

    for triple in triples:
        s_id, r_id, o_id = triple
        
        if r_id not in triple_dict[s_id]:
            triple_dict[s_id][r_id] = set()
        triple_dict[s_id][r_id].add(o_id)

    return triple_dict


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l
    
def process_qa_file(qa_file, d_word2id, keep_ratio):
    l_data = []
    with open(qa_file, "r") as f:
        for i, line in enumerate(f):
            qa_data = json.loads(line.strip())

            qid = qa_data["id"]
            question_pattern = qa_data["question_pattern"]
            topic_entity_id = qa_data["topic_entity_id"]
            answer_entities_id = qa_data["answer_entities_id"]

            question_pattern_id = [token_to_id(word, d_word2id) for word in question_pattern.strip().split(" ")]
            l_data.append([qid, question_pattern_id, topic_entity_id, answer_entities_id])
    
    random.shuffle(l_data)
    data_len = len(l_data)
    keep_num = ceil(keep_ratio * data_len)

    l_data = l_data[:keep_num]
            
    return l_data

def getEntityActions(subject, triple_dict, NO_OP_RELATION = 2):
    action_space = []

    if subject in triple_dict:
        for relation in triple_dict[subject]:
            objects = triple_dict[subject][relation]
            for obj in objects: 
                action_space.append((relation, obj))
        
    action_space.insert(0, (NO_OP_RELATION, subject))

    return action_space


def vectorize_action_space(action_space_list, action_space_size, DUMMY_ENTITY = 0, DUMMY_RELATION = 0):
    bucket_size = len(action_space_list)
    r_space = torch.zeros(bucket_size, action_space_size) + DUMMY_ENTITY 
    e_space = torch.zeros(bucket_size, action_space_size) + DUMMY_RELATION
    r_space = r_space.long()
    e_space = e_space.long()
    action_mask = torch.zeros(bucket_size, action_space_size)
    for i, action_space in enumerate(action_space_list):
        for j, (r, e) in enumerate(action_space):
            r_space[i, j] = r
            e_space[i, j] = e
            action_mask[i, j] = 1
    action_mask = action_mask.long()
    return (r_space, e_space), action_mask
    
def initialize_action_space(l_entities, triple_dict, bucket_interval):
    d_action_space_buckets = {}
    d_action_space_buckets_discrete = defaultdict(list)
    tensor_entity2bucketid = torch.zeros(len(l_entities), 2).long()
    num_facts_saved_in_action_table = 0

    for e1 in l_entities:
        action_space = getEntityActions(e1, triple_dict)
        key = int(len(action_space) / bucket_interval) + 1 
        tensor_entity2bucketid[e1, 0] = key
        tensor_entity2bucketid[e1, 1] = len(d_action_space_buckets_discrete[key])
        d_action_space_buckets_discrete[key].append(action_space)
        num_facts_saved_in_action_table += len(action_space)
    
    for key in d_action_space_buckets_discrete:
        d_action_space_buckets[key] = vectorize_action_space(
            d_action_space_buckets_discrete[key], key * bucket_interval)
    
    return tensor_entity2bucketid, d_action_space_buckets

def tensor_id_mapping(tensor, d_vocab):
    tensor_dim = tensor.dim()
    l_tensor = tensor.cpu().numpy().tolist()

    if tensor_dim == 1:
        l_new_id = list(map(lambda x: d_vocab[x], l_tensor))
    
    elif tensor_dim == 2:
        l_new_id = list(map(
            lambda x: list(map(lambda y: d_vocab[y], x)),
            l_tensor
        ))

    return torch.LongTensor(l_new_id).cuda()

def kb_relation_global2local(r_tensor):
    l_r_tensor = r_tensor.cpu().numpy().tolist()
    
    l_all_relations_id = []
    for l_r in l_r_tensor:
        l_all_relations_id += l_r
    l_all_relations_id = list(set(l_all_relations_id))
    
    d_batch_global2local_id = {}
    
    for i, global_id in enumerate(l_all_relations_id): 
        if global_id not in d_batch_global2local_id:
            d_batch_global2local_id[global_id] = i
        
    r_tensor_localId = list(map(
            lambda x: list(map(lambda y: d_batch_global2local_id[y], x)),
            l_r_tensor
        ))

    r_tensor_localId = torch.LongTensor(r_tensor_localId).cuda()
    r_tensor_globalId = torch.LongTensor(l_all_relations_id).cuda()

    return r_tensor_globalId, r_tensor_localId

def generate_local_map(d_entity_global2local, global_ent_id):
    if global_ent_id not in d_entity_global2local:
        d_entity_global2local[global_ent_id] = len(d_entity_global2local)
    
    return d_entity_global2local