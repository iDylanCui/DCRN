import os
import torch
import pickle
import itertools
import numpy as np
from parse_args import args
from utils import set_seed, get_dataset_path, get_id_vocab, build_qa_vocab, initialize_word_embedding, flatten
from training_strategy import run_train

def get_hyperparameter_range():
    hp_KEQA = ["training_data_percentage", "kb_triples_percentage", "seed", "max_hop", "loss_with_beam", "use_dynamic_action_space", "total_supplement_action_num", "supplement_relation_num", "use_relation_pruner"]

    hp_KEQA_range = {
        "training_data_percentage": [1.0],
        "kb_triples_percentage": [0.5, 0.3, 0.1],
        "seed": [2023], 
        "max_hop": [args.max_hop],
        "loss_with_beam": [True],
        "use_dynamic_action_space": [True],
        "total_supplement_action_num": [10],
        "supplement_relation_num": [2, 5, 10],
        "use_relation_pruner": [True],
    }

    return hp_KEQA, hp_KEQA_range

def grid_search(train_path, valid_path, output_path, d_entity2id, d_relation2id_kb, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, l_relation_names):
    hp_model, hp_model_range = get_hyperparameter_range()
    grid = hp_model_range[hp_model[0]]
    for hp in hp_model[1:]:
        grid = itertools.product(grid, hp_model_range[hp])
    
    grid_results = {}
    grid = list(grid)

    out_log_path = os.path.join(output_path, "log.txt")
    if not os.path.exists(out_log_path):
        with open(out_log_path, "w")  as ft:
            ft.write("** Grid Search **\n")
            ft.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
            ft.write('Signature\tbest_epoch_valid\tHits@1_valid\n')
    
    for i, grid_entry in enumerate(grid):
        if type(grid_entry) is not list:
            grid_entry = [grid_entry]
        
        grid_entry = flatten(grid_entry)
        print('* Hyperparameter Set {} = {}'.format(i, grid_entry)) 

        signature = ''
        for j in range(len(grid_entry)):
            hp = hp_model[j]
            value = grid_entry[j]

            if hp in ["loss_with_beam", "use_dynamic_action_space", "use_relation_pruner", "use_candidate_entities_constraint"]:
                setattr(args, hp, bool(value))
            elif hp in ["training_data_percentage", "kb_triples_percentage"]:
                setattr(args, hp, float(value))
            else:
                setattr(args, hp, int(value))
            
            signature += '{}_{} '.format(hp, value)
        
        signature = signature.strip()

        set_seed(args.seed)

        grid_entry_path = os.path.join(output_path, signature)

        if not os.path.exists(grid_entry_path):
            os.makedirs(grid_entry_path)
        
        best_epoch, best_dev_metrics = run_train(train_path, valid_path, grid_entry_path, d_entity2id, d_relation2id_kb, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, l_relation_names)

        if best_dev_metrics != -1:
            with open(out_log_path, "a") as f:
                f.write(signature + "\t" + str(best_epoch) + "\t" + str(round(best_dev_metrics, 4)) + "\n")
        
        grid_results[signature] = best_dev_metrics

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    train_path, valid_path, test_path, entity2id_path, kb_relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, output_path, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path = get_dataset_path(args)

    d_entity2id, d_relation2id_kb, l_relation_names = get_id_vocab(entity2id_path, kb_relation2id_path)

    if not os.path.isfile(word2id_path):
        d_word2id = build_qa_vocab(train_path, valid_path, word2id_path, args.min_freq)
    else:
        d_word2id = pickle.load(open(word2id_path, 'rb'))
    
    if not os.path.isfile(word_embedding_path):
        glove_path = "/home/hai/hai_disk/Codes_hub/My_multi-agent_p5/datasets/glove.840B.300d.zip"
        word_embeddings = initialize_word_embedding(d_word2id, glove_path, word_embedding_path)
    else:
        word_embeddings = np.load(word_embedding_path)
    word_embeddings = torch.from_numpy(word_embeddings)

    if os.path.isfile(entity_embedding_path):
        entity_embeddings = np.load(entity_embedding_path)
        entity_embeddings = torch.from_numpy(entity_embeddings)
    
    if os.path.isfile(relation_embedding_path):
        relation_embeddings = np.load(relation_embedding_path)
        relation_embeddings = torch.from_numpy(relation_embeddings)

    if args.grid_search:
        grid_search(train_path, valid_path, output_path, d_entity2id, d_relation2id_kb, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, l_relation_names)