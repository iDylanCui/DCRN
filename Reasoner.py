import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Transformer import TransformerModel
from transformers import AutoModel, AutoTokenizer
from ConvE import KGE_framework
from parse_args_ConvE import args as args_convE
from utils import kb_relation_global2local, index2word, tensor_id_mapping


class Reasoner_Network(nn.Module):
    def __init__(self, args, kb_relation_num, word_embeddings, entity_embeddings, relation_embeddings, entity2types_pkl_path, type2relations_pkl_path, kge_ckpt_path, retriever_ckpt_path, d_entity2id, d_relation2id, d_word2id, l_relation_names):
        super(Reasoner_Network, self).__init__()

        self.max_hop = args.max_hop
        self.kb_relation_num = kb_relation_num

        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.DUMMY_RELATION_idx = args.DUMMY_RELATION_idx
        self.DUMMY_ENTITY_idx = args.DUMMY_ENTITY_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_emb = args.is_train_emb
        self.max_question_len_global = args.max_question_len_global
        self.use_ecm_tokens_internal_memory = args.use_ecm_tokens_internal_memory
        self.use_dynamic_action_space = args.use_dynamic_action_space
        self.temperature_theta = args.temperature_theta
        self.use_relation_pruner = args.use_relation_pruner
        self.use_candidate_entities_constraint = args.use_candidate_entities_constraint

        self.total_supplement_action_num = args.total_supplement_action_num
        self.supplement_relation_num = args.supplement_relation_num
        assert(self.total_supplement_action_num % self.supplement_relation_num == 0)
        self.supplement_entity_num = int(self.total_supplement_action_num / self.supplement_relation_num)

        self.entity2types_pkl_path = entity2types_pkl_path
        self.type2relations_pkl_path = type2relations_pkl_path
        self.kge_ckpt_path = kge_ckpt_path
        self.retriever_ckpt_path = retriever_ckpt_path
        
        self.d_entity2id = d_entity2id
        self.d_relation2id = d_relation2id
        self.d_word2id = d_word2id
        self.d_id2entity = index2word(self.d_entity2id)
        self.d_id2relation = index2word(self.d_relation2id)
        self.d_id2word = index2word(self.d_word2id)
        self.l_relation_names = l_relation_names

        self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_ckpt_path)
        self.retriever = AutoModel.from_pretrained(self.retriever_ckpt_path).cuda()

        if not args.train_retriever:
            for param in self.retriever.parameters():
                param.requires_grad = False
            self.retriever.eval()

        self.kge_model = KGE_framework(args_convE, len(d_entity2id), len(d_relation2id), use_cuda = True).cuda()

        self.kge_model.load(self.kge_ckpt_path)

        if not args.train_kge:
            for param in self.kge_model.parameters():
                param.requires_grad = False
            self.kge_model.eval()

        with open(self.entity2types_pkl_path, 'rb') as f:
            self.d_entity2types = pickle.load(f)
        
        with open(self.type2relations_pkl_path, 'rb') as f:
            self.d_type2relations = pickle.load(f)

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.transform_output_dim = self.word_dim

        self.relation_only = args.relation_only
        self.history_dim = args.history_dim
        self.history_layers = args.history_layers
        self.rl_dropout_rate = args.rl_dropout_rate

        self.word_embeddings = Embedding(word_embeddings, self.word_dropout_rate, self.is_train_emb, self.word_padding_idx)

        self.entity_embeddings = Embedding(entity_embeddings, self.emb_dropout_rate, self.is_train_emb, self.DUMMY_ENTITY_idx)
        self.relation_embeddings = Embedding(relation_embeddings, self.emb_dropout_rate, self.is_train_emb, self.DUMMY_RELATION_idx)

        self.Transformer = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate)

        if self.use_ecm_tokens_internal_memory:
            self.l_internal_state_W_read_KG = nn.ModuleList([nn.Linear(self.transform_output_dim + self.history_dim + self.relation_dim, self.max_question_len_global) for i in range(self.max_hop)])

            self.l_internal_state_W_write_KG = nn.ModuleList([nn.Linear(self.transform_output_dim + self.relation_dim, self.max_question_len_global) for i in range(self.max_hop)])
        
        else:
            self.l_question_linears = nn.ModuleList([nn.Linear(self.word_dim, self.relation_dim) for i in range(self.max_hop)])
            self.relation_linear = nn.Linear(self.relation_dim, self.relation_dim)
            self.W_att = nn.Linear(self.relation_dim, 1)

        self.input_dim = self.history_dim + self.word_dim
        
        if self.relation_only:
            self.action_dim = self.relation_dim
        else:
            self.action_dim = self.relation_dim + self.entity_dim
        
        self.lstm_input_dim = self.action_dim
        
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W2Dropout = nn.Dropout(self.rl_dropout_rate)

        self.use_entity_embedding_in_vn = args.use_entity_embedding_vn
        if self.use_entity_embedding_in_vn:
            self.W_value = nn.Linear(self.history_dim + self.entity_dim, 1)
        else:
            self.W_value = nn.Linear(self.history_dim, 1) 

        self.path_encoder = nn.LSTM(input_size=self.lstm_input_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_layers,
                                    batch_first=True)
        self.initialize_modules()

    def get_question_representation(self, batch_question, batch_sent_len):
        batch_question_embedding = self.word_embeddings(batch_question) 
        mask = self.batch_sentence_mask(batch_sent_len)
        transformer_output = self.Transformer(batch_question_embedding.permute(1, 0 ,2), mask)
        
        transformer_output = transformer_output.permute(1, 0 ,2)

        return transformer_output, mask
    
    def question_max_pooling(self, transformer_out, question_mask):
        _, _, output_dim = transformer_out.shape
        question_mask = question_mask.unsqueeze(-1).repeat(1,1, output_dim)
        transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))
        question_transformer_masked = transformer_out_masked.transpose(1, 2)
        question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)

        return question_mp
    
    def get_relation_aware_question_vector_attention(self, t, b_question_vectors, b_question_mask, b_r_embeddings, b_r_localId):
        relation_num, _ = b_r_embeddings.shape
        b_size, seq_len = b_question_mask.shape
        b_question_vectors = b_question_vectors.unsqueeze(1).repeat(1, relation_num, 1, 1).view(b_size * relation_num, seq_len, -1)
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, relation_num, 1).view(b_size * relation_num, seq_len)

        b_question_project = self.l_question_linears[t](b_question_vectors)
        
        b_relation_vector = b_r_embeddings.unsqueeze(1).unsqueeze(0).repeat(b_size, 1, seq_len, 1).view(b_size * relation_num, seq_len, -1)
        b_relation_project = self.relation_linear(b_relation_vector)

        b_att_features = b_question_project + b_relation_project
        b_att_features_tanh = torch.tanh(b_att_features)
        b_linear_result = self.W_att(b_att_features_tanh).squeeze(-1)

        b_linear_result_masked = b_linear_result.masked_fill(b_question_mask, float('-inf'))
        b_matrix_alpha = F.softmax(b_linear_result_masked, 1).unsqueeze(1)

        b_relation_aware_question_vector = torch.matmul(b_matrix_alpha, b_question_vectors).squeeze(1).view(b_size, relation_num, -1)

        b_matrix_alpha = b_matrix_alpha.squeeze(1).view(b_size, relation_num, -1)

        l_relation_aware_question_vector = []
        l_matrix_alpha = []

        for batch_i in range(b_size):
            output_i = b_relation_aware_question_vector[batch_i]
            matrix_i = b_matrix_alpha[batch_i]
            relation_i = b_r_localId[batch_i]
            new_output_i = output_i[relation_i]
            new_matrix_i = matrix_i[relation_i]

            l_relation_aware_question_vector.append(new_output_i)
            l_matrix_alpha.append(new_matrix_i)
        
        b_relation_aware_question_vector = torch.stack(l_relation_aware_question_vector, 0)
        b_matrix_alpha = torch.stack(l_matrix_alpha, 0)

        return b_relation_aware_question_vector, b_matrix_alpha

    def get_relation_aware_question_vector_ecm(self, t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_embeddings, b_token_memory_state):
        b_size, action_num, hidden_dim = b_r_embeddings.shape 

        b_question_mp = b_question_mp.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_question_vectors = b_question_vectors.unsqueeze(1).repeat(1, action_num, 1, 1).view(b_size * action_num, self.max_question_len_global, -1)

        b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_relation_vector = b_r_embeddings.view(b_size * action_num, -1)
        b_token_memory_state = b_token_memory_state.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)

        b_internal_state_input = torch.cat([b_question_mp, b_path_hidden, b_relation_vector], dim = -1)

        b_internal_state_read = torch.sigmoid(self.l_internal_state_W_read_KG[t](b_internal_state_input))

        b_internal_memory = b_internal_state_read * b_token_memory_state

        b_internal_memory_masked = b_internal_memory.masked_fill(b_question_mask, float('-inf'))
        b_internal_memory_softmax = F.softmax(b_internal_memory_masked, 1).unsqueeze(1)

        b_question_vectors_att = torch.matmul(b_internal_memory_softmax, b_question_vectors).squeeze(1)
        
        b_internal_state_output = torch.cat([b_relation_vector, b_question_vectors_att], dim = -1)

        b_internal_state_write = torch.sigmoid(self.l_internal_state_W_write_KG[t](b_internal_state_output))

        b_token_memory_state_new = b_internal_state_write * b_token_memory_state

        return b_question_vectors_att.view(b_size, action_num, -1), b_token_memory_state_new.view(b_size, action_num, -1)

    def get_text_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True,
                        truncation=True, return_tensors="pt")

        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = self.retriever(**inputs, output_hidden_states=True,
                        return_dict=True).pooler_output

        return embeddings

    def get_dynamic_action_space(self, b_e_t, b_question, b_relation_paths, b_cadidates_onehot, relation_vectors_retriever):
        l_question_id = b_question.cpu().numpy().tolist()
        l_question_text = list(map(
            lambda x: list(map(lambda y: self.d_id2word[y], x)),
            l_question_id
        ))

        l_question_sentences = []
        for l_question in l_question_text:
            question = " ".join([word for word in l_question if word != '<pad>'])
            l_question_sentences.append(question)
        
        l_relation_paths_id = b_relation_paths.cpu().numpy().tolist()
        l_relation_paths_text = list(map(
            lambda x: list(map(lambda y: self.d_id2relation[y], x)),
            l_relation_paths_id
        )) 
        l_retriever_query_input = []

        for question, l_relation_paths in zip(l_question_sentences, l_relation_paths_text):
            l_relation_paths = l_relation_paths[1:]
            query = ' </s> '.join([question] + l_relation_paths) + ' </s>'
            l_retriever_query_input.append(query)
        
        
        b_query_vectors = self.get_text_embeddings(l_retriever_query_input).unsqueeze(1)

        b_sim_score = torch.cosine_similarity(b_query_vectors, relation_vectors_retriever, dim=2) / self.temperature_theta 

        if self.use_relation_pruner:
            b_e_ids = b_e_t.cpu().numpy().tolist()
            b_e_names = list(map(lambda y: self.d_id2entity[y], b_e_ids))
            b_relation_prune_mask = torch.zeros_like(b_sim_score).long()
            l_possible_relation_num = []
            
            for i, current_ent in enumerate(b_e_names):
                if current_ent in self.d_entity2types:
                    s_current_ent_types = self.d_entity2types[current_ent]
                else:
                    s_current_ent_types = None
                
                l_candidate_relations = []

                if s_current_ent_types is not None:
                    for ent_type in s_current_ent_types:
                        if ent_type in self.d_type2relations:
                            l_candidate_relations.extend(self.d_type2relations[ent_type])
                    
                    l_candidate_relations = list(set(l_candidate_relations))
                    l_candidate_relations_id = [self.d_relation2id[rel] for rel in l_candidate_relations]
                    b_relation_prune_mask[i][l_candidate_relations_id] = 1 
                else:
                    b_relation_prune_mask[i] = 1
                
                b_relation_prune_mask[i][:3] = 0
                candidate_relations_num = torch.sum(b_relation_prune_mask[i]).item()
                l_possible_relation_num.append(int(candidate_relations_num))

        else:
            b_relation_prune_mask = torch.ones_like(b_sim_score).long()
            b_relation_prune_mask[:, :3] = 0
            b_relation_num = torch.sum(b_relation_prune_mask, dim = -1)

            l_possible_relation_num = b_relation_num.cpu().numpy().tolist()

        b_sim_score_masked = b_sim_score.masked_fill((1 - b_relation_prune_mask).bool(), float('-inf'))
        b_size = b_sim_score_masked.shape[0]

        b_selected_relation_idx = torch.zeros(b_size, self.supplement_relation_num, dtype=torch.long).cuda() + self.DUMMY_RELATION_idx

        for i in range(b_size):
            selected_num = min(self.supplement_relation_num, l_possible_relation_num[i])
            _, relation_idx = torch.topk(b_sim_score_masked[i], selected_num, dim=-1)
            b_selected_relation_idx[i][:selected_num] = relation_idx

        b_e_t_expand = b_e_t.repeat_interleave(self.supplement_relation_num, dim=0)

        b_entity_scores = self.kge_model.predict_rl(b_e_t_expand, b_selected_relation_idx.view(-1))


        if self.use_candidate_entities_constraint:
            b_cadidates_onehot_expand = b_cadidates_onehot.repeat_interleave(self.supplement_relation_num, dim=0)
            b_entity_num = torch.sum(b_cadidates_onehot_expand, dim = -1)
            l_possible_entity_num = b_entity_num.cpu().numpy().tolist()

            b_entity_score_masked = b_entity_scores.masked_fill((1 - b_cadidates_onehot_expand).bool(), float('-inf'))

            b_selected_entity_idx = torch.zeros(b_size * self.supplement_relation_num, self.supplement_entity_num, dtype=torch.long).cuda() + self.DUMMY_ENTITY_idx

            for i in range(b_size * self.supplement_relation_num):
                selected_num = min(self.supplement_entity_num, l_possible_entity_num[i])
                _, entity_idx = torch.topk(b_entity_score_masked[i], selected_num, dim=-1)
                b_selected_entity_idx[i][:selected_num] = entity_idx

        else:
            b_entity_scores[:, :2] = float('-inf')
            _, b_selected_entity_idx = torch.topk(b_entity_scores, self.supplement_entity_num, dim=-1)

        new_r_space = b_selected_relation_idx.repeat_interleave(self.supplement_entity_num, dim=1)
        new_e_space = b_selected_entity_idx.view(b_size, -1)

        new_action_mask = (new_r_space != self.DUMMY_RELATION_idx).long()
        new_action_mask = new_action_mask * (new_e_space != self.DUMMY_ENTITY_idx).long()

        return new_e_space, new_r_space, new_action_mask

    def get_action_space_in_buckets(self, batch_e_t_global, batch_question, batch_cadidates_onehot, batch_relation_paths, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local,  d_entity_global2local, d_entity_local2global):
        l_action_spaces_global, l_references, l_action_flags = [], [], []

        if self.use_dynamic_action_space:
            relation_vectors_retriever = self.get_text_embeddings(self.l_relation_names).unsqueeze(0)

        batch_e_t_local = tensor_id_mapping(batch_e_t_global, d_entity_global2local)

        entity2bucketid_local = tensor_entity2bucketid_batch_local[batch_e_t_local.tolist()]
        key1 = entity2bucketid_local[:, 0]
        key2 = entity2bucketid_local[:, 1]
        batch_ref = {}

        for i in range(len(batch_e_t_local)):
            key = int(key1[i])
            if not key in batch_ref:
                batch_ref[key] = []
            batch_ref[key].append(i)
    
        for key in batch_ref:
            action_space = d_action_space_buckets_batch_local[key]

            l_batch_refs = batch_ref[key]
            g_bucket_ids = key2[l_batch_refs].tolist()
            r_space_b_kg_global = action_space[0][0][g_bucket_ids]
            e_space_b_kg_local = action_space[0][1][g_bucket_ids]
            action_mask_b_kg = action_space[1][g_bucket_ids]

            e_space_b_kg_global = tensor_id_mapping(e_space_b_kg_local, d_entity_local2global)

            r_space_b_kg_global = r_space_b_kg_global.cuda()
            e_space_b_kg_global = e_space_b_kg_global.cuda()
            action_mask_b_kg = action_mask_b_kg.cuda()

            if self.use_dynamic_action_space:
                b_e_t = batch_e_t_global[l_batch_refs]
                b_question = batch_question[l_batch_refs]
                b_relation_paths = batch_relation_paths[l_batch_refs]
                b_cadidates_onehot = batch_cadidates_onehot[l_batch_refs]

                new_e_space_b, new_r_space_b, new_action_mask_b = self.get_dynamic_action_space(b_e_t, b_question, b_relation_paths, b_cadidates_onehot, relation_vectors_retriever)

                action_flags_b = torch.cat([torch.ones_like(e_space_b_kg_global), torch.ones_like(new_e_space_b) + 1], dim=-1)
                e_space_b_merge = torch.cat([e_space_b_kg_global, new_e_space_b], dim=-1)
                r_space_b_merge = torch.cat([r_space_b_kg_global, new_r_space_b], dim=-1)
                action_mask_b_merge = torch.cat([action_mask_b_kg, new_action_mask_b], dim=-1)
                action_space_b = ((r_space_b_merge, e_space_b_merge), action_mask_b_merge)
            else:
                action_flags_b = torch.ones_like(e_space_b_kg_global)
                action_space_b = ((r_space_b_kg_global, e_space_b_kg_global), action_mask_b_kg)
            
            l_action_spaces_global.append(action_space_b)
            l_references.append(l_batch_refs)
            l_action_flags.append(action_flags_b)

        return l_action_spaces_global, l_references, l_action_flags 

    def policy_linear(self, b_input_vector):
        X = self.W1(b_input_vector)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)
        return X2
    
    def get_action_embedding_kg(self, action):
        r, e = action
        relation_embedding = self.relation_embeddings(r)
        
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = self.entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def calculate_kg_policy(self, t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_space, b_e_space_global, b_token_memory_state):
        b_size, action_num = b_r_space.shape
        if self.use_ecm_tokens_internal_memory:
            b_r_embeddings = self.relation_embeddings(b_r_space)
            b_question_vectors_att, b_token_memory_state_new = self.get_relation_aware_question_vector_ecm(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_embeddings, b_token_memory_state)
        else:
            r_tensor_globalId, b_r_localId = kb_relation_global2local(b_r_space)
            b_r_embeddings = self.relation_embeddings(r_tensor_globalId)

            b_question_vectors_att, _ = self.get_relation_aware_question_vector_attention(t, b_question_vectors, b_question_mask, b_r_embeddings, b_r_localId)

            b_token_memory_state_new = torch.zeros(b_size, action_num, self.max_question_len_global).cuda()

        b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1) 
        b_policy_network_input = torch.cat([b_path_hidden, b_question_vectors_att], -1)

        b_policy_network_output = self.policy_linear(b_policy_network_input)
        b_action_embedding = self.get_action_embedding_kg((b_r_space, b_e_space_global))
        b_action_embedding = b_action_embedding.view(-1, self.action_dim).unsqueeze(1) 
        b_output_vector = b_policy_network_output.view(-1, self.action_dim).unsqueeze(-1)
        b_action_logit = torch.matmul(b_action_embedding, b_output_vector).squeeze(-1).view(-1, action_num)

        return b_token_memory_state_new, b_action_logit


    def transit(self, t, e_t_global, batch_question, batch_question_seq_lengths, batch_cadidates_onehot, batch_path_hidden, batch_token_memory_state, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, batch_relation_paths, d_entity_global2local, d_entity_local2global):
        l_action_spaces_global, l_references, l_action_flags = self.get_action_space_in_buckets(e_t_global, batch_question, batch_cadidates_onehot, batch_relation_paths,tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local,  d_entity_global2local, d_entity_local2global)

        batch_question_vectors, batch_question_mask = self.get_question_representation(batch_question, batch_question_seq_lengths)

        batch_question_mp = self.question_max_pooling(batch_question_vectors, batch_question_mask)

        references = []
        values_list = []
        l_2D = []
        l_internal_token_states = []

        for b_action_spaces_global, b_reference_kg, b_action_flags in zip(l_action_spaces_global, l_references, l_action_flags):
            b_e_t_global = e_t_global[b_reference_kg]
            b_question_vectors = batch_question_vectors[b_reference_kg]
            b_question_mask = batch_question_mask[b_reference_kg]
            b_question_mp = batch_question_mp[b_reference_kg]
            b_path_hidden = batch_path_hidden[b_reference_kg]
            b_token_memory_state = batch_token_memory_state[b_reference_kg]

            (b_r_space_global_kg, b_e_space_global_kg), b_action_mask_kg = b_action_spaces_global
            b_size, action_num_kg = b_r_space_global_kg.shape

            if self.use_entity_embedding_in_vn:
                b_e_t_global_embeddings = self.entity_embeddings(b_e_t_global)
                value_input = torch.cat([b_e_t_global_embeddings, b_path_hidden], dim = -1)
                b_value = self.W_value(value_input).view(-1)
            else:
                b_value = self.W_value(b_path_hidden).view(-1)

            b_value = torch.sigmoid(b_value)
            values_list.append(b_value)

            b_token_memory_state_new_kg, b_action_logit_kg = self.calculate_kg_policy(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_space_global_kg, b_e_space_global_kg, b_token_memory_state)
        
            b_action_logit_kg_masked = b_action_logit_kg.masked_fill((1 - b_action_mask_kg).bool(), float('-inf'))
            b_reasoner_action_dist_kg = F.softmax(b_action_logit_kg_masked, 1)

            b_2D_kg = (b_r_space_global_kg, b_e_space_global_kg, b_action_mask_kg, b_reasoner_action_dist_kg, b_action_flags)

            references.extend(b_reference_kg)
            l_2D.append(b_2D_kg)
            l_internal_token_states.append(b_token_memory_state_new_kg)
        
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        
        return inv_offset, values_list, l_2D, l_internal_token_states

    def initialize_path(self, init_action):
        init_action_embedding = self.get_action_embedding_kg(init_action)
        init_action_embedding.unsqueeze_(1)
        init_h = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])
        init_c = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])

        init_h = init_h.cuda()
        init_c = init_c.cuda()

        h_n, c_n = self.path_encoder(init_action_embedding, (init_h, init_c))[1]
        return (h_n, c_n)

    def update_path(self, action, path_list, offset=None):
        
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]


        action_embedding = self.get_action_embedding_kg(action)
        action_embedding.unsqueeze_(1) 

        if offset is not None:
            offset_path_history(path_list, offset)

        h_n, c_n = self.path_encoder(action_embedding, path_list[-1])[1]
        return path_list, (h_n, c_n)

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.constant_(self.W_value.bias, 0.0)

        if not self.use_ecm_tokens_internal_memory:
            nn.init.xavier_uniform_(self.relation_linear.weight)
            nn.init.constant_(self.relation_linear.bias, 0.0)
            nn.init.xavier_uniform_(self.W_att.weight)
            nn.init.constant_(self.W_att.bias, 0.0)


        for i in range(self.max_hop):
            if self.use_ecm_tokens_internal_memory:
                nn.init.xavier_uniform_(self.l_internal_state_W_read_KG[i].weight)
                nn.init.constant_(self.l_internal_state_W_read_KG[i].bias, 0.0)

                nn.init.xavier_uniform_(self.l_internal_state_W_write_KG[i].weight)
                nn.init.constant_(self.l_internal_state_W_write_KG[i].bias, 0.0)

            else:
                nn.init.xavier_uniform_(self.l_question_linears[i].weight)
                nn.init.constant_(self.l_question_linears[i].bias, 0.0)

        for name, param in self.path_encoder.named_parameters(): 
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def batch_sentence_mask(self, batch_sent_len):
        batch_size = len(batch_sent_len)
        if self.use_ecm_tokens_internal_memory:
            max_sent_len = self.max_question_len_global
        else:
            max_sent_len = batch_sent_len[0]
        
        mask = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

        for i in range(batch_size):
            sent_len = batch_sent_len[i]
            mask[i][sent_len:] = 1
        
        mask = (mask == 1)
        mask = mask.cuda()
        return mask
    
    def judge_wiki_relation_in_path(self, b_is_wiki_in_history, b_r_global_merge): 
        b_r_is_wiki = b_r_global_merge >= self.kb_relation_num
        b_result = b_is_wiki_in_history | b_r_is_wiki 
        return b_result
    
    def return_entity_kg_embeddings(self, b_e_global):
        return self.entity_embeddings(b_e_global)
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)