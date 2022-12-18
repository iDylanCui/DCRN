import torch
import torch.nn as nn
from utils import safe_log
from parse_args import args
    
def pad_and_cat1d(a, padding_value, inv_offset, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()[inv_offset]

def pad_and_cat2d(a, padding_value, inv_offset, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad2d((0, 0, 0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()[inv_offset]


def top_k_actions(batch_size, beam_size, log_reasoner_action_dist_sum, batch_buckets_r_global, batch_buckets_e_global, batch_buckets_reasoner_action_dist_log, batch_buckets_action_flags, batch_buckets_token_states):
    full_size, action_space_size, max_question_len_global = batch_buckets_token_states.shape
    last_k = int(full_size / batch_size)

    log_reasoner_action_dist_sum = log_reasoner_action_dist_sum.view(batch_size, -1)
    batch_buckets_r_global = batch_buckets_r_global.view(batch_size, -1)
    batch_buckets_e_global = batch_buckets_e_global.view(batch_size, -1)
    batch_buckets_action_flags = batch_buckets_action_flags.view(batch_size, -1)

    batch_buckets_reasoner_action_dist_log = batch_buckets_reasoner_action_dist_log.view(batch_size, -1)
    batch_buckets_token_states = batch_buckets_token_states.view(batch_size, -1, max_question_len_global)

    beam_action_space_size = log_reasoner_action_dist_sum.size()[1]
    topK = min(beam_size, beam_action_space_size)

    batch_log_action_prob_sum_topK, batch_action_index_topK = torch.topk(log_reasoner_action_dist_sum, topK)

    batch_buckets_r_global_topK = torch.gather(batch_buckets_r_global, 1, batch_action_index_topK).view(-1) 
    batch_buckets_e_global_topK = torch.gather(batch_buckets_e_global, 1, batch_action_index_topK).view(-1) 
    batch_buckets_reasoner_action_dist_log_topK = torch.gather(batch_buckets_reasoner_action_dist_log, 1, batch_action_index_topK).view(-1)
    batch_buckets_action_flags_topK = torch.gather(batch_buckets_action_flags, 1, batch_action_index_topK).view(-1) 
    batch_log_action_prob_sum_topK = batch_log_action_prob_sum_topK.view(-1)

    new_token_states_list = []
    for b_index in range(batch_size):
        action_indices = batch_action_index_topK[b_index]
        state_vector = batch_buckets_token_states[b_index][action_indices]
        new_token_states_list.append(state_vector)
    
    token_states_matrix = torch.stack(new_token_states_list, dim=0).view(-1, max_question_len_global) 
    
    action_beam_offset = torch.div(batch_action_index_topK, action_space_size, rounding_mode = 'trunc')

    action_batch_offset = (torch.arange(batch_size).cuda() * last_k).unsqueeze(1)
    action_offset = (action_batch_offset + action_beam_offset).view(-1)

    return action_offset, batch_log_action_prob_sum_topK, batch_buckets_r_global_topK, batch_buckets_e_global_topK, batch_buckets_action_flags_topK, batch_buckets_reasoner_action_dist_log_topK, token_states_matrix

def rollout_beam(env, reasoner):
    batch_question, batch_question_seq_lengths, batch_candidates_onehot, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, d_entity_global2local, d_entity_local2global = env.return_batch_data()

    batch_size, _ = batch_question.shape
    
    log_reasoner_action_prob_history = torch.zeros(batch_size).cuda()

    for t in range(0, env.max_hop):
        path_trace_global, l_path_hidden, l_token_memory_state = env.observe()
        batch_relation_paths = env.get_relation_paths()

        _, e_t_global = path_trace_global[-1]
        batch_path_hidden = l_path_hidden[-1][0][-1, :, :]
        batch_token_memory_state = l_token_memory_state[-1]

        k = int(e_t_global.size()[0] / batch_size)

        beam_question = batch_question.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1)
        beam_question_len = batch_question_seq_lengths.unsqueeze(1).repeat(1, k).view(batch_size * k)
        beam_cadidates_onehot = batch_candidates_onehot.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1)
        
        inv_offset, values_list, l_2D, l_internal_token_states = reasoner.transit(t, e_t_global, beam_question, beam_question_len, beam_cadidates_onehot, batch_path_hidden, batch_token_memory_state, tensor_entity2bucketid_batch_local, d_action_space_buckets_batch_local, batch_relation_paths, d_entity_global2local, d_entity_local2global)

        values_t = torch.cat(values_list, dim=0)[inv_offset]

        l_buckets_r_global = [r for r, _, _, _, _ in l_2D]
        l_buckets_e_global = [e_g for _, e_g, _, _, _ in l_2D]
        l_buckets_reasoner_action_dist = [r_a for _, _, _, r_a, _ in l_2D]
        l_buckets_action_flags = [a_f for _, _, _, _, a_f in l_2D]

        batch_buckets_r_global = pad_and_cat1d(l_buckets_r_global, args.DUMMY_RELATION_idx, inv_offset, padding_dim=1)
        batch_buckets_e_global = pad_and_cat1d(l_buckets_e_global, args.DUMMY_ENTITY_idx, inv_offset, padding_dim=1)
        
        batch_buckets_reasoner_action_dist = pad_and_cat1d(l_buckets_reasoner_action_dist, 0, inv_offset, padding_dim=1)
        batch_buckets_reasoner_action_dist_log = safe_log(batch_buckets_reasoner_action_dist)
        batch_buckets_action_flags = pad_and_cat1d(l_buckets_action_flags, 0, inv_offset, padding_dim=1)

        batch_buckets_token_states = pad_and_cat2d(l_internal_token_states, 0, inv_offset, padding_dim=1)

        log_reasoner_action_dist_sum = log_reasoner_action_prob_history.view(-1, 1) + batch_buckets_reasoner_action_dist_log

        action_offset, log_reasoner_action_prob_history, batch_buckets_r_global_topK, batch_buckets_e_global_topK, batch_buckets_action_flags_topK, batch_buckets_reasoner_action_dist_log_topK, token_states_matrix = top_k_actions(batch_size, args.beam_size, log_reasoner_action_dist_sum, batch_buckets_r_global, batch_buckets_e_global, batch_buckets_reasoner_action_dist_log, batch_buckets_action_flags, batch_buckets_token_states)
    
        path_list, (h_t, c_t) = reasoner.update_path( (batch_buckets_r_global_topK, batch_buckets_e_global_topK), l_path_hidden, offset = action_offset)
        new_hidden = (h_t, c_t)

        env.step(path_list, new_hidden, batch_buckets_r_global_topK, batch_buckets_e_global_topK, token_states_matrix, values_t, batch_buckets_reasoner_action_dist_log_topK, batch_buckets_action_flags_topK, offset = action_offset)
    
    return log_reasoner_action_prob_history
