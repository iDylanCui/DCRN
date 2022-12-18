import torch
import numpy as np

class Environment(object):
    def __init__(self, args, kb_relation_num, dummy_start_r = 1):
        self.args = args
        self.max_hop = self.args.max_hop
        self.dummy_start_r = dummy_start_r
        self.max_question_len_global = self.args.max_question_len_global
        self.kb_relation_num = kb_relation_num
        
    def reset(self, bath_data, reasoner):
        self.batch_question, self.batch_question_seq_lengths, self.batch_topic_ent_global, self.batch_answers_onehot, self.batch_candidates_onehot, self.tensor_entity2bucketid_batch_local, self.d_action_space_buckets_batch_local, self.d_entity_global2local, self.d_entity_local2global = bath_data

        batch_size, _ = self.batch_question.shape
        self.batch_size = batch_size

        r_s = torch.zeros_like(self.batch_topic_ent_global) + self.dummy_start_r

        self.path_trace_global = [(r_s, self.batch_topic_ent_global)]

        self.path_hidden = [reasoner.initialize_path(self.path_trace_global[0])]
        
        self.token_memory_state = torch.zeros(self.batch_size, self.max_question_len_global, dtype=torch.float).cuda()

        for i in range(self.batch_size):
            seq_len = self.batch_question_seq_lengths[i].item()
            self.token_memory_state[i].narrow(0,0,seq_len).copy_(torch.ones(seq_len, dtype=torch.float).cuda())

        self.l_token_memory_state = [self.token_memory_state]
        self.l_values = []
        self.l_reasoner_action_dist_log = []
        self.l_action_flags = []

    def observe(self):
        return self.path_trace_global, self.path_hidden, self.l_token_memory_state

    def step(self, path_list, new_hidden, r_global, e_global, token_states_matrix, values, reasoner_action_dist_log, action_flags, offset):
        self.path_hidden = path_list
        self.path_hidden.append(new_hidden)
        self.update_token_memory_states(token_states_matrix, offset)
        self.update_path_trace_global(r_global, e_global, offset)
        self.update_values(values, offset)
        self.update_reasoner_action_dist_log(reasoner_action_dist_log, offset)

        self.update_action_flags(action_flags, offset)

    def get_pred_entities(self):
        return self.path_trace_global[-1][1]
    
    def return_batch_data(self):
        return self.batch_question, self.batch_question_seq_lengths, self.batch_candidates_onehot, self.tensor_entity2bucketid_batch_local, self.d_action_space_buckets_batch_local, self.d_entity_global2local, self.d_entity_local2global
    
    def get_relation_paths(self):
        l_relation_paths = []

        for timestep in range(len(self.path_trace_global)):
            batch_r, _ = self.path_trace_global[timestep]
            l_relation_paths.append(batch_r.unsqueeze(-1))
        
        batch_relation_paths = torch.cat(l_relation_paths, dim = -1)

        return batch_relation_paths
    
    def dc_relation_distribution(self, flag_Top1 = True):
        l_relation_flag_paths = []

        for timestep in range(len(self.l_action_flags)):
            l_relation_flag_paths.append(self.l_action_flags[timestep].unsqueeze(-1))
        
        batch_relation_flags = torch.cat(l_relation_flag_paths, dim = -1)

        if flag_Top1 == True:
            batch_relation_flags = batch_relation_flags.view(self.batch_size, -1, self.max_hop)
            batch_relation_flags = batch_relation_flags[:, 0, :]
            bath_has_dc_relation = batch_relation_flags == 2
            batch_results = torch.all(bath_has_dc_relation, dim=-1)

            has_dc_overall = torch.sum(batch_results).item()
            has_dc_step = torch.sum(bath_has_dc_relation, dim = 0).cpu().numpy()
            
            return has_dc_overall, has_dc_step
            
    def update_token_memory_states(self, token_states_matrix, offset=None):
        
        def offset_state_history(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset, :]
        
        if offset is not None:
            offset_state_history(self.l_token_memory_state, offset)
        
        self.l_token_memory_state.append(token_states_matrix)
    
    def update_path_trace_global(self, r_global, e_global, offset=None):

        def offset_path_trace(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[offset] for _x in x])
                    p[i] = new_tuple
        
        if offset is not None: 
            offset_path_trace(self.path_trace_global, offset)

        self.path_trace_global.append((r_global, e_global))
    
    def update_values(self, values, offset=None):
        def offset_values(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset]

        self.l_values.append(values)

        if offset is not None: 
            offset_values(self.l_values, offset)
    
    def update_action_flags(self, action_flags, offset= None):
        def offset_action_flags(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset]

        if offset is not None: 
            offset_action_flags(self.l_action_flags, offset)
        
        self.l_action_flags.append(action_flags)
    
    def update_reasoner_action_dist_log(self, reasoner_action_dist_log, offset= None):
        def offset_reasoner_action_dist_log(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset]

        if offset is not None: 
            offset_reasoner_action_dist_log(self.l_reasoner_action_dist_log, offset)
        
        self.l_reasoner_action_dist_log.append(reasoner_action_dist_log)
    
    def calculate_beam_loss(self):
        reasoner_loss = None

        batch_pred_ans = self.get_pred_entities()
        batch_golden_ans = self.batch_answers_onehot

        full_size = batch_pred_ans.size()[0]
        beam_size = int(full_size / self.batch_size)

        batch_golden_ans = batch_golden_ans.unsqueeze(1).repeat(1, beam_size, 1).view(self.batch_size * beam_size, -1)

        batch_binary_reward = torch.gather(batch_golden_ans, 1, batch_pred_ans.unsqueeze(-1)).squeeze(-1).float()

        reasoner_loss = self.calculate_reasoner_loss(batch_binary_reward)

        return reasoner_loss
    
    def calculate_reasoner_loss(self, batch_binary_reward):
        cum_rewards = [0] * self.max_hop

        if self.args.loss_with_beam:
            full_size = batch_binary_reward.size()[0]
            R = torch.zeros(full_size).cuda()
            policy_loss = torch.zeros(full_size).cuda()
            value_loss = torch.zeros(full_size).cuda()

            l_action_values = self.l_values
            l_reasoner_action_dist_log = self.l_reasoner_action_dist_log

            if self.args.use_gae:
                gae = torch.zeros(full_size).cuda()
                l_action_values.append(R)

        else:
            R = torch.zeros(self.batch_size).cuda()
            policy_loss = torch.zeros(self.batch_size).cuda()
            value_loss = torch.zeros(self.batch_size).cuda()

            batch_binary_reward = batch_binary_reward.view(self.batch_size, -1)
            batch_binary_reward = batch_binary_reward[:, 0]
            l_action_values = [value.view(self.batch_size, -1)[:, 0] for value in self.l_values]
            l_reasoner_action_dist_log = [dist_log.view(self.batch_size, -1)[:, 0] for dist_log in self.l_reasoner_action_dist_log]

            if self.args.use_gae:
                gae = torch.zeros(self.batch_size).cuda()
                l_action_values.append(R)

        
        cum_rewards[-1] = batch_binary_reward
        for i in reversed(range(self.max_hop)):
            R = self.args.gamma * R + cum_rewards[i]
            total_R = R
            
            if self.args.use_actor_critic:
                advantage = total_R - l_action_values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                policy_loss = policy_loss - advantage.detach() * l_reasoner_action_dist_log[i]

                if self.args.use_gae:
                    delta_t = total_R + self.args.gamma * l_action_values[i+1] - l_action_values[i]
                    gae = gae * self.args.gamma * self.args.tau + delta_t
                    policy_loss = policy_loss - gae.detach() * l_reasoner_action_dist_log[i]

            else:
                policy_loss = policy_loss - total_R.detach() * l_reasoner_action_dist_log[i]

        if self.args.use_actor_critic:
            rl_loss = policy_loss + self.args.value_loss_coef * value_loss
        else:
            rl_loss = policy_loss
        
        reasoner_loss = rl_loss.mean()

        if self.args.use_ecm_tokens_internal_memory and self.args.use_tokens_memory_normalization:
            token_memory_state_T = self.l_token_memory_state[-1]
            token_memory_state_T_norm = torch.norm(token_memory_state_T, p=2)

            reasoner_loss += token_memory_state_T_norm
        
        return reasoner_loss

    def inference_hits1(self):
        batch_pred_ans = self.get_pred_entities()
        batch_pred_ans = batch_pred_ans.view(self.batch_size, -1)
        batch_golden_ans = self.batch_answers_onehot

        batch_pred_e2_top1 = batch_pred_ans[:, 0].view(self.batch_size, -1)
        hits1_item = torch.sum(torch.gather(batch_golden_ans, 1, batch_pred_e2_top1).view(-1)).item()

        return hits1_item
    
    def inference_hits1_batch(self):
        batch_pred_ans = self.get_pred_entities()
        batch_pred_ans = batch_pred_ans.view(self.batch_size, -1)
        batch_golden_ans = self.batch_answers_onehot

        batch_pred_e2_top1 = batch_pred_ans[:, 0].view(self.batch_size, -1)
        hits1_item = torch.gather(batch_golden_ans, 1, batch_pred_e2_top1).view(-1)

        return hits1_item
    
    def get_token_memory_states(self, flag_Top1 = True):
        l_token_memory_states_new = []

        if flag_Top1 == True:
            for t in range(self.max_hop + 1):
                memory_state_t = self.l_token_memory_state[t]
                memory_state_t = memory_state_t.view(self.batch_size, -1, self.max_question_len_global)
                memory_state_t_top1 = memory_state_t[:, 0, :].unsqueeze(1)
                l_token_memory_states_new.append(memory_state_t_top1)
        
            token_memory_states_new = torch.cat(l_token_memory_states_new, dim = 1)

            print(token_memory_states_new)
    
    def get_intermediate_rel_ent(self):
        l_relations = []
        l_entities = []

        for timestep in range(len(self.path_trace_global)):
            batch_r, batch_e = self.path_trace_global[timestep]
            l_relations.append(batch_r.unsqueeze(-1))
            l_entities.append(batch_e.unsqueeze(-1))
        
        batch_relations = torch.cat(l_relations, dim = -1)
        batch_entities = torch.cat(l_entities, dim = -1)


        l_relation_flag_paths = []
        for timestep in range(len(self.l_action_flags)):
            l_relation_flag_paths.append(self.l_action_flags[timestep].unsqueeze(-1))
        batch_relation_flags = torch.cat(l_relation_flag_paths, dim = -1)

        batch_relations = batch_relations.cpu().numpy().tolist()
        batch_entities = batch_entities.cpu().numpy().tolist()
        batch_relation_flags = batch_relation_flags.cpu().numpy().tolist()

        return batch_relations, batch_entities, batch_relation_flags