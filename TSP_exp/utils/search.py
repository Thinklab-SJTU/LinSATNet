import torch
import numpy as np

def beamsearch(assign_matrix, beam_width, cords_2d, task = 'StartEnd', priority_level = 5):
    #assign_matrix shape[batch_size, node_num, node_num] [i,j] the i-th node in position j
    batch_size, node_num, _ = assign_matrix.shape
    batch_index = torch.arange(batch_size)
    
    log_prob = torch.log(assign_matrix)
    
    current_beam_num = 1
    path_matrix_beam = torch.zeros([batch_size, beam_width, node_num, node_num]).to(assign_matrix.device)
    path_matrix_beam[:, :, 0, 0] = 1; path_matrix_beam[:, :, -1, -1] = 1
    visited_node = torch.zeros([batch_size, beam_width, node_num], dtype = torch.bool).to(assign_matrix.device)
    visited_node[:, :current_beam_num, 0] = 1; visited_node[:, :current_beam_num, -1] = 1
    beam_score = torch.zeros([batch_size, beam_width]).to(assign_matrix.device)
    beam_score[:, 0] += log_prob[:, 0, 0] + log_prob[:, -1, -1]
    
    for i in range(1, node_num - 1, 1):
        #expansion
        step_log_prob = log_prob[:, :, i][:, None, :].repeat(1, current_beam_num, 1)  #shape [batch_size, curent_beam_num, node_num]
        step_log_prob.masked_fill_(visited_node[:, :current_beam_num, :], -np.infty)
        candidate_score = beam_score[:, :current_beam_num, None] + step_log_prob   #shape [batch_size, current_beam_num, node_num]
        
        candidate_beam_num = current_beam_num * node_num
        candidate_score_flatten = candidate_score.view(batch_size, -1)   #[batch_size, new_beam_num]
        
        #select topk beam
        if (candidate_beam_num < beam_width):
            new_beam_num = candidate_beam_num
        else:
            new_beam_num = beam_width
        topkScores, topkScoresId = candidate_score_flatten.topk(new_beam_num, dim = 1, largest = True, sorted = True)
        prev_beam_id = (topkScoresId / node_num).long()      #[batch_size, candidate_beam_num]
        added_node_id = topkScoresId - prev_beam_id * node_num

        #update path
        prev_path = path_matrix_beam[:, :, :, :i].gather(dim = 1, index = prev_beam_id[:, :, None, None].expand(-1, -1, node_num, i))
        path_matrix_beam[:, :new_beam_num, :, :i] = prev_path
        path_matrix_beam[:, :new_beam_num, :, i].scatter_(-1, added_node_id[:, :, None], 1.)

        #update visited_node
        prev_visited = visited_node.gather(dim = 1, index = prev_beam_id[:, :, None].expand(-1, -1, node_num))
        visited_node[:, :new_beam_num, :] = prev_visited
        visited_node[:, :new_beam_num, :].scatter_(-1, added_node_id[:, :, None], 1.)

        #update beam_score
        beam_score[:, :new_beam_num] = topkScores

        #if the priority hase not been selected within the given steps, set the score to -inf
        if (task == 'Priority' and i == priority_level):
            not_visited_index = (visited_node[:, :new_beam_num, 1] == False)  #shape [batch_size, new_beam_num]
            beam_score[not_visited_index] = -np.infty

        current_beam_num = new_beam_num

    distance_matrix = torch.norm(cords_2d[:, None, :, :] - cords_2d[:, :, None, :], dim=-1)
    distance_matrix_expand = distance_matrix[:, None, :, :].expand(-1, beam_width, -1, -1)
    assign_current = path_matrix_beam[:, :, :, :-1]
    assign_next = path_matrix_beam[:, :, :, 1:]
    edge_samples = torch.matmul(assign_current, assign_next.permute(0, 1, 3, 2))
    distance_samples = (distance_matrix_expand * edge_samples).sum((-1, -2))
    best_index = distance_samples.argmin(dim =  -1)
    path_matrix_best = path_matrix_beam[batch_index, best_index, :, :]
    shortest_length = distance_samples[batch_index, best_index]
    
    return path_matrix_best, shortest_length