import torch
import numpy as np
from utils.search import beamsearch
from utils.data_process import TSPDataset
from models.attention_model import AttentionModel
from torch.utils.data import DataLoader
from utils.project import StartEnd_constrain, Priority_constrain, project_one_batch
from utils.train import process_one_batch
import json
import argparse
import os
import time

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

parser = argparse.ArgumentParser(description='Evaluate the trianed LinSATNet for TSP with extra constraints')

parser.add_argument('--model_dict', type = str, required=True, help="The dictionary of the trained model")
parser.add_argument('--test_epoch', type = int, default=10, help="The epoch of the trained model to be evaluated")
parser.add_argument('--test_dataset', type = str, default='datasets/tsp20_test_seed1234.pkl', help="Test dataset path")
parser.add_argument('--test_batch_size', type=int, default=10000, help="Number of instances per batch to be input to the model during testing")
parser.add_argument('--search_batch_size', type=int, default=512, help="Number of instances per batch to perform beam search together")
parser.add_argument('--beam_width', type=int, default=2048, help="Beam width for beam search")
parser.add_argument('--gpu', type = int, default=0, help="id of gpu to use, -1 for cpu")

args = parser.parse_args()

train_args = load_args(os.path.join(args.model_dict, 'args.json'))
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.temp = train_args['temp']; args.max_iter = train_args['max_iter']

#load the trained model
model = AttentionModel(
    train_args['graph_size'], train_args['embedding_dim'], train_args['hidden_dim'], train_args['n_encode_layers'], 
    train_args['tanh_clipping'], train_args['normalization'], train_args['n_heads'], train_args['task']
).to(args.device)
load_path = os.path.join(args.model_dict, 'epoch-{}.pt'.format(args.test_epoch))
load_model = torch_load_cpu(load_path)
model.load_state_dict(load_model['model'])

#load the test dataset
test_dataset = TSPDataset(filename=args.test_dataset, num_samples=10000, size=train_args['graph_size']) #there are 10000 test instances in the test dataset
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=False, shuffle=False)

#prepare the constraints for LinSATNet
if (train_args['task'] == 'StartEnd'):
    constrain_left, constrain_right = StartEnd_constrain(train_args['graph_size'])
elif (train_args['task']  == 'Priority'):
    constrain_left, constrain_right = Priority_constrain(train_args['graph_size'], train_args['priority_level'])
constrain_left = constrain_left.to(args.device)
constrain_right = constrain_right.to(args.device)


#evaluate the trained model
beamsearch_paths = []
beamsearch_lengths = []
step_probs = []
cords_2ds = []
NN_time = 0
Proj_time = 0

model.eval()
with torch.no_grad():
    #get the post project matrix with the trained model
    for batch_id, batch_input in enumerate(test_loader):
        step_prob, cords_2d, batch_loss, t1, t2 = process_one_batch(model, batch_input, args, \
                                        constrain_left = constrain_left, constrain_right = constrain_right)
        step_probs.append(step_prob)
        cords_2ds.append(cords_2d)
        NN_time += t1; Proj_time += t2
    step_probs = torch.cat(step_probs, dim = 0)
    cords_2ds = torch.cat(cords_2ds, dim = 0)
    
    #perform beam search
    inference_index = 0
    start_time = time.time()
    while(inference_index < step_probs.shape[0]):
        batch_index_end = min(inference_index + args.search_batch_size, step_probs.shape[0])
        batch_prob = step_probs[inference_index : batch_index_end]
        batch_cord = cords_2ds[inference_index : batch_index_end]
        beamsearch_path, beamsearch_length = beamsearch(batch_prob, beam_width = args.beam_width, cords_2d = batch_cord, task = train_args['task'], priority_level = train_args['priority_level'])
        beamsearch_paths.append(beamsearch_path)
        beamsearch_lengths.append(beamsearch_length)
        inference_index = batch_index_end
    end_time = time.time()
    search_time = end_time - start_time
beamsearch_lengths = torch.cat(beamsearch_lengths, dim = 0)
avg_length = beamsearch_lengths.mean().item()
beamsearch_path_np = torch.cat(beamsearch_paths, dim = 0).cpu().detach().numpy()
overall_time = NN_time + Proj_time + search_time

#save the metric into a text file
with open(os.path.join(args.model_dict, 'test_result.txt'), 'w') as f:
    f.write('Task: {}\n'.format(train_args['task']))
    f.write('Node Number: {}\n'.format(train_args['graph_size']))
    f.write('Priority Level (only for TSP-PRI): {}\n'.format(train_args['priority_level']))
    f.write('Test Dataset: {}\n'.format(args.test_dataset))
    f.write('\n')
    f.write('Test Epoch: {}\n'.format(args.test_epoch))
    f.write('Beamsearch Width: {}\n'.format(args.beam_width))
    f.write('\n')
    f.write('Avg Tour Length: {:.6f}\n'.format(avg_length))
    f.write('Test Time: {:.4f}s\n'.format(overall_time))
    f.write('Neural Network Time: {:.4f}s | Project Time: {:.4f}s | BeamSearch Time: {:.4f}s\n'.format(NN_time, Proj_time, search_time))

print('Beamsearch Width {}, Avg length {:.6f}, Test Time {:.4f}s, NN Time {:.4f}s, Project Time {:.4f}s, Search Time {:.4f}s'.format(\
                                        args.beam_width, avg_length, overall_time, NN_time, Proj_time, search_time))

