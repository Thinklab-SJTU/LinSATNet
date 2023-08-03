import os
import json
import time
import argparse

import torch
import torch.optim as optim

from models.attention_model import AttentionModel
from utils.data_process import TSPDataset
from utils.train import train_epoch

parser = argparse.ArgumentParser(description='LinSATNet for TSP with Extra Constraints')


#Data
parser.add_argument('--graph_size', type = int, default=20, help="The size of the problem graph")
parser.add_argument('--batch_size', type = int, default=1024, help="Number of instances per batch during training")
parser.add_argument('--epoch_size', type = int, default=256000, help="Number of instances per epoch during training")
parser.add_argument('--val_size', type = int, default=10000, help="Number of instances used for reporting validation performance")
parser.add_argument('--val_dataset', default='datasets/tsp20_validation_seed4321.pkl', help="Validation dataset path")

#Model
parser.add_argument('--embedding_dim', type = int, default=256, help="Dimension of input embedding")
parser.add_argument('--hidden_dim', type = int, default=256, help="Dimension of hidden layers in Enc/Dec")
parser.add_argument('--n_encode_layers', type = int, default=3, help="Number of layers in the encoder/critic network")
parser.add_argument('--tanh_clipping', type = int, default=1, help="Clip the parameters to within +- this value using tanh,Set to 0 to not perform any clipping.")
parser.add_argument('--n_heads', type = int, default=8, help="Number of heads in the multi-head attention")
parser.add_argument('--normalization', default='instance', help="Normalization type, 'batch' (default) or 'instance'")

#Projection
parser.add_argument('--task', default='StartEnd', help="The task to perform, 'StartEnd' (default) or 'Priority'")
parser.add_argument('--temp', type = float, default=1e-1, help="Temperature to control the closeness to integer")
parser.add_argument('--max_iter', type = int, default=50, help="Max number of iterations for projection")
parser.add_argument('--priority_level', type = int, default=5, help="The priority level of the emergency node, only valid for 'Priority' task")

# Training
parser.add_argument('--lr_model', type = float, default=1e-4, help="Set the learning rate for the network")
parser.add_argument('--lr_decay', type = float, default=1., help="Learning rate decay per epoch")
parser.add_argument('--n_epochs', type = int, default=50, help="The number of epochs to train")
parser.add_argument('--seed', type = int, default=1234, help="Random seed to use")
parser.add_argument('--max_grad_norm', type = float, default=1., help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)")
parser.add_argument('--gpu', type = int, default=0, help="id of gpu to use, -1 for cpu")

# Misc
parser.add_argument('--log_step', type = int, default=50, help="Log info every log_step steps")
parser.add_argument('--run_name', default='StartEnd-20', help="Name to identify the run")
parser.add_argument('--output_dir', default='outputs', help="Directory to write output models to")
parser.add_argument('--epoch_start', type = int, default=1, help="Start at epoch (relevant for learning rate decay)")
parser.add_argument('--checkpoint_epochs', type = int, default=2, help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints")
parser.add_argument('--no_progress_bar', action='store_true', help="Disable progress bar", default=False)

args = parser.parse_args()

args.save_dir = os.path.join(args.output_dir, "{}_{}_{}".format(args.task, args.graph_size, time.strftime("%Y%m%dT%H%M%S")))
args.log_path = os.path.join(args.save_dir, "log.txt")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# Save arguments so exact configuration can always be found
with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
    json.dump(vars(args), f, indent=True)

torch.manual_seed(args.seed)

args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu")


model = AttentionModel(
    args.graph_size, args.embedding_dim, args.hidden_dim, args.n_encode_layers, 
    args.tanh_clipping, args.normalization, args.n_heads, args.task
).to(args.device)

optimizer = optim.Adam(model.parameters(), lr = args.lr_model)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)

val_dataset = TSPDataset(size=args.graph_size, num_samples=args.val_size, filename=args.val_dataset)

log_file = open(args.log_path, 'w')
for epoch in range(args.epoch_start, args.epoch_start + args.n_epochs):
    train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, args, log_file)
log_file.close()