import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        default = 'WebQSP',
                        help="dataset for training")

argparser.add_argument("--KGE_model",
                        type=str,
                        default='ConvE',
                        help="KGE model")

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--seed', type=int, default=2023,
                    help='random seed')

argparser.add_argument('--partition', type=float, default= 0.1,
                    help='valid, test set partition ratio')

argparser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')

argparser.add_argument('--train', action='store_true',
                    help='train model')

argparser.add_argument('--eval', action='store_true',
                    help='eval model')

argparser.add_argument('--group_examples_by_query', action='store_true',
                    help='group examples by topic entity + query relation (default: False)')

argparser.add_argument('--add_reversed_edges', action='store_true',
                    help='add reversed edges to extend training set')

# general parameters
argparser.add_argument("--num_epochs",
                        type=int,
                        default=1000,
                        help="maximum # of epochs")

argparser.add_argument("--early_stop_patience",
                        type=int,
                        default=50,
                        help="early stop epoch")

argparser.add_argument("--num_wait_epochs",
                        type=int,
                        default=5,
                        help="valid wait epochs")

argparser.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size')

argparser.add_argument('--entity_dim', type=int, default=200,
                    help='entity embedding dimension')

argparser.add_argument('--relation_dim', type=int, default=200,
                    help='relation embedding dimension')
                    
argparser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help='epsilon used for label smoothing')

argparser.add_argument('--learning_rate', type=float, default=0.005,
                    help='learning rate')

argparser.add_argument('--grad_norm', type=float, default=100,
                    help='norm threshold for gradient clipping')

argparser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate')

argparser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='ConvE hidden layer dropout rate')
argparser.add_argument('--feat_dropout_rate', type=float, default=0.2,
                    help='ConvE feature dropout rate')
argparser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
argparser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
argparser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer')
argparser.add_argument('--kernel_size', type=int, default=4,
                    help='ConvE kernel size (default: 3)')


args = argparser.parse_args()