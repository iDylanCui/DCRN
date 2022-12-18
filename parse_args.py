import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        default = 'MetaQA-1H',
                        help="dataset for training")

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--num_workers', type=int, default=4, help="Dataloader workers")

argparser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')

argparser.add_argument('--train', action='store_true',
                    help='train model')

argparser.add_argument('--eval', action='store_true',
                    help='evaluate the results on the test set')

argparser.add_argument('--total_epoch', type=int, default=1,
                    help='adversarial learning epoch number.')

argparser.add_argument("--min_freq", type=int, default=0, help="Minimum frequency for words")

argparser.add_argument("--max_question_len_global", type=int, default=20, help="Maximum question pattern words length")

argparser.add_argument("--max_hop",
                        type=int,
                        default=3,
                        help="max reasoning hop")

argparser.add_argument("--num_wait_epochs",
                        type=int,
                        default=1,
                        help="valid wait epochs")

argparser.add_argument('--entity_dim', type=int, default=200,
                    help='entity embedding dimension')

argparser.add_argument('--relation_dim', type=int, default=200,
                    help='relation embedding dimension')

argparser.add_argument('--word_dim', type=int, default=300,
                    help='word embedding dimension')

argparser.add_argument('--word_dropout_rate', type=float, default=0.3,
                    help='word embedding dropout rate')

argparser.add_argument('--word_padding_idx', type=int, default=0,
                    help='word padding index')

argparser.add_argument('--DUMMY_RELATION_idx', type=int, default=0,
                    help='DUMMY_RELATION index')

argparser.add_argument('--DUMMY_ENTITY_idx', type=int, default=0,
                    help='DUMMY_ENTITY index')  

argparser.add_argument('--is_train_emb', type=bool, default=True,
                    help='train word/entity/relation embedding or not')

argparser.add_argument('--grad_norm', type=float, default=50,
                    help='norm threshold for gradient clipping')

argparser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate')

argparser.add_argument('--head_num', type=int, default=4,
                    help='Transformer head number')

argparser.add_argument('--hidden_dim', type=int, default=100,
                    help='Transformer hidden dimension')

argparser.add_argument('--encoder_layers', type=int, default=2,
                    help='Transformer encoder layers number')

argparser.add_argument('--encoder_dropout_rate', type=float, default=0.3,
                    help='Transformer encoder dropout rate')

argparser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay rate')

argparser.add_argument('--history_dim', type=int, default=200,
                    help='path encoder LSTM hidden dimension')

argparser.add_argument('--relation_only', type=bool, default=False,
                    help='search with relation information only, ignoring entity representation')

argparser.add_argument('--rl_dropout_rate', type=float, default=0.3,
                    help='reinforce learning dropout rate')

argparser.add_argument('--history_layers', type=int, default=2,
                    help='path encoder LSTM layers')

argparser.add_argument('--gamma', type=float, default=0.95,
                    help='moving average weight') 

argparser.add_argument('--tau', type=float, default=1.00,
                    help='GAE tau')

argparser.add_argument("--early_stop_patience", type=int, default=10,
                        help="early stop epoch")

argparser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')

argparser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size')

argparser.add_argument('--batch_size', type=int, default=1,
                    help='mini-batch size')

argparser.add_argument('--beam_size', type=int, default=5, help='size of beam used in train')

argparser.add_argument('--use_entity_embedding_vn', type=bool, default=True, help='use entity embedding in value netwok or not')

argparser.add_argument('--use_actor_critic', type=bool, default=True, help='use actor critic optimization.')

argparser.add_argument('--use_gae', type=bool, default=True, help='use gae in actor critic optimization.')

argparser.add_argument('--use_ecm_tokens_internal_memory', type=bool, default=True, help='use emotional chatting machine tokens internal memory.')

argparser.add_argument('--use_tokens_memory_normalization', type=bool, default=True, help='use tokens memory normalization.')

argparser.add_argument('--loss_with_beam', type=bool, default = True, help='calculate full beam size loss not top-1.')

argparser.add_argument('--temperature_theta', type=float, default=0.07,
                    help='temperature hyperparameter')

argparser.add_argument('--seed', type=int, default=2022, help='random seed')

argparser.add_argument('--value_loss_coef', type=float, default=0.1,
                    help = "value loss coefficient")

argparser.add_argument('--train_retriever', type=bool, default=False, help='train retriever along with reasoner or not')

argparser.add_argument('--train_kge', type=bool, default=True, help='train ConvE along with reasoner or not')

argparser.add_argument('--use_dynamic_action_space', type=bool, default=True, help='complement action space or not')

argparser.add_argument('--use_relation_pruner', type=bool, default=True, help='use relation pruner or not')

argparser.add_argument('--use_candidate_entities_constraint', type=bool, default=False, help='use candidate entities constraint or not')

argparser.add_argument('--total_supplement_action_num', type=int, default=20, help='random seed')

argparser.add_argument('--supplement_relation_num', type=int, default=2, help='random seed')

argparser.add_argument('--training_data_percentage', type=float, default=1.0,
                    help = "qa data percentage used for training")

argparser.add_argument('--kb_triples_percentage', type=float, default=1.0,
                    help = "kg triples percentage used for training")


args = argparser.parse_args()