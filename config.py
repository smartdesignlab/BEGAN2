#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_scale_size', type=int, default=128,
                     help='input image will be resized with the given value as width and height')
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128],help='n in the paper')
net_arg.add_argument('--z_num', type=int, default=128, choices=[64, 128])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='Wheel') #'wheeldesign'
data_arg.add_argument('--split', type=str, default='train')#'topology_181213'
data_arg.add_argument('--batch_size', type=int, default=48)
data_arg.add_argument('--grayscale', type=str2bool, default=True)
data_arg.add_argument('--num_worker', type=int, default=4)
# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=40000)
train_arg.add_argument('--lr_update_step', type=int, default=100, choices=[1000, 750])
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
train_arg.add_argument('--gamma', type=float, default=0.7) # 0.5->0.7 제대로 학습 D보다 G에 더 무게
train_arg.add_argument('--lambda_k', type=float, default=0.001)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=30)
misc_arg.add_argument('--save_step', type=int, default=100)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='./data') #'/home/ubuntu/app1/input'
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=128,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument("--n_buffer", type=int, default=1)
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
