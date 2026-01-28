# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse
import time
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    begin = time.time()
    parameter_dict = {
        # 'neg_sampling': None  # Commented out - use config file setting instead
        # 'gpu_id':3,
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='TASIF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Amazon_Beauty.yaml', help='config files')
    parser.add_argument('--use_ls', action='store_true', help='enable label smoothing (sets label_smoothing=0.1 if --label_smoothing not specified)')
    parser.add_argument('--label_smoothing', type=float, default=None, help='label smoothing value (e.g., 0.1)')

    args, _ = parser.parse_known_args()

    # Handle label smoothing arguments
    if args.use_ls or args.label_smoothing is not None:
        ls_value = args.label_smoothing if args.label_smoothing is not None else 0.1
        parameter_dict['label_smoothing'] = ls_value

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,config_dict=parameter_dict)
    end=time.time()
    print(end-begin)
