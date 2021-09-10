import os
import sys


ABBREVIATION = {
    '--fp32': 'fp32',
    '--per_device_train_batch_size': 'b',
    '--per_device_eval_batch_size': 'b',
    '--num_train_epochs': 'ep',
    '--learning_rate': 'lr',
    '--lr_scheduler_type': '',
    '--adam_epsilon': 'eps',
    '--warmup_ratio': 'wr',
    '--metric_for_best_model': '',
    '--logging_steps': 'log',
    '--early_stopping_patience': 'sp',
    '--save_start_ratio': 'sr',
    '--input_limit': 'lim'
}


def check_arg(arg):
    return len(arg) > 2 and arg[:2] == '--'

def parse_args(args):
    abbreviation = ''
    state = False
    for arg in args:
        if check_arg(arg):
            assert arg in ABBREVIATION, f'Cannot find abbreviation for arg: {arg}'
            abbreviation = f'{abbreviation}_{ABBREVIATION[arg]}'
            state = True
        elif state:
            abbreviation = f'{abbreviation}{arg}'
            state = False
        else:
            raise ValueError(f'Invalid arg: {arg}')
    if abbreviation == '': return 'default'
    return abbreviation[1:]


if __name__ == '__main__':
    print(parse_args(sys.argv[1:]))
