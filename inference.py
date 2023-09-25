#when the weights of LLaMA2 will be available by meta on my account i will load the model and the weights and write
#inference code here !!

#simple test

import torch
from model import Config, Transformer


if __name__ == '__main__':
    x = torch.randint(0, 10, (1, 1))
    model_args = Config()
    model_args.dim = 512
    model_args.max_batch_size = 1
    model_args.max_seq_len = 10
    model_args.device = 'cuda'
    model_args.vocab_size = 100
    print(Transformer(model_args)(x, 0))