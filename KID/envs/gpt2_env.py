import gc

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from KID.envs import BaseEnv
from KID.infra import Frame


class GPT2Env(BaseEnv):

    def __init__(self, model_name: str, max_len: int = 100, device: str = 'cuda:0'):
        super(GPT2Env, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        self.model.to(device)
        self.max_len = max_len
        self.device = device

        # set to evaluation mode. no update on model itself.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.cur_len = 0

    def step(self, frame: 'Frame') -> 'Frame':

        past, last = frame['past'], frame['last']

        # If the action is at the beginning, and the past is None,
        # we need do a dummy generation step to get it.
        if past is None:
            while len(last.size()) < 2:
                last.unsqueeze_(0)  # so that it is a batch
            logits, past = self.model(last[:, :-1], use_cache=True, return_dict=False)
        else:
            logits, past = self.model(
                last, use_cache=True, past_key_values=past, return_dict=False
            )

        self.cur_len += 1

        frame['logits'] = logits
        frame['past'] = past
        frame['is_done'] = self.cur_len > self.max_len

        return frame

    def reset(self) -> None:
        self.cur_len = 0
        gc.collect()
        torch.cuda.empty_cache()
