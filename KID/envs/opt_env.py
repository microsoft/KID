import gc

import torch
from transformers import AutoTokenizer, OPTForCausalLM

from KID.envs import BaseEnv
from KID.infra import Frame


class OPTEnv(BaseEnv):

    def __init__(self, model_name: str, max_len: int = 100, device: str = 'cuda:0'):
        super(OPTEnv, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = OPTForCausalLM.from_pretrained(model_name)

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
            # HF bug related to past_key_values on OPT,
            # see https://github.com/huggingface/transformers/issues/21685
            past_key_values_length = past[0][0].shape[2]
            bs = past[0][0].shape[0]
            attn = torch.ones(bs, past_key_values_length +  last.size(-1)).to(last.device)

            logits, past = self.model(
                last, use_cache=True, past_key_values=past, return_dict=False,
                attention_mask=attn
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
