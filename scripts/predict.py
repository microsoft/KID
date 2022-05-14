import colorama
import numpy as np
import torch
from tqdm import trange
from transformers import GPT2Tokenizer
from cog import BasePredictor, Input, BaseModel

import KID
from KID.agent import KIDAgent
from KID.envs import GPT2Env
from KID.infra import Frame
from KID.policy import KIDPolicy


class Output(BaseModel):
    vanilla_sampling_output: str
    kid_output: str


class Predictor(BasePredictor):
    def setup(self):

        model_path = "pretrained_models/gpt2-medium"
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        knowledge_trie_path = "KID/assets/dummy_trie.pickle"
        self.max_seq_len = 63

        # envs
        env = GPT2Env(
            model_name=model_path, max_len=self.max_seq_len, device=self.device
        )
        env.reset()

        kid_policy = KIDPolicy(
            model_name=model_path, device=self.device, kt_path=knowledge_trie_path
        )
        self.kid_agent = KIDAgent(env=env, policy=kid_policy, is_kid=True)
        self.norm_agent = KIDAgent(env=env, policy=kid_policy, is_kid=False)

    def predict(
        self,
        question: str = Input(default="Does marijuana impair driving ability?"),
    ) -> Output:
        num_gen_seq = 5
        seeds = [0, 128, 256, 512, 1024, 2048]
        tokenized_q = self.tokenizer.encode(self.tokenizer.bos_token + question)
        tokenized_q_ids = torch.tensor(
            tokenized_q, device=self.device, dtype=torch.long
        )

        norm_gen_text = ""
        print("Generating sequences for vanilla decoding...")
        for n in range(num_gen_seq):
            print(f"Generating sequence {n + 1}...")
            norm_action = Frame(past=None, last=tokenized_q_ids)
            norm_next_obs = None

            torch.manual_seed(seeds[n])
            np.random.seed(seeds[n])

            for _ in trange(self.max_seq_len):
                norm_next_obs = self.norm_agent.forward(norm_action)
                norm_gen_text = norm_next_obs["gen_text"]
                norm_action["past"] = norm_next_obs["past"]
                norm_action["last"] = norm_next_obs["last"]

        kid_gen_text, kid_ppls = [], []
        print("Generating sequences for KID decoding...")
        for n in range(num_gen_seq):
            print(f"Generating sequence {n + 1}...")
            kid_action = Frame(past=None, last=tokenized_q_ids)
            kid_next_obs = None

            torch.manual_seed(seeds[n])
            np.random.seed(seeds[n])

            for _ in trange(self.max_seq_len):
                kid_next_obs = self.kid_agent.forward(kid_action)
                kid_gen_text.append(kid_next_obs["gen_text"])
                kid_action["past"] = kid_next_obs["past"]
                kid_action["last"] = kid_next_obs["last"]
                kid_ppls.append(kid_next_obs["ppl"])

        return Output(
            vanilla_sampling_output=norm_gen_text,
            kid_output=kid_gen_text[kid_ppls.index(min(kid_ppls))],
        )
