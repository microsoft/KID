import os

import colorama
import numpy as np
import torch
from tqdm import trange
from transformers import GPT2Tokenizer

import KID
from KID.agent import KIDAgent
from KID.envs import GPTNeoEnv
from KID.infra import Frame
from KID.policy import KIDPolicy

model_path = 'EleutherAI/gpt-neo-1.3B'

if __name__ == '__main__':

    max_seq_len = 63
    num_gen_seq = 5
    trained_model_name = 'EleutherAI/gpt-neo-1.3B'
    device = 'cuda:0'

    seeds = [0, 128, 256, 512, 1024, 2048]

    is_kid = True

    # replace with the proper trie path
    assets_path = os.path.join(list(KID.__path__)[0], "assets")
    knowledge_trie_path = os.path.join(assets_path, "dummy_trie.pickle")

    # init
    tokenizer = GPT2Tokenizer.from_pretrained(trained_model_name)
    q = "Does marijuana impair driving ability?"
    tokenized_q = tokenizer.encode(tokenizer.bos_token + q)
    tokenized_q_ids = torch.tensor(tokenized_q, device=device, dtype=torch.long)

    # envs
    env = GPTNeoEnv(model_name=model_path, max_len=max_seq_len, device=device)
    env.reset()

    # policy
    kid_policy = KIDPolicy(
        model_name=model_path, device=device, kt_path=knowledge_trie_path
    )

    # agent
    if is_kid:
        agent = KIDAgent(env=env, policy=kid_policy, is_kid=True)
    else:
        agent = KIDAgent(env=env, policy=kid_policy, is_kid=False)

    norm_gen_text, kid_gen_text, kid_ppls = '', [], []
    for n in range(num_gen_seq):
        print(f"Generating sequence {n+1}...")
        norm_action = Frame(past=None, last=tokenized_q_ids)
        kid_action = Frame(past=None, last=tokenized_q_ids)
        norm_next_obs, kid_next_obs = None, None

        torch.manual_seed(seeds[n])
        np.random.seed(seeds[n])

        for _ in trange(max_seq_len):
            if is_kid:
                kid_next_obs = agent.forward(kid_action)
                kid_gen_text.append(kid_next_obs['gen_text'])
                kid_action['past'] = kid_next_obs['past']
                kid_action['last'] = kid_next_obs['last']
                kid_ppls.append(kid_next_obs['ppl'])
            else:
                norm_next_obs = agent.forward(norm_action)
                norm_gen_text = norm_next_obs['gen_text']
                norm_action['past'] = norm_next_obs['past']
                norm_action['last'] = norm_next_obs['last']

    print(
        "= The context for generation is: {}{}{}".format(
            colorama.Fore.YELLOW, q, colorama.Style.RESET_ALL
        )
    )
    if is_kid:
        print("= KID Decoding: {}".format(kid_gen_text[kid_ppls.index(min(kid_ppls))]))
    else:
        print("= Sampling Decoding: {}".format(norm_gen_text))
