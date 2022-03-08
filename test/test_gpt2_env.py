import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from KID.envs.gpt2_env import GPT2Env
from KID.infra import Frame


def test_gpt2_env():
    # init
    torch.manual_seed(2)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenized_text = tokenizer.encode(
        tokenizer.bos_token + "Does marijuana impair driving ability?"
    )
    tokenized_text = torch.tensor(tokenized_text, device='cuda:0', dtype=torch.long)

    # envs
    env = GPT2Env(model_name='gpt2-medium', max_len=64, device='cuda:0')
    env.reset()

    # let's try first step
    action = Frame(past=None, last=tokenized_text, is_kid=False)
    is_done = False
    cur_gen_ids = None

    while not is_done:
        observation = env.step(action)
        logits = observation['logits']
        past = observation['past']
        is_done = observation['is_done']

        last_logits = logits[:, -1, :]
        last_probs = F.softmax(last_logits, dim=-1)

        last_probs = last_probs[~torch.any(last_probs.isnan(), dim=-1)]
        last = torch.multinomial(last_probs, num_samples=1)
        action['past'], action['last'] = past, last
        cur_gen_ids = last if cur_gen_ids is None else torch.cat(
            (cur_gen_ids, last), dim=1
        )

    generated_text = tokenizer.decode(cur_gen_ids.tolist()[0])  # decode to actual text
    print("Does marijuana impair driving ability?" + generated_text)


if __name__ == '__main__':
    test_gpt2_env()
