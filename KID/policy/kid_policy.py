import pickle
from operator import add
from typing import Any, List

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, OPTForCausalLM

from KID.infra import Frame
from KID.policy import BasePolicy

SMALL_CONST = 1e-15
BIG_CONST = 1e10

getter = lambda tk: tk.is_stop or tk.lower_ in STOP_WORDS or tk.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=getter, force=True)  # set attribute with getter

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)


class KIDPolicy(BasePolicy):

    def __init__(self, model_name: str, device: str, kt_path: str):
        super(KIDPolicy, self).__init__()
        self.device = device
        if '/opt-' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            #self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.OPT_FLAG = False # Flag for HF bug related to OPT
        if 'gpt2' in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        elif 'gpt-neo' in model_name:
            self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        elif '/opt-' in model_name:
            self.model = OPTForCausalLM.from_pretrained(model_name)
            self.OPT_FLAG = True
        self.model.to(self.device)

        # freezing weights of the LM,
        # so that KID is orthogonal to the contribution of LM.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.kid_steps = 3
        self.kl_scale = 0.01
        self.step_size = 0.01
        self.gamma = 1.5
        self.max_hops = 1
        self.stride = 512

        # load the external knowledge trie
        with open(kt_path, 'rb') as handle:
            self.kg_trie = pickle.load(handle)

    def update_policy(self, frame: 'Frame', **kwargs) -> 'Frame':

        past, last, = frame['past'], frame['last']
        norm_logits, cur_gen_ids = frame['logits'], frame['cur_gen_ids']

        cur_ppl = self._compute_ppl(cur_gen_ids)

        cur_gen_toks = self.tokenizer.decode(
            cur_gen_ids.tolist()[0]
        )  # decode to actual text

        related_kgs = list(set(self.update_knowledge(cur_gen_toks)))

        kg_indices = [
            #self.tokenizer.encode(word.strip(), add_prefix_space=True)
            self.tokenizer.encode(word.strip())
            for word in related_kgs
        ]

        past = [torch.cat([pp.unsqueeze(0) for pp in p], dim=0) for p in past]
        past = [p.detach_() for p in past]

        norm_last_logits = norm_logits[:, -1, :].detach()
        norm_last_probs = torch.softmax(norm_last_logits, dim=-1)

        # we are imitating optimizer.step() manually, for easy gradient accumulation
        grad_accumulator = [(np.zeros(p.size()).astype("float32")) for p in past]
        grad_norms = None

        for _ in range(self.kid_steps):
            # this part has been inspired by DExperts https://arxiv.org/abs/2105.03023
            # and PPLM https://arxiv.org/abs/1912.02164
            kid_controlling = [
                torch.from_numpy(p_).requires_grad_(True).to(device=self.device)
                for p_ in grad_accumulator
            ]

            for p_ in kid_controlling:
                p_.retain_grad()

            kid_controlled_past = list(map(add, past, kid_controlling))

            # reward
            policy_reward = self.compute_reward(
                last=last,
                kg_indices=kg_indices,
                norm_last_probs=norm_last_probs,
                kid_past=kid_controlled_past
            )

            policy_reward.backward()

            if grad_norms is not None:
                grad_norms = [
                    torch.max(grad_norms[index], torch.norm(p_.grad))
                    for index, p_ in enumerate(kid_controlling)
                ]
            else:
                grad_norms = []
                for p_ in kid_controlling:
                    grad_norms.append(torch.norm(p_.grad) + SMALL_CONST)

            grad = [
                -self.step_size *
                (p_.grad / grad_norms[index]**self.gamma).detach().cpu().numpy()
                for index, p_ in enumerate(kid_controlling)
            ]

            # accumulate gradient
            grad_accumulator = list(map(add, grad, grad_accumulator))

            for p_ in kid_controlling:
                p_.grad.detach().zero_()

            # removing past from the computing graph
            new_past = []
            for p_ in past:
                new_past.append(p_.detach())
            past = new_past

        # apply the accumulated grad controlling to the past
        grad_accumulator = [
            torch.from_numpy(p_).requires_grad_(True).to(self.device)
            for p_ in grad_accumulator
        ]

        frame['past'] = list(map(add, past, grad_accumulator))
        frame['kg_indices'] = kg_indices
        frame['ppl'] = cur_ppl

        return frame

    def update_knowledge(self, cur_gen_toks: str) -> List[Any]:
        """Get local knowledge (local_kg), and query external knowledge (kg_trie)

        :param cur_gen_toks: current local generated sentence
        :return: merged kgs (list of triplet nodes)
        """
        local_kg = [
            token.lemma_.lower() for token in nlp(cur_gen_toks)
            if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
        ]
        local_kg = list(set(local_kg))

        # if len(local_kg) > self.max_hops:
        #     local_kg = local_kg[-self.max_hops]

        related_kgs = []
        for _ in range(self.max_hops):
            for ent in local_kg:
                for span in self.kg_trie.keys(ent):
                    related_kgs.extend(span.split(' '))
            local_kg = related_kgs

        return related_kgs

    def compute_reward(
        self, last: torch.Tensor, kg_indices: List[List[int]],
        norm_last_probs: torch.Tensor, kid_past: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute reward: knowledge part + KL penalty

        :param last: the last generated token id
        :param kg_indices: List of knowledge term token ids (inner list) in vocabulary
        :param norm_last_probs: the vanilla policy (pi_t)
        :param kid_past: the KID updated policy (pi_t^*)
        :return:
        """

        if self.OPT_FLAG:
            past_kv_length = kid_past[0][0].shape[2]
            bs = kid_past[0][0].shape[0]
            attn = torch.ones(bs, past_kv_length + last.size(-1)).to(last.device)
        else:
            attn = None
            
        kid_logits, _, kid_all_hidden = self.model(
            last,
            past_key_values=kid_past,
            attention_mask=attn,
            return_dict=False,
            output_hidden_states=True
        )

        kid_last_logits = kid_logits[:, -1, :]
        kid_last_probs = F.softmax(kid_last_logits, dim=-1)

        # tackle case where no knowledge term token id has been retrieved (otherwise it crashes)
        if len(kg_indices) == 0:
            kg_indices = [[self.tokenizer.eos_token_id]]
        one_hot_neg_vec = self._build_one_hot_vec(kg_indices).detach()

        policy_Q = -torch.log(torch.sum(torch.mm(kid_last_probs, one_hot_neg_vec.T)))

        # the KL tricks from PPO https://arxiv.org/pdf/1707.06347.pdf
        norm_last_probs = norm_last_probs + SMALL_CONST * (
            norm_last_probs <= SMALL_CONST
        ).float().to(self.device).detach()
        kid_last_probs = kid_last_probs + SMALL_CONST * (
            kid_last_probs <= SMALL_CONST
        ).float().to(self.device).detach()
        kl_penalty = self.kl_scale * (
            (kid_last_probs * (kid_last_probs / norm_last_probs).log()).sum()
        )

        policy_Q += kl_penalty

        return policy_Q

    def _build_one_hot_vec(self, indices: List[List[int]]):
        """Helper function. convert indices into one hot vectors

        :param indices: list of knowledge term token ids (inner list) in vocabulary
        :return: one hot vectors of the encoded knowledge terms.
                 size of (num of kg terms, vocab size)
        """
        if indices is None or len(indices) == 0:
            return None

        indices = [ind[0] for ind in indices]
        #oh_vec = torch.zeros(len(indices), len(self.tokenizer)).to(self.device)
        oh_vec = torch.zeros(len(indices), self.model.config.vocab_size).to(self.device)
        oh_vec.scatter_(1, torch.tensor(indices).to(self.device).unsqueeze(1), 1)
        return oh_vec

    def _compute_ppl(self, encodings: torch.Tensor) -> torch.Tensor:
        """Compute the perplexity

        :param encodings:
        :return:
        """
        if hasattr(self.model.config, 'n_positions'):
            max_length = self.model.config.n_positions
        else:
            max_length = self.model.config.max_position_embeddings

        nlls = []
        for i in range(0, encodings.size(1), self.stride):
            begin_loc = max(i + self.stride - max_length, 0)
            end_loc = min(i + self.stride, encodings.size(1))
            trg_len = end_loc - i  # may be different from self.stride on last loop
            input_ids = encodings[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / end_loc)
