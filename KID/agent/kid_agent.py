import colorama
import spacy
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from transformers import AutoTokenizer

from KID.agent import BaseAgent
from KID.agent.agent_utils import calc_banned_ngram_tokens, top_k_filter, top_p_filter
from KID.envs import BaseEnv
from KID.infra import Frame
from KID.policy import BasePolicy

STOP_WORDS = set(stopwords.words('english'))
getter = lambda token: token.is_stop \
                       or token.lower_ in STOP_WORDS or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=getter, force=True)  # set attribute with getter

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)


class KIDAgent(BaseAgent):

    def __init__(
        self,
        env: BaseEnv,
        policy: BasePolicy,
        is_kid: bool,
        model_name : str = 'gpt2-medium'
    ):
        super(KIDAgent, self).__init__()
        self.env = env
        self.policy = policy
        self.is_kid = is_kid

        self.temperature = 1
        self.repetition_penalty = 1
        self.banned_ngram_size = 2
        self.min_length = 0
        self.gm_scale = 0.99

        self.top_p = 0.92
        self.top_k = 20

        self.sampling = True
        self.color = False

        self._cur_norm_ids = torch.empty(0)
        self._cur_kid_ids = torch.empty(0)

        if '/opt-' in model_name: # hacky way to detect if using OPT
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, action: 'Frame') -> 'Frame':

        # action should have past and last
        past, last = action['past'], action['last']
        if past is None:  # initial step
            action = self.env.step(action)  # now action['past'] should not be None
            self._cur_norm_ids = last
            if self.is_kid:
                self._cur_kid_ids = last

        observation = self.env.step(action)  # do the real step with valid action
        norm_logits = observation['logits']
        norm_past = observation['past']

        if not self.is_kid:
            # by default, we are using sampling decoding
            norm_last_prob = F.softmax(norm_logits[:, -1, :], dim=-1)

            if self.sampling:
                norm_last_prob = norm_last_prob[~torch.
                                                any(norm_last_prob.isnan(), dim=-1)]
                norm_last = torch.multinomial(norm_last_prob, num_samples=1)
            else:  # or we can choose greedy decoding
                _, norm_last = torch.topk(norm_last_prob, k=1, dim=-1)

            self._cur_norm_ids = norm_last if self._cur_norm_ids is None \
                else torch.cat((self._cur_norm_ids, norm_last), dim=1)

            gen_text_norm = self.tokenizer.decode(
                self._cur_norm_ids.tolist()[0], skip_special_tokens=True
            )
            observation['last'] = norm_last
            observation['gen_text'] = gen_text_norm
            observation['cur_gen_id'] = self._cur_norm_ids
            return observation

        else:  # if it is for KID
            norm_last_prob = self.pos_func(norm_logits, is_kid=True)

            kid_action = self.policy.update_policy(
                Frame(
                    past=norm_past,
                    last=last,
                    logits=norm_logits,
                    cur_gen_ids=self._cur_kid_ids,
                    is_kid=True,
                )
            )

            kid_observation = self.env.step(kid_action)

            kid_kg_indices = kid_observation['kg_indices']
            kid_kg_indices = list(filter(lambda x: len(x) <= 1, kid_kg_indices))
            kid_kg_indices = [ind[0] for ind in kid_kg_indices]

            kid_logits = kid_observation['logits']
            kid_last_prob = self.pos_func(kid_logits, is_kid=True)

            # fuse the two probabilities
            kid_last_prob = (kid_last_prob**
                             self.gm_scale) * (norm_last_prob**(1 - self.gm_scale))
            kid_last_prob = top_k_filter(
                kid_last_prob,
                top_k=self.top_k,
                min_tokens_to_keep=self.min_length,
                is_probs=True
            )

            kid_last_prob = top_p_filter(
                kid_last_prob,
                top_p=self.top_p,
                min_tokens_to_keep=self.min_length,
                is_probs=True
            )

            # rescale
            if torch.sum(kid_last_prob) <= 1:
                kid_last_prob = kid_last_prob / torch.sum(kid_last_prob)

            if self.sampling:
                kid_last_prob = kid_last_prob[~torch.
                                              any(kid_last_prob.isnan(), dim=-1)]
                kid_last = torch.multinomial(kid_last_prob, num_samples=1)
            else:  # or we can choose greedy decoding
                _, kid_last = torch.topk(kid_last_prob, k=1, dim=-1)

            self._cur_kid_ids = kid_last if self._cur_kid_ids is None \
                else torch.cat((self._cur_kid_ids, kid_last), dim=1)

            if self.color:
                gen_text_kid = ""
                for word_id in self._cur_kid_ids.tolist()[0]:
                    if word_id in kid_kg_indices:
                        gen_text_kid += "{}{}{}".format(
                            colorama.Fore.RED,
                            self.tokenizer.decode([word_id], skip_special_tokens=True),
                            colorama.Style.RESET_ALL,
                        )
                    else:
                        gen_text_kid += self.tokenizer.decode(
                            [word_id], skip_special_tokens=True
                        )
            else:
                gen_text_kid = self.tokenizer.decode(
                    self._cur_kid_ids.tolist()[0], skip_special_tokens=True
                )
            kid_observation['last'] = kid_last
            kid_observation['gen_text'] = gen_text_kid
            kid_observation['cur_gen_id'] = self._cur_kid_ids
            return kid_observation

    def pos_func(self, logits: torch.Tensor, is_kid: bool = False) -> torch.Tensor:

        last_logits = logits[:, -1, :]

        # load already generated tokens
        if is_kid:
            gen_toks_ids = self._cur_kid_ids
        else:
            gen_toks_ids = self._cur_norm_ids

        # repetition penalty
        for token_idx in set(gen_toks_ids[0].tolist()):
            if last_logits[0, token_idx] < 0:
                last_logits[0, token_idx] *= self.repetition_penalty
            else:
                last_logits[0, token_idx] /= self.repetition_penalty

        # ban duplicated ngrams
        cur_length = gen_toks_ids.size(1)
        banned_batch_tokens = calc_banned_ngram_tokens(
            self.banned_ngram_size, gen_toks_ids, cur_length
        )
        # print("banned tokens", banned_batch_tokens)

        for banned_tokens in enumerate(banned_batch_tokens):
            last_logits[:, banned_tokens] = -float("inf")
        # # del banned_batch_tokens

        # min_length guarantee
        if cur_length < self.min_length:
            last_logits[:, self.tokenizer.eos_token_id] = -float("inf")

        last_prob = F.softmax(last_logits, dim=-1)

        return last_prob
