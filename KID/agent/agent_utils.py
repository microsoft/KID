from typing import Dict, List, Tuple

import torch

BIG_CONST = 1e10


def top_k_filter(
    logits: torch.Tensor,
    top_k: int,
    min_tokens_to_keep: int,
    is_probs: bool = False
) -> torch.Tensor:
    """Helper function for top-k sampling decoding.

    :param logits:
    :param top_k:
    :param min_tokens_to_keep:
    :param is_probs:
    :return:
    """
    if top_k == 0:
        return logits
    else:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        values = torch.topk(logits, top_k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if is_probs:
            return torch.where(
                logits < batch_mins,
                torch.ones_like(logits) * 0.0, logits
            )
        return torch.where(
            logits < batch_mins,
            torch.ones_like(logits) * -BIG_CONST, logits
        )


def top_p_filter(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int,
    is_probs: bool = False
) -> torch.Tensor:
    """Helper function for nucleus sampling decoding, aka. top-p decoding.

    :param logits:
    :param top_p:
    :param min_tokens_to_keep:
    :param is_probs:
    :return:
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # keep at least min tokens
        sorted_indices_to_remove[..., :min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    if is_probs:
        scores = logits.masked_fill(indices_to_remove, 0.0)
    else:
        scores = logits.masked_fill(indices_to_remove, -float("Inf"))
    return scores


def calc_banned_ngram_tokens(
    banned_size: int, prev_input_ids: torch.Tensor, cur_len: int
) -> List[torch.Tensor]:
    """Calculate tokens ids that need to be banned in terms of ban size.

    :param banned_size: the n of the n-gram you want to keep no duplication.
    :param prev_input_ids: previously generated token ids.
    :param cur_len: current length of generated tokens.
    :return:
    """
    if cur_len + 1 < banned_size:
        return []

    def _get_ngrams(
        ngram_size: int, prev_ids: torch.Tensor
    ) -> Dict[Tuple[torch.Tensor], List[torch.Tensor]]:
        gen_ngrams: Dict[Tuple[torch.Tensor], List[torch.Tensor]] = {}
        gen_tokens = prev_ids[0].tolist()
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            gen_ngrams[prev_ngram_tuple] = gen_ngrams.get(prev_ngram_tuple,
                                                          []) + [ngram[-1]]
        return gen_ngrams

    def _get_generated_ngrams(
        banned_ngrams: Dict[Tuple[torch.Tensor], List[torch.Tensor]],
        prev_ids: torch.Tensor, ngram_size: int, cur_length: int
    ) -> List[torch.Tensor]:
        """Get the ngrams of the generated text.
        :param banned_ngrams:
        :param prev_ids:
        :param ngram_size:
        :param cur_length:
        :return:
        """
        start_idx = cur_length + 1 - ngram_size
        ngram_idx = tuple(prev_ids[0][start_idx:cur_length].tolist())
        return banned_ngrams.get(ngram_idx, [])

    generated_ngrams = _get_ngrams(banned_size, prev_input_ids)
    banned_tokens = _get_generated_ngrams(
        generated_ngrams, prev_input_ids, banned_size, cur_len
    )
    return banned_tokens
