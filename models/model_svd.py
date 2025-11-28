import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, GPTConfig  # assuming model.py is in the same directory


class GPTSVD(GPT):
    """GPT model with an SVD factorization method for the lm_head """

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def apply_svd_to_lm_head(self, rank: int):
        """
        Replace model.lm_head (final Linear) with a low-rank factorization U·S·Vᵀ ≈ W, i.e.
        lm_head(x) = (U·S) @ (Vᵀ x) so that input/output dims are unchanged but with an 
        intermediate dim = rank. Keep the head factorized for an inference time speed-up.
        """
        # 1) pull out the weight matrix W of shape [vocab_size, final_dim]
        W = self.lm_head.weight.data               # torch.Tensor[vocab_size, final_dim]
        vocab_size, final_dim = W.shape

        # 2) compute SVD (full_matrices=False gives U:[vocab,final_dim], Vh:[final_dim,final_dim])
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 3) truncate to rank
        Ur  = U[:, :rank]                            # [vocab_size, rank]
        Sr  = S[:rank]                               # [rank]
        Vhr = Vh[:rank, :]                           # [rank, final_dim]

        # 4) build two new Linear layers
        #    first: final_dim → rank  (weights = Vᵀ)
        linear1 = nn.Linear(final_dim, rank, bias=False)
        #    second: rank → vocab_size  (weights = U·S)
        linear2 = nn.Linear(rank, vocab_size, bias=False)

        # 5) copy over the truncated factors
        with torch.no_grad():
            linear1.weight.copy_(Vhr)               # [rank, final_dim]
            # U·diag(S): shape [vocab_size, rank]
            linear2.weight.copy_(Ur * Sr.unsqueeze(0))

        # 6) replace the old lm_head with a Sequential of the two factors
        self.lm_head = nn.Sequential(linear1, linear2)
