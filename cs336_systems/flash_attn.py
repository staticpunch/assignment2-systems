from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool, Int
# from nn_utils import softmax
from .attention import (
    scaled_dot_product_attention,
    annotated_scaled_dot_product_attention
)

import lovely_tensors as lt
lt.monkey_patch()

logger = logging.getLogger("flash_attn")

FLASH_ATTENTION_DOCSTRING = """
FlashAttention-2 forward pass with tiled computation and online softmax.

This implementation computes attention output O = softmax(QK^T)V using tiling
to avoid materializing the full attention matrix S = QK^T in HBM. Each tile
of the output P is computed independently, enabling memory-efficient attention.

Algorithm Overview:
-------------------
To avoid reading and writing the attention matrix to/from HBM, we use tiling
to compute each tile of the output independently. This requires computing tiles
of P that are ideally tiled in both dimensions (for queries and keys).

Online Softmax:
---------------
Since softmax(S) requires entire rows of S to compute the denominator, we cannot
compute P in tiles directly. FlashAttention-2 solves this using online softmax,
which incrementally computes softmax statistics as we process key tiles.

Notation:
---------
- i: subscript index denoting the current query tile
- j: superscript index denoting the current key tile
- B_q: tile size along the query dimension
- B_k: tile size along the key dimension
- d: hidden dimension (not tiled)

Running Values (per query tile):
--------------------------------
For each query tile i, we maintain row-wise running statistics:

- m_i^(j) ∈ R^(B_q): Running maximum across key tiles
    * Tracks the maximum value seen so far for numerical stability
    * Updated as: m_i^(j) = max(m_i^(j-1), rowmax(S_ij))
    * Used to compute numerically stable softmax

- l_i^(j) ∈ R^(B_q): Running proxy for softmax denominator
    * Accumulates unnormalized softmax values
    * Updated as: l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(exp(S_ij - m_i^(j)))

- P̃_i^(j) = exp(S_ij - m_i^(j)): Unnormalized softmax numerators
    * Computed for the current tile using the running maximum

Final Output:
-------------
After processing all T_k key tiles, normalize the output using the final
running denominator l_i^(T_k) to obtain the complete attention output.

Notes:
------
This algorithm enables O(1) memory complexity with respect to sequence length
by never materializing the full attention matrix in HBM, only processing tiles
that fit in SRAM.   
"""

EXAMPLE_DOCSTRING = """
Concrete Example: Two-Block Computation
----------------------------------------
For the simple case of 2 key/value tiles (j=1,2), the computation with online 
softmax algorithm proceeds as follows:

**First Iteration (j=1):**

    m^(1) = rowmax(S^(1)) ∈ ℝ^(B_r)
        Initialize running maximum from first attention score tile
    
    ℓ^(1) = rowsum(exp(S^(1) - m^(1))) ∈ ℝ^(B_r)
        Initialize running softmax denominator (unnormalized)
    
    Õ^(1) = exp(S^(1) - m^(1)) V^(1) ∈ ℝ^(B_r × d)
        Compute initial weighted output (unnormalized)

**Second Iteration (j=2):**

    m^(2) = max(m^(1), rowmax(S^(2))) = m
        Update running maximum across both tiles (becomes final m)
    
    ℓ^(2) = exp(m^(1) - m^(2)) ℓ^(1) + rowsum(exp(S^(2) - m^(2)))
            = rowsum(exp(S^(1) - m)) + rowsum(exp(S^(2) - m)) = ℓ
        Rescale previous denominator and add current tile's contribution
        (becomes final ℓ after all tiles processed)
    
    P̃^(2) = diag(ℓ^(2))^(-1) exp(S^(2) - m^(2))
        Normalized attention weights for second tile only
    
    Õ^(2) = diag(exp(m^(1) - m^(2)))^(-1) Õ^(1) + exp(S^(2) - m^(2)) V^(2)
            = exp(S^(1) - m) V^(1) + exp(S^(2) - m) V^(2)
        Rescale previous output and add current tile's contribution
        (accumulated unnormalized output)
    
    O^(2) = diag(ℓ^(2))^(-1) Õ^(2) = O
        Final normalization produces correct attention output

Key Insights:
-------------
- m: Final running maximum used for numerical stability
- ℓ: Final softmax denominator = sum of exp(S_ij - m) over all tiles
- Õ: Accumulated weighted sum that gets rescaled as m updates
- O: Final normalized output = Õ / ℓ

The rescaling factors (exp(m^(j-1) - m^(j))) ensure correctness when
the running maximum changes, allowing us to process tiles sequentially
without materializing the full attention matrix.

Dimensions:
-----------
- B_r: Number of rows in query tile (block size)
- d: Hidden dimension
- S^(j): Attention scores for tile j, shape (B_r, B_c)
- V^(j): Value vectors for tile j, shape (B_c, d)
- m, ℓ: Row-wise statistics, shape (B_r,)
- O, Õ: Output matrices, shape (B_r, d)
"""
def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def attention_debug(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    is_causal: Bool = False,
):
    n_queries = Q.shape[-2]
    n_keys = K.shape[-2]
    d = Q.shape[-1]
    scale = 1 / (d ** 0.5)

    # Equation 4
    S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[None, :, None] >= 
            torch.arange(n_keys, device=S.device)[None, None, :],
            S, -1e6
        )

    # Equation 5
    P = softmax(S, dim=-1)

    # Equation 6
    O = einsum(P, V, '... q k, ... k d -> ... q d')

    # Equation 12
    L = torch.logsumexp(S, dim=-1)
    return (S, L, O, P)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal: Bool | None = False,
    ):
        """
        Running Values (per query tile):
        --------------------------------
        For each query tile i, we maintain row-wise running statistics:

        - m_i^(j) ∈ R^(B_q): Running maximum across key tiles
            * Tracks the maximum value seen so far for numerical stability
            * Definition: m_i^{j} = rowmax(Si[:jth-tile])
            * Recursive formula: m_i^(j) = max(m_i^(j-1), rowmax(S_ij))
            * Used to compute numerically stable softmax

        - l_i^(j) ∈ R^(B_q): Running proxy for softmax denominator
            * Accumulates unnormalized softmax values
            * Definition: l_i^{j} = sum(exp(Si[:jth-tile] - rowmax(Si[:jth-tile])))
            * Recursive formula: l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(exp(S_ij - m_i^(j)))
            * Used for easier backward recomputation

        - P̃_i^(j) = exp(S_ij - m_i^(j)): Unnormalized softmax numerators
            * Computed for the current tile using the running maximum

        Final Output:
        -------------
        After processing all T_k key tiles, normalize the output using the final
        running denominator l_i^(T_k) to obtain the complete attention output.
        """
        debug = True
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        Bq, Bk = 16, 32
        Tq, Tk = math.ceil(n_queries / Bq), math.ceil(n_keys / Bk)

        # Reshaping Q, K, V so that each outer for-loop iteration 
        # deals with elements from a single batch index
        original_shape = Q.shape
        Q = rearrange(Q, "... queries d -> (...) queries d")  # (n_programs, n_queries, d)
        K = rearrange(K, "... keys d -> (...) keys d")        # (n_programs, n_keys, d)
        V = rearrange(V, "... keys d -> (...) keys d")        # (n_programs, n_keys, d)
        O = torch.zeros(Q.shape)                              # (n_programs, n_queries, d)
        L = torch.zeros(Q.shape[:-1])                         # (n_programs, n_queries)
        n_programs = Q.shape[0]

        if debug:
            S_debug = torch.zeros(n_programs, n_queries, n_keys)  # (n_programs, n_queries, n_keys)

        for pid in range(n_programs):
            # Extract tensors for current batch element
            q, k, v, o, l = Q[pid], K[pid], V[pid], O[pid], L[pid]  # (n_queries, d), (n_keys, d), (n_keys, d), (n_queries, d), (n_queries,)
            
            for i in range(Tq):
                # Load query tile
                Qi = q[i*Bq:(i+1)*Bq, :]                    # (Bq, d)
                Oi = o[i*Bq:(i+1)*Bq, :]                    # (Bq, d)
                li = l[i*Bq:(i+1)*Bq]                       # (Bq,)
                mi = torch.full((Bq,), -torch.inf)          # (Bq,) - initialized to -inf

                for j in range(Tk):
                    # Load key and value tiles
                    Kj = k[j*Bk:(j+1)*Bk, :]                # (Bk, d)
                    Vj = v[j*Bk:(j+1)*Bk, :]                # (Bk, d)

                    # Step 1: Compute tile of pre-softmax attention scores
                    # Sij = (Bq, Bk) <- Qi @ Kj.T = (Bq, d) @ (d, Bk)
                    Sij = (Qi @ Kj.T) * scale
                    if debug: 
                        S_debug[pid, i*Bq:(i+1)*Bq, j*Bk:(j+1)*Bk] = Sij

                    # Step 2: Update running maximum
                    # mi_next = (Bq,) <- max(mi, rowmax(Sij))
                    mi_next = torch.max(mi, Sij.max(dim=-1).values)

                    # Step 3: Compute unnormalized softmax numerator
                    # Pij = (Bq, Bk) <- exp(Sij - mi_next[:, None])
                    Pij = torch.exp(Sij - mi_next[:, None])
                    
                    # Step 4: Update running denominator proxy
                    # li_next = (Bq,) <- exp(mi - mi_next) * li + rowsum(Pij)
                    li_next = torch.exp(mi - mi_next) * li + Pij.sum(dim=-1)
        
                    # Step 5: Update output accumulator
                    # Oi_next = (Bq, d) <- exp(mi - mi_next)[:, None] * Oi + Pij @ Vj
                    Oi_next = torch.exp(mi - mi_next)[:, None] * Oi + Pij @ Vj
                    
                    # Step 6: Update running values
                    mi, li, Oi = mi_next, li_next, Oi_next

                # Normalize output by final denominator
                # Oi = (Bq, d) <- Oi / li[:, None]
                Oi = Oi / li[:, None]
                
                # Compute logsumexp for backward pass
                # Li = (Bq,) <- mi + log(li)
                Li = mi + torch.log(li)

                # Write tile results back
                o[i*Bq:(i+1)*Bq, :] = Oi
                l[i*Bq:(i+1)*Bq] = Li

            # Write batch results back
            L[pid] = l
            O[pid] = o
        
        # Reshape outputs to original shape
        L = L.view(original_shape[:-1])  # (..., n_queries)
        O = O.view(original_shape)       # (..., n_queries, d)

        result = (S_debug, L, O) if debug else O
        return result

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class AttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal: Bool | None = None,
    ):
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)

        # Equation 4
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
        if is_causal:
            S = torch.where(
                torch.arange(n_queries, device=S.device)[None, :, None] >= 
                torch.arange(n_keys, device=S.device)[None, None, :],
                S, -1e6
            )

        # Equation 5
        P = torch.softmax(S, dim=-1)

        # Equation 6
        O = einsum(P, V, '... q k, ... k d -> ... q d')

        # Equation 12
        L = torch.logsumexp(S, dim=-1)
        ctx.save_for_backward(L, Q, K, V, O)
        return O
        
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

def flash_attention_forward_pytorch(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    is_causal: Bool | None = None,
):
    """
    Implements the FlashAttention-2 forward pass as a pure PyTorch autograd.Function.

    This implementation does not use Triton and is intended to be a slower,
    more debuggable reference for a Triton kernel implementation. It computes the
    attention output `O` and the log-sum-exp `L` value using a tiled approach.

    Args:
        ctx (torch.autograd.function.Context): The context object for `autograd.Function`.
            It is used to save tensors like Q, K, V, O, and L for the backward pass.
        Q (torch.Tensor): The query tensor.
        K (torch.Tensor): The key tensor.
        V (torch.Tensor): The value tensor.
        is_causal (bool, optional): If True, a causal mask is applied to prevent
            attention to future tokens. Defaults to False.

    Returns:
        torch.Tensor: The attention output tensor `O`.

    Notes:
        - The implementation should use a tiled algorithm. Tile sizes can be
        chosen by the implementer but should be at least 16x16.
        - Input tensor dimensions will always be clean powers of 2 and at least 16,
        so out-of-bounds accesses do not need to be handled.
        - The backward method for this function should be defined but can simply
        raise `NotImplementedError` for this task.
        - The log-sum-exp value `L` should be calculated and saved to the context
        using `ctx.save_for_backward()` for the (unimplemented) backward pass.
    """
    pass


def _make_attn_inputs(
    device=None,
    dtype=torch.float32,
    batch_size=8,
    n_queries=128,
    n_keys=128,
    head_dim=64
):
    torch.random.manual_seed(42)
    q = torch.randn(batch_size, n_queries, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, n_queries, head_dim, device=device, dtype=dtype)

    return q, k, v, do

if __name__ == "__main__":
    Q, K, V, dO = _make_attn_inputs(device="cpu")
    # manual_outputs = scaled_dot_product_attention(Q, K, V)
    S_ref, L_ref, O_ref, P_ref = attention_debug(Q, K, V, is_causal=False)
    S_debug, L_debug, O = FlashAttentionPytorch.apply(Q, K, V, False)
    print(f"S_ref: {S_ref}")
    print(f"L_ref: {L_ref}")
    print(f"O_ref: {O_ref}")
    print(f"P_ref: {P_ref}")
    print("---------------")
    print(f"S_debug: {S_debug}")
    print(f"L_debug: {L_debug}")
    print(f"O_debug: {O}")
    # print(f"P_debug: {P_debug}")
    torch.testing.assert_close(S_ref, S_debug, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(L_ref, L_debug, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(O_ref, O, rtol=1e-2, atol=1e-2)
    # print("lovely tensors")
