---
title: >-
  [Paper Note] Non-Markovian Discrete Diffusion with Causal Language Models
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion models] This paper proposes CaDDi, a framework that enables each denoising step to access the full generation trajectory via a non-Markovian discrete diffusion process…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "discrete diffusion models"
  - "non-Markovian"
  - "causal language models"
  - "sequence generation"
  - "autoregressive"
date: 2026-05-08
content_hash: 7dd7d635aba8d518
---

# Non-Markovian Discrete Diffusion with Causal Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.09767](https://arxiv.org/abs/2502.09767)
**Code**: [https://github.com/](https://github.com/) (not provided)
**Area**: Image/Text Generation (Discrete Diffusion)
**Keywords**: discrete diffusion models, non-Markovian, causal language models, sequence generation, autoregressive

## TL;DR

This paper proposes CaDDi, a framework that enables each denoising step to access the full generation trajectory via a non-Markovian discrete diffusion process, and unifies this process within a causal language model architecture, allowing pretrained LLMs to be directly reused as discrete diffusion models.

## Background & Motivation

Discrete diffusion models (e.g., D3PM, SEDD, MDLM) have demonstrated flexible and controllable advantages in structured sequence generation, particularly in tasks such as text infilling and bidirectional generation. However, their generation quality still lags behind autoregressive language models.

**Limitations of Prior Work**: Existing discrete diffusion models rely on the Markov assumption—each denoising step can only observe the current state $\mathbf{x}_t$ and cannot exploit previous generation history. This forces all information to be compressed into a single state, causing errors at any step to accumulate irreversibly across time steps. Furthermore, the factorization assumption $p_\theta(\mathbf{x}_0|\mathbf{x}_t) = \prod_i p_\theta(\mathbf{x}_0^i|\mathbf{x}_t)$ prevents the model from capturing inter-token dependencies, further limiting its self-correction capability.

**Key Challenge**: Autoregressive LMs achieve high generation quality but lack bidirectional flexibility; discrete diffusion models are flexible but produce lower quality. Can the two be unified?

**Key Insight**: This work reinterprets discrete diffusion as a generalized hierarchical VAE (HVAE), breaking the Markov constraint so that the reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T})$ can access the complete future trajectory. The key observation is that this non-Markovian autoregressive inference structure naturally aligns with the architecture of causal language models—requiring only an additional time dimension. Consequently, the sequence dimension (token order) and the time dimension (diffusion timestep) can be unified within a single decoder-only Transformer, and the standard causal LM corresponds to the special case $T=1$, enabling pretrained LLMs to be directly fine-tuned as discrete diffusion models.

## Method

### Overall Architecture

CaDDi is a non-Markovian discrete diffusion framework that jointly models the sequence (causal) and time (diffusion) dimensions. The input is formed by concatenating the complete non-Markovian forward trajectory $(\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_0)$, processed by a decoder-only Transformer with a block-wise causal mask. At inference time, the model starts from pure noise $\mathbf{x}_T$ and autoregressively predicts clean data $\tilde{\mathbf{x}}_0$, then re-corrupts via the forward kernel to obtain $\mathbf{x}_{t-1}$.

### Key Designs

1. **Non-Markovian Forward Process**: Unlike standard Markov chain corruption $q(\mathbf{x}_t|\mathbf{x}_{t-1})$, the paper adopts independent corruption: $q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_0)$. Noise at each timestep is independent of the others conditioned on $\mathbf{x}_0$, yielding a fundamentally different trajectory structure in which different timesteps carry complementary information. The forward process requires only the marginal corruption kernel $q(\mathbf{x}_t|\mathbf{x}_0)$, and standard absorbing or uniform kernels can be reused.

2. **Non-Markovian Reverse Inference**: The reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T}) = q(\mathbf{x}_{t-1}|\mathbf{x}_0 = \mu_\theta(\mathbf{x}_{t:T}, t))$ delegates full-history dependence to the denoiser $\mu_\theta$. The independent corruption kernel greatly simplifies the posterior form—$\mathbf{x}_{t:T}$ need not participate directly in posterior computation; instead, the denoiser predicts $\tilde{\mathbf{x}}_0$, and $\mathbf{x}_{t-1}$ is sampled via $\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1}|\tilde{\mathbf{x}}_0)$.

3. **2D Rotary Position Encoding (2D RoPE)**: Existing causal LM RoPE encodes only the sequence dimension. This paper extends it to a block-diagonal structure: $\mathbf{R}_t^{(i)} = \text{diag}[\mathbf{R}_{\text{seq}}^{(i)}, \mathbf{R}_{\text{time}}^{(t)}]$, encoding token position and diffusion timestep in separate subspaces of the query/key representations. Within each timestep, the attention pattern is identical to that of a standard causal LM, ensuring full backward compatibility.

4. **CaDDi-AR Variant**: At each timestep, a further token-level autoregressive factorization is applied: $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T}) = \prod_{i} p_\theta(\mathbf{x}_{t-1}^i|\mathbf{x}_{t-1}^{0:i-1}, \mathbf{x}_{t:T})$, where the historical trajectory serves as a "prompt" for autoregressive generation. When $T=1$, this reduces to a standard causal LM, allowing pretrained LLM weights to be loaded directly for fine-tuning.

5. **Semi-Speculative Decoding**: Naïve generation with CaDDi-AR requires $O(L \times T)$ forward passes. Exploiting the fact that all timesteps share the same denoising target $\mathbf{x}_0$, the prediction $\tilde{\mathbf{x}}_0^{\text{prev}}$ from the previous timestep is reused as a draft for the current step. The model verifies all tokens in parallel and resamples only from the first rejected position, substantially accelerating inference.

### Loss & Training

For the absorbing kernel, the ELBO simplifies to a weighted cross-entropy: $\mathcal{L}_{\text{absorb}} = \mathbb{E} \sum_{t=1}^T [\alpha_{t-1} \mathbf{x}_0^\top \log \mu_\theta(\mathbf{x}_{t:T}, t)]$, where the weight $\alpha_{t-1}$ reflects the degree of corruption. CaDDi-AR uses a next-token prediction loss. In practice, latent truncation and trajectory re-composition are employed to compress the context window.

## Key Experimental Results

### Main Results

**LM1B Dataset — Generation Perplexity (PPL, lower is better)**

| Model | GPT2 (T=0.5) | Llama-2 (T=0.7) | Llama-3 (T=0.5) |
|-------|-------------|-----------------|-----------------|
| UDLM | 328.99 | 111.86 | 231.05 |
| D3PM | 133.38 | 55.54 | 110.86 |
| SEDD | 81.44 | 66.54 | 60.00 |
| MDLM | 106.20 | 62.34 | 104.71 |
| DFM | 106.03 | 38.93 | 102.89 |
| **CaDDi** | **45.96** | **35.40** | **36.79** |
| **CaDDi-AR** | 67.59 | **35.38** | 44.54 |

**Text8 Dataset — BPD (64-step discretization)**

| Model | BPD↓ | PPL↓ | NLL↓ |
|-------|------|------|------|
| D3PM | ≤1.51 | ≤2.85 | ≤1.05 |
| SEDD | ≤1.46 | ≤2.75 | ≤1.01 |
| MDLM | ≤1.46 | ≤2.75 | ≤1.01 |
| **CaDDi** | **≤1.41** | **≤2.66** | **≤0.98** |

### Ablation Study

**Inference Robustness Test (noise injection)**

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| CaDDi (no noise injection) | Best PPL | Baseline |
| CaDDi (early noise injection) | Small PPL drop | Non-Markovian trajectory provides error-correction capability |
| D3PM/MDLM (equivalent noise) | Large PPL drop | Markovian models suffer severe error accumulation |

**LLM Fine-tuning on Reasoning Tasks (CaDDi-AR based on QWen2-1.5B)**

| Model | ARC-Chal. | BoolQ | LAMBADA |
|-------|-----------|-------|---------|
| QWen2-1.5B | 33.7 | 72.6 | 63.9 |
| CaDDi-AR | **34.2** (+1.9%) | 71.6 | **66.3** (+2.4%) |

### Key Findings

- CaDDi reduces PPL on LM1B by approximately 57% compared to MDLM under low-temperature settings, substantially closing the gap with autoregressive LMs.
- Under low temperature, block-level CaDDi even outperforms token-level CaDDi-AR, as low temperature alleviates the long-tail problem in block generation.
- CaDDi-AR fine-tuned from pretrained LLMs surpasses the base model on reasoning tasks, suggesting that the "review and revise" capability of non-Markovian diffusion is beneficial for reasoning.
- Semi-speculative decoding substantially reduces the inference overhead of CaDDi-AR.

## Highlights & Insights

- **Strong Unification**: The framework unifies autoregressive LMs and discrete diffusion ($T=1$ recovers AR), enabling the broad ecosystem of pretrained LLMs to be directly leveraged for diffusion-based generation.
- **Elegant 2D RoPE Design**: The 2D RoPE encodes the time dimension while maintaining full compatibility with standard RoPE, requiring no architectural modifications.
- **Solid Information-Theoretic Foundation**: Proposition 3.1 establishes the mutual information equivalence between non-Markovian and Markovian diffusion, providing theoretical guidance for noise scheduling.
- **Semi-Speculative Decoding** presents a novel idea—using the diffusion model's own prediction from the previous timestep as a draft, which naturally fits the speculative decoding paradigm.

## Limitations & Future Work

- The full trajectory $\mathbf{x}_{t:T}$ consumes a large portion of the context window; although truncation strategies are employed, the approach remains constrained by context length.
- CaDDi-AR offers higher quality but incurs $O(L \times T)$ inference cost; semi-speculative decoding mitigates but does not fully resolve this issue.
- Experiments are primarily conducted on NLP tasks; other discrete sequence domains such as images and proteins have not been explored.
- A more thorough comparison with the latest variants of Discrete Flow Matching is lacking.

## Related Work & Insights

- CaDDi is complementary to DART (non-Markovian diffusion in continuous space): DART operates in continuous space, while CaDDi operates in discrete space.
- The 2D RoPE design is generalizable to other settings requiring multi-dimensional positional encoding (e.g., video, multimodal).
- The combination of non-Markovian diffusion and pretrained LLMs opens new pathways for controllable text generation (e.g., infilling, conditional generation).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The perspective of unifying AR and diffusion via non-Markovian discrete diffusion is highly novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple dimensions including LM1B, Text8, and reasoning tasks, but lacks validation across broader domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, motivation is thoroughly articulated, and the theory-experiment connection is tight.
- Value: ⭐⭐⭐⭐⭐ The unified framework and reuse of pretrained LLMs carry significant value and may open new research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)

</div>

<!-- RELATED:END -->
