---
title: >-
  [Paper Note] Esoteric Language Models: A Family of Any-Order Diffusion LLMs
description: >-
  [ICML 2026][Image Generation][Masked Diffusion LM] Eso-LMs deeply integrate AR and Masked Diffusion at three levels: loss, attention, and sampling. By utilizing a denoising Transformer with a causal-on-shuffled-sequence…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Masked Diffusion LM"
  - "Any-Order AR"
  - "KV Cache"
  - "Causal Attention"
  - "Hybrid Training"
date: 2026-05-08
content_hash: 764a6ae99afa00d5
---

# Esoteric Language Models: A Family of Any-Order Diffusion LLMs

**Conference**: ICML 2026  
**arXiv**: [2506.01928](https://arxiv.org/abs/2506.01928)  
**Code**: https://s-sahoo.com/Eso-LMs (Available)  
**Area**: LLM Pre-training / Discrete Diffusion Language Models  
**Keywords**: Masked Diffusion LM, Any-Order AR, KV Cache, Causal Attention, Hybrid Training

## TL;DR
Eso-LMs deeply integrate AR and Masked Diffusion at three levels: loss, attention, and sampling. By utilizing a denoising Transformer with a causal-on-shuffled-sequence architecture, the model simultaneously supports parallel diffusion and left-to-right AR. This enables **the first precise KV cache for MDM during the diffusion phase**, achieving 14–65× speedups over MDLM and 3–4× over BD3-LM on OWT long contexts, while setting a new SOTA on the speed–quality Pareto frontier.

## Background & Motivation
**Background**: Language models are evolving from pure AR toward a dual-path of "AR + Discrete Diffusion." AR models offer the best quality but are restricted to token-by-token decoding. Masked Diffusion LMs (MDM), represented by MDLM, support parallel and controllable generation, nearing LLaMA performance in math/code/science at the 8B scale.

**Limitations of Prior Work**: MDM deployment faces two critical shortcomings. First, inference is slow—despite being "parallel," the denoising Transformer uses **bidirectional attention**, requiring a full Q/K/V recalculation of the entire sequence at every step, which prevents **KV caching** and makes it slower than AR for long sequences. Second, exact likelihood calculation is impossible; the NELBO is only an upper bound, and MDMs lack a usable policy log-prob for RL fine-tuning like GRPO. BD3-LM segments sequences into blocks for inter-block AR and intra-block MDM, but it **only caches between blocks** while requiring full forwards within blocks. Furthermore, small blocks (≤16) suffer from "parallel decoding conflicts," collapsing sample quality at low NFE.

**Key Challenge**: The "causal attention" of AR is the prerequisite for KV caching, whereas the "bidirectional attention" of MDM is the prerequisite for parallel denoising—these architectures are mutually exclusive. Any solution seeking the benefits of both must answer: "What attention mechanism supports both random-order denoising and KV reuse?"

**Goal**: (1) Design a shared denoising Transformer that accommodates both parallel diffusion and left-to-right AR; (2) Enable **precise KV cache** (not an approximation) during the diffusion phase; (3) Provide the first exact likelihood formula for MDM to enable RL-type objectives.

**Key Insight**: The authors leverage the equivalence revealed by Ou et al. (2025): the MDM NELBO is equivalent to the "Any-Order AR" loss $L_\text{AO} = -\mathbb{E}_\sigma \sum_\ell \log p_\theta(x^{\sigma(\ell)} \mid x^{\sigma(<\ell)})$ averaged over all permutations $\sigma$. Since MDM is essentially AO-AR, it can be **trained directly as an AR model**: by shuffling clean tokens to the front and masked tokens to the back of $z_t$, and applying standard causal attention, the model functions as both an MDM and an AR model.

**Core Idea**: A denoising Transformer using "clean-tokens-first + causal attention on shuffled sequence" implements parallel diffusion and AR. By adding an AR loss and a specialized sparse attention mask, the AR phase can reuse the random-order KV cache established during the diffusion phase, forming a two-stage sampler that "paves a layer in parallel with MDM, then fills blanks with AR."

## Method

### Overall Architecture
Eso-LMs decompose the generation process as $p_\theta(x) = \sum_{z_0} p^\text{AR}_\theta(x \mid z_0)\, p^\text{MDM}_\theta(z_0)$. The MDM component first denoises in parallel to produce a **partially masked intermediate sequence** $z_0$ (where a fraction $\alpha_0$ of positions are clean). The AR component then completes the remaining masks in $z_0$ from left to right. $\alpha_0$ is a continuous hyperparameter—$\alpha_0=1$ reduces to pure MDM, while $\alpha_0=0$ reduces to pure AR. The intermediate values provide a smooth interpolation between AR and MDM perplexity. The entire workflow uses **a single shared denoising Transformer** $x_\theta$, with phases distinguished by different attention masks. The variational bound consists of an AR cross-entropy term and an MDM NELBO term. During training, batches are split by a ratio $\kappa$ (default 0.5) between the AR and MDM losses.

### Key Designs

1.  **"Clean-tokens-first + Causal Attention" Denoising Transformer for Diffusion**:
    - **Function**: Transforms the traditional bidirectional denoising Transformer of MDM into a causal version while maintaining "any-position random-order denoising," thereby unlocking **precise KV cache** in the diffusion phase.
    - **Mechanism**: Given $z_t \sim q_t(\cdot \mid x)$, the authors **shuffle clean tokens along with their original positional embeddings to the front of the sequence, with mask tokens placed at the back**, then train the denoising with standard left-to-right causal attention. Thus: (i) clean tokens are causally visible to each other, corresponding exactly to "clean tokens solved in previous steps" during sampling—allowing persistent KV cache reuse; (ii) mask tokens only attend to clean tokens on their left, never seeing masks to be denoised in the future, satisfying causal constraints. During sampling, each forward pass only processes "currently clean tokens + the current mask to be denoised" rather than the full sequence—saving more than just a constant factor for long sequences.
    - **Design Motivation**: The fundamental reason MDM cannot use KV cache is that bidirectional attention makes "predicted tokens" dependent on "future tokens." Removing this edge solves the problem. From an Any-Order AR perspective, random-order MDM is **simply one permutation of AR**. Thus, reordering the sequence into a causal sequence based on generation order allows parallel denoising without abandoning KV reuse, reducing inference complexity from $O(L \cdot L)$ to $O(L)$.

2.  **$z_0 \oplus x$ Concatenation + Sparse Attention Mask for Sequential Phase**:
    - **Function**: Enables the AR phase (filling the remaining masks in $z_0$) to **reuse the random-order KV cache created during the diffusion phase** instead of restarting from scratch.
    - **Mechanism**: During training, the clean+masked $z_0$ and the full $x$ are concatenated into a $z_0 \oplus x$ sequence of length $2L$ and fed into the same Transformer. A $2L \times 2L$ structured sparse attention bias $A$ (dependent on a permutation $\sigma$) is designed: (i) clean tokens precede masks under $\sigma$; (ii) mask tokens maintain natural order; (iii) each mask position $i$ to be predicted via AR can only attend to true tokens $x_{<i}$ on its left. The Transformer output on the $x$ side is discarded, and only the logits of mask positions on the $z_0$ side are used for the AR loss. Since clean tokens are generated and cached in $\sigma$ order during diffusion, the AR phase directly reuses this KV cache to solve masks causally. Implementation via FlexAttention requires less than one screen of code (Fig. 9).
    - **Design Motivation**: Pure AR training requires each predicted token to have a "full clean left context," which $z_0$ (interspersed with masks) lacks. Conventional methods would require a new forward pass without caching. The authors bypass this with concatenation and sparse bias to simulate a "pseudo-left context," forcing the AR to learn conditional prediction based on a non-natural order KV sequence. This is the engineering key to seamless cache transition between phases. The cost is doubling the sequence length, but since only half the batch undergoes AR training, total training is only ~1.37× slower than MDLM.

3.  **First Exact Likelihood for MDM + Single-Forward NELBO**:
    - **Function**: Provides the **first (asymptotically) exact likelihood formula** for MDM (using Eso-LMs with $\alpha_0=1$ as a proxy) and reduces the Monte Carlo estimation of NELBO from $L$ forwards to 1, enabling RL algorithms like GRPO.
    - **Mechanism**: Based on $L_\text{AO}$ equivalence, the authors prove an importance-weighted upper bound (Theorem 3.1): $L^K_\text{AO} = -\mathbb{E}_{\sigma_{1:K}}\left[\log \tfrac{1}{K} + \log\sum_{k=1}^K \exp\sum_\ell \log p_\theta(x^{\sigma_k(\ell)} \mid x^{\sigma_k(<\ell)})\right]$, showing that $-\log p_\theta(x) \le L^K_\text{AO} \le L_\text{MDM}$, where $L^K_\text{AO}$ is monotonically decreasing in $K$ and converges to the true likelihood as $K\to\infty$. Critically, one permutation $\sigma$ characterizes an entire diffusion trajectory of $L$ latents, so $L_\text{AO}$ **requires only one forward pass** in Eso-LMs (impossible for MDLM due to bidirectional attention). Table 2 shows that while MDLM with 10 MC samples has an $L_\text{MDM}$ std of 0.56, Eso-LMs with a single $\sigma$ has an $L_\text{AO}$ std of only 0.03.
    - **Design Motivation**: To perform RL fine-tuning (e.g., GRPO) on MDM, calculating the policy's $\log p$ is essential. Original MDM NELBO estimation required $L$ forward passes per data point, which is infeasible for long sequences, and exact likelihood was non-existent. The causal architecture of Eso-LMs solves both. Subsequent work by Wang et al. (2025b) has already used this estimator for GRPO, outperforming Black et al. (2024) and Zhao et al. (2025) at 0.1B and 8B scales.

### Loss & Training
The total objective is the variational upper bound in Eq. (7): $-\log p_\theta(x) \le \mathbb{E}_{z_0}[\text{AR loss}] + \mathbb{E}_{q_t,t}[\text{MDM loss}]$. Batches are split by $\kappa$: $\kappa=0.5$ for diffusion loss and $1-\kappa$ for AR loss (for $\alpha_0=1$, $\kappa=1$). In the AR loss, a replacement operator $\odot$ substitutes the first $\ell-1$ positions of $z_0$ with true $x_{<\ell}$ to ensure clean left contexts for predicted masks. The noise schedule uses linear $\alpha_t = \alpha_0(1-t)$. When $\alpha_0=1$, the MDM loss coefficient $\alpha'_t/(1-\alpha_t)$ is replaced by $-1$, which empirically reduces variance and speeds up convergence.

## Key Experimental Results

### Main Results
Test perplexity on LM1B ($L=128$, 1M steps) and OWT ($L=1024$, 250K steps) shows smooth AR/MDM interpolation:

| Method | LM1B PPL (NELBO) | LM1B PPL (Exact) | OWT PPL (NELBO) | OWT PPL (Exact) |
|------|------------------|------------------|-----------------|-----------------|
| AR Transformer | – | 21.86 | – | 17.78 |
| MDLM | 31.78 | 26.82 | 25.19 | – |
| BD3-LM ($L'=4$) | 28.23 | – | 20.96 | – |
| **Eso-LM ($\alpha_0=1$)** | 36.12 | 31.65 | 30.06 | 29.31 |
| **Eso-LM ($\alpha_0=0.5$)** | 32.53 | 28.07 | 27.94 | 26.61 |
| **Eso-LM ($\alpha_0=0.125$)** | 26.29 | 23.02 | 21.92 | 20.53 |
| **Eso-LM ($\alpha_0=0)$** | – | 21.86 | – | 17.78 |

Long-context sampling latency (OWT, $T \gg L$, same NFE level as AR):

| Context $L$ | Speedup vs MDLM | Speedup vs BD3-LM ($L'{=}16$) | Speedup vs BD3-LM ($L'{=}4$) |
|--------|--------------|--------------------------|--------------------------|
| 2048 | ~14× | Significant | Significant |
| 8192 | ~65× | ~3.2× | ~3.8× |
| 10240 (tuned) | ~5× at same quality as BD3-LM | – | – |

### Ablation Study
| Configuration | Key Observation | Description |
|------|---------|------|
| Eso-LM ($\alpha_0=1$, full) | LM1B NELBO 36.12 | ~4 points worse than MDLM |
| Eso-LM (A): Only causal attention on masks, clean tokens remain bidirectional | On par with MDLM at $\alpha_0=1$ | Suggests PPL gap stems from "causal among clean tokens"—the price for KV cache. |
| $\kappa$ scan (Table 4) | $\kappa=0.5$ is optimal | Best performance when AR/MDM loss split 50/50. |
| MC NELBO Estimation (Table 2) | $L_\text{AO}$ single sample σ=0.03 vs $L_\text{MDM}$ 10 samples σ=0.56 | Single forward is more precise and efficient. |
| Block sampler vs original ancestral | Significant MAUVE boost at low NFE | Parallelizes only distant masks to avoid local conflicts. |

### Key Findings
- On the speed–quality Pareto frontier (Fig. 4, MAUVE vs sampling time), Eso-LMs dominate both MDLM and BD3-LM across the board. While BD3-LM quality collapses in the low NFE range, Eso-LM remains robust.
- $\alpha_0=1$ training is sufficient: The authors find that a model with $\alpha_0^\text{train}=1$ can cover the entire Pareto frontier by adjusting $\alpha_0^\text{eval}$ during sampling, eliminating the need to train models for every operating point (Remark 2).
- Smaller $\alpha_0$ values yield results closer to AR, and the gap between exact PPL and NELBO PPL narrows—validating the tightness differences between IW bound and NELBO at different interpolation points.

## Highlights & Insights
- While the "Any-Order AR ≡ MDM" equivalence was known, this work is the first to **implement it at the architectural level**. By combining "shuffling + causal attention," MDM is converted into a KV-cacheable AR variant without adding parameters. This is an engineering-heavy but extremely powerful insight.
- The $z_0 \oplus x$ concatenation + sparse mask design—"doubling sequence during training but not during inference"—is ingenious. It resolves the conflict between "AR's need for left context" and "MDM's random-order KV" by offloading the burden to the training phase, allowing inference to reuse the diffusion phase cache.
- Exact likelihood and single-forward NELBO are more than theoretical curiosities: they integrate MDM into the mainstream RL toolchain (like GRPO). Subsequent 8B scale work (Wang et al. 2025b) has already proven this approach superior.
- The remark that "perplexity does not reflect quality at finite NFE" is a critical reflection on the diffusion LM evaluation paradigm. Although $\alpha_0=1$ Eso-LMs have worse PPL than MDLM, they produce higher-quality samples under any fixed time budget.

## Limitations & Future Work
- The authors acknowledge: (i) Training with $\alpha_0<1$ is ~1.37× slower than MDLM due to doubled sequences (though still faster than BD3-LM); (ii) At $\alpha_0=1$, the NELBO is ~4 points worse than MDLM (traced back to "causal clean tokens"); (iii) KV reuse introduces a step of latency, making it slightly slower than AR at the same NFE.
- Additional limitations: (i) Experiments were conducted at academic pretraining scales (LM1B/OWT, ~9K H200 GPU hours) without instruction tuning; scaling relies on citations of 1.7B results from Sahoo et al. 2026; (ii) The sufficiency of $\alpha_0^\text{train}=1$ was only verified on OWT; (iii) Memory friendliness of $2L$ concatenation in the sequential phase remains a concern for long-context fine-tuning stability.
- Future Work: Developing a variant like "Eso-LMs (A)" that restores bidirectional clean token attention might reclaim PPL without losing cache; another direction is applying the IW bound directly to RLHF/RLAIF pipelines for MDM-based DPO/GRPO.

## Related Work & Insights
- **vs MDLM (Sahoo et al., 2024a)**: Both are MDMs, but MDLM uses a bidirectional DiT which prevents caching. Eso-LMs uses causal-on-shuffled-sequence, providing 1–2 orders of magnitude faster inference on long sequences at the cost of slightly higher NELBO at $\alpha_0=1$.
- **vs BD3-LMs (Arriola et al., 2025)**: Both interpolate AR and MDM. BD3-LM interpolates via block size with caching only between blocks, causing quality collapse at small block sizes. Eso-LMs interpolates at the token level via $\alpha_0$, with caching throughout the process and a superior Pareto frontier.
- **vs Pannatier et al. (2024) / Xue et al. (2025)**: These are special cases of Eso-LMs at $\alpha_0=1$. While Xue uses AdaLN for position information, Eso-LMs achieves it entirely through attention masks without extra parameters.
- **vs Concurrent KV Cache Work (Hu 2025, Wu 2025, Ma 2025)**: These provide **approximate** cache (requiring full forwards or refreshes within blocks), whereas Eso-LMs provides **precise** cache.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Successfully grounded Any-Order AR in architecture, providing the first exact likelihood and precise KV cache for diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered LM1B/OWT, long context, ablations, and Pareto frontiers, lacking only large-scale instruction tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear formulas and figures (Fig. 1-3) that make complex designs intuitive.
- Value: ⭐⭐⭐⭐⭐ A critical engineering breakthrough for diffusion LMs—14–65× speedup and single-forward NELBO make GRPO feasible for MDM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[ICML 2026\] A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] CoFrGeNet: Continued Fraction Architectures for Language Generation](cofrgenet_continued_fraction_architectures_for_language_generation.md)
- [\[CVPR 2026\] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching](../../CVPR2026/image_generation/2ndmatch_finetuning_pruned_diffusion_models_via_second-order_jacobian_matching.md)

</div>

<!-- RELATED:END -->
