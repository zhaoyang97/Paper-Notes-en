---
title: >-
  [Paper Note] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion
description: >-
  [ACL 2026][LLM/NLP][Glauber Dynamics] This paper constructs discrete text diffusion using Glauber dynamics from statistical physics. By treating a pretrained UL2 model as the "energy function / noise distribution" and us…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Glauber Dynamics"
  - "Discrete Diffusion Language Models"
  - "UL2"
  - "score entropy loss"
  - "Energy Function"
date: 2026-05-08
content_hash: a5a458e7fd260aab
---

# Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion

**Conference**: ACL 2026  
**arXiv**: [2605.04291](https://arxiv.org/abs/2605.04291)  
**Code**: TBD  
**Area**: Energy-based Models / Text Generation / Discrete Diffusion  
**Keywords**: Glauber Dynamics, Discrete Diffusion Language Models, UL2, score entropy loss, Energy Function

## TL;DR
This paper constructs discrete text diffusion using Glauber dynamics from statistical physics. By treating a pretrained UL2 model as the "energy function / noise distribution" and using mask infilling as the Markov transition kernel, the resulting Glauber-UL2 **matches the generation perplexity of GPT-2-M/L equivalent AR models for the first time**. It outperforms MDLM on search-planning tasks like Sudoku/Zebra and surpasses AR in best-of-N performance under equal compute.

## Background & Motivation
**Background**: Autoregressive (AR) LMs dominate text generation but possess structural weaknesses in global planning, complex structural constraints, and self-correction (Bachmann & Nagarajan 2024). Discrete diffusion LMs (D3PM, SEDD, MDLM, GGM, etc.) are promising alternatives but currently suffer from instability, slow training, weak theoretical foundations, or low sampling efficiency.

**Limitations of Prior Work**: Though current popular masked diffusion LMs (MDLM, SEDD-Absorb) offer fast inference, Zheng et al. 2025 rigorously proved they **cannot surpass AR models**—the optimal solution for their loss is effectively equivalent to a time-invariant masked LM, and their perplexity advantage vanishes under 64-bit precision (revealing "temperature hacking" at lower precision). Liu et al. 2025 further proved some problems are inherently non-parallelizable, making MDLM's parallel sampling gains come at the cost of quality.

**Key Challenge**: As a **stochastic process for path-wise relative entropy minimization** (Föllmer 1985, Lehec 2013), a diffusion model's performance relies heavily on two factors: (a) the distance between the noisy distribution and the data distribution, and (b) the curvature / entropy decay rate of the underlying Markov chain. Current discrete diffusion uses uniform, unigram, or absorbing distributions as the noisy distribution, which are too far from the real data distribution, and uses independent token transition kernels with poor curvature.

**Goal**: (1) Bring the noisy distribution as close to the data distribution as possible to reduce required steps; (2) Select a Markov chain with favorable entropy decay properties; (3) Reuse the compute invested in AR pretraining to avoid the sample inefficiency of training diffusion LMs from scratch.

**Key Insight**: The authors noted that:
- The **natural candidate** closer to the data distribution than uniform/unigram is a pretrained LM itself;
- The sampler specifically designed for the "energy function $p(x) \propto e^{f(x)}$" in statistical physics is **Glauber dynamics**, which updates one position $x_k$ at a time given all other positions $x_{\setminus k}$;
- The "conditional sampling" step in Glauber is essentially **mask infilling**;
- The UL2 model weights simultaneously support causal generation (to sample approximate steady states) and mask infilling (to act as the conditional transition kernel), making it a perfect backbone.

**Core Idea**: Treat a pretrained UL2 as the energy function for Glauber dynamics, use causal generation for steady-state initialization, mask infilling as the Markov transition kernel, and fine-tune the entire system as a diffusion transformer using score entropy loss.

## Method

### Overall Architecture
- **State Space**: Token sequences of length $L=1024$, vocabulary $\Sigma$, data distribution $p_D$.
- **Energy Function / Steady State**: Pretrained UL2 provides $p_{\text{base}}(x) \propto e^{f(x)}$, where $f$ is implicitly defined by UL2.
- **Forward Markov Process**: Glauber dynamics with $N$ rounds × $L$ steps = $T = N \cdot L$ total steps. Each round uses a pre-fixed random permutation $\sigma_i$ of 1..L; the $j$-th step of the $i$-th round updates only position $\sigma_i(j)$ by sampling from $p(x_k \mid x_{\setminus k})$ (provided by UL2 mask infilling).
- **Training Loss**: Diffusion Weighted Denoising Score Entropy (DWDSE), learning the probability ratio $s_\theta(x, t)_y \approx p_{t|0}(y|x_0)/p_{t|0}(x_t|x_0)$.
- **Architecture**: UL2 (FLAN-T5 encoder-decoder) + AdaLN-Zero time embeddings + RoPE, with ~15% temporal parameters initialized to zero. One copy is frozen as the transition kernel, while the other is the learnable reverse score model.
- **Inference**: First use UL2 at $t=T$ with causal generation to obtain initial $x$; then perform $N$ rounds in reverse, using mask infilling for each token according to the reversed permutation, totaling $L + N \cdot L = (N+1)L$ model calls.

### Key Designs

1. **Pretrained LM as Energy Function = Shortening noisy ↔ data distance**:
    - **Function**: Uses the implicit distribution $p_{\text{base}} \propto e^{f_{\text{UL2}}(x)}$ of pretrained UL2 as the steady state/noise distribution of the diffusion process, replacing uniform/unigram distributions.
    - **Mechanism**: Since diffusion models are path-wise relative entropy minimizers, starting from an initial value closer to the data significantly shortens the "distance" the reverse process must travel, increasing sample efficiency. AR models learn faster due to $L$ teacher-forcing signals per step compared to diffusion models (1 signal per step); instead of training diffusion from scratch, it is more efficient to reuse established linguistic structures from AR/MLM.
    - **Design Motivation**: Addresses the fundamental problem of slow training and inferior performance in discrete diffusion LMs—previous noisy distributions were too far from the data, which was the root of training inefficiency.

2. **Glauber Dynamics + Mask Infilling Equivalence**:
    - **Function**: Implements the "conditional sampling $p(x_k \mid x_{\setminus k})$" of Glauber dynamics directly via UL2's mask infilling operation.
    - **Mechanism**: Glauber updates $x_k$ based on the conditional distribution given all positions except $k$. With a Metropolis filter, $p(x_k = x \mid x_{\setminus k}) = \min\{1, e^{f(x_{\setminus k}, x)}\}$. This corresponds exactly to "filling in one masked token given context." Utilizing UL2's unified objective (R/S/X-denoising), the same weights can perform both causal generation and mask infilling, avoiding the need for two separate models.
    - **Design Motivation**: Theoretically, Glauber dynamics ensures convergence to a unique steady state given conditional distributions. It also aligns the reverse update positions strictly with the forward noise positions (same round, same permutation), solving the "training-inference misalignment" caused by independent kernels in MDLM.

3. **Synchronized Permutations + UL2 Dual Mode + Score Entropy Training**:
    - **Function**: Uses a frozen UL2 as the transition kernel to add noise and a learnable UL2 as the score model to learn probability ratios, refreshing the frozen copy every epoch to allow the steady state to evolve.
    - **Mechanism**: Unlike standard Glauber which picks indices randomly, this method **predefines permutations $\sigma_1, ..., \sigma_N$**. Reverse updates follow the exact same token indices as the forward process. During training, $t$ is sampled, the frozen UL2 runs $t$ forward steps to produce $x_t$, and the learnable UL2 calculates the SEDD loss $\mathcal{L}_{\text{DWDSE}}$. Since mask infilling directly outputs token probabilities, the score $s_\theta = p_t(y)/p_t(x)$ is calculated by dividing outputs from the frozen and learnable models.
    - **Design Motivation**: Fixed permutations ensure precise coupling of forward/reverse positions (a pain point in MDLM). Refreshing the frozen copy every epoch allows the steady state to progressively converge to the score-entropy-fitted model, acting as a form of progressive self-distillation.

### Loss & Training
- **DWDSE loss**: $\mathcal{L}_{\text{DWDSE}} = \mathbb{E}_{x_0, x_t \sim p_{t|0}}[\int_0^T \sum_{y \sim x_t} Q_t(x_t, y)(s_\theta(x_t, t)_y - p_{t|0}(y|x_0)/p_{t|0}(x_t|x_0) \log s_\theta(x_t, t)_y + K(\cdot))dt]$, where $K(a) = a(\log a - 1)$.
- **Training Data**: OpenWebText.
- **Model Scale**: Glauber-UL2-M (419M, vs GPT-2-M), Glauber-UL2-L (898M, vs GPT-2-L).
- **Compute Cost**: ~6 days on 32 H100s for the large model; iso-TFLOPs equivalent to GGM on 24 H100s for 8 days.
- **Hyperparameters**: $L=1024$, $N \in \{1, 3\}$ (corresponding to $2L$ and $4L$ inference calls).

## Key Experimental Results

### Main Results: Unconditional Generation Perplexity (evaluated by GPT-2-L/XL/NEO, lower is better)

| Model | Parameters | Eval Steps | Gen PPL (GPT2-L) | Gen PPL (GPT2-XL) | Gen PPL (GPT-NEO) |
|------|--------|----------|-------------------|--------------------|--------------------|
| GPT-2-M (AR baseline) | 345M | $L=1024$ | 12.4 | 13.0 | 14.5 |
| GPT-2-L (AR baseline) | 774M | $L$ | 6.5 | — | 7.4 |
| SEDD-M | 424M | $T=2048$ | 27.3 | 28.0 | 25.2 |
| MDLM | 170M | $T=10$ | 4.2* | 45.4 | 40.9 |
| GGM | 387M | $T=4096$ | 19.5 | 19.9 | 18.0 |
| Plaid (continuous) | 1.3B | $T=4096$ | 19.7 | 19.7 | 17.9 |
| **Glauber-UL2-M ($N=1$)** | 419M | $T=2048$ | 17.1 | 17.5 | 16.6 |
| **Glauber-UL2-M ($N=3$)** | 419M | $T=4096$ | **13.2** | 13.7 | 14.9 |
| **Glauber-UL2-L ($N=1$)** | 898M | $T=2048$ | 9.5 | 9.9 | — |
| **Glauber-UL2-L ($N=3$)** | 898M | $T=4096$ | **6.9** | 7.8 | — |

(*The 4.2 PPL for MDLM on GPT-2-L is an artifact of "temperature hacking" at low precision, as noted in the paper.)

**Key Observation**: Glauber-UL2-M ($N=3$) at 13.2 ≈ GPT-2-M at 12.4, **matching AR for the first time with an equivalent scale discrete diffusion model**. Glauber-UL2-L ($N=3$) at 6.9 ≈ GPT-2-L at 6.5 (diff < 0.5 PPL), significantly better than SEDD/GGM/Plaid.

### Ablation Study / Key Findings Table

| Configuration | LAMB | WT2 | WT103 | 1BW | Implications |
|------|------|-----|-------|-----|------|
| GPT-2-M (AR) | 15.60 | 22.76 | 26.37 | 55.72 | AR Baseline |
| UL2-M pre-SEDD (causal only) | 21.7 | — | — | — | Baseline without diffusion training |
| UL2-M post-SEDD CAUSAL-GEN | 19.1 | — | — | — | Causal performance improves after SEDD |
| Glauber-UL2-M ($N=1$) | 17.89 | 23.95 | 30.21 | 56.12 | One-round reverse is already better than baseline |
| Glauber-UL2-M ($N=3$) | **17.14** | **20.98** | **25.47** | **52.18** | Three-round reverse catches up to GPT-2-M |
| Glauber-UL2-L ($N=3$) | **10.14** | 20.35 | 20.83 | **44.12** | Large scale approaches GPT-2-L |

### Iso-Compute Best-of-N (AR samples $2K$ vs Glauber samples $K$)

| Task | AR BoN=2 | AR BoN=4 | Glauber BoN=1 | Glauber BoN=2 |
|------|----------|----------|---------------|---------------|
| GSM8K | 43.9 | 46.1 | 46.9 | **50.4** |
| Winogrande | 68.3 | 69.7 | 69.4 | **71.2** |
| PIQA | 77.6 | 79.9 | 79.1 | **80.8** |
| SIQA | 48.9 | 50.3 | 49.5 | **50.2** |

Glauber BoN=1 (one round iterative refinement) ≈ AR BoN=2 (two independent samples); Glauber BoN=2 outperforms AR BoN=4 consistently.

### Key Findings
- **Rounds $N$ is the critical control knob**: $N=1$ allows UL2 to match GPT-2 on most tasks, while $N=3$ allows it to **catch up or even surpass** AR in PPL—demonstrating that "iterative refinement" is the path to overcoming AR, rather than single-step parallel sampling.
- **Post-SEDD causal generation is stronger than pre-SEDD** (21.7→19.1 PPL): Score entropy training implicitly improves the base UL2 model's language modeling ability as a byproduct.
- **Diffusion > AR under equal compute**: The iso-compute BoN experiment is the most strategic conclusion—since RL/test-time training already treats BoN as standard, Glauber's "self-correction" is more compute-efficient than "independent multiple sampling."
- **Common Sense Reasoning**: Glauber-M ($N=3$) outperforms GPT-2-M and SEDD-M across the board, showing that iterative refinement offers structural advantages for tasks requiring coherence.

## Highlights & Insights
- **Applying statistical physics tools (Glauber + Energy Functions) to LMs** is a genuine cross-disciplinary innovation—score entropy combined with Markov chain curvature analysis provides a **first-principles** explanation for discrete diffusion.
- **UL2 as a backbone is a brilliant choice**: By using causal and masking modes in the same weights, a single pretraining run provides both the steady-state sampler and the Glauber transition kernel.
- **Fixed permutations instead of random indices** solve the training-inference mismatch in MDLM: forcing noise and denoising to occur at the same positions is critical for convergence.
- **The Iso-Compute framework rewrites the Diffusion vs. AR comparison**: Instead of debating "who generates faster in one pass," it focuses on who produces higher quality given a specific budget (RL/BoN)—positioning diffusion LMs clearly for reasoning agent scenarios.
- **Outperforming MDLM+AR on planning tasks (Sudoku/Zebra) without explicit ordering training**: Suggests Glauber's inherent backtracking capability is structurally superior to AR, which must simulate it via long CoT.

## Limitations & Future Work
- **Inference Speed**: $2L$ to $4L$ model calls make it 2-4x slower than AR. This is a trade-off unsuitable for single-turn latency-sensitive applications like chat.
- **Training Cost**: 32 H100s for 6 days for a 0.9B model is high for academic research; scalability to 7B+ remains unproven.
- **UL2 Checkpoint Dependency**: As Google only released the 20B version, the authors had to retrain the mixture-of-denoisers on FLAN-T5, setting a high bar for reproduction.
- **Static $N$ and Permutations**: Adaptive $N$ (stopping when "good enough") is unexplored; permutation strategies are not yet optimized.
- **Lacks Benchmarking against SOTA AR**: Only compared against GPT-2-era models; whether diffusion maintains advantages at the scale of GPT-4 is an open question.
- **Future Work**: (1) Use Flow Matching for more stable training; (2) Use parallel Glauber (Lee 2024) to reduce $4L$ to sub-linear time; (3) Accelerate training with 2nd-order optimizers like Muon; (4) Replace AR+GRPO with Glauber+GRPO during reasoning finetuning.

## Related Work & Insights
- **vs. MDLM/SEDD-Absorb**: These use absorbing/uniform Markov chains where the noisy distribution is far from data and locations are mismatched. Zheng et al. proved their optimal solution degrades to static MLM. Glauber-UL2 fixes this with pretrained LM energy functions and fixed permutations.
- **vs. GGM (Varma et al. 2024)**: Both use Glauber dynamics, but GGM uses unigram noisy distributions and treats training as an $O(L)$ binary classification; Glauber-UL2's use of UL2 makes the noisy distribution significantly closer to data, yielding PPL 13.2 vs GGM's 19.5.
- **vs. Plaid / SSD-LM (Continuous Diffusion)**: Continuous diffusion relies on heavy re-annealing/heuristics to catch AR and is slow; Glauber-UL2 works directly in discrete space with better quality and theoretical grounding.
- **Insight**: "Using pretrained LMs as steady-state distributions" can be generalized to any discrete generation (audio tokens, image tokens, actions) to recycle massive pretraining investments.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefines discrete diffusion LMs through statistical physics, providing a first-principles foundation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Good coverage (PPL, MAUVE, Reasoning, Planning, Iso-compute), though scaling is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical motivation; symbols are dense but the logic flow is very persuasive.
- **Value**: ⭐⭐⭐⭐⭐ First discrete diffusion LM to match GPT-2 scale AR; finds a compute-positive niche in the RL/BoN era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](../../ICLR2026/llm_nlp/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/llm_nlp/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)

</div>

<!-- RELATED:END -->
