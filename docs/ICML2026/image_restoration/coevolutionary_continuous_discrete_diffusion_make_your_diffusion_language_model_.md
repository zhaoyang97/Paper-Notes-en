---
title: >-
  [Paper Note] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner
description: >-
  [ICML 2026][Image Restoration][Diffusion LM] This paper systematically compares continuous diffusion, discrete masked diffusion, and looped transformers across the dimensions of expressivity and trainability. It proves t…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Diffusion LM"
  - "Latent Reasoning"
  - "Joint Continuous-Discrete Diffusion"
  - "Looped Transformer"
  - "CFG"
date: 2026-05-08
content_hash: 3f5efd3b7c417bff
---

# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

**Conference**: ICML 2026  
**arXiv**: [2510.03206](https://arxiv.org/abs/2510.03206)  
**Code**: https://github.com/zhouc20/CCDD (Available)  
**Area**: Diffusion Language Models / Latent Reasoning / Multimodal Diffusion  
**Keywords**: Diffusion LM, Latent Reasoning, Joint Continuous-Discrete Diffusion, Looped Transformer, CFG

## TL;DR
This paper systematically compares continuous diffusion, discrete masked diffusion, and looped transformers across the dimensions of expressivity and trainability. It proves that "continuous diffusion" is strictly more expressive than discrete diffusion and can simulate looped transformers, though its practical performance is limited by decoding and representation spaces. Based on this, it proposes **CCDD (Coevolutionary Continuous Discrete Diffusion)**—a model that diffuses simultaneously in the discrete token space and the contextualized embedding space of a pretrained LLM, with joint denoising by a single model. On LM1B/OWT, it reduces perplexity by 25-35% compared to MDLM and outperforms MDLM with 256 steps using only 8 sampling steps.

## Background & Motivation
**Background**: Language modeling is currently dominated by autoregressive LLMs. Non-autoregressive approaches split into two branches: Continuous Diffusion Models (CDM, SDE/PF-ODE, early but weak) and Discrete Diffusion Models (DDM, especially masked diffusion like MDLM/SEDD, recently surpassing CDM). Simultaneously, there is a "latent reasoning" branch: looped transformers (LT) and continuous CoT, which theoretically break the expressivity upper bound of transformers in $\mathsf{TC^0}$.

**Limitations of Prior Work**: (1) LT is theoretically strong but lacks intermediate supervision, and the rollout depth suffers from severe OOD issues during training, making it impractical. (2) CDM can theoretically be stronger, but its practical performance is overtaken by DDM; the authors attribute this to the "triple trainability problem" of large decision spaces, poor embedding spaces, and complex decoding combinations. (3) While masked DDM is trainable, quantizing logits into tokens at each step loses uncertainty memory across steps and sacrifices self-correction capabilities.

**Key Challenge**: The fundamental trade-off between **expressivity upper bounds $\leftrightarrow$ practical trainability**. Continuous representations preserve full information conducive to reasoning but are hard to train and decode; discrete representations have clear training targets but suffer from information bottlenecks.

**Goal**: Construct a unified framework that, without compromising either side, achieves (a) the high expressivity of continuous CDM (covering LT), (b) the good trainability of discrete DDM, (c) the semantic priors of pretrained LLM embeddings, and (d) flexible NFE sampling.

**Key Insight**: Redefine "language diffusion" on a **joint multimodal space** of $\mathcal{X} \times \mathcal{Z}$—where discrete tokens provide an easily decodable "skeleton," and contextualized embeddings from pretrained LLMs provide smooth, information-rich "flesh." Two sets of noise are injected in parallel, and a single network denoises them simultaneously.

**Core Idea**: Utilize a joint CTMC×SDE process of "discrete token diffusion + continuous contextualized embedding diffusion" for language modeling. The continuous part handles latent reasoning memory across steps, while the discrete part manages high-confidence decoding.

## Method

### Overall Architecture
CCDD consists of three layers:
1.  **Forward Process**: Independent noise is applied to clean data $(x_0, z_0)$—$x_t \sim \text{Cat}(\eta_t x_0 + (1-\eta_t)\pi_t)$ (masked or uniform CTMC) and $z_t \sim \mathcal{N}(\alpha_t z_0, \sigma_t^2 I)$ (VP-SDE).
2.  **Backward Process**: A single network $f_\theta(x_t, z_t, t)$ takes both noisy states as input and outputs token logits and embedding predictions $\hat{x}_{0,\theta}, \hat{z}_{0,\theta}$, which are updated independently according to their respective modal rules: DDPM/DDIM for $z$, and the Bayes posterior (8) for $x$.
3.  **Embedding Space Selection**: $z$ is not a newly learned embedding but the **contextualized embeddings from the final layers of Qwen3-Embedding-0.6B** (hidden dim=32 after normalization). This injects the semantic priors of a pretrained LLM into the diffusion process and acts as representation guidance to accelerate training convergence.

The final training loss is a weighted sum of continuous and discrete ELBO: $\mathcal{L}_{\text{CCDD}} = \gamma_{\text{cont}} \mathcal{L}_{\text{cont}} + \gamma_{\text{disc}} \mathcal{L}_{\text{disc}}$.

### Key Designs

1.  **Joint Continuous-Discrete Diffusion Process (Joint CTMC × SDE)**:
    *   **Function**: Allows the model to see both the "current state of discrete tokens" and the "current state of continuous semantics" at each step, preserving probabilistic history throughout while benefiting from strong supervision of discrete labels.
    *   **Mechanism**: The forward kernel $q_t(x_t,z_t|x_0,z_0) = q_t^{\text{disc}}(x_t|x_0) q_t^{\text{cont}}(z_t|z_0)$ is fully separable; the backward kernel $p_\theta(x_s, z_s | x_t, z_t) = p_\theta^{\text{disc}}(x_s|x_t,z_t) p_\theta^{\text{cont}}(z_s|x_t,z_t)$ in factored form **still allows each factor to depend on both inputs simultaneously** (Remark 4.1). The authors prove that this "forward independent + backward conditional coupling" scheme is asymptotically equivalent in expressivity to a fully coupled backward kernel as the step size $\to 0$ (Theorem B.19), yet significantly simplifies parameterization.
    *   **Design Motivation**: The continuous path takes on "cross-step memory/planning"—preserving logit geometry instead of quantizing each step (Lemma B.9 proves that DDM's "logits→sample→embed" is a hard information bottleneck); the discrete path takes on "high-confidence decoding"—avoiding the combinatorial explosion of decoding tokens from continuous space in CDM. Forward factorization ensures simple noising, while backward conditional coupling ensures expressivity.

2.  **Pretrained LLM Contextualized Embeddings as Continuous Space**:
    *   **Function**: Uses a continuous target space that is "easy to generate, decodable, and semantic" to bypass the three trainability issues of CDM.
    *   **Mechanism**: Frozen contextualized embeddings from Qwen3-Embedding are used as the source for $z_0$. In the key ablation of Figure 2, the authors compare the 0-th layer (near token-wise, pure look-up table) vs. the 28-th layer (fully contextualized) as generation targets—the former has the lowest token cross-entropy (easy to decode) but highest MSE (hard to generate), while the latter is the opposite; middle layers (e.g., 12-th, 20-th) strike a balance. Eventually, a contextualized layer is chosen as the target space for $z$. Table 1 also systematically compares simplex / token-wise $\mathbb{R}^d$ / contextualized $\mathbb{R}^d$ spaces, concluding that contextualized is optimal in dimension, smoothness, and decoding ambiguity (though ambiguity is higher, the discrete branch provides a safety net).
    *   **Design Motivation**: The authors prove via Proposition E.1 that token-wise embedding dimensions $d \le V$ do not exceed the expressivity of a simplex, and the generation target is a discrete set of codebooks, which is extremely unfriendly to CDM; simplex faces high-dimensional hard constraints. Contextualized embeddings provide a smooth generation target and carry semantic priors of pretrained LLMs, acting as "proxy representation guidance" (similar to REPA, Yu 2024) to accelerate convergence—experiments show CCDD reaches the PPL of MDLM at 1000k steps in just 40k steps, a 25× training speedup.

3.  **Representation-guided Classifier-Free Guidance (Representation-CFG) + Multi-architecture Choice**:
    *   **Function**: Treats continuous $z$ as a "self-generated representation condition," using CFG during inference to adjust its influence on token generation, realizing a flexible quality-efficiency trade-off.
    *   **Mechanism**: During training, $z_t$ is zeroed out with probability $p_{\text{drop}}$, allowing the model to learn both conditional ($z$ in) and unconditional ($z$ zero) forwards; during sampling, $\text{logits} = w \cdot \text{logits}_c + (1-w) \cdot \text{logits}_\phi$, where $w$ is the guidance strength. Three architecture choices are provided: (a) **MDiT** with zero extra parameters, directly adding $x_t, z_t$ embeddings into the DiT; (b) **MMDiT** drawing from MM-DiT dual-stream cross-attention, doubling parameters for best performance; (c) **MoEDiT** using MoE to route different modalities to experts, with low parameter inflation but high FLOPs efficiency.
    *   **Design Motivation**: CFG explicitly turns "continuous reasoning" into a controllable guidance signal; multi-architecture options allow users with different compute budgets to find a suitable solution—MDiT achieves "learning from joint diffusion with zero extra params," while MMDiT swaps "parameters for performance," and MoEDiT achieves "optimal cost-performance."

### Loss & Training
The loss is a weighted sum of both modalities; the architecture is based on SEDD's DiT modified with rotary embeddings; LM1B sequence length 128, OWT sequence length 512, 1M steps with batch 512 (33B / 131B tokens). Qwen-2 and GPT-2 tokenizers' PPL are not directly comparable, so baselines are retrained using Qwen-2. Hidden dim is set to 32 (consistent with Qwen3-Embedding), using $x_0$-prediction parameterization.

## Key Experimental Results

### Main Results
Comparing PPL on LM1B and OWT, with parameters aligned to the MDLM 92.1M baseline:

| Dataset | Model | Params | Training Tokens | Val PPL ↓ | Gain vs. MDLM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LM1B | MDLM (reimpl.) | 92.1M | 33B | ≤39.17 | — |
| LM1B | **CCDD-MDiT w/ Qwen3** | 92.1M | 33B | ≤29.22 | **-25.4%** |
| LM1B | CCDD-MoEDiT w/ Qwen3 | 104M | 33B | ≤28.50 | -27.2% |
| LM1B | CCDD-MMDiT w/ Qwen3 | 216M | 33B | ≤25.76 | -34.2% |
| OWT (Qwen-2) | MDLM (reimpl.) | 92.1M | 131B | ≤33.78 | — |
| OWT (Qwen-2) | **CCDD-MoEDiT w/ Qwen3** | 104M | 131B | ≤21.90 | **-35.2%** |
| OWT (GPT-2) | MDLM (reimpl.) | 92.1M | 131B | ≤27.39 | — |
| OWT (GPT-2) | CCDD-MoEDiT w/ RoBERTa | 104M | 131B | ≤24.56 | -10.3% |
| OWT (GPT-2) | GIDD+ (reimpl.) | 92.1M | 131B | ≤25.82 | -5.7% |

Comparison of 6M small models on three complex reasoning tasks:

| Task | GPT2(6M) | Llama-7B | MDM(20 steps) | LT(2 layers) | LT(3 layers) | **CCDD(2 steps)** | **CCDD(3 steps)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Sudoku | 16.2 | 27.1 | 99.9 | 100.0 | 100.0 | **100.0** | **100.0** |
| 3-SAT | 73.1 | — | 87.0 | 91.3 | — | **91.9** | — |
| Countdown | 31.9 | 41.1 | 52.0 | 60.6 | 68.2 | **67.8** | **73.7** |

### Ablation Study

| Config | Val PPL / Metric | Description |
| :--- | :--- | :--- |
| Qwen3-Embedding layer 0 (token-wise) | Min token CE, max representation MSE | Easy to decode but hard to generate |
| Qwen3-Embedding layer 28 (contextualized) | Max token CE, min representation MSE | Easy to generate but needs token branch safety net |
| Qwen3-Embedding middle layer | Both losses moderate | Balanced; used as the final configuration |
| CCDD w=0 (joint) | Gen NLL 9.06 | Already surpasses MDLM 9.19 |
| CCDD w=1 (discrete-only forward) | Gen NLL 8.38 | CFG significantly improves performance |
| CCDD w=1.5 | Gen NLL 8.25 | Inference guidance further improves quality |
| CCDD 8 steps sampling | Better than MDLM 256 steps | **16× Sampling Acceleration** |

### Key Findings
-   **Disruptive Advantage in Few-step Sampling**: CCDD surpasses MDLM with 256 steps using only 8 steps—a direct dividend of the continuous part's ability to model joint distributions and support ODE sampling, whereas DDM requires many SDE steps for uniformity.
-   **25× Training Efficiency**: On LM1B, CCDD reaches MDLM’s 1000k-step PPL in just 40k steps, with pretrained LLM embeddings providing significant representation regularization.
-   **CCDD 2 steps ≈ LT optimal depth in inference**: Sudoku/3-SAT are maximized by CCDD in 2 steps; on Countdown, CCDD 3 steps surpasses LT's 3-layer peak score, validating the hypothesis that "continuous paths handle cross-step reasoning."
-   **Architectural Sensitivity**: MDiT (zero extra params) already achieves a 25% PPL reduction, indicating that performance gains stem mainly from the joint diffusion design rather than parameter stacking; MMDiT/MoEDiT provide further improvements.

## Highlights & Insights
-   **Unified Perspective**: The two theoretical conclusions—"CDM ⊋ DDM" and "CDM simulates LT"—place continuous diffusion, discrete diffusion, and looped transformers on a single expressivity hierarchy, providing clear direction: continuous is the upper bound; the issue is trainability.
-   **Three-factor Decomposition of Trainability** (large decision space, poor embeddings, complex decoding) is profound, **directly guiding the use of pretrained LLM contextualized embeddings to solve "poor embeddings" and a discrete branch to solve "decoding difficulty"**—an exceptionally clean logical chain.
-   **CFG-as-representation-guidance**: Merges "continuous representation" and "classifier-free guidance"—random zeroing during training and strengthening during inference. This paradigm can migrate to any "main modality + auxiliary modality conditional generation" task (e.g., Code + AST, Molecule + Graph).
-   **8 steps beats 256 steps** is more industrially significant than PPL gains: the biggest bottleneck for diffusion LM adoption is slow sampling; CCDD provides a systematic breakthrough—reducing NFE via a more expressive continuous branch rather than a new sampler.
-   **Tight Theory-Experiment Coupling**: Theorem 3.2 and Prop 3.4 explain "why this path," Figure 2 explains "why contextualized layers," and Table 6's reasoning tasks show "theoretical predictions validated"—making the paper exceptionally self-consistent.

## Limitations & Future Work
-   **Reliance on External Pretrained Embeddings**: Performance is heavily tied to Qwen3-Embedding quality; using a smaller/weaker encoder (RoBERTa) drops gains from 35% to ~10%. This approach degrades if no suitable pretrained encoder exists (minority languages, niche domains).
-   **Experimental Scale is Still Small**: 92M-216M parameters are much smaller than modern LLMs; pretraining was limited to 1B-level datasets (LM1B/OWT). Scaling laws and performance at 3.2B/7B scales remain unverified.
-   **Overhead of Joint Diffusion on Long Sequences**: While efficient, joint input and CFG require two forwards, making single-step costs approximately 2×. There is no end-to-end wall-clock comparison with AR LLMs at same FLOPs.
-   **Loss of Discrete Self-correction**: Masked DDM sacrifices self-correction for trainability; as CCDD also uses a masked discrete process, it's unclear if a uniform DDM could be used with the continuous branch to regain self-correction.

## Related Work & Insights
-   **vs MDLM / SEDD (masked DDM)**: This paper proves these are strictly weaker than CDM and adds a continuous branch to break their ceiling while keeping their trainability.
-   **vs Continuous DLM (SED, Score Diffusion)**: Diagnoses that CDM failures were due to "poor embedding spaces" rather than "theory failures," pointing to pretrained LLM embeddings as the solution.
-   **vs Looped Transformer / Universal Transformer**: CDM can simulate LT and provides intermediate supervision; authors suggest CCDD as a latent reasoning alternative to LT.
-   **vs DiT / MM-DiT / MoE**: Successfully ports vision diffusion architectures to language diffusion with significant gains.
-   **vs REPA / RCG (Representation-guided Diffusion)**: Successfully migrates the core idea of using pretrained encoder representations as diffusion guidance from vision to language.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Joint CTMC × SDE diffusion is a paradigm-shifting structure, unifying independent paths under an expressivity-trainability framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete horizontal/vertical comparisons across datasets, architectures, CFG, and reasoning tasks, but lacks scaling experiments and wall-clock comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent logical flow from motivation to theory to experiments; Figure 1/2/3 and Table 1/6 are highly self-consistent.
- Value: ⭐⭐⭐⭐⭐ Provides a feasible path for diffusion LMs to exceed AR LLMs in reasoning; the practical impact of few-step sampling is massive. Likely to become a standard baseline for future DLM work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
