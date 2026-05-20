---
title: >-
  [Paper Note] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner
description: >-
  [ICML 2026][LLM Pretraining][Diffusion LM] This work systematically compares continuous diffusion, discrete masked diffusion, and looped transformer from the perspectives of expressiveness and trainability…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Diffusion LM"
  - "Latent Reasoning"
  - "Joint Continuous-Discrete Diffusion"
  - "Looped Transformer"
  - "CFG"
date: 2026-05-08
content_hash: 7746196824f88c11
---

# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

**Conference**: ICML 2026  
**arXiv**: [2510.03206](https://arxiv.org/abs/2510.03206)  
**Code**: https://github.com/zhouc20/CCDD (available)  
**Area**: Diffusion Language Models / Latent Reasoning / Multimodal Diffusion  
**Keywords**: Diffusion LM, Latent Reasoning, Joint Continuous-Discrete Diffusion, Looped Transformer, CFG

## TL;DR
This work systematically compares continuous diffusion, discrete masked diffusion, and looped transformer from the perspectives of expressiveness and trainability, proving that "continuous diffusion" is strictly more expressive than discrete diffusion and can simulate looped transformers, but its practical performance is limited by decoding and representation space. Based on this, it proposes **CCDD (Coevolutionary Continuous Discrete Diffusion)**—diffusing simultaneously in the discrete token space and the contextual embedding space of a pretrained LLM, with a single model jointly denoising. On LM1B/OWT, it reduces perplexity by 25-35% compared to MDLM, and surpasses MDLM's 256-step performance with only 8 sampling steps.

## Background & Motivation
**Background**: The mainstream in language modeling is currently autoregressive LLMs. Non-autoregressive approaches split into two branches: continuous diffusion language models (CDM, SDE/PF-ODE, early but weak) and discrete diffusion language models (DDM, especially masked diffusion like MDLM/SEDD, which recently surpassed CDM). There is also the "latent reasoning" line: looped transformer (LT) and continuous CoT, which theoretically can break the expressiveness ceiling of transformers at $\mathsf{TC^0}$.

**Limitations of Prior Work**: (1) LT is theoretically strong but lacks intermediate supervision, and rollout depth leads to severe OOD issues during inference, making it impractical; (2) CDM is theoretically stronger, but in practice is outperformed by DDM, attributed by the authors to a "triple trainability problem" of excessive decision space, poor embedding space, and complex decoding combinations; (3) Masked DDM is trainable, but quantizes logits to tokens at each step, losing cross-step uncertainty memory and self-correction ability.

**Key Challenge**: The fundamental trade-off between **expressiveness ceiling ↔ practical trainability**. Continuous representations retain complete information for reasoning but are hard to train and decode; discrete representations have clear training objectives but suffer from information bottlenecks.

**Goal**: Without sacrificing either side, construct a model that simultaneously possesses (a) the high expressiveness of continuous CDM (covering LT), (b) the trainability of discrete DDM, (c) semantic priors from pretrained LLM embeddings, and (d) a unified framework for flexible NFE sampling.

**Key Insight**: Redefine "language diffusion" in the **joint multimodal space** $\mathcal{X} \times \mathcal{Z}$—discrete tokens provide an easily decodable "skeleton," while pretrained LLM contextual embeddings provide smooth, information-rich "flesh." Two types of noise are injected in parallel, and a single network denoises both.

**Core Idea**: Use a joint CTMC×SDE process of "discrete token diffusion + continuous contextual embedding diffusion" for language modeling, letting the continuous part handle latent reasoning memory across steps, and the discrete part handle high-confidence decoding.

## Method

### Overall Architecture
CCDD consists of three layers:
1. **Forward Process**: Independent noise is applied to clean data $(x_0, z_0)$—$x_t \sim \text{Cat}(\eta_t x_0 + (1-\eta_t)\pi_t)$ (masked or uniform CTMC) and $z_t \sim \mathcal{N}(\alpha_t z_0, \sigma_t^2 I)$ (VP-SDE);
2. **Reverse Process**: A single network $f_\theta(x_t, z_t, t)$ takes both noisy states as input and outputs token logits and embedding predictions $\hat{x}_{0,\theta}, \hat{z}_{0,\theta}$, but updates each modality independently according to their rules: DDPM/DDIM for $z$, Bayes posterior (8) for $x$;
3. **Embedding Space Selection**: $z$ is not a newly learned embedding, but uses **Qwen3-Embedding-0.6B's last few layers of contextual embeddings** (hidden dim=32, normalized), effectively injecting pretrained LLM semantics into the diffusion process and serving as representation guidance to accelerate convergence.

The final training loss is a weighted sum of continuous and discrete ELBO: $\mathcal{L}_{\text{CCDD}} = \gamma_{\text{cont}} \mathcal{L}_{\text{cont}} + \gamma_{\text{disc}} \mathcal{L}_{\text{disc}}$.

### Key Designs

1. **Joint Continuous-Discrete Diffusion Process (Joint CTMC × SDE)**:

    - **Function**: Enables the model to observe both the "current discrete token state" and the "current continuous semantic state" at each step, retaining the full probabilistic history while benefiting from strong supervision of discrete labels.
    - **Mechanism**: The forward kernel $q_t(x_t,z_t|x_0,z_0) = q_t^{\text{disc}}(x_t|x_0) q_t^{\text{cont}}(z_t|z_0)$ is fully factorized; the reverse kernel $p_\theta(x_s, z_s | x_t, z_t) = p_\theta^{\text{disc}}(x_s|x_t,z_t) p_\theta^{\text{cont}}(z_s|x_t,z_t)$ allows each factor to depend on both inputs (Remark 4.1). The authors prove that this "forward independence + reverse conditional coupling" is asymptotically equivalent in expressiveness to a fully coupled reverse kernel as step size $\to 0$ (Theorem B.19), while greatly simplifying parameterization.
    - **Design Motivation**: The continuous path handles "cross-step memory/planning"—retaining logit geometry instead of quantizing at each step (Lemma B.9 proves DDM's "logits→sample→embed" is a hard information bottleneck); the discrete path handles "high-confidence decoding"—avoiding CDM's combinatorial explosion when decoding tokens from continuous space; forward factorization ensures simple noising, reverse conditional coupling ensures expressiveness.

2. **Pretrained LLM Contextual Embedding as Continuous Space (Contextualized Embedding Space)**:

    - **Function**: Provides a "generatable, decodable, semantically rich" continuous target space, circumventing CDM's three major trainability issues.
    - **Mechanism**: Freezes Qwen3-Embedding's contextual embeddings as the source of $z_0$. Figure 2's key ablation compares the 0-th layer (token-wise, pure lookup embedding) vs. the 28th layer (fully contextualized) as generation targets—the former yields lowest cross-entropy (easy to decode) but highest MSE (hard to generate), the latter the opposite; intermediate layers (e.g., 12th, 20th) strike a balance. The final choice is a contextualized layer as the $z$ target space. Table 1 systematically compares simplex / token-wise $\mathbb{R}^d$ / contextualized $\mathbb{R}^d$ as generation spaces, concluding that contextualized is optimal overall in dimension, smoothness, and decoding ambiguity (though ambiguity is higher, the discrete branch provides fallback).
    - **Design Motivation**: Proposition E.1 shows token-wise embedding with dimension $d \le V$ is no more expressive than the simplex, and the generation target is a discrete codebook set, which is unfriendly to CDM; simplex faces high-dimensional hard constraints. Contextualized embedding offers a smooth generation target and carries pretrained LLM semantic priors, serving as "proxy representation guidance" (akin to REPA, Yu 2024, etc.) to accelerate convergence—experiments show CCDD reaches MDLM's PPL in just 40k steps vs. 1000k, a 25× speedup.

3. **Representation-Guided Classifier-Free Guidance (Representation-CFG) + Multiple Architecture Choices**:

    - **Function**: Treats continuous $z$ as a "self-generated representation condition," allowing CFG to adjust its influence on token generation during inference, achieving a flexible quality-efficiency trade-off.
    - **Mechanism**: During training, with probability $p_{\text{drop}}$, $z_t$ is zeroed out, so the model learns both conditional ($z$ in) and unconditional ($z$ all zero) forward passes; during sampling, $\text{logits} = w \cdot \text{logits}_c + (1-w) \cdot \text{logits}_\phi$, where $w$ is the guidance strength. Three architecture options: (a) **MDiT** adds $x_t, z_t$ embeddings directly into DiT with no extra parameters; (b) **MMDiT** adopts MM-DiT dual-stream cross-attention, doubling parameters but achieving best results; (c) **MoEDiT** uses MoE to route modalities to experts, with minimal parameter increase but high FLOPs utilization.
    - **Design Motivation**: CFG explicitly turns "continuous reasoning" into a controllable guidance signal; multiple architectures provide options for different compute budgets—MDiT achieves "zero extra parameters yet benefits from joint diffusion," MMDiT "parameters for performance," MoEDiT "best cost-performance ratio."

### Loss & Training
The loss is a weighted sum of the two modalities; the architecture is based on a DiT variant from SEDD with rotary embedding. LM1B uses sequence length 128, OWT uses 512, 1M steps with batch size 512 (33B / 131B tokens). Qwen-2 tokenizer and GPT-2 tokenizer are not directly comparable in PPL, so all baselines are retrained with Qwen-2. Hidden dim is 32 (matching Qwen3-Embedding), with $x_0$-prediction parameterization.

## Key Experimental Results

### Main Results
PPL comparison on LM1B and OWT, parameter counts aligned with MDLM 92.1M baseline:

| Dataset | Model | Params | Training tokens | Val PPL ↓ | vs MDLM |
|---------|-------|--------|----------------|-----------|---------|
| LM1B | MDLM (reimpl.) | 92.1M | 33B | ≤39.17 | — |
| LM1B | **CCDD-MDiT w/ Qwen3** | 92.1M | 33B | ≤29.22 | **-25.4%** |
| LM1B | CCDD-MoEDiT w/ Qwen3 | 104M | 33B | ≤28.50 | -27.2% |
| LM1B | CCDD-MMDiT w/ Qwen3 | 216M | 33B | ≤25.76 | -34.2% |
| OWT (Qwen-2) | MDLM (reimpl.) | 92.1M | 131B | ≤33.78 | — |
| OWT (Qwen-2) | CCDD-MoEDiT w/ Qwen3 | 104M | 131B | ≤21.90 | **-35.2%** |
| OWT (GPT-2) | MDLM (reimpl.) | 92.1M | 131B | ≤27.39 | — |
| OWT (GPT-2) | CCDD-MoEDiT w/ RoBERTa | 104M | 131B | ≤24.56 | -10.3% |
| OWT (GPT-2) | GIDD+ (reimpl.) | 92.1M | 131B | ≤25.82 | -5.7% |

On three complex reasoning tasks (Sudoku / 3-SAT / Countdown) with 6M-parameter small models:

| Task | GPT2(6M) | Llama-7B | MDM(20 steps) | LT(2 layers) | LT(3 layers) | **CCDD(2 steps)** | **CCDD(3 steps)** |
|------|----------|----------|--------------|--------------|--------------|-------------------|-------------------|
| Sudoku | 16.2 | 27.1 | 99.9 | 100.0 | 100.0 | **100.0** | **100.0** |
| 3-SAT | 73.1 | — | 87.0 | 91.3 | — | **91.9** | — |
| Countdown | 31.9 | 41.1 | 52.0 | 60.6 | 68.2 | **67.8** | **73.7** |

### Ablation Study

| Configuration | Val PPL / Metric | Notes |
|---------------|------------------|-------|
| Qwen3-Embedding layer 0 (token-wise) | Min token CE, max representation MSE | Easy to decode, hard to generate |
| Qwen3-Embedding layer 28 (contextualized) | Max token CE, min representation MSE | Easy to generate, needs token branch fallback |
| Qwen3-Embedding intermediate layer | Both losses moderate | Balanced, used in final config |
| CCDD w=0 (joint) | Gen NLL 9.06 | Already surpasses MDLM 9.19 |
| CCDD w=1 (discrete-only forward) | Gen NLL 8.38 | CFG significantly improves |
| CCDD w=1.5 | Gen NLL 8.25 | Stronger guidance further improves |
| CCDD 8-step sampling | Outperforms MDLM 256 steps | **16× sampling speedup** |

### Key Findings
- **Disruptive advantage in few-step sampling**: CCDD surpasses MDLM's 256-step performance with just 8 steps—this is a direct benefit of the continuous branch modeling the joint distribution and supporting ODE sampling, while DDM can only use SDE sampling and thus needs more steps.
- **25× training efficiency**: On LM1B, CCDD reaches MDLM's PPL in 40k steps vs. 1000k, with pretrained LLM embeddings providing strong representation regularization.
- **CCDD 2 steps ≈ LT optimal depth on reasoning tasks**: Sudoku/3-SAT are solved by CCDD in 2 steps, and on Countdown, CCDD 3 steps surpasses LT 3-layer's highest score, validating the theoretical hypothesis that the continuous path handles cross-step reasoning.
- **Architecture sensitivity**: MDiT (zero extra parameters) already achieves a 25% PPL reduction, indicating that performance mainly comes from the joint diffusion design rather than parameter scaling; MMDiT/MoEDiT further enhance results.

## Highlights & Insights
- **Unified perspective**: Theoretical results "CDM ⊋ DDM, CDM simulates LT" place the previously independent lines (continuous diffusion / discrete diffusion / looped transformer) on a single expressiveness ladder, clarifying that continuous is the upper bound, with trainability as the main issue.
- **Three-factor decomposition of trainability** (large decision space, poor embedding, complex decoding) is insightful, **directly guiding the use of pretrained LLM contextual embeddings to solve "poor embedding" and the discrete branch to solve "hard decoding"**, forming a rare, clean logical chain.
- **CFG-as-representation-guidance**: Seamlessly combines "continuous representation" and "classifier-free guidance"—randomly zeroed during training, strength-adjusted during inference; this paradigm can transfer to any "primary modality + auxiliary modality conditional generation" task (e.g., code generation + AST, molecule generation + graph).
- **8 steps vs. 256 steps** is more industrially significant than PPL improvement: the main bottleneck for diffusion LM deployment is slow sampling, and CCDD provides a systematic solution—reducing NFE via a more expressive continuous branch, not just new samplers.
- **Tight theory-experiment connection**: Theorem 3.2, Prop 3.4 explain "why this approach," Figure 2 explains "why contextualized layer," Table 6 on reasoning tasks validates "theoretical predictions," forming a rare, self-consistent diffusion LM work.

## Limitations & Future Work
- **Dependence on external pretrained embeddings**: Performance is strongly tied to Qwen3-Embedding quality; using smaller or weaker encoders (RoBERTa) yields only ~10% gain instead of 35%. If the target domain lacks suitable pretrained embedders (low-resource languages, special domains), this approach degrades significantly.
- **Still limited experimental scale**: 92M-216M parameters are much smaller than modern LLMs; pretraining is only on 1B-scale datasets (LM1B/OWT), with no scaling laws or results at 3.2B/7B scale.
- **Joint diffusion overhead on long sequences**: Despite theoretical and empirical efficiency, joint input of two states and CFG require two forward passes, making per-step cost about 2×; no end-to-end wall-clock comparison with AR LLMs under equal FLOPs is provided.
- **Loss of discrete self-correction**: Masked DDM already sacrifices self-correction for trainability, and CCDD also uses masked discrete processes. The authors do not discuss whether uniform DDM combined with the continuous branch could restore self-correction.

## Related Work & Insights
- **vs MDLM / SEDD (masked DDM)**: This work proves such models are strictly less expressive than CDM; adding a continuous branch preserves trainability while breaking the ceiling.
- **vs Continuous DLM (SED, Score Diffusion)**: The authors diagnose that CDM's failure is not "theoretical weakness" but "embedding space weakness," and identify pretrained LLM embeddings as the solution.
- **vs Looped Transformer / Universal Transformer**: CDM can theoretically simulate LT and provides intermediate supervision; the authors suggest using CDM instead of LT for latent reasoning—opening a new direction for diffusion-based latent reasoning.
- **vs DiT / MM-DiT / MoE**: The architecture directly ports DiT systems, making it one of the few works to successfully transfer vision diffusion architectures to language diffusion with significant effect.
- **vs REPA / RCG (representation-guided diffusion)**: Successfully transfers the core idea of "using pretrained encoder representations as diffusion guidance" from vision to language.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Joint CTMC × SDE diffusion is a paradigm-level new structure, unifying multiple independent lines under the expressiveness-trainability framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + three architectures + CFG + comprehensive reasoning tasks, but lacks scaling experiments and wall-clock comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical closure from motivation to theory to method to experiment, with highly self-consistent Figures 1/2/3 and Tables 1/6, excellent readability.
- Value: ⭐⭐⭐⭐⭐ Provides a feasible path for "how diffusion LMs can surpass AR LLMs in reasoning," with few-step sampling of great practical significance, likely to become a standard baseline for future DLM work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)
- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](../../NeurIPS2025/image_generation/itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[CVPR 2026\] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](../../CVPR2026/image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
