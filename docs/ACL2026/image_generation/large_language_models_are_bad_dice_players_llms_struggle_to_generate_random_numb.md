---
title: >-
  [Paper Note] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions
description: >-
  [ACL 2026][Image Generation][probabilistic sampling] This paper presents the first large-scale systematic audit of the native sampling capability of 11 frontier LLMs across 15 probability distributions, demonstrating that LLMs severely lack intrinsic probabilistic sampling mechanisms and that this deficiency propagates into downstream applications as systematic bias.
tags:
  - ACL 2026
  - Image Generation
  - probabilistic sampling
  - random number generation
  - distributional fidelity
  - intrinsic LLM capability
  - downstream bias
date: 2026-05-08
content_hash: 80afe3882047234e
---

# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions

**Conference**: ACL 2026
**arXiv**: [2601.05414](https://arxiv.org/abs/2601.05414)
**Code**: [GitHub](https://github.com/Mininda/LLM_Bad_Dice_Player)
**Area**: Image Generation
**Keywords**: probabilistic sampling, random number generation, distributional fidelity, intrinsic LLM capability, downstream bias

## TL;DR

This paper presents the first large-scale systematic audit of the native sampling capability of 11 frontier LLMs across 15 probability distributions, demonstrating that LLMs severely lack intrinsic probabilistic sampling mechanisms and that this deficiency propagates into downstream applications as systematic bias.

## Background & Motivation

**Background**: LLMs are evolving from conversational interfaces into core components of complex application pipelines, including synthetic data generation, agent-based simulation, educational material construction, and text-to-image prompt synthesis. These scenarios increasingly require LLMs to faithfully sample from specified probability distributions.

**Limitations of Prior Work**: Existing studies have sporadically identified biases in simple random generation tasks—such as preferences for "lucky numbers" and coin-flip biases—but remain limited to small sample sizes ($N=100$), few distributions (5 types), and a single sampling protocol, making it impossible to comprehensively evaluate the native sampling capability of LLMs. Current practice relies on external libraries (e.g., `numpy.random`) to generate distribution-conforming data, a workaround that implicitly acknowledges the absence of this capability in LLMs.

**Key Challenge**: If LLMs pursuing general intelligence cannot faithfully sample from basic probability distributions, they will introduce uncontrollable systematic bias into downstream applications requiring statistical guarantees. Yet no large-scale, statistically valid benchmark exists to verify this hypothesis.

**Goal**: To conduct the first large-scale probabilistic sampling audit of frontier LLMs. **Key Insight**: A dual-protocol experimental framework (batch vs. independent) is designed to disentangle distinct failure modes. **Core Idea**: LLMs lack a functional internal sampler; batch generation works marginally through context dependence, while independent requests result in near-complete failure.

## Method

### Overall Architecture

The evaluation pipeline covers 11 frontier LLMs × 15 probability distributions (organized into three complexity tiers) × 2 sampling protocols. Distributional fidelity is quantified using three metrics: Wasserstein-1 distance $\mathcal{W}_1$, KL divergence, and statistical tests (KS/χ²). Two downstream application experiments are additionally designed to validate the propagation of sampling deficiencies.

### Key Designs

1. **Dual-Protocol Experimental Design**:

   - **Function**: Disentangles two failure modes in LLM sampling—context dependence and intrinsic prior bias.
   - **Mechanism**: Protocol A (batch generation) generates $N=1000$ samples in a single response, allowing the model to self-correct using prior context; Protocol B (independent requests) issues $N=1000$ stateless calls each generating 1 sample, isolating the model's intrinsic prior.
   - **Design Motivation**: Prior studies used only batch protocols and could not determine whether LLMs possess genuine independent sampling capability.

2. **Theoretical Analysis of the Context-Fidelity Dilemma**:

   - **Function**: Characterizes the non-monotonic relationship between sampling budget $N$ and distributional fidelity.
   - **Mechanism**: Under independent requests, the expected error $\mathcal{E}(N) = \Delta_{\text{ind}} + \mathcal{O}(N^{-1/2})$ converges to an irreducible bias $\Delta_{\text{ind}}$; under batch generation, the error decomposes into a Correction Gain (improvement from in-context self-correction) and Drift (degradation from autoregressive drift), with drift dominating beyond a critical length.
   - **Design Motivation**: Explains why larger $N$ leads to worse distributional fit.

3. **Three-Tier Distribution Complexity Classification**:

   - **Function**: Organizes 15 distributions into three tiers according to entropy characteristics, support constraints, and tail behavior.
   - **Mechanism**: Tier I (basic distributions: Uniform, Gaussian, Bernoulli); Tier II (bounded/discrete: Beta, Binomial, Poisson, Exponential); Tier III (heavy-tailed/multi-parameter: Cauchy, t, Chi-squared, F, Gamma, Weibull, Laplace, Logistic).
   - **Design Motivation**: Systematically assesses whether sampling fidelity degrades monotonically with distributional complexity.

### Loss & Training

This is an evaluation study and involves no model training. Statistical tests serve as binary diagnostics (KS/χ², $\alpha=0.01$); $\mathcal{W}_1$ serves as a continuous fidelity measure; and KL divergence serves as an information loss measure.

## Key Experimental Results

### Main Results

| Protocol | Median Pass Rate | Best Model | Best Pass Rate |
|----------|-----------------|------------|----------------|
| Batch generation | 7% | GPT-4o | 40% |
| Independent requests | 0% | Llama-4-Scout | 7% (Bernoulli only) |

### Ablation Study

| Configuration | Key Metric | Remark |
|---------------|-----------|--------|
| Tier I distributions | Highest Pass Rate | Basic distributions relatively tractable |
| Tier III distributions | Pass Rate ~0% | Complete failure on complex distributions |
| $N=50 \to 2000$ | $\mathcal{W}_1$ increases monotonically | More samples expose larger bias |
| MCQ position bias | All models $p < 0.001$ | GPT-OSS: position C at 54.6%, position A at only 4.5% |
| Attribute-constrained prompts | Gender/ethnicity severely deviate from targets | GPT-4o: Asian 33.5% vs. target 6.5% |

### Key Findings

- A **sharp protocol asymmetry** exists between batch generation and independent requests: effective sampling depends on long-context dependence rather than intrinsic capability.
- Sampling fidelity degrades monotonically with distributional complexity, with $\mathcal{W}_1$ rising from ~0.1 on Tier I to ~1.5 on Tier III.
- Larger sampling budgets $N$ lead to worse distributional fit, violating the standard $\mathcal{O}(N^{-1/2})$ convergence expectation.
- In downstream applications, LLMs fail to comply with explicit distributional constraint instructions; MCQ position bias and demographic attribute bias are both pervasive and severe.

## Highlights & Insights

- The dual-protocol design is the key methodological innovation, refining the sampling problem from "whether approximation is possible" to "whether dependence is contextual or intrinsic."
- The theoretical framework of the Context-Fidelity Dilemma elegantly explains the non-monotonic behavior observed in batch generation.
- The downstream application experiments (MCQ, text-to-image prompts) directly link the abstract sampling problem to practical application risks, strengthening the paper's impact.
- The conclusion is unambiguous: current LLMs lack a functional internal sampler and require external tools to provide statistical guarantees.
- The three-tier distributional complexity taxonomy provides a reusable evaluation framework that can be extended to additional distribution families in future work.

## Limitations & Future Work

- Only default decoding parameters are tested (T=1.0, top-p=1.0); different decoding strategies (e.g., low temperature, top-k) may yield different results.
- It remains unexplored whether augmentation strategies such as chain-of-thought or code generation—common workarounds in practice—can improve sampling capability.
- Only 1D distributions are covered; multivariate joint distributions present greater complexity (preliminary experiments on bivariate Gaussian are reported in the appendix).
- Future work could investigate whether LLMs can acquire intrinsic sampling capability through specialized fine-tuning, RLHF, or dedicated training objectives.
- The scale and scope of the downstream experiments could be further extended to a broader range of real-world application domains.

## Related Work & Insights

- **vs. Gu et al. (2024)**: The previously most comprehensive study covered only 5 distributions × $N=100$ × a single protocol; this paper substantially expands both the scale and methodological depth.
- **vs. Hopkins et al. (2023)**: Identified "lucky number" preferences but was limited to uniform integer distributions; this paper systematically covers 15 continuous and discrete distributions.
- **vs. Xiao et al. (2025)**: Revealed Bernoulli sampling bias; this paper extends the analysis to continuous distributions and adds downstream application validation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First large-scale systematic audit of native probabilistic sampling in LLMs; the dual-protocol design carries methodological value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 models × 15 distributions × 2 protocols × downstream applications; coverage is exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical framework is clear and elegant; data presentation is intuitive; conclusions are convincing.
- **Value**: ⭐⭐⭐⭐ — Exposes a fundamental capability deficiency in LLMs, serving as an important warning for applications that rely on LLM-based sampling.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/image_generation/diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/image_generation/mmada_multimodal_large_diffusion_language_models.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)

<!-- RELATED:END -->
