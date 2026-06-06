---
title: >-
  [Paper Note] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions
description: >-
  [ACL 2026][LLM Evaluation][Probabilistic Sampling] This paper presents the first large-scale systematic audit of the native sampling capabilities of 11 frontier LLMs across 15 probability distributions. It reveals that L…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Probabilistic Sampling"
  - "Random Number Generation"
  - "Distribution Fidelity"
  - "LLM Intrinsic Capabilities"
  - "Downstream Bias"
date: 2026-05-08
content_hash: af7ab534f8d57b1a
---

# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions

**Conference**: ACL 2026  
**arXiv**: [2601.05414](https://arxiv.org/abs/2601.05414)  
**Code**: [GitHub](https://github.com/Mininda/LLM_Bad_Dice_Player)  
**Area**: Image Generation  
**Keywords**: Probabilistic Sampling, Random Number Generation, Distribution Fidelity, LLM Intrinsic Capabilities, Downstream Bias

## TL;DR

This paper presents the first large-scale systematic audit of the native sampling capabilities of 11 frontier LLMs across 15 probability distributions. It reveals that LLMs fundamentally lack intrinsic probabilistic sampling mechanisms, and these deficiencies propagate to downstream applications, causing systematic biases.

## Background & Motivation

**Background**: LLMs are evolving from conversational interfaces into core components of complex application pipelines, including synthetic data generation, agent simulation, educational material construction, and text-to-image prompt synthesis. These scenarios increasingly require LLMs to faithfully sample from specified probability distributions.

**Limitations of Prior Work**: Existing studies have sporadically identified biases in LLMs during simple random generation tasks—such as preferences for "lucky numbers" or coin-flip biases. However, these studies are limited by small sample sizes ($N=100$), few distributions (around 5), and single sampling protocols, failing to comprehensively assess the native sampling capabilities of LLMs. Current practices rely on external libraries (e.g., `numpy.random`) to generate distribution-compliant data, a "workaround" that suggests LLMs lack underlying functional capabilities.

**Key Challenge**: If LLMs aiming for general intelligence cannot faithfully sample from basic probability distributions, they will introduce uncontrollable systematic biases in downstream applications requiring statistical guarantees. However, there is a lack of large-scale, statistically valid benchmarks to verify this hypothesis.

**Goal**: To conduct the first large-scale probabilistic sampling audit of frontier LLMs. **Key Insight**: Design a dual-protocol experimental framework (Batch vs. Independent) to decouple different failure modes. **Core Idea**: LLMs lack functional internal samplers; batch generation barely works due to context dependency, while they fail almost entirely under independent requests.

## Method

### Overall Architecture

The evaluation pipeline covers 11 frontier LLMs × 15 probability distributions (across three complexity tiers) × 2 sampling protocols. Distribution fidelity is quantified using a triple metric approach: Wasserstein-1 distance $\mathcal{W}_1$, KL divergence, and statistical tests (KS/$\chi^2$). Additionally, two downstream application experiments are designed to verify the transmission of sampling defects.

### Key Designs

1.  **Dual-Protocol Experimental Design**:
    *   **Function**: Decouples two failure modes in LLM sampling: context dependency and intrinsic priors.
    *   **Mechanism**: Protocol A (Batch Generation) generates $N=1000$ samples in a single response, allowing the model to utilize historical context for self-correction. Protocol B (Independent Requests) generates 1 sample per call across $N=1000$ stateless calls, isolating the model's intrinsic priors.
    *   **Design Motivation**: Previous studies only used batch protocols, making it impossible to determine whether LLMs possess true independent sampling capabilities.

2.  **Theoretical Analysis of the Context-Fidelity Dilemma**:
    *   **Function**: Characterizes the non-monotonic relationship between sampling budget $N$ and distribution fidelity.
    *   **Mechanism**: Under independent requests, the expected error $\mathcal{E}(N) = \Delta_{\text{ind}} + \mathcal{O}(N^{-1/2})$ converges to an irreducible bias $\Delta_{\text{ind}}$. Under batch generation, the error decomposes into Correction Gain (improvement from context self-correction) and Drift (degradation from autoregressive drift); drift dominates after exceeding a critical length.
    *   **Design Motivation**: To explain why a larger $N$ can lead to poorer distribution fitting.

3.  **Three-Tier Distribution Complexity Classification**:
    *   **Function**: Organizes 15 distributions into three tiers based on entropy characteristics, support constraints, and tail behavior.
    *   **Mechanism**: Tier I (Basic: Uniform, Gaussian, Bernoulli), Tier II (Bounded/Discrete: Beta, Binomial, Poisson, Exponential), Tier III (Heavy-tailed/Multi-parameter: Cauchy, t, Chi-squared, F, Gamma, Weibull, Laplace, Logistic).
    *   **Design Motivation**: To systematically evaluate whether sampling fidelity degrades monotonically with distribution complexity.

### Loss & Training

This work is an evaluation study and does not involve model training. Statistical tests are used as binary diagnostics (KS/$\chi^2$, $\alpha=0.01$), $\mathcal{W}_1$ as a continuous fidelity measure, and KL divergence as an information loss measure.

## Key Experimental Results

### Main Results

| Protocol | Median Pass Rate | Best Model | Best Pass Rate |
| :--- | :--- | :--- | :--- |
| Batch Generation | 7% | GPT-4o | 40% |
| Independent Requests | 0% | Llama-4-Scout | 7% (Passed Bernoulli only) |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Tier I Distributions | Highest Pass Rate | Basic distributions are relatively manageable |
| Tier III Distributions | Pass Rate $\approx 0\%$ | Complex distributions fail completely |
| $N=50 \to 2000$ | $\mathcal{W}_1$ increases monotonically | More samples expose larger biases |
| MCQ Position Bias | $p < 0.001$ for all models | GPT-OSS: 54.6% for Pos C, only 4.5% for Pos A |
| Attribute Constraint Prompts | Gender/Race deviate severely | GPT-4o: 33.5% Asian vs. 6.5% Target |

### Key Findings
*   There is a **sharp protocol asymmetry** between batch generation and independent requests: effective sampling relies on long-context dependencies rather than intrinsic capabilities.
*   Sampling fidelity degrades monotonically with distribution complexity, with $\mathcal{W}_1$ rising from $\approx 0.1$ in Tier I to $\approx 1.5$ in Tier III.
*   A larger sampling budget $N$ results in worse distribution fitting, violating the standard $\mathcal{O}(N^{-1/2})$ convergence expectation.
*   In downstream applications, LLMs fail to adhere to explicit distribution constraint instructions; MCQ position bias and demographic attribute bias are prevalent and severe.

## Highlights & Insights
*   The dual-protocol design is a key methodological innovation, refining the sampling problem from "can it approximate" to "does it rely on context or intrinsic capability."
*   The theoretical framework of the Context-Fidelity Dilemma elegantly explains the non-monotonic behavior in batch generation.
*   Downstream application experiments (MCQ, T2I prompts) directly link abstract sampling issues with practical application risks, enhancing the paper's impact.
*   The conclusion is clear: Current LLMs lack functional internal samplers and require external tools to provide statistical guarantees.
*   The three-tier distribution complexity taxonomy provides a reusable evaluation framework that can be extended to more distribution families in the future.

## Limitations & Future Work
*   Only default decoding parameters were tested ($T=1.0, \text{top-p}=1.0$); different decoding strategies (e.g., low temperature, top-k) might yield different behaviors.
*   The potential for enhancement strategies like Chain-of-Thought or code generation to improve sampling was not explored; these are common workarounds in practice.
*   The study only covers 1D distributions; multivariate joint distributions are more complex (preliminary bivariate Gaussian experiments are reported in the appendix).
*   Future research could investigate whether specialized fine-tuning, RLHF, or specific training objectives could grant LLMs intrinsic sampling capabilities.
*   The scale and scenarios of downstream experiments could be further expanded to more practical application domains.

## Related Work & Insights
*   **vs. Gu et al. (2024)**: Previous comprehensive research only covered 5 distributions × $N=100$ × single protocol; this paper significantly expands the scale and methodological depth.
*   **vs. Hopkins et al. (2023)**: Identified "lucky number" preferences but limited to uniform integer distributions; this paper systematically covers 15 continuous and discrete distributions.
*   **vs. Xiao et al. (2025)**: Revealed Bernoulli sampling bias; this paper extends to continuous distributions and adds downstream application verification.

## Rating
*   Novelty: ⭐⭐⭐⭐ First large-scale systematic audit of LLM native probabilistic sampling; dual-protocol design has methodological value.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models × 15 distributions × 2 protocols × downstream applications; extremely comprehensive coverage.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear and elegant theoretical framework, intuitive data presentation, and persuasive conclusions.
*   Value: ⭐⭐⭐⭐ Highlights fundamental capability flaws in LLMs, serving as an important warning for applications relying on LLM sampling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
