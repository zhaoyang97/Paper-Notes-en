---
title: >-
  [Paper Note] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty
description: >-
  [ICML 2026][LLM Safety][Free-form Generation UQ] SemGrad introduces gradient-based uncertainty quantification (UQ) to free-form LLM generation scenarios for the first time. By utilizing the Semantic Preserving Score (SPS…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Free-form Generation UQ"
  - "Semantic Gradient"
  - "Semantic Preserving Score"
  - "Single Forward + Backward"
  - "Multiple Valid Answers"
date: 2026-05-08
content_hash: de4e0dfce6af88d8
---

# SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty

**Conference**: ICML 2026  
**arXiv**: [2605.04638](https://arxiv.org/abs/2605.04638)  
**Code**: https://github.com/mingdali6717/SemGrad (Available)  
**Area**: LLM Safety / Uncertainty Quantification / Hallucination Detection  
**Keywords**: Free-form Generation UQ, Semantic Gradient, Semantic Preserving Score, Single Forward + Backward, Multiple Valid Answers

## TL;DR
SemGrad introduces gradient-based uncertainty quantification (UQ) to free-form LLM generation scenarios for the first time. By utilizing the Semantic Preserving Score (SPS) to identify hidden states that encode input semantics, it uses the norm of the log-likelihood gradient with respect to these states as a measure of LLM confidence. Without requiring sampling, it outperforms 11 SOTA baselines across 3 QA datasets via a single backward pass, specifically exceeding SAR by 3.27 AUROC on the multi-answer TruthfulQA dataset.

## Background & Motivation

**Background**: While LLMs are increasingly deployed in medical, educational, and financial sectors, the issue of hallucinations makes quantifying "how confident the model is in its answer" a critical requirement. SOTA UQ methods (e.g., Semantic Entropy, SAR, Semantic Density) typically follow a "sampling + cross-sample semantic clustering" route: generating $K$ outputs for the same query and calculating distribution divergence.

**Limitations of Prior Work**: (i) Sampling methods incur a cost of $K \times$ generation, which is slow and high-variance, leading to high deployment costs; (ii) Established "parameter gradient norm" UQ in classification tasks assumes a single ground truth label, equivalent to a Dirac distribution where the formula $\nabla_\theta \log p(y^\star|x)=0$ holds at the optimum. However, natural language inherently involves aleatoric uncertainty (multiple valid answers), meaning gradients do not vanish even at the optimal $\theta^\star$. Consequently, parameter gradient norms misinterpret "task-inherent randomness" as "model uncertainty."

**Key Challenge**: In free-form generation, aleatoric (task-inherent) and epistemic (lack of knowledge) uncertainties are conflated, and gradients in the parameter space cannot decouple them; meanwhile, sampling methods remain too expensive.

**Goal**: (1) Propose the first gradient-based UQ truly suitable for free-form generation; (2) ensure its effectiveness in multi-answer scenarios; (3) maintain high efficiency through "single forward + single backward" passes.

**Key Insight**: Following linguistic intuition, if a model truly understands a query, then a semantic-preserving perturbation $\boldsymbol{x}+\Delta\boldsymbol{x}$ to the query should not alter the output distribution. This local stability can be quantified by the "gradient norm with respect to semantics-preserving embeddings," which is independent of whether the ground truth distribution is unimodal or multimodal.

**Core Idea**: Shift gradients from "parameter space" to "semantic space"—identifying intermediate hidden states $\boldsymbol{h}_E$ that preserve input semantics and using $\|\nabla_{\boldsymbol{h}_E}\log p(\hat{\boldsymbol{y}}|\boldsymbol{x};\boldsymbol{h}_E)\|$ as the uncertainty metric.

## Method

### Overall Architecture
During inference, a single forward pass obtains the answer $\hat{\boldsymbol{y}}$ and all hidden states. Hidden states of semantic-preserving tokens $t^\star$ in the deeper half of the model (layers $L/2+1$ to $L-1$) are concatenated as $\boldsymbol{h}^\uparrow_{t^\star}$. A single backward pass is performed on the entropy-weighted log-likelihood $\sum_t \omega_t \log p(\hat{y}_t|\hat{y}_{<t},\boldsymbol{x};\boldsymbol{h}^\uparrow_{t^\star})$, using the $\ell_1$ norm divided by dimensionality to obtain SemGrad. Finally, HybridGrad is formed by interpolating the parameter gradient ParaGrad and SemGrad using the average token entropy $\bar\omega$. The entire process requires only one forward and one backward pass without sampling.

### Key Designs

1.  **Semantic Preserving Score (SPS) and Semantic Preserving Token $t^\star$**:
    -   **Function**: Identifies "which token positions / hidden states at which layers best encode input semantics."
    -   **Mechanism**: For each query, GPT generates $K$ semantically equivalent paraphrases. Within-paraphrase similarity $S_{w/i}^{l,t}$ and across-query similarity $S_{a/c}^{l,t}$ are calculated. A high value of $\mathrm{SPS}=S_{w/i}-S_{a/c}$ indicates that the token/layer maps synonymous inputs closer together while pushing non-synonymous inputs apart. Findings show: (i) Each model has a stable $t^\star$ (e.g., `<|start_header_id|>` for LLaMA-3.1) consistent across datasets; (ii) High SPS values are concentrated in the deeper half of the layers; (iii) The high SPS region is a band rather than a single point.
    -   **Design Motivation**: The location of gradient computation directly determines UQ performance; one cannot arbitrarily choose the last layer (primary use is next-token decoding) or lower layers (dominated by lexical features). SPS provides a quantifiable, data-driven selection criterion.

2.  **Entropy-Weighted Semantic Gradient (SemGrad)**:
    -   **Function**: Compresses "how sensitive the output is to semantic perturbations" into a scalar score.
    -   **Mechanism**: Defined as $S_{\text{SemGrad}}=\frac{1}{|\boldsymbol{h}^\uparrow_{t^\star}|}\|\nabla_{\boldsymbol{h}^\uparrow_{t^\star}}\sum_{t=1}^T \omega_t \log p(\hat{y}_t|\hat{y}_{<t},\boldsymbol{x};\boldsymbol{h}^\uparrow_{t^\star})\|_1$, where $\omega_t=H(p(y_t|\hat{y}_{<t},\boldsymbol{x}))$ is the token entropy at the current step, detached from the computational graph. Low-entropy tokens (e.g., stopwords) receive lower weights, while high-entropy tokens (e.g., key factual words) receive higher weights.
    -   **Design Motivation**: Token contributions are uneven; treating all tokens equally dilutes the signal with redundant words. Entropy weights characterize token importance cheaply without third-party models. Theoretically, $\|\nabla_{\boldsymbol{h}_E}\log p\|\approx 0$ holds only when the model matches the true distribution, remaining effective in multi-answer scenarios.

3.  **HybridGrad: Adaptive Fusion of Semantic and Parameter Gradients**:
    -   **Function**: Leverages numerical stability of parameter gradients in single-answer scenarios and the theoretical robustness of SemGrad in multi-answer scenarios.
    -   **Mechanism**: $S_{\text{HybridGrad}}=(1-e^{-\bar\omega})S_{\text{SemGrad}}+e^{-\bar\omega}S_{\text{ParaGrad}}$, where $\bar\omega=\frac{1}{T}\sum_t \omega_t$. Low entropy biases the score toward ParaGrad; high entropy biases it toward SemGrad. ParaGrad is the "parameter version" of SemGrad, replacing $\nabla_{\boldsymbol{h}_E}$ with $\nabla_{\boldsymbol{W}_{\text{head}}}$.
    -   **Design Motivation**: Parameter gradients are most stable under single ground-truth settings but fail under multi-answer settings. Using $\bar\omega$ as a proxy for how aleatoric the input is allows for dynamic switching.

### Loss & Training
The method is inference-only with no training. The only offline step is a single SPS scan on a small development set to determine $t^\star$.

## Key Experimental Results

### Main Results
Evaluated on 3 LLMs across 3 QA datasets (SciQ, TriviaQA for single-answer; TruthfulQA for multi-answer) using BEM for correctness and AUROC for UQ performance:

| Method | SciQ avg | TriviaQ avg | TruthfulQ avg | **Overall avg** |
| :--- | ---: | ---: | ---: | ---: |
| SAR (Prev. SOTA, Sampling) | 74.86 | 84.13 | 66.99 | 75.33 |
| ExGrad (Param Gradient) | 74.33 | 83.37 | 64.06 | 73.92 |
| ParaGrad (Ours baseline) | 75.02 | 84.81 | 66.95 | 75.59 |
| SemGrad | 74.50 | 82.50 | **70.25** | 75.75 |
| **HybridGrad** | **75.35** | 83.90 | **70.53** | **76.59** |

On TruthfulQA, SemGrad outperforms SAR by +3.27, ExGrad by +6.82, and ParaGrad by +3.30 AUROC.

### Ablation Study

| Configuration | TruthfulQA AUROC (LLaMA) | Description |
| :--- | ---: | :--- |
| Full SemGrad (Deep-half + $t^\star$ + $\ell_1$ + entropy weight) | 69.42 | Default |
| $\ell_2$ instead of $\ell_1$ | 69.42 | Almost no difference |
| Remove $\omega_t$ entropy weight | 68.98 | Drop more significant on TriviaQA (3.4 pts) |
| Using only the last layer ($L-1$) | 68.13 | Band > single layer |
| Token changed to last input token | 69.07 | $t^\star$ > last |
| Using low SPS hidden states | Significant drop | Strong positive correlation between SPS and AUROC |

### Key Findings
-   **Positive Correlation between SPS and AUROC**: Hidden states with higher SPS yield higher SemGrad performance. Low SPS states (early layers or misaligned tokens) capture almost no uncertainty. This validates that gradients must be computed in the semantic space.
-   **SemGrad Dominates in Multi-Answer Scenarios**: Parameter gradients fail on TruthfulQA due to aleatoric uncertainty, whereas SemGrad's theoretical independence provides qualitative improvements.
-   **HybridGrad as a Robust All-rounder**: Combining semantic and parameter gradients adaptively results in the highest and most stable average AUROC across 9 (model, dataset) combinations.
-   **Efficiency Advantage**: SemGrad/HybridGrad is an order of magnitude faster than sampling baselines per example.

## Highlights & Insights
-   **First Gradient UQ for Free-form Generation**: Moves beyond "sampling + clustering," proving that gradient-based approaches are even superior in multi-answer scenarios.
-   **SPS as a Tool-grade Byproduct**: Locating "semantic bottleneck tokens" via paraphrase consistency has direct value for mechanistic interpretability and representation engineering.
-   **Entropy-Weighted Token Importance**: Replacing expensive third-party models for token scoring with cheap token-level entropy is a lightweight technique worth generalizing.
-   **Adaptive Fusion Paradigm**: Using $\bar\omega$ as an aleatoric indicator to interpolate between SemGrad and ParaGrad is a versatile approach for scenarios where two estimators excel in different regimes.

## Limitations & Future Work
-   Limited to white-box models (requires gradients and hidden states), making it inapplicable to closed-source APIs.
-   Primarily validated on claim-level short-answer QA; gradient signals in long-form outputs might be diluted by low-information tokens.
-   Current implementation computes gradients for all tokens simultaneously due to framework constraints; optimization for specific positions could improve efficiency.
-   $t^\star$ must be re-determined via SPS scanning for new models; the impact of special tokens in different chat templates is significant.

## Related Work & Insights
-   **vs. Semantic Entropy / SAR / Semantic Density**: Sampling routes capture "distribution divergence" via cross-sample clustering; SemGrad solves this via a single backward pass and naturally handles multi-answer cases.
-   **vs. ExGrad / ParaGrad**: Parameter gradient routes for classification; this work clarifies why they fail on multi-answer tasks (violation of Dirac assumption) and proposes SemGrad as a remedy.
-   **vs. INSIDE / Self-Consistency / P(True)**: Methods based on internal states or self-scoring; SemGrad provides a more principled "semantic stability" metric.
-   **Transferable Insights**: Shifting gradients from parameter space to representation space is useful for LLM diagnostics like OOD detection and prompt sensitivity analysis.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ Advances gradient UQ from the classification era to free-form generation and clarifies the failure modes of previous methods in multi-answer scenarios.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across models, datasets, and baselines; however, lacks long-form or out-of-distribution (OOD) validation.
-   Writing Quality: ⭐⭐⭐⭐ Clear derivations and well-explained motivations; intuitive figures.
-   Value: ⭐⭐⭐⭐⭐ Significant improvement in AUROC on multi-answer QA with lower deployment costs than sampling-based routes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](../../AAAI2026/llm_safety/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](../../ACL2026/llm_safety/agentmark_utility-preserving_behavioral_watermarking_for_agents.md)

</div>

<!-- RELATED:END -->
