---
title: >-
  [Paper Note] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty
description: >-
  [ICML 2026][Interpretability][Free-form Generation UQ] SemGrad is the first to bring "gradient-based" uncertainty quantification to LLM free-form generation. It uses the Semantics-Preserving Score (SPS) to identify hidde…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Free-form Generation UQ"
  - "Semantic Gradient"
  - "Semantics-Preserving Score"
  - "Single Forward+Backward"
  - "Multiple Valid Answers"
date: 2026-05-08
content_hash: b295c7d1a1b3b778
---

# SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty

**Conference**: ICML 2026  
**arXiv**: [2605.04638](https://arxiv.org/abs/2605.04638)  
**Code**: https://github.com/mingdali6717/SemGrad (available)  
**Area**: LLM Safety / Uncertainty Quantification / Hallucination Detection  
**Keywords**: Free-form Generation UQ, Semantic Gradient, Semantics-Preserving Score, Single Forward+Backward, Multiple Valid Answers

## TL;DR
SemGrad is the first to bring "gradient-based" uncertainty quantification to LLM free-form generation. It uses the Semantics-Preserving Score (SPS) to identify hidden states encoding input semantics, and treats the norm of the log-likelihood gradient with respect to these states as a measure of LLM confidence. Without sampling and with only a single backward pass, it outperforms 11 SOTA baselines on 3 QA datasets, especially surpassing SAR by 3.27 AUROC on the multi-answer TruthfulQA.

## Background & Motivation

**Background**: LLMs are increasingly deployed in medical, educational, and financial scenarios, but hallucination issues make "how confident is the model in its answer" a critical need. SOTA UQ methods (Semantic Entropy / SAR / Semantic Density, etc.) follow a "sampling + cross-sample semantic clustering" approach: generate $K$ outputs for the same query, then compute distributional divergence.

**Limitations of Prior Work**: (i) Sampling methods incur $K\times$ generation cost, have high variance, are slow, and expensive to deploy; (ii) In classification, mature "parameter gradient norm" UQ assumes a single ground truth label, equivalent to a Dirac distribution, so $\nabla_\theta\log p(y^\star|x)=0$ at optimum. However, natural language inherently has aleatoric uncertainty (multiple valid answers), so even at optimal $\theta^\star$, the parameter gradient norm is nonzero, misinterpreting "task randomness" as "model uncertainty".

**Key Challenge**: In free-form generation, aleatoric (task-intrinsic randomness) and epistemic (model ignorance) uncertainties are entangled, and parameter-space gradients cannot disentangle them; sampling-based methods are too costly.

**Goal**: (1) Propose the first truly suitable gradient-based UQ for free-form generation; (2) Ensure effectiveness in multi-answer scenarios; (3) Maintain high efficiency with "single forward + single backward".

**Key Insight**: Drawing from linguistic intuition—"If the model truly understands the query, then a semantics-preserving perturbation $\boldsymbol{x}+\Delta\boldsymbol{x}$ should not change the output distribution." This local stability can be quantified by the gradient norm with respect to semantics-preserving embeddings, independent of whether the ground truth distribution is unimodal or multimodal.

**Core Idea**: Shift the gradient from "parameter space" to "semantic space"—identify intermediate hidden states $\boldsymbol{h}_E$ that preserve input semantics, and use $\|\nabla_{\boldsymbol{h}_E}\log p(\hat{\boldsymbol{y}}|\boldsymbol{x};\boldsymbol{h}_E)\|$ as the uncertainty metric.

## Method

### Overall Architecture
During inference, a single forward pass yields the answer $\hat{\boldsymbol{y}}$ and all hidden states; select the semantics-preserving token $t^\star$ and concatenate its hidden states from the deeper half of layers ($L/2+1$ to $L-1$) to form $\boldsymbol{h}^\uparrow_{t^\star}$. Backpropagate once on the entropy-weighted log-likelihood $\sum_t\omega_t\log p(\hat{y}_t|\hat{y}_{<t},\boldsymbol{x};\boldsymbol{h}^\uparrow_{t^\star})$, take the $\ell_1$ norm divided by dimension as SemGrad. Then, interpolate with average token entropy $\bar\omega$ to fuse parameter gradient (ParaGrad) and SemGrad into HybridGrad. The entire process requires only one forward and one backward pass, with no sampling.

### Key Designs

1. **Semantics-Preserving Score (SPS) and Semantics-Preserving Token $t^\star$**:

    - **Function**: Identify "which token positions / layer hidden states best encode input semantics".
    - **Mechanism**: For each query, use GPT to generate $K$ semantically equivalent paraphrases; compute within-paraphrase similarity $S_{w/i}^{l,t}$ and across-query similarity $S_{a/c}^{l,t}$; the difference $\mathrm{SPS}=S_{w/i}-S_{a/c}$ indicates that tokens/layers with high SPS pull together synonymous inputs and push apart different ones. Experiments show: (i) Each model has a stable $t^\star$ (LLaMA-3.1: `<|start_header_id|>`, Qwen3: `<|im_start|>`, Mistral-Nemo: last user token), consistent across datasets; (ii) High SPS concentrates in the deeper half of layers, while lower layers mainly capture lexical features; (iii) High SPS forms a band rather than a single point, so the final approach concatenates hidden states from the deeper half.
    - **Design Motivation**: The location of gradient computation directly determines UQ performance; cannot arbitrarily choose the last layer (mainly for next-token decoding, not attended by subsequent tokens) or lower layers (lexical-dominated); SPS provides a quantifiable, data-driven selection criterion.

2. **Entropy-Weighted Semantic Gradient (SemGrad)**:

    - **Function**: Compress "output sensitivity to semantic perturbation" into a scalar score.
    - **Mechanism**: Define $S_{\text{SemGrad}}=\frac{1}{|\boldsymbol{h}^\uparrow_{t^\star}|}\|\nabla_{\boldsymbol{h}^\uparrow_{t^\star}}\sum_{t=1}^T\omega_t\log p(\hat{y}_t|\hat{y}_{<t},\boldsymbol{x};\boldsymbol{h}^\uparrow_{t^\star})\|_1$, where $\omega_t=H(p(y_t|\hat{y}_{<t},\boldsymbol{x}))$ is the token entropy at each step, detached from the computation graph. Low-entropy tokens (stopwords/subwords) get low weight, high-entropy tokens (key factual words) get high weight.
    - **Design Motivation**: In free-form generation, token contributions are uneven; treating all tokens equally dilutes the score with redundant words. Entropy weighting provides a cheap way to capture token importance without third-party models. Theoretically, $\|\nabla_{\boldsymbol{h}_E}\log p\|\approx 0$ only holds when the model's output matches the true distribution, regardless of the ground truth distribution's shape, thus remaining effective in multi-answer scenarios.

3. **HybridGrad: Adaptive Fusion of Semantic and Parameter Gradients**:

    - **Function**: Leverage parameter gradient's numerical stability in single-answer scenarios and SemGrad's theoretical robustness in multi-answer scenarios.
    - **Mechanism**: $S_{\text{HybridGrad}}=(1-e^{-\bar\omega})S_{\text{SemGrad}}+e^{-\bar\omega}S_{\text{ParaGrad}}$, where $\bar\omega=\frac{1}{T}\sum_t\omega_t$ is the average token entropy (approximating sequence-level entropy). Low entropy → favor ParaGrad (task is deterministic, parameter gradient is reliable); high entropy → favor SemGrad (task is multi-solution, semantic gradient is more reliable). ParaGrad is the "parameter version" of SemGrad: replace $\nabla_{\boldsymbol{h}_E}$ with $\nabla_{\boldsymbol{W}_{\text{head}}}$ and apply the same entropy weighting.
    - **Design Motivation**: With a single ground truth, parameter gradient directly aligns with the training objective and is numerically stable; but becomes unstable with multiple answers. Using $\bar\omega$ as a proxy for "how aleatoric is the input", the method dynamically switches, avoiding a hard choice.

### Loss & Training
The method is purely inference-time; the only offline step is running SPS scanning on a small dev set to determine $t^\star$.

## Key Experimental Results

### Main Results
3 LLMs × 3 QA datasets (SciQ, TriviaQA single-answer + TruthfulQA multi-answer), answer correctness evaluated by BEM, UQ performance measured by AUROC:

| Method | SciQ avg | TriviaQ avg | TruthfulQ avg | **Overall avg** |
|--------|---------:|------------:|--------------:|----------------:|
| SAR (Prev. SOTA, sampling) | 74.86 | 84.13 | 66.99 | 75.33 |
| ExGrad (parameter gradient) | 74.33 | 83.37 | 64.06 | 73.92 |
| ParaGrad (Ours, baseline) | 75.02 | 84.81 | 66.95 | 75.59 |
| SemGrad | 74.50 | 82.50 | **70.25** | 75.75 |
| **HybridGrad** | **75.35** | 83.90 | **70.53** | **76.59** |

On the multi-answer TruthfulQA, SemGrad outperforms SAR by +3.27, ExGrad by +6.82, and ParaGrad by +3.30 AUROC.

### Ablation Study

| Configuration | TruthfulQA AUROC (LLaMA) | Notes |
|---------------|-------------------------:|-------|
| Full SemGrad (deep half + $t^\star$ + $\ell_1$ + entropy weight) | 69.42 | Default |
| $\ell_2$ instead of $\ell_1$ | 69.42 | Almost no difference |
| Remove $\omega_t$ entropy weight | 68.98 | TriviaQA drops 3.4 points, more significant |
| Only last layer ($L-1$) | 68.13 | Band > single layer |
| Token replaced by last input token | 69.07 | $t^\star$ > last |
| Use low SPS hidden states | Significant drop | SPS-AUROC strongly correlated |

### Key Findings
- **SPS strongly correlates with AUROC**: Hidden states with high SPS yield better SemGrad performance; low SPS (early layers/misaligned tokens) barely capture uncertainty. This directly validates that "the gradient must be computed in semantic space".
- **SemGrad outperforms parameter gradient in multi-answer scenarios**: On TruthfulQA, parameter gradient fails due to task aleatoric nature, while SemGrad's theoretical independence yields a qualitative improvement.
- **HybridGrad is the most robust all-rounder**: Adaptive fusion of semantic and parameter gradients achieves the highest and most stable average AUROC across 9 (model, dataset) combinations.
- **Significant efficiency advantage**: Table 3 shows SemGrad/HybridGrad are an order of magnitude faster per example than sampling baselines; the paper notes that current implementation computes gradients for all tokens due to PyTorch grad limitations, but there is ample room for optimization.

## Highlights & Insights
- **First truly suitable gradient UQ for LLM free-form generation**: Breaks away from the "sampling + clustering" mainstream, proving that the gradient approach is equally or more effective in multi-answer scenarios, opening a new direction for the UQ community.
- **SPS is a standalone tool**: Using paraphrase consistency difference to locate "semantic encoding tokens" has direct value for mechanistic interpretability, probing, and representation engineering.
- **Entropy-weighted token importance**: Using cheap token-level entropy instead of expensive third-party models for token scoring (as in MARS, SAR's importance score) is a lightweight technique worth promoting.
- **Adaptive fusion paradigm**: Using $\bar\omega$ as an aleatoric indicator to interpolate between SemGrad and ParaGrad is a general idea—any scenario where "two estimators excel in different regimes" can benefit.

## Limitations & Future Work
- Only applicable in white-box settings (requires gradients and hidden states), not usable for closed-source APIs.
- Mainly validated on short-answer, claim-level QA; in long-form outputs, gradient signals may be diluted by many low-information tokens.
- Current implementation computes hidden state gradients for all tokens at once, leading to high memory and time costs due to framework constraints; the authors note that, in theory, only a few positions are needed, so there is significant engineering optimization potential.
- $t^\star$ still requires SPS scanning on new models, with no "zero-shot" automatic determination; special tokens introduced by different chat templates have significant impact.

## Related Work & Insights
- **vs Semantic Entropy / SAR / Semantic Density**: Sampling-based approaches rely on cross-sample clustering to capture "distributional divergence"; SemGrad solves this with a single backward pass and naturally handles multiple answers; on TruthfulQA, prior methods are overwhelmed by aleatoric noise.
- **vs ExGrad / ParaGrad**: Parameter gradient approaches from classification; this work reveals their theoretical failure in multi-answer settings (Dirac assumption breaks), and proposes SemGrad as a remedy.
- **vs INSIDE / Self-Consistency / P(True)**: Internal state or self-labeling methods; SemGrad provides a more principled "semantic stability" metric.
- **Transferable insights**: Shifting "gradient from parameter space to representation space" is useful for many LLM internal diagnostics (OOD detection, confidence calibration, prompt sensitivity analysis); the SPS "paraphrase consistency difference" approach can also locate the model's "semantic bottleneck layers".

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Advances gradient UQ from the classification era to free-form generation, and is the first to clarify the failure reason in multi-answer scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 3 datasets + 11 baselines + 3-dimension ablation + SPS-AUROC correlation curves, comprehensive coverage; lacks long-form and OOD validation.
- Writing Quality: ⭐⭐⭐⭐ Clear formula derivations, well-explained motivation; Figure 1 is intuitive, Figure 3's SPS-AUROC scatter plot is convincing.
- Value: ⭐⭐⭐⭐⭐ +3 AUROC on multi-answer QA with a single backward pass, deployment cost far lower than sampling, highly practical for hallucination detection.

## Related Papers

- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](../../ACL2026/llm_safety/agentmark_utility-preserving_behavioral_watermarking_for_agents.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](../../ACL2026/llm_safety/agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](../../ICML2025/llm_safety/is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](../../NeurIPS2025/interpretability/deep_modularity_networks_with_diversity-preserving_regularization.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)

</div>

<!-- RELATED:END -->
