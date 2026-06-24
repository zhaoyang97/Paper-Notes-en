---
title: >-
  [Paper Note] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles
description: >-
  [ICCV 2025][Multimodal VLM][MLLM-as-a-Judge] This paper proposes Multimodal Mixture-of-Bayesian Prompt Ensembles (MMB), which learns image-cluster-conditioned prompt weights to substantially improve calibration and judgment accuracy of MLLMs used as evaluators, addressing the failure of standard prompt ensemble methods in multimodal settings.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "MLLM-as-a-Judge"
  - "prompt ensembles"
  - "Bayesian inference"
  - "calibration"
  - "text-to-image generation"
date: 2026-05-08
content_hash: 900dfabcf07743d2
---

# Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles

**Conference**: ICCV 2025
**arXiv**: [2509.08777](https://arxiv.org/abs/2509.08777)  
**Code**: None  
**Area**: Multimodal Evaluation / Image Generation
**Keywords**: MLLM-as-a-Judge, prompt ensembles, Bayesian inference, calibration, text-to-image generation

## TL;DR

This paper proposes Multimodal Mixture-of-Bayesian Prompt Ensembles (MMB), which learns image-cluster-conditioned prompt weights to substantially improve calibration and judgment accuracy of MLLMs used as evaluators, addressing the failure of standard prompt ensemble methods in multimodal settings.

## Background & Motivation

MLLMs are increasingly employed as automated judges for evaluating text-to-image (TTI) generation systems, yet face several critical challenges:

**Bias**: Judge models tend to favor outputs aligned with their own training lineage, reward verbosity, and exhibit sensitivity to minor prompt variations.

**Overconfidence**: Predicted probabilities fail to accurately reflect actual correctness frequencies, leading to high-confidence erroneous judgments.

**Cross-domain inconsistency**: Different image types (photographs vs. abstract art) require distinct evaluation strategies.

**Black-box constraints**: The strongest MLLMs are typically closed-source APIs, precluding fine-tuning or internal inspection.

Existing Bayesian Prompt Ensemble (BPE) methods are effective on purely textual tasks, but **assume all prompts are equally relevant to all samples**—an assumption that breaks down in multimodal settings, where a prompt designed to assess lighting quality may be effective for photographs but irrelevant for digital art.

## Method

### Overall Architecture

The core idea of MMB is that different visual content requires different combinations of evaluation prompts. By clustering image embeddings, samples are partitioned into groups, each of which learns independent prompt weights. During inference, an adaptive prompt combination is achieved through a soft group-assignment mechanism.

### Key Designs

1. **Soft Image Grouping**:

    - Pre-trained CLIP-ViT-B16 is used to extract image embeddings $\phi_I(x)$.
    - Spherical k-means clustering is applied to image embeddings from an unlabeled support set $\mathcal{D}_{sup}$, yielding $K$ groups.
    - Soft assignment probabilities based on cosine similarity are defined as:
    $p(z|x) \propto \exp(\text{sim}(\phi_I(x), g_z) / \tau)$
    - The temperature parameter $\tau$ controls assignment sharpness: $\tau \to 0$ yields hard assignment (independent BPE per group), while $\tau \to \infty$ yields uniform assignment (degenerating to global BPE).

2. **Group-Conditioned Posterior Prompt Weight Learning**:

    - An independent variational distribution $q(a|z)$, parameterized by learnable weights $w_{za}$, is introduced for each group $z$.
    - Optimization maximizes the image-conditioned ELBO:
    $\sum_{j=1}^{M} \sum_z p(z|x_j) \left[ \sum_a w_{za} \log p(y_j^*|x_j, a) - \sum_a w_{za} \log w_{za} \right]$
    - The first term (group-conditioned log-likelihood) rewards prompts that perform well within the group.
    - The second term (group entropy regularization) prevents weight collapse onto a single prompt.

3. **Adaptive Prompt Fusion at Inference**:

    - Given a new sample $x$, soft assignment probabilities $p(z|x)$ over $K$ groups are first computed.
    - Prompt predictions across groups are then combined via weighted summation:
    $p(y|x) \approx \sum_z p(z|x) \sum_a w_{za}^* p(y|x, a_i)$
    - Key insight: the same prompt may carry different importance across different image groups.

### Loss & Training

- Optimization objective: maximize the ELBO in Eq.(9), solvable directly with standard optimizers.
- No modification to the underlying MLLM parameters is required; the method is fully black-box compatible.
- The validation set $\mathcal{D}_{val}$ can be very small (as few as 5–50 samples).
- The support set $\mathcal{D}_{sup}$, used for clustering, has size $256 \times K$.
- Statistical significance is ensured through repeated experiments across multiple random seeds (3 training runs × 50 data samplings × 5 clusterings = 52.2K configurations).

## Key Experimental Results

### Main Results (HPSv2 dataset, 20-prompt setting)

| Method | ECE↓ | MCE↓ | AUC-PR↑ |
|--------|------|------|---------|
| Std. (random single prompt) | 0.263 | 0.422 | 0.716 |
| Best (best single prompt) | 0.153 | 0.342 | 0.812 |
| Avg. (uniform ensemble) | 0.133 | 0.210 | 0.849 |
| BPE (prev. SOTA) | 0.111 | 0.274 | 0.841 |
| **MMB (Ours)** | **0.080** | **0.172** | **0.847** |

Under the 5-sample/20-prompt configuration, MMB reduces ECE by 39.8% relative to the Avg. baseline and by 27.9% relative to BPE.

### Ablation Study (ECE under varying prompt and sample counts)

| # Prompts | # Samples | Avg. ECE | BPE ECE | MMB ECE | MMB Gain vs. Avg. |
|-----------|-----------|----------|---------|---------|-------------------|
| 5 | 5 | 0.155 | 0.127 | **0.113** | −27.1% |
| 10 | 20 | 0.142 | 0.114 | **0.091** | −35.9% |
| 20 | 50 | 0.133 | 0.113 | **0.076** | −42.9% |
| 5 | 50 | 0.155 | 0.121 | **0.107** | −31.0% |

### Key Findings

- **Limitations of standard BPE in multimodal settings**: BPE tends to reduce ECE at the cost of F1 (discriminative power), whereas MMB improves both simultaneously (+1.8% F1 vs. Avg., −36% ECE vs. Avg.).
- **Interaction between prompt count and sample count**: Increasing either the number of prompts or the number of samples improves MMB performance, with orthogonal gains from both factors.
- **MJBench bias experiments**: MMB brings predicted confidence in socially biased scenarios closer to the ideal 50%, suggesting that improved calibration helps mitigate bias.
- **All statistically significant differences are verified via permutation tests at 95% confidence**, with Benjamini-Yekutieli FDR correction applied to control Type-I error inflation.

## Highlights & Insights

- The paper precisely identifies the broken assumption when transferring BPE from text to multimodal settings: prompt relevance is image-conditioned.
- The ELBO derivation is elegant, naturally extending BPE to a group-conditioned form, with the two temperature extremes recovering known methods.
- The approach has direct practical value: low-confidence judgments can be deferred to human review, enabling hybrid evaluation pipelines.
- The experimental design is exceptionally rigorous, with 52.2K configurations covering a full factorial design over prompt counts × sample counts × random seeds.

## Limitations & Future Work

- Clustering quality depends on CLIP embeddings and may degrade for image domains poorly covered by CLIP (e.g., specialized medical imagery).
- The hyperparameters $K$ (number of groups) and $\tau$ (temperature) require tuning, increasing optimization overhead.
- Experiments are conducted solely on GPT-4o; effectiveness on open-source MLLMs (e.g., LLaVA, InternVL) remains unknown.
- The current framework addresses only pairwise preference judgments; more complex evaluation formats (e.g., scoring, ranking) are not covered.
- The support set $\mathcal{D}_{sup}$ must be sampled from the same generator, raising questions about cross-generator generalization.

## Related Work & Insights

- BPE (Tonolini et al.) is the direct predecessor; MMB achieves the critical extension from text to multimodal settings through image conditioning.
- The work is closely related to the selective prediction/deferral literature, with MMB providing a probabilistic foundation for multimodal scenarios.
- Inspiration: other settings employing MLLMs as judges (e.g., video quality assessment, 3D generation evaluation) may similarly benefit from multimodal-aware prompt calibration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multimodal-conditioned prompt ensembles represent a natural yet important generalization, supported by rigorous derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers two benchmarks, multi-dimensional factorial designs, and strict statistical testing with extensive data.
- **Writing Quality**: ⭐⭐⭐⭐ Background and derivations are clear, though the notation density somewhat increases reading load.
- **Value**: ⭐⭐⭐⭐ Directly contributes to the trustworthiness of MLLM evaluation pipelines, though applicability remains relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReBaPL: Repulsive Bayesian Prompt Learning](../../CVPR2026/multimodal_vlm/rebapl_repulsive_bayesian_prompt_learning.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](../../CVPR2025/multimodal_vlm/mllm-as-a-judge_for_image_safety_without_human_labeling.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Information Density Principle for MLLM Benchmarks](information_density_principle_for_mllm_benchmarks.md)

</div>

<!-- RELATED:END -->
