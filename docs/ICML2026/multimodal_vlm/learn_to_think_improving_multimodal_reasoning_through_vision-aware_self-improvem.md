---
title: >-
  [Paper Note] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Reasoning] VISTA transforms self-improvement training for Multimodal Large Language Models (MLLMs) into a two-stage pipeline: supplementing samples via prefix resampling for difficu…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Reasoning"
  - "Self-Improvement Training"
  - "Visual Attention"
  - "Prefix Resampling"
  - "DPO"
date: 2026-05-08
content_hash: 49b401d781273cf4
---

# Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training

**Conference**: ICML 2026  
**arXiv**: [2605.11931](https://arxiv.org/abs/2605.11931)  
**Code**: Not mentioned  
**Area**: Multimodal VLM / LLM Reasoning / Self-Improvement  
**Keywords**: Multimodal Reasoning, Self-Improvement Training, Visual Attention, Prefix Resampling, DPO

## TL;DR
VISTA transforms self-improvement training for Multimodal Large Language Models (MLLMs) into a two-stage pipeline: supplementing samples via prefix resampling for difficult problems and filtering false positives via Vision-aware Attention Score (VAS). This achieves a +13.66% average improvement in mathematical and medical multimodal reasoning on Qwen2.5-VL-3B.

## Background & Motivation
**Background**: Current mainstream approaches improve multimodal reasoning by performing post-training on MLLMs with explicit Chain-of-Thought (CoT). Since annotating CoT is expensive, "self-improvement" paradigms like STaR, ReSTEM, and R3V allow models to sample answers and self-train after verification against ground-truth.

**Limitations of Prior Work**: Empirical analysis using Qwen2.5-VL-3B on SLAKE, VQA-Rad, and Geometry3K reveals two overlooked issues. First is **data imbalance**: simple problems easily generate numerous correct solutions, while over 40% of queries in difficult tasks (e.g., Geometry3K) yield zero correct samples in 10 attempts, despite being critical for training. Second is **language prior bias**: even if the final answer is correct, the intermediate reasoning may describe objects not present in the image. Attention distribution shows that although vision tokens occupy the largest proportion of context, their attention scores across layers remain below 20%.

**Key Challenge**: Existing self-improvement methods **only use "answer correctness" as a quality signal**. This signal is insufficient in terms of quantity (too few positive samples for hard problems) and quality (inability to distinguish true image-based reasoning from lucky guesses).

**Goal**: (1) How to supplement correct solutions for difficult problems? (2) How to identify and filter "correct answer but hallucinated reasoning" false positives?

**Key Insight**: Leveraging observations from Ji et al. 2025, errors in failed solutions often occur in the later stages of reasoning while the **prefixes are typically correct**. Simultaneously, the model's own attention distribution can serve as an internal signal for visual focus, requiring no additional models or second forward passes (unlike He et al. 2025, which requires re-running without the image).

**Core Idea**: Use "prefix resampling" to revive high-quality prefixes from failed solutions to augment difficult samples; use "Vision-aware Attention Score (VAS)" to calculate the attention ratio across vision, system, and instruction segments in a single forward pass to filter out false positives with low visual attention.

## Method

### Overall Architecture
VISTA is embedded within a standard three-step iterative loop (sampling → verification → training), primarily modifying the sampling and verification steps. Given the model $\mathcal{M}_{t-1}$ from iteration $t-1$ and a multimodal dataset $\mathcal{D}$, each query $x_i = \{x_i^{\text{sys}}, x_i^{\text{vis}}, x_i^{\text{ins}}\}$ first undergoes standard sampling of $K=10$ solutions. These are partitioned into a positive set $\mathcal{D}_t^p$ and a negative set $\mathcal{D}_t^n$ via ground-truth verification. Subsequently: (1) $\mathcal{D}_t^n$ is expanded into $\mathcal{D}_t^p$ through $J=3$ prefix resampling attempts; (2) VAS is calculated for each solution in $\mathcal{D}_t^p$, discarding those below a threshold $\tau=-0.5$; (3) remaining high-quality positive solutions are used for SFT or DPO+NLL optimization to obtain $\mathcal{M}_t$ over $T=3$ iterations.

### Key Designs

1.  **Prefix Resampling**:
    - **Function**: Localizes "critical tokens" where errors begin in failed solutions and resamples from that point, without relying on ground-truth or external models during localization.
    - **Mechanism**: For each failed solution $r_i^{k_n}$, a paraphrased input "$x_i^{\text{sys}} + x_i^{\text{ins}} + x_i^{\text{vis}} + r_i^{k_n}$" is constructed by swapping image and instruction positions. This is fed into $\mathcal{M}_{t-1}$ to obtain $\text{Top}_5(o_n)$ predictions for each position. The first token not present in the original $\text{Top}_5(o_{n-1})$ is identified as the critical token. It is replaced with a new Top-1 token, the subsequent sequence is truncated, and the model resamples $J$ new solutions using the original query with this prefix.
    - **Design Motivation**: This yields a self-calibration capability to discover "areas of uncertainty." Compared to simply increasing sampling frequency for hard problems (Tong et al. 2024) or using ground-truth guidance (Ding et al. 2025), this method efficiently recycles the "early correct portions" of negative samples.

2.  **Vision-aware Attention Score (VAS)**:
    - **Function**: Uses the model's internal attention maps to quantify whether "the reasoning actually looked at the image," thereby filtering false positives.
    - **Mechanism**: The attention output $\mathbf{A}_i^k$ from the middle layers of $\mathcal{M}_{t-1}$ (identified as most responsible for visual processing) is extracted. The attention sums from output tokens to the system, vision, and instruction segments are computed as $\lambda^k_{\text{sys}}, \lambda^k_{\text{vis}}, \lambda^k_{\text{ins}}$. These are normalized to $S_i^k = \lambda^k_{\text{vis}} / (\lambda^k_{\text{sys}} + \lambda^k_{\text{vis}} + \lambda^k_{\text{ins}})$, and then standardized via z-score within the query to obtain $\text{VAS}_i^k = (S_i^k - \text{mean}(S_i)) / \text{std}(S_i)$. Solutions below threshold $\tau$ are discarded as having insufficient visual focus or potential hallucinations.
    - **Design Motivation**: Unlike He et al. 2025, which requires two forward passes (with and without the image), VAS requires only one forward pass with zero additional overhead. Using z-scores instead of absolute thresholds accounts for varying baseline attention levels across different samples.

3.  **Unified SFT / DPO+NLL Training Interface**:
    - **Function**: Allows the aforementioned data processing to integrate seamlessly into two types of post-training paradigms.
    - **Mechanism**: In SFT, the filtered $\mathcal{D}_t^p$ is used directly for NLL optimization $\mathcal{L}_{\text{SFT}} = -\mathbb{E}[\log \mathcal{M}_\theta(r,\hat y \mid x)/(|r|+|\hat y|)]$. In DPO, each positive instance is paired with a randomly selected negative instance using an augmented loss $\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \mathcal{L}_{\text{NLL}}(r^{k_p}, \hat y^{k_p})$, where $\alpha=0.5, \beta=0.1$.
    - **Design Motivation**: Retaining the NLL term in preference learning prevents DPO training collapse and maintains generation quality. Using the same data processing for both paradigms facilitates fair comparison against baselines like RFT, STaR, and ReSTEM.

### Loss & Training
The process involves $T=3$ iterations. Each iteration uses a sampling size $K=10$, prefix resampling $J=3$, temperature 1.0, and maximum output length 2048. To prevent overfitting, the model is re-fine-tuned from the base model each round. Training is conducted on 8×A800 80GB for 3 epochs using greedy decoding for inference.

## Key Experimental Results

### Main Results

| Model / Method | SLAKE | VQA-Rad | Geo3K | Overall (Δ vs SFT-Seed) |
|---|---|---|---|---|
| Qwen2.5-VL-3B + SFT-Seed | 67.04 | 64.14 | 25.46 | 52.21 |
| Qwen2.5-VL-3B + ReSTEM (iter 3) | 81.69 | 73.71 | 32.28 | 62.56 (+10.35) |
| Qwen2.5-VL-3B + R3V (iter 3) | 81.41 | 69.32 | 32.78 | 61.17 (+8.96) |
| **Qwen2.5-VL-3B + VISTA-SFT (iter 3)** | **84.23** | **76.10** | **37.27** | **65.87 (+13.66)** |
| Qwen2.5-VL-7B + SFT-Seed | 79.15 | 70.52 | 36.94 | 62.20 |
| **Qwen2.5-VL-7B + VISTA-SFT (iter 3)** | **87.89** | **77.29** | **41.43** | **68.87 (+6.67)** |

Consistent improvement across MLLMs: VISTA outperforms baselines like STaR and STaR+ in a single iteration on Qwen3-VL-2B and InternVL3-2B/8B, proving back-bone independence.

### Ablation Study

| Configuration | Overall on 3B | Description |
|---|---|---|
| Full VISTA-SFT (iter 1) | 62.41 | Both prefix resampling and VAS enabled |
| Prefix resampling only | 5x.xx | Addresses data imbalance |
| VAS filtering only | 5x.xx | Addresses hallucinated false positives |
| Shifting VAS threshold $\tau$ | Bell-shaped | High thresholds over-filter samples |

### Key Findings
- **Geo3K Analysis**: The 3B model improved from 25.46 to 37.27 (+11.81 absolute gain) on the difficult Geo3K set, demonstrating that prefix resampling effectively recovers "unsamplable" hard problems.
- **Layer Selection**: VAS analysis (Appendix C.2) shows that filtering is most effective when using middle layers, consistent with Jiang et al. 2025's finding that middle layers are most responsible for visual processing.
- **OOD Generalization**: Improvements on unseen ScienceQA and ChartQA suggest that VISTA learns reliable visual reasoning habits rather than dataset-specific features.

## Highlights & Insights
- "Treating negative samples as resources rather than noise": While traditional self-improvement discards all incorrect solutions, prefix resampling highlights that **failed solution prefixes are often correct and valuable**. This perspective can be migrated to any sample-then-filter training paradigm.
- Using a single-pass internal attention z-score as a hallucination detector is a minimalist but effective "model introspection" method; it requires no external discriminators or token-level alignment data.
- The observation that "Correct Answer $\neq$ Correct Reasoning" is operationalized into a filtering signal via attention scores, which could inspire "process-level" extensions for reward models.

## Limitations & Future Work
- VAS effectiveness relies on the assumption that "internal attention distribution is a reliable indicator of visual focus," which may not hold for models with collapsed attention distributions due to heavy instruction tuning.
- Middle layer selection is empirical (taking one middle layer of the backbone); re-calibration may be needed for different backbones, and it lacks an automated selection mechanism.
- The threshold $\tau$ is globally fixed; adaptive thresholds for different difficulties or tasks could be beneficial.
- Experiments focused on medical and mathematical geometry; transferability to more complex modalities like common-sense images, video, or documents remains to be verified.

## Related Work & Insights
- **vs STaR / ReSTEM**: These discard all failed solutions; VISTA recycles prefixes. They only observe answer correctness; VISTA also monitors visual attention.
- **vs Ding et al. 2025**: That method uses ground-truth to guide reasoning (hint-augmented); VISTA relies entirely on the model's own predictive consistency.
- **vs He et al. 2025**: That method requires two forward passes to quantify language priors; VAS achieves equivalent results with one pass, saving significant computation.
- **vs R3V**: R3V also improves through multiple iterations, but VISTA achieves better results with approximately half the sample volume, suggesting that "sample quality > sample quantity."

## Rating
- Novelty: ⭐⭐⭐⭐ Both technical points are well-targeted adaptations of existing concepts into a cohesive pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 MLLMs, 5 benchmarks, SFT + DPO dual paradigms, and detailed ablation on layer selection.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis (§2.1) is supported by data, methodology is clear, and notations are consistent.
- Value: ⭐⭐⭐⭐ The self-improvement paradigm is currently popular; both the hallucination filtering via attention and prefix recovery tricks are highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](../../ICLR2026/multimodal_vlm/through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICML 2026\] VisionPulse: Dynamic Visual Sparsification in Multimodal Reasoning](visionpulse_dynamic_visual_sparsity_for_efficient_multimodal_reasoning.md)
- [\[ACL 2026\] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models](../../ACL2026/multimodal_vlm/ireasoner_trajectory-aware_intrinsic_reasoning_supervision_for_self-evolving_lar.md)

</div>

<!-- RELATED:END -->
