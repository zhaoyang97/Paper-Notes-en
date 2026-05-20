---
title: >-
  [Paper Note] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training
description: >-
  [ICML 2026][Multimodal VLM][Multimodal reasoning] VISTA transforms self-improvement training for multimodal large models into a two-stage pipeline: "augmenting hard samples via prefix resampling…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal reasoning"
  - "self-improvement training"
  - "visual attention"
  - "prefix resampling"
  - "DPO"
date: 2026-05-08
content_hash: 4774f4d3d1891c1e
---

# Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training

**Conference**: ICML 2026  
**arXiv**: [2605.11931](https://arxiv.org/abs/2605.11931)  
**Code**: Not mentioned  
**Area**: Multimodal VLM / LLM Reasoning / Self-Improvement  
**Keywords**: Multimodal reasoning, self-improvement training, visual attention, prefix resampling, DPO

## TL;DR
VISTA transforms self-improvement training for multimodal large models into a two-stage pipeline: "augmenting hard samples via prefix resampling, filtering pseudo-positive samples via Visual Attention Score (VAS)." On Qwen2.5-VL-3B, it achieves an average improvement of +13.66% on mathematical/medical multimodal reasoning.

## Background & Motivation
**Background**: The mainstream approach to improving multimodal reasoning in MLLMs is post-training with explicit CoT. However, annotating CoT is expensive, so self-improvement paradigms such as STaR, ReSTEM, and R3V let the model sample its own answers and retrain itself after verification with ground-truth.

**Limitations of Prior Work**: Empirical analysis on Qwen2.5-VL-3B over SLAKE, VQA-Rad, and Geometry3K reveals two overlooked issues. First, **data imbalance**: simple questions easily yield many correct answers, but for hard questions (e.g., Geometry3K), over 40% of queries have zero correct answers in 10 samples, even though these are most critical for training. Second, **language prior bias**: even when the final answer is correct, the model's intermediate reasoning may describe objects not present in the image; attention maps show that although visual tokens dominate the context, their attention scores are below 20% across layers.

**Key Challenge**: Existing self-improvement methods **use only "answer correctness" as the quality signal**, which is insufficient both quantitatively (too few positives for hard questions) and qualitatively (cannot distinguish genuine image-based reasoning from lucky guesses).

**Goal**: (1) How to supplement correct answers for hard questions? (2) How to identify and filter pseudo-positive samples where the answer is correct but the reasoning is hallucinated?

**Key Insight**: Citing Ji et al. 2025, the authors note that errors in failed solutions often occur in the latter part of reasoning—**prefixes are usually correct**. They also leverage the model's own attention distribution as an internal signal of visual focus, requiring neither extra models nor a second forward pass (unlike He et al. 2025, which needs to rerun without the image).

**Core Idea**: Use "prefix resampling" to revive good prefixes from failed solutions to augment hard samples; use "Vision-aware Attention Score (VAS)"—computed in a single forward pass as the proportion of attention to visual/system/instruction segments—to filter pseudo-positives with low visual attention.

## Method

### Overall Architecture
VISTA is embedded in the standard three-step iteration (sampling → verification → training), mainly modifying the sampling and verification steps. Given the $(t-1)$-th model $\mathcal{M}_{t-1}$ and multimodal dataset $\mathcal{D}$, each query $x_i = \{x_i^{\text{sys}}, x_i^{\text{vis}}, x_i^{\text{ins}}\}$ is first sampled for $K=10$ solutions as usual; ground-truth is used to separate positives $\mathcal{D}_t^p$ and negatives $\mathcal{D}_t^n$. Then: (1) For $\mathcal{D}_t^n$, prefix resampling is applied for $J=3$ times to augment $\mathcal{D}_t^p$; (2) For $\mathcal{D}_t^p$, VAS is computed for each solution, and those below threshold $\tau=-0.5$ are discarded; (3) The remaining high-quality positives are used for SFT or DPO+NLL optimization to obtain $\mathcal{M}_t$, iterated for $T=3$ rounds.

### Key Designs

1. **Prefix Resampling**:

    - **Function**: Locates the "critical token" where failed solutions start to go wrong, truncates there, and resamples from that point, without relying on ground-truth or external models.
    - **Mechanism**: For each failed solution $r_i^{k_n}$, swap the image and instruction positions in the query to construct a paraphrased input "$x_i^{\text{sys}} + x_i^{\text{ins}} + x_i^{\text{vis}} + r_i^{k_n}$", feed it into $\mathcal{M}_{t-1}$ to get Top-5 predictions $\text{Top}_5(o_n)$ at each position; the first original token not in $\text{Top}_5(o_{n-1})$ is deemed the critical token. Replace it with the new Top-1 token, truncate the rest, and concatenate this prefix back to the original query to resample $J$ new solutions.
    - **Design Motivation**: This leverages the model's self-calibration to find "uncertain spots." Compared to simply increasing sampling for hard questions (Tong et al. 2024) or using ground-truth guidance (Ding et al. 2025), this method recycles the "correct early part" of negatives, making it more efficient.

2. **Vision-aware Attention Score (VAS)**:

    - **Function**: Quantifies whether a reasoning sequence actually "looks at the image" using the model's own attention maps, thus filtering pseudo-positives.
    - **Mechanism**: Take the attention output $\mathbf{A}_i^k$ from an intermediate layer of $\mathcal{M}_{t-1}$ (intermediate layers are found to be most responsible for visual processing), sum the attention from output tokens to system/visual/instruction input segments to get $\lambda^k_{\text{sys}}, \lambda^k_{\text{vis}}, \lambda^k_{\text{ins}}$, normalize to $S_i^k = \lambda^k_{\text{vis}} / (\lambda^k_{\text{sys}} + \lambda^k_{\text{vis}} + \lambda^k_{\text{ins}})$, then apply z-score normalization within the query to get $\text{VAS}_i^k = (S_i^k - \text{mean}(S_i)) / \text{std}(S_i)$. Solutions below threshold $\tau$ are considered visually inattentive and filtered out.
    - **Design Motivation**: Unlike He et al. 2025, which requires two forward passes (with and without image), VAS needs only one, incurring zero extra cost. Using z-score instead of an absolute threshold adapts to varying overall attention levels across samples.

3. **Unified SFT / DPO+NLL Training Interface**:

    - **Function**: Seamlessly integrates the above data processing into both post-training paradigms.
    - **Mechanism**: For SFT, directly use the filtered $\mathcal{D}_t^p$ for NLL optimization $\mathcal{L}_{\text{SFT}} = -\mathbb{E}[\log \mathcal{M}_\theta(r,\hat y \mid x)/(|r|+|\hat y|)]$; for DPO, pair each positive with a randomly selected negative, using the enhanced loss $\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \mathcal{L}_{\text{NLL}}(r^{k_p}, \hat y^{k_p})$, where $\alpha=0.5, \beta=0.1$.
    - **Design Motivation**: Retaining the NLL term in preference learning prevents DPO collapse and maintains generation quality. The unified data processing supports both paradigms, enabling fair comparison with SFT-Seed, SFT-Oracle, RFT, STaR, ReSTEM, R3V, etc.

### Loss & Training
Iterate for $T=3$ rounds; each round samples $K=10$, prefix resampling $J=3$, temperature 1.0, max output 2048. Each round fine-tunes from the base model to prevent overfitting. Training runs for 3 epochs on 8×A800 80GB; inference uses greedy decoding.

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

Consistent improvements across MLLMs: On Qwen3-VL-2B, InternVL3-2B/8B, a single round of training stably outperforms STaR / STaR+ baselines, demonstrating the method's backbone-agnostic nature.

### Ablation Study

| Configuration | Overall on 3B | Notes |
|---|---|---|
| Full VISTA-SFT (iter 1) | 62.41 | Both prefix resampling and VAS enabled |
| Prefix resampling only | Between SFT-Seed and Full | Addresses data imbalance |
| VAS filtering only | Between SFT-Seed and Full | Addresses hallucinated pseudo-positives |
| Adjusting VAS threshold $\tau$ | Bell-shaped performance | Too high a threshold filters out too many samples |

### Key Findings
- On the hard set Geo3K: 3B model improves from 25.46 to 37.27 (absolute +11.81), showing that prefix resampling truly revives hard samples that otherwise yield no positives.
- VAS layer selection analysis (Appendix C.2) shows filtering is most effective with intermediate layers, consistent with Jiang et al. 2025's finding that intermediate layers are most responsible for visual processing.
- OOD generalization: Gains are also observed on unseen ScienceQA and ChartQA, indicating VISTA learns more reliable visual reasoning habits rather than dataset-specific features.

## Highlights & Insights
- "Treating negatives as resources, not noise": Traditional self-improvement discards all failed solutions, but prefix resampling reveals that **prefixes of failed solutions are often correct and valuable**—this perspective can be transferred to almost any sample-then-filter training paradigm.
- Using internal attention z-score from a single forward pass as a hallucination detector is a minimalist yet effective "model introspection" method; it requires no extra discriminator or token-level alignment data.
- The observation "correct answer ≠ correct reasoning" is operationalized as a filtering signal via attention quantification, which could inspire "process-level" extensions to reward models in the future.

## Limitations & Future Work
- The effectiveness of VAS relies on the assumption that the model's own attention distribution is a reliable indicator of visual focus; this may not hold for models heavily instruction-tuned or with collapsed attention distributions.
- The choice of intermediate layer is empirical (middle layer of the backbone); switching backbones requires recalibration, and there is no automatic layer selection mechanism.
- The threshold $\tau$ is globally fixed; different difficulties or tasks may require adaptive thresholds.
- Experiments are mainly on medical and mathematical geometry; transfer to more complex visual modalities (common-sense images, video, documents) remains to be validated.

## Related Work & Insights
- **vs STaR / ReSTEM**: They discard all failed solutions, VISTA recycles prefixes; they only consider answer correctness, VISTA also considers visual attention.
- **vs Ding et al. 2025 (ground-truth guided reasoning)**: That approach uses answer leakage to guide reasoning, essentially hint-augmented; VISTA relies solely on the model's own prediction consistency.
- **vs He et al. 2025 (rerun without image to quantify language prior)**: That method requires two forward passes; VAS achieves an equivalent signal in a single pass, saving computation.
- **vs R3V**: R3V also improves via multiple iterations, but its sample size is about twice that of VISTA yet achieves worse results, indicating "sample quality > sample quantity."

## Rating
- Novelty: ⭐⭐⭐⭐ Both techniques are not entirely new, but their combination is highly effective and well-targeted
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 MLLMs, 5 benchmarks, both SFT and DPO paradigms, with ablation and layer selection studies
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis (§2.1) includes figures and data, method description is clear, formulae are consistently marked
- Value: ⭐⭐⭐⭐ Self-improvement paradigms are trending; both "attention-based hallucination filtering" and "prefix recycling" tricks are easily reusable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](../../ICLR2026/multimodal_vlm/through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICML 2026\] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)

</div>

<!-- RELATED:END -->
