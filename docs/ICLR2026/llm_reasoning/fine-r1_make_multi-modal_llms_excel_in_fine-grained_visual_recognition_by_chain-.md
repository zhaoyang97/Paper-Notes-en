---
title: >-
  [Paper Note] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Fine-grained recognition] Fine-R1 combines CoT supervised fine-tuning (structured reasoning chains following "visual analysis → candidate sub-classes → comparison → prediction") with Triplet-Augmented Policy Optimization (TAPO)—intra-class augmentation for robustness and inter-class augmentation for discriminability—achieving superior performance over CLIP and general/reasoning MLLMs on fine-grained visual recognition using only 4-shot training.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Fine-grained recognition
  - CoT reasoning
  - triplet-augmented policy optimization
  - few-shot FGVR
  - DAPO
date: 2026-05-08
content_hash: de1e510edbb59615
---

# Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning

**Conference**: ICLR 2026
**arXiv**: [2602.07605](https://arxiv.org/abs/2602.07605)
**Code**: [https://github.com/PKU-ICST-MIPL/FineR1_ICLR2026](https://github.com/PKU-ICST-MIPL/FineR1_ICLR2026)
**Area**: LLM Reasoning
**Keywords**: Fine-grained recognition, CoT reasoning, triplet-augmented policy optimization, few-shot FGVR, DAPO

## TL;DR
Fine-R1 combines CoT supervised fine-tuning (structured reasoning chains following "visual analysis → candidate sub-classes → comparison → prediction") with Triplet-Augmented Policy Optimization (TAPO)—intra-class augmentation for robustness and inter-class augmentation for discriminability—achieving superior performance over CLIP and general/reasoning MLLMs on fine-grained visual recognition using only 4-shot training.

## Background & Motivation

**Background**: MLLMs perform well on coarse-grained visual tasks but lag significantly behind contrastive CLIP models on fine-grained visual recognition (FGVR), such as distinguishing different bird species.

**Limitations of Prior Work**:
   - Adapting general-purpose MLLMs to FGVR requires large amounts of labeled data, with high annotation costs (e.g., domain experts labeling thousands of bird subspecies).
   - MLLMs tend to overfit to seen sub-classes and generalize poorly to unseen ones.
   - Even frontier models such as GPT-4V underperform specialized CLIP models on FGVR.

**Key Challenge**: The "high intra-class variance + low inter-class variance" problem inherent to FGVR—the same bird species can look vastly different from different angles, while distinct species may appear nearly identical.

**Key Insight**: MLLMs have already internalized rich fine-grained knowledge; the bottleneck is not a lack of knowledge but an inability to effectively **retrieve** it. CoT reasoning is used to guide knowledge retrieval, while RL optimizes how that knowledge is utilized.

**Core Idea**: Rather than teaching the model to "learn more knowledge," Fine-R1 teaches it to "better use existing knowledge"—by activating the MLLM's inherent fine-grained recognition capability through structured CoT and triplet-contrastive RL.

## Method

### Overall Architecture
Two-stage training: **Stage 1**: CoT SFT (fine-tuning on 404 high-quality CoT samples to establish structured reasoning capability) → **Stage 2**: TAPO (Triplet-Augmented Policy Optimization, reinforcing fine-grained discriminability via intra-class and inter-class contrastive signals).

### Key Designs

1. **Structured CoT Data Construction**:

    - Function: Generate four-step reasoning chains following "visual analysis → candidate sub-classes → comparison → prediction."
    - Mechanism: (1) Image-level visual concept selection—the MLLM describes the same image multiple times; aggregated descriptions are filtered via an information bottleneck to retain the most discriminative features. (2) Structured CoT prompting—the model is guided to first enumerate candidate sub-classes (those most likely to be confused), then systematically compare and eliminate them. Only 404 samples are used, with quality assured through multiple sampling rounds and manual verification.
    - Design Motivation: Generic CoT ("analyze then predict") is insufficient—FGVR specifically requires a reasoning pattern of "first narrow the search space (candidate sub-classes), then perform precise comparison."

2. **Intra-class Augmentation**:

    - Function: Mixes rollout trajectories from different images of the same class to improve robustness against intra-class variance.
    - Mechanism: For each anchor image $x$, a positive example $x_{pos}$ is sampled from the same sub-class. The prior approach generates rollouts for $(x, q)$ and $(x_{pos}, q)$ separately and merges them into a shared reward pool to compute advantages. Policy updates are conditioned on the anchor only.
    - Design Motivation: When two images of the same class yield different predictions, the reward discrepancy provides an informative signal, encouraging the model to focus on class-level rather than image-specific cues.

3. **Inter-class Augmentation**:

    - Function: Maximizes the divergence between output distributions for the anchor and the most similar negative example.
    - Mechanism: A hard negative $x_{neg}$ is sampled from the most similar but distinct sub-class. A discriminative ratio $g^{inter}(\theta) = \pi_\theta(o|q,x_*) / \pi_\theta(o|q,x_{neg})$ is defined, and discriminability is enhanced by maximizing the KL divergence $D_{KL}[\pi_\theta \| \pi_\theta^{neg}]$. Dual entropy regularization stabilizes training.
    - Design Motivation: If the model's prediction remains unchanged when the image is replaced by a visually similar but different class, it indicates the model is not exploiting fine-grained discriminative cues—this behavior is penalized.

### Loss & Training
- **Stage 1**: Standard SFT on 404 CoT samples.
- **Stage 2**: TAPO = DAPO base + Intra-class Augmentation (mixing positive rollouts) + Inter-class Augmentation (maximizing KL divergence from negatives) + dual entropy regularization.
- 4-shot per category setting (only 4 training samples per class).

## Key Experimental Results

### Main Results (6 FGVR Datasets, Closed-world)

| Method | Seen Avg↑ | Unseen Avg↑ | Overall Avg↑ |
|--------|-----------|-------------|--------------|
| SigLIP-L (CLIP) | 88.33 | 80.54 | 84.44 |
| Qwen2.5-VL-7B | ~84% | ~57% | ~70% |
| DeepPerception-7B | ~87% | ~50% | ~68% |
| **Fine-R1-3B** | **~93%** | **~81%** | **~87%** |

### Ablation Study

| Configuration | Seen↑ | Unseen↑ |
|---------------|-------|---------|
| SFT only | baseline | baseline |
| + Standard RL (CLS-RL) | +5% | -2% (overfitting) |
| + TAPO (full) | **+8%** | **+13%** |
| — w/o Intra-class Aug | -3% | -5% |
| — w/o Inter-class Aug | -2% | -4% |

### Key Findings
- **Surpasses specialized CLIP models**: Fine-R1-3B outperforms SigLIP-L by approximately 3% on average across 6 datasets—the first generative MLLM to exceed contrastive models on FGVR.
- **Strong open-world generalization**: Outperforms Qwen2.5-VL-7B by +23.75% on unseen categories, demonstrating that the model learns a reasoning methodology rather than memorizing class labels.
- **4-shot sufficiency**: Only 4 samples per category are sufficient to elicit strong fine-grained recognition capability.
- **Knowledge and visual features remain unchanged**: Internal representations before and after training are nearly identical—improvements stem from "better use of knowledge" rather than "acquisition of new knowledge."
- **Strong cross-domain transfer**: Achieves +3.6% improvement on QA tasks requiring object recognition, such as ImageWikiQA.

## Highlights & Insights
- The finding that **"it is not about learning more knowledge, but about using existing knowledge better"** is particularly insightful—the FGVR bottleneck in MLLMs lies not in perception or knowledge, but in knowledge retrieval. Structured CoT essentially functions as a "knowledge retrieval strategy," guiding the model to first narrow the search space before performing precise comparison.
- **Triplet-contrastive RL is a natural fit for FGVR**—it incorporates the intuition of metric learning (triplet loss) into policy optimization, where intra-class augmentation aligns positives and inter-class augmentation repels negatives. This is better suited to the structural properties of FGVR than general-purpose GRPO.
- **Efficient training with only 404 CoT samples** is impressive—through quality control (multiple sampling rounds and manual verification), a small amount of high-quality data outperforms large amounts of low-quality data.

## Limitations & Future Work
- Only 3B/7B models are evaluated; effectiveness on larger MLLMs remains to be verified.
- Negative example selection relies on predefined "most similar sub-classes"—more dynamic online hard negative mining may be more effective.
- CoT data construction depends on Qwen2.5-VL-32B, introducing a dependency on an external large model.
- The approach is not evaluated on non-classification tasks such as fine-grained detection or segmentation.
- All 6 datasets are classical FGVR benchmarks—evaluation on newer and more challenging datasets (e.g., the full iNaturalist) remains to be explored.

## Related Work & Insights
- **vs. CLS-RL (Li et al.)**: Directly applying classification reward-based RL leads to overfitting on seen classes; Fine-R1 achieves generalization through CoT and TAPO.
- **vs. SigLIP/CLIP**: Contrastive models are the gold standard for FGVR, but Fine-R1 demonstrates that generative MLLMs can surpass them with appropriate training.
- **vs. DeepPerception**: Focused on visual perception but lacks a mechanism for fine-grained knowledge retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Triplet-augmented policy optimization + structured CoT for FGVR; first to enable MLLMs to surpass CLIP.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, open/closed-world settings, ablation studies, knowledge analysis, and cross-domain transfer.
- Writing Quality: ⭐⭐⭐⭐ Motivation and method are clearly articulated, though the formulations are somewhat dense.
- Value: ⭐⭐⭐⭐⭐ A new paradigm for 4-shot FGVR with significant practical value for knowledge-intensive domains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/red_rationale_enhanced_decoding_cot.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](doxing_via_the_lens_revealing_location-related_privacy_leakage_on_multi-modal_la.md)
- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](../../AAAI2026/llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)

<!-- RELATED:END -->
