---
title: >-
  [Paper Note] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment
description: >-
  [CVPR 2026][Multimodal VLM][noisy labels] This paper proposes the NA-MVP framework, which employs a bi-directional (clean + noise-aware) multi-view prompt design coupled with Unbalanced Optimal Transport (UOT) for fine-grained patch-to-prompt alignment, and applies classical OT for selective label correction on identified noisy samples, consistently surpassing state-of-the-art methods in noisy few-shot learning scenarios.
tags:
  - CVPR 2026
  - Multimodal VLM
  - noisy labels
  - prompt learning
  - optimal transport
  - CLIP
  - few-shot learning
date: 2026-05-08
content_hash: 8fe057a7f246df92
---

# Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.11617](https://arxiv.org/abs/2603.11617)
**Code**: None (no code link provided in the paper)
**Area**: Multimodal VLM
**Keywords**: noisy labels, prompt learning, optimal transport, CLIP, few-shot learning

## TL;DR
This paper proposes the NA-MVP framework, which employs a bi-directional (clean + noise-aware) multi-view prompt design coupled with Unbalanced Optimal Transport (UOT) for fine-grained patch-to-prompt alignment, and applies classical OT for selective label correction on identified noisy samples, consistently surpassing state-of-the-art methods in noisy few-shot learning scenarios.

## Background & Motivation
- Vision-language models such as CLIP can be efficiently adapted to downstream tasks via prompt learning; however, when training labels are noisy, even a small number of incorrect labels can disproportionately affect gradient updates.
- Existing noisy prompt learning methods suffer from three major limitations:
    - **Insufficient prompt expressiveness**: Most methods rely on only 1–2 prompts (e.g., positive/negative pairs), and single-view alignment fails to capture fine-grained semantic cues.
    - **Rigid explicit negative labels**: Assigning a hard negative label to each image produces a fixed counter-class signal that is often inaccurate or uninformative under noise.
    - **Coarse denoising**: Reliance on fixed confidence thresholds or non-selective pseudo-labels leads to error propagation.
- Core insight: Robust noisy few-shot learning requires a shift from global matching toward **region-aware fine-grained alignment** that adaptively distinguishes clean from noisy semantics.

## Core Problem
How, within VLM prompt learning under severe label noise and few-shot settings, can one (1) adaptively distinguish clean from noisy semantic signals; (2) achieve fine-grained image-region-to-prompt alignment to suppress noisy regions; and (3) selectively correct erroneous labels without over-correction.

## Method

### Overall Architecture
NA-MVP comprises two core modules that operate jointly:
1. **Noise-Aware Alignment** (blue pathway): For each class, multiple clean and noise-aware prompts are constructed and aligned with local image patches via UOT for fine-grained matching, yielding clean/noisy probabilities.
2. **Selective Label Correction** (green pathway): Based on bi-directional alignment signals, mislabeled samples are adaptively identified and their labels are corrected using classical OT.

The two modules iteratively update the training set and optimize the prompts, producing a denoised dataset for robust prediction.

### Key Designs
1. **Bi-directional Multi-View Prompt Construction**:

    - For each class $k$, two sets of learnable prompts are constructed: clean-oriented $\{Prompt_{m,k}^c\}_{m=1}^N$ and noise-aware $\{Prompt_{m,k}^n\}_{m=1}^N$.
    - Each prompt consists of $M$ learnable context tokens concatenated with a class-specific token.
    - Clean prompts capture class-relevant semantics, while noise-aware prompts serve as adaptive filters to suppress misleading signals.
    - Non-target classes function as implicit negative samples, avoiding the rigidity of explicit negative labels.

2. **Fine-Grained Noise-Aware Alignment via UOT**:

    - Local image features $F_i \in \mathbb{R}^{L \times d}$ and prompt features $G_k \in \mathbb{R}^{N \times d}$ are treated as discrete distributions.
    - A cost matrix is computed via cosine similarity: $C_k = 1 - F_i G_k^\top$.
    - UOT relaxes the strict mass conservation constraint ($T\mathbf{1}_N \leq \mu$ rather than equality), permitting partial matching.
    - Solved efficiently via a Dykstra-based implementation (Sinkhorn + entropic regularization).
    - Core advantage: Not all features are forced to align; noisy or irrelevant patches can be safely "discarded."

3. **Selective Label Correction**:

    - **Noise identification**: UOT distances between each sample and the clean/noise-aware prompts yield similarities $s_{i,k}^c$ and $s_{i,k}^n$. An adaptive threshold $\phi_{i,k} = \frac{\exp(s_{i,k}^n/\tau)}{\exp(s_{i,k}^c/\tau) + \exp(s_{i,k}^n/\tau)}$ determines whether a sample is noisy.
    - **Label correction**: For samples identified as noisy, classical OT (with strict mass conservation) computes an optimal transport plan $T^*$ between global image features and class prompt features; $\arg\max_j T^*_{ij}$ is assigned as the pseudo-label.
    - Only samples satisfying $p_{ik}^c < \phi_{i,k}$ are corrected; reliable samples are left unchanged to avoid over-correction.

### Loss & Training
- **Early stage** (first $T_{sup}$ epochs): $\mathcal{L}_{sup} = \mathcal{L}_{gce} + \lambda_i \cdot \mathcal{L}_{itbp}$
    - GCE (Generalized Cross-Entropy): a noise-robust loss function.
    - ITBP Loss: an auxiliary bi-directional contrastive loss encouraging image features to align with clean prompts and diverge from noise-aware prompts.
- **Later stage**: Label correction is activated and training continues with GCE on the denoised dataset.
- Inference: $p(y=k|x_i) = (1 - p_{ik}^n) \cdot p_{ik}^c$, jointly leveraging clean and noise-aware probabilities.
- SGD optimizer (lr=0.002, momentum=0.9, weight_decay=5e-4), 50 epochs, 16 shared context tokens, ResNet-50 image encoder.

## Key Experimental Results

### Synthetic Noise (Average over 5 Datasets, 16-shot)

| Dataset | Metric | NA-MVP | NLPrompt | Gain |
|--------|------|--------|----------|------|
| OxfordPets (75% Sym) | Acc | **86.23** | 70.77 | +15.46 |
| OxfordPets (50% Sym) | Acc | **88.13** | 83.17 | +4.96 |
| DTD (75% Sym) | Acc | **48.63** | 39.80 | +8.83 |
| Caltech101 (75% Sym) | Acc | **89.37** | 86.70 | +2.67 |

### Real-World Noise (Food101N)

| Method | 4-shot | 8-shot | 16-shot | 32-shot |
|------|--------|--------|---------|---------|
| NLPrompt | 70.57 | 73.93 | 76.46 | 76.87 |
| **NA-MVP** | **76.10** | **76.27** | **76.90** | **77.03** |

The advantage is especially pronounced under extreme few-shot conditions (4-shot: +5.53), where the impact of noisy labels is most severe.

### Ablation Study
- **Bi-directional prompts**: single prompt (48.08%) → + explicit negative labels (48.82%) → + implicit bi-directional (49.63%) → + multi-view (51.83%), showing consistent incremental gains.
- **UOT vs. OT vs. KL**: UOT (54.18%) > OT (53.37%) > KL (52.60%); the relaxed constraints of UOT yield a clear advantage under noise.
- **Selective correction**: Global OT correction incorrectly modifies clean labels at low noise (25% noise: 59.60%), whereas selective correction is more stable (63.13%).
- **Number of prompts N**: $N=4$ achieves the best balance ($N=1$ is insufficient; $N=8$ is redundant).
- **vs. DEFT**: NA-MVP consistently outperforms DEFT at all noise levels (75% noise: 86.23 vs. 75.87).

## Highlights & Insights
- **A novel conceptual reframing**: The noise robustness problem is reformulated as "region-aware clean-noisy semantic decomposition" rather than simple global matching.
- **Judicious use of UOT**: Relaxing the mass conservation constraint naturally suits noisy settings—noisy patches need not be forcibly aligned and can be safely discarded.
- **Complementary design of two OT variants**: UOT handles local fine-grained alignment (noise identification), while classical OT handles global label correction (strict mass conservation ensures assignment validity).
- Improvements are most pronounced at high noise rates (75%), demonstrating that the framework's noise robustness genuinely surpasses prior work.
- The inference formula $(1-p^n) \cdot p^c$ is concise and elegant.
- Introducing optimal transport theory into prompt-based noisy label learning represents a novel cross-domain transfer.
- Bi-directional multi-view alignment jointly addresses both prompt noise and label noise.
- Strong accuracy is maintained even under 60% symmetric noise, demonstrating practical utility.

## Limitations & Future Work
- Validated only on classification tasks; not extended to more complex visual tasks such as detection or segmentation.
- Relies on CLIP's ResNet-50 backbone; performance under stronger backbones (e.g., ViT-L/14) has not been verified.
- Sinkhorn iterations in UOT introduce additional computational overhead; the paper does not discuss computational cost in detail.
- The framework assumes standard symmetric/asymmetric noise patterns; real-world noise distributions may be more complex (e.g., instance-dependent noise).
- The optimal number of multi-view prompts ($N=4$) may depend on dataset-specific characteristics.
- UOT computational cost grows with the number of prompts and sample size.

## Related Work & Insights
- **vs. CoOp/CoCoOp**: Standard prompt learning methods that do not handle noisy labels; NA-MVP achieves substantially higher performance in the presence of noise.
- **vs. NLPrompt**: Employs OT-Filter for noise identification and global OT for relabeling—a coarse-grained global approach. NA-MVP achieves finer-grained noise handling via patch-level UOT and adaptive thresholding.
- **vs. DEFT**: Uses a fixed threshold (0.5) for clean sample identification; NA-MVP's adaptive threshold $\phi_{i,k}$ is more flexible.
- **vs. PLOT**: PLOT applies OT for multi-prompt alignment on clean data; NA-MVP extends this to a noise-aware bi-directional design.
- **vs. CLIPN**: CLIPN uses positive/negative prompt pairs for OOD detection; NA-MVP transfers this idea to noisy label learning with added multi-view prompts.
- **vs. ProGrad**: Uses gradient filtering to suppress noisy prompts, but is less effective than the globally optimal transport alignment in UOT.
- The "relaxed matching" intuition of UOT under noise is transferable to other noisy settings (e.g., noisy annotations in object detection, imprecise labels in medical image segmentation).
- The finding that implicit negatives outperform explicit negative labels also has implications for contrastive learning in NLP.
- The bi-directional prompt design can be extended to "multi-dimensional prompts" (e.g., a three-way clean/noisy/ambiguous formulation).

## Relevance to My Research
- Possibly related: `20260316_adaptive_model_routing.md`
- Possibly related: `20260316_cross_species_framework.md`
- Possibly related: `20260316_concept_bottleneck_world_model.md`

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing UOT into prompt-based noisy label learning is novel; the bi-directional multi-view design is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five synthetic noise datasets plus one real-world noise dataset; ablation studies comprehensively cover components, alignment methods, prompt count, and thresholding strategies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; Figure 1 effectively summarizes the three key limitations; the method description is systematic.
- Value: ⭐⭐⭐⭐ Provides an effective solution for the practically important setting of noisy few-shot learning; substantial gains under high noise rates carry real-world significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)

<!-- RELATED:END -->
