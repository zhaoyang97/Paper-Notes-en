---
title: >-
  [Paper Note] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition
description: >-
  [CVPR 2026][Video Understanding][Zero-shot action recognition] This paper proposes SkeletonContext, a framework that recovers the missing environmental and object context semantics in skeleton data from pretrained language models via a cross-modal context prompt module, and enhances the discriminability of motion-critical joints through a key part decoupling module. The method achieves state-of-the-art performance on NTU-60/120 and PKU-MMD under both zero-shot (ZSL) and generalized zero-shot (GZSL) settings.
tags:
  - CVPR 2026
  - Video Understanding
  - Zero-shot action recognition
  - skeleton sequences
  - context prompt learning
  - cross-modal alignment
  - key part decoupling
date: 2026-05-08
content_hash: 68d8c338a5f332ad
---

# SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition

**Conference**: CVPR 2026
**arXiv**: [2603.29692](https://arxiv.org/abs/2603.29692)
**Code**: [https://github.com/NingWang2049/skeletoncontext](https://github.com/NingWang2049/skeletoncontext)
**Area**: Video Understanding / Action Recognition
**Keywords**: Zero-shot action recognition, skeleton sequences, context prompt learning, cross-modal alignment, key part decoupling

## TL;DR

This paper proposes SkeletonContext, a framework that recovers the missing environmental and object context semantics in skeleton data from pretrained language models via a cross-modal context prompt module, and enhances the discriminability of motion-critical joints through a key part decoupling module. The method achieves state-of-the-art performance on NTU-60/120 and PKU-MMD under both zero-shot (ZSL) and generalized zero-shot (GZSL) settings.

## Background & Motivation

1. **State of the Field**: Zero-shot skeleton-based action recognition (ZSSAR) identifies unseen action categories by aligning skeleton features with text embeddings in a shared space. Existing methods primarily focus on improving skeleton encoders, data augmentation, or external knowledge augmentation.
2. **Limitations of Prior Work**: Skeleton sequences contain only joint coordinates, lacking contextual cues such as objects and environments. The skeletal motions of "typing on a keyboard" and "writing on paper" are highly similar, yet cannot be distinguished without the context of "keyboard" and "paper."
3. **Root Cause**: Skeleton modality inherently lacks contextual information, while semantic descriptions are rich in such context. This fundamental semantic gap makes direct alignment between the two modalities limited in effectiveness.
4. **Paper Goals**: Inject language-driven contextual semantics into skeleton representations to bridge the semantic gap in cross-modal alignment.
5. **Starting Point**: Use an LLM to generate structured contextual descriptions (environment + used object + target object), then train the model to "reconstruct" these contexts from skeleton motion, enabling the skeleton encoder itself to acquire context-aware representations.
6. **Core Idea**: Train the skeleton encoder to infer contextual semantics (e.g., interacted objects and environments) from motion patterns via masked reconstruction.

## Method

### Overall Architecture

Skeleton sequences are first encoded by Shift-GCN, and the resulting features are fed into two modules in parallel: (1) the **Cross-Modal Context Prompt module**, which obtains fine-grained skeleton features via a differential joint encoder, performs bidirectional cross-attention with BERT-processed masked context prompts, and reconstructs the masked context words (environment, objects) to produce context-enhanced skeleton features; and (2) the **Key Part Decoupling module**, which predicts a joint importance map to highlight motion-critical joints. Each branch is aligned with its corresponding semantic embedding via a contrastive loss.

### Key Designs

1. **Cross-Modal Context Prompt (CMCP) Module**:

    - **Function**: Equips the skeleton encoder with the ability to infer contextual semantics (interacted objects, environments).
    - **Mechanism**: An LLM (ChatGPT-4) first generates structured descriptions for each action class in the format "In [environment], [body part] uses [object] to [sub-action] on [target object]," with 10 descriptions generated per class. During training, the three slots—environment, used object, and target object—are replaced with [MASK] tokens. Skeleton features interact with BERT token representations via bidirectional cross-attention, and BERT's masked prediction head reconstructs the masked context words. The context reconstruction loss $\mathcal{L}_{ccr}$ drives the skeleton features to encode contextual information.
    - **Design Motivation**: Unlike methods such as SCoPLe that enhance the text encoder, this approach directly enriches the skeleton-side representation so that skeleton features themselves carry contextual information.

2. **Differential Joint Encoder (DJE)**:

    - **Function**: Captures subtle inter-joint differences to model pose-specific spatial dependencies.
    - **Mechanism**: Skeleton features are pooled to the topology level and projected into queries and keys. A differential topology representation is computed as $A^{diff} = \phi(\mathcal{T}_1(H_x^Q) - \mathcal{T}_2(H_x^K))$, i.e., a difference matrix over all joint pairs. This difference matrix is then used to reweight and aggregate the original features into a topology-enhanced embedding $F_x^{diff}$.
    - **Design Motivation**: The "fingerprint" of different poses lies in inter-joint differences—"bending forward" implies a desk-level scene, while "raising a hand" implies head-level interaction. Differential encoding implicitly reflects such contextual cues.

3. **Progressive Partial Masking (PPM)**:

    - **Function**: A curriculum learning strategy that progressively increases the difficulty of context reconstruction.
    - **Mechanism**: The masking ratio is defined as $r_t = \min(1, t/T)$, growing linearly with training steps. In early training, only a small proportion of context slots are masked (e.g., only the environment slot), making reconstruction relatively easy. As training progresses, the masking ratio increases until all slots are masked, forcing the model to infer the complete context solely from skeleton motion and BERT's language priors.
    - **Design Motivation**: The structured prompt format differs substantially from the natural language seen during BERT pretraining, and full masking from the start makes reconstruction too difficult, leading to unstable training. The progressive strategy bridges this distributional gap.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{align} + \mathcal{L}_{ccr} + \mathcal{L}_{kpd}$:
- $\mathcal{L}_{align}$: Contrastive cross-entropy loss that aligns the context-enhanced skeleton features with context semantic embeddings, and the key part features with action semantic embeddings, respectively.
- $\mathcal{L}_{ccr}$: Masked context reconstruction loss that supervises BERT in recovering the masked context words.
- $\mathcal{L}_{kpd}$: Joint importance calibration loss that guides joint weight learning using LLM-generated body part priors $K_{gt}$.

At inference, calibrated stacking is applied to mitigate domain shift in GZSL by aggregating predictions from the context branch and the key part branch.

## Key Experimental Results

### Main Results

ZSL Accuracy (%):

| Method | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|--------|-------------|--------------|----------------|---------------|
| STAR (ACMM24) | 81.4 | 45.1 | 63.3 | 44.3 |
| Neuron (CVPR25) | 86.9 | 62.7 | 71.5 | 57.1 |
| FS-VAE (ICCV25) | 86.9 | 57.2 | 74.4 | 62.5 |
| **Ours** | **89.6** | **64.4** | 74.2 | 60.1 |

GZSL Harmonic Mean H (%):

| Method | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|--------|-------------|--------------|----------------|---------------|
| ScoPLe (CVPR25) | 70.8 | 57.9 | 52.2 | 52.2 |
| Neuron (CVPR25) | 71.4 | 59.1 | 63.3 | 53.6 |
| FS-VAE (ICCV25) | 75.7 | 52.1 | 63.3 | 54.7 |
| **Ours** | **77.1** | **61.1** | 63.1 | **56.1** |

### Ablation Study

| DJE | SCG | PPM | KPD | NTU60-ZSL | NTU120-GZSL |
|-----|-----|-----|-----|-----------|-------------|
| ✗ | ✗ | ✗ | ✗ | 79.4 | 49.4 |
| ✓ | ✗ | ✗ | ✗ | 81.4 | 51.4 |
| ✓ | ✓ | ✗ | ✗ | 83.9 | 55.4 |
| ✓ | ✓ | ✓ | ✗ | 87.4 | 55.9 |
| ✓ | ✓ | ✓ | ✓ | **89.6** | **56.1** |

### Key Findings

- **Context reconstruction is the primary contribution**: Introducing SCG yields the largest single improvement (81.4→83.9 ZSL), and PPM further stabilizes gains to 87.4.
- On hard, visually similar action classes (Hard Level), the proposed method achieves 55.8% GZSL, outperforming Neuron by 12.0 points and FS-VAE by 5.1 points, validating the critical role of context inference in fine-grained discrimination.
- Removing $\mathcal{L}_{ccr}$ (i.e., removing context reconstruction supervision) drops ZSL from 89.6 to 86.4, confirming the necessity of LLM-generated context for cross-modal alignment.
- Object-related slots (Use Object + Target Object) contribute more than the environment slot (87.0 vs. 84.4), as skeleton actions are primarily defined by hand-object interactions.
- On PKU-MMD, the proposed method achieves a GZSL harmonic mean of 71.4%, surpassing the second-best method Neuron by 2.2 points.

## Highlights & Insights

- **Reverse thinking—enhancing the skeleton side rather than the text side**: Prior methods (SCoPLe, Neuron) predominantly enhance the text encoder to better match skeleton features. SkeletonContext takes the opposite direction by enriching skeleton representations to carry contextual semantics, fundamentally addressing the information asymmetry.
- **Masked reconstruction as a bridge for cross-modal knowledge transfer**: The approach draws inspiration from masked reconstruction in vision-language pretraining (e.g., VL-BEiT), but innovatively applies it to the skeleton modality—which has no visual component—enabling BERT's linguistic knowledge to "flow into" the skeleton encoder.
- **Convincing qualitative analysis**: At inference, the model infers contextual objects such as "keyboard" or "pen/paper" from skeleton motion without any text input, intuitively demonstrating that the model has genuinely learned motion-to-context mappings.

## Limitations & Future Work

- The approach depends on the quality of ChatGPT-4-generated descriptions and the validity of the structured template; different LLMs may yield varying results.
- Only three slots (environment, used object, target object) are considered, without accounting for fine-grained body part interaction patterns.
- Shift-GCN is no longer the strongest skeleton encoder; adopting more powerful alternatives (e.g., CTR-GCN, InfoGCN) could yield further improvements.
- On the NTU-120 110/10 split, the proposed method does not surpass FS-VAE (74.2 vs. 74.4), suggesting that the marginal benefit of context augmentation may diminish when the number of seen classes is large.

## Related Work & Insights

- **vs. SCoPLe (CVPR25)**: Achieves data-driven semantic alignment by jointly tuning text and skeleton prompts, but introduces no additional contextual information. SkeletonContext fundamentally compensates for the informational deficiency of skeletons through reconstruction.
- **vs. Neuron (CVPR25)**: Uses multi-round LLM-generated side information to dynamically guide skeleton-semantic co-learning, but still operates at the alignment level. SkeletonContext directly injects context into the skeleton encoder.
- **vs. FS-VAE (ICCV25)**: Models skeleton motion as high- and low-frequency components via frequency-semantic decomposition, representing a complementary direction—frequency decomposition and context injection could potentially be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying masked reconstruction for cross-modal context injection from language into skeleton representations is a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple splits, ZSL + GZSL evaluation, hard-class analysis, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, though some notation in the formulations is slightly redundant.
- **Value**: ⭐⭐⭐⭐ A clear contribution to zero-shot skeleton-based action recognition; the paradigm of "enhancing the skeleton side rather than the text side" merits broader adoption.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[CVPR 2026\] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](../../AAAI2026/video_understanding/finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)

<!-- RELATED:END -->
