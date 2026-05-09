---
title: >-
  [Paper Note] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models
description: >-
  [CVPR 2026][Multimodal VLM][Compositional understanding] C2LIP proposes a contrastive learning fine-tuning approach that requires no hard negatives: by decomposing text into noun-phrase concepts and introducing cross-modal attention pooling, it achieves state-of-the-art performance on the SugarCrepe/SugarCrepe++ compositionality benchmarks while maintaining or improving zero-shot and retrieval performance.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Compositional understanding
  - contrastive learning
  - CLIP fine-tuning
  - noun phrases
  - zero-shot generalization
date: 2026-05-08
content_hash: 59ba6ecc392ba35d
---

# No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models

**Conference**: CVPR 2026
**arXiv**: [2603.25722](https://arxiv.org/abs/2603.25722)
**Code**: [https://github.com/SamsungLabs/concept_centric_clip](https://github.com/SamsungLabs/concept_centric_clip)
**Area**: Multimodal VLM / Contrastive Learning
**Keywords**: Compositional understanding, contrastive learning, CLIP fine-tuning, noun phrases, zero-shot generalization

## TL;DR

C2LIP proposes a contrastive learning fine-tuning approach that requires no hard negatives: by decomposing text into noun-phrase concepts and introducing cross-modal attention pooling, it achieves state-of-the-art performance on the SugarCrepe/SugarCrepe++ compositionality benchmarks while maintaining or improving zero-shot and retrieval performance.

## Background & Motivation

1. **State of the Field**: Contrastive vision-language models (CLIP, SigLIP) are foundational to computer vision, supporting open-world tasks such as zero-shot classification and retrieval.

2. **Limitations of Prior Work**:
   - **Poor compositional understanding**: CLIP tends to learn Bag-of-Words (BoW) representations and cannot distinguish "a red couch" from "a couch next to a red object," failing to correctly bind nouns and attributes.
   - **Limitations of hard negative approaches**: Existing methods (NegCLIP, DAC, SLVC, etc.) improve compositionality via hard-negative fine-tuning, but (a) are effective only on specific benchmarks with poor generalization; (b) severely degrade zero-shot classification and retrieval performance; and (c) require complex data generation pipelines involving LLMs and text-to-image models.
   - **Architectural issue**: The final global pooling operations in both text and visual encoders mix noun and attribute information across regions, causing binding relationships to be irreversibly lost.

3. **Root Cause**: Long descriptive captions do not inherently require compositional representations for contrastive learning (BoW suffices), while global pooling destroys binding information — these two fundamental causes make compositionality impossible to address through simple post-hoc hard-negative training.

4. **Paper Goals**: To simultaneously improve compositional understanding and preserve zero-shot/retrieval performance without using hard negatives.

5. **Starting Point**: (a) Replace long captions with short noun phrases for contrastive learning, forcing the model to learn fine-grained binding; (b) extract concept-specific visual representations via cross-modal attention before global pooling, propagating compositionality learning signals to pre-pooling features.

6. **Core Idea**: Achieve compositionality through noun-phrase concept contrastive learning and cross-modal attention-based binding before pooling — without any hard negatives.

## Method

### Overall Architecture

C2LIP fine-tunes SigLIP while keeping the original global contrastive loss $\mathcal{L}_{contrastive}$ unchanged, and introduces two auxiliary losses: (1) a noun-phrase concept loss $\mathcal{L}_{npc}$ that aligns global visual representations with each noun phrase; and (2) a cross-attention concept loss $\mathcal{L}_{xac}$ that uses noun phrases as queries to extract concept-specific representations from visual tokens and aligns them accordingly. No additional overhead is incurred at inference time; the pipeline is identical to standard SigLIP.

### Key Designs

1. **Noun-Phrase Concept Contrastive Loss ($\mathcal{L}_{npc}$)**:

   - **Function**: Forces the model to encode all concept information into the global visual representation.
   - **Mechanism**: Noun phrases (e.g., "a red couch") are extracted from each caption using spaCy; the corresponding text tokens are pooled to obtain concept embeddings $\{c_k\}$. Each image's visual embedding $v$ is contrasted against all its noun-phrase concepts via a multi-positive contrastive objective (extending SigLIP's sigmoid loss to support multiple positives). Short noun phrases cannot be resolved by BoW ("a red couch" requires distinguishing a red couch from a couch near a red object), compelling the model to learn more discriminative representations.
   - **Design Motivation**: Addresses the first root cause — long captions do not require compositionality. Noun phrases are short enough to defeat BoW shortcuts, and they use real data positives rather than synthetic hard negatives, minimizing distributional shift.

2. **Cross-Modal Attention Pooling + Cross-Attention Concept Loss ($\mathcal{L}_{xac}$)**:

   - **Function**: Learns concept binding before global pooling.
   - **Mechanism**: The value projection and MLP from SigLIP's attention pooling layer are reused to project visual tokens into the joint space, yielding $\bar{V}'$. Noun-phrase concept embeddings $c$ serve as queries in a cross-attention operation over $\bar{V}'$, producing concept-specific visual embeddings $\hat{v}(c) = \bar{V}'^T \cdot \text{attn}(c, \bar{V}')$. A contrastive loss analogous to $\mathcal{L}_{npc}$ then aligns $\hat{v}(c_k)$ with $c_k$. Since the attention pooling has **no learnable parameters**, the learning signal is propagated directly to pre-pooling visual representations.
   - **Design Motivation**: Addresses the second root cause — global pooling destroys binding information. Establishing concept-visual correspondences before pooling enables the encoder to internally learn compositional representations. The parameter-free design ensures zero inference overhead.

3. **Total Training Loss**:

   - **Function**: Balances global alignment, concept alignment, and cross-modal concept binding.
   - **Mechanism**: $\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \lambda_{npc}\mathcal{L}_{npc} + \lambda_{xac}\mathcal{L}_{xac}$, where $\lambda_{npc} = 1$ and $\lambda_{xac} = 0.01$.
   - **Design Motivation**: The small $\lambda_{xac}$ reflects that the cross-attention loss already produces sufficiently strong gradient signals; larger values would degrade global representation quality.

### Loss & Training

- Fine-tunes a pretrained SigLIP ViT-B/16 on CC3M (DreamLIP version) for only 5 epochs.
- Adam optimizer with learning rate 1e-5, 8 A40 GPUs, effective batch size 768.
- Noun phrases are extracted offline from captions using spaCy.
- Inference pipeline is identical to standard SigLIP with no additional parameters or computation.

## Key Experimental Results

### Main Results

Comprehensive evaluation across compositionality, zero-shot classification, and retrieval (ViT-B/16):

| Method | SC Add | SC Replace | SC Swap | SC++ Replace I2T | SC++ Swap I2T | ImNet1K | Flickr30k | MSCOCO | Avg. |
|--------|--------|------------|---------|-------------------|---------------|---------|-----------|--------|------|
| SigLIP (original) | 86.5 | 84.1 | 65.8 | 73.8 | 62.8 | 76.1 | 95.2 | 78.9 | 70.0 |
| NegCLIP | 85.8 | 85.0 | 75.3 | 69.1 | 70.9 | 55.7 | 92.4 | 73.9 | 67.7 |
| DAC-LLM | 93.7 | 89.5 | 74.6 | 53.7 | 59.6 | 51.1 | 83.7 | 59.0 | 57.2 |
| FG-CLIP | 84.7 | 85.1 | 69.9 | 75.8 | 67.5 | 69.0 | 95.8 | 78.4 | 70.7 |
| SigLIP (CC3M ft) | 87.9 | 85.6 | 69.7 | 73.5 | 67.9 | 75.9 | 95.6 | 80.3 | 71.5 |
| **C2LIP** | **94.2** | **88.3** | **73.1** | **79.7** | **75.3** | 73.5 | **97.0** | **82.7** | **75.0** |

### Ablation Study

Attribute binding breakdown (SugarCrepe + SugarCrepe++ attribute subsets):

| Method | SC Replace | SC Swap | SC++ Replace I2T/TOT | SC++ Swap I2T/TOT | Avg. |
|--------|------------|---------|----------------------|-------------------|------|
| SigLIP | 86.7 | 71.5 | 75.5 / 64.2 | 56.3 / - | - |
| NegCLIP | 85.3 | 80.0 | 66.1 / - | 73.2 / - | - |
| **C2LIP** | **89.3** | **77.6** | **82.5** / - | **78.2** / - | - |

### Key Findings

- **C2LIP is the only method that ranks highly across all benchmarks**: Hard-negative methods (NegCLIP/DAC) suffer severe degradation on zero-shot/retrieval tasks (ImageNet dropping to 40–55%), whereas C2LIP incurs only a 2.6% drop (76.1→73.5).
- **Fine-tuning on CC3M alone provides limited compositional gains** (SigLIP ft: 70.0→71.5), but adding C2LIP's concept losses yields a substantial jump to 75.0.
- **The parameter-free cross-modal attention pooling** is critical — it propagates gradient signals directly to pre-pooling feature representations, enabling the encoder to internally learn binding.
- Flickr30k retrieval improves from 95.2 to 97.0, and MSCOCO from 78.9 to 82.7, demonstrating that concept alignment also benefits retrieval tasks.

## Highlights & Insights

- **Precise problem diagnosis**: Identifying the BoW shortcut and the information loss from global pooling as the two root causes leads to targeted and principled solutions, rather than the brute-force addition of hard negatives.
- **Minimalist design**: No additional learnable parameters, no inference overhead, only 5 epochs of fine-tuning, and no dependency on LLMs or text-to-image models — only spaCy noun-phrase extraction and standard attention operations are required.
- **The hyperparameter $\lambda_{xac} = 0.01$** reflects the efficiency of the cross-attention loss gradient signal; a small weight suffices, and larger values would impair global representation quality.
- **Generality**: Although validated on SigLIP, the methodology is applicable in principle to any CLIP-like model.
- **Deployment-friendly**: Inference is entirely zero-cost, introducing no additional runtime overhead.

## Limitations & Future Work

- ImageNet zero-shot classification drops by 2.6%; the authors attribute this to the narrow training data domain and a conflict between scene-centric representations and ImageNet's object-centric task, yet this trade-off remains incompletely resolved.
- Fine-tuning is conducted only on CC3M (3M scale); performance at larger data scales has not been validated.
- The quality of noun-phrase extraction via spaCy is bounded by the accuracy of the NLP tool.
- Effects on ViT-L and larger models remain unexplored.
- The cross-modal attention pooling is used only during training; whether applying it at inference time could further improve concept-level retrieval is an open question.

## Related Work & Insights

- **vs. NegCLIP/DAC**: Hard-negative methods can achieve strong performance on specific benchmarks (DAC reaches 93.7 on SugarCrepe Add), but severely degrade zero-shot capability (ImageNet: 51.1). C2LIP achieves consistently strong performance across all tasks.
- **vs. CLIC**: CLIC performs well on SugarCrepe++ Swap-I2T but extremely poorly on text-only (TOT) tasks, indicating that its text encoder has not genuinely learned compositionality.
- **vs. FG-CLIP**: FG-CLIP is pretrained on LAION-2B with extensive hard-sample data, achieving an average of 70.7; C2LIP fine-tuned on CC3M for only 5 epochs reaches 75.0.
- **vs. Assouel et al.**: That work also uses cross-attention for concept binding but requires LLM-based scene graph decomposition and multiple forward passes, incurring high training and inference costs; C2LIP requires no additional parameters or forward passes.

## Rating

- Novelty: ⭐⭐⭐⭐ — Root-cause analysis is insightful and the solution is elegant and concise, though the overall direction is not entirely unexpected.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Full coverage of compositionality, zero-shot, retrieval, and fine-grained retrieval with fair comparisons against numerous baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ — Textbook-level clarity in problem formulation and experimental design.
- Value: ⭐⭐⭐⭐⭐ — An extremely practical post-training approach with zero inference overhead, directly applicable to industrial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)
- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)

<!-- RELATED:END -->
