---
title: >-
  [Paper Note] SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge
description: >-
  [AAAI 2026][Multimodal VLM][CLIP safety fine-tuning] This paper proposes SafeR-CLIP, a framework that improves upon Safe-CLIP by introducing proximity-based alignment (redirecting unsafe embeddings to their semantically…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "CLIP safety fine-tuning"
  - "NSFW content filtering"
  - "nearest-neighbor alignment"
  - "representation space preservation"
  - "progressive training"
date: 2026-05-08
content_hash: c8fd7f6fd9ab2e87
---

# SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge

**Conference**: AAAI 2026  
**arXiv**: [2511.16743](https://arxiv.org/abs/2511.16743)  
**Code**: [Project Page](https://adeelyousaf.github.io/SC-Project-Page/)  
**Area**: Multimodal VLM Safety  
**Keywords**: CLIP safety fine-tuning, NSFW content filtering, nearest-neighbor alignment, representation space preservation, progressive training

## TL;DR

This paper proposes SafeR-CLIP, a framework that improves upon Safe-CLIP by introducing proximity-based alignment (redirecting unsafe embeddings to their semantically nearest safe targets rather than fixed pairs) and a relative cross-modal redirection loss (using only unsafe representations as negatives rather than random in-batch negatives), recovering zero-shot classification accuracy by 8.0% over Safe-CLIP while maintaining stronger safety guarantees.

## Background & Motivation

- **Background**: Vision-language models such as CLIP, pre-trained on large-scale web data (e.g., LAION-5B), inevitably learn representations associated with NSFW content. Existing safety fine-tuning methods (e.g., Safe-CLIP) mitigate this by redirecting unsafe embeddings toward safe ones.
- **Limitations of Prior Work**: Safe-CLIP's safety fine-tuning incurs approximately 22% degradation in zero-shot classification accuracy, revealing a severe trade-off between safety and generalization.
- **Key Challenge**: Safe-CLIP relies on fixed unsafe–safe pair mappings, which introduces two fundamental problems: (1) a single unsafe concept may have multiple semantically valid safe alternatives, yet is forced to map to one fixed target; (2) under the standard InfoNCE contrastive loss, other semantically plausible safe descriptions are incorrectly treated as negatives and pushed away, disrupting the pre-trained semantic structure.
- **Key Insight**: The observation that a single unsafe description (e.g., "a child sitting next to a gun on a table") can correspond to multiple reasonable safe descriptions (e.g., "a child sitting at a table eating" or "a child sitting next to objects piled on a table") motivates a minimal-intervention approach: rather than forcing a mapping to a potentially mismatched fixed target, one should find the semantically nearest safe substitute and redirect along the shortest path.
- **Core Idea**: Proximity-aware re-alignment — identifying the semantically nearest safe substitute to an unsafe input in the embedding space and redirecting along the shortest path, so as to minimally perturb the pre-trained representation space.

## Method

### Overall Architecture

SafeR-CLIP is a CLIP safety fine-tuning framework trained on ViSU quadruple data (safe image–text pairs + unsafe image–text pairs). The key improvements are: (1) offline construction of nearest-neighbor safe pairs to replace fixed pairs; (2) a relative cross-modal redirection loss replacing standard InfoNCE; and (3) progressive curriculum training that introduces samples in order of increasing difficulty.

### Key Designs

1. **Proximity-Based Alignment**

    - **Function**: For each unsafe description $t_i^*$, identify the semantically nearest safe description $\hat{t}_i$.
    - **Mechanism**: The frozen CLIP text encoder computes cosine similarities between the unsafe description and all safe descriptions: $s_{ij} = \cos(\mathcal{T}_0(t_i^*), \mathcal{T}_0(t_j))$, and selects $\hat{t}_i = t_{j^*}$ where $j^* = \arg\max_j s_{ij}$.
    - **Design Motivation**: Fixed pairings frequently introduce semantic mismatches (e.g., mapping "gun" to "cake"), whereas nearest-neighbor pairing ensures the redirection direction is the shortest in embedding space, minimizing representational perturbation.
    - Computed offline once before training, incurring no additional training overhead.

2. **Relative Cross-Modal Redirection Loss**

    - **Function**: Replaces Safe-CLIP's standard InfoNCE loss to prevent incorrectly pushing away semantically valid safe concepts.
    - **Mechanism**: Instead of using random in-batch negatives, the loss uses only the corresponding unsafe cross-modal embedding as the sole hard negative. The image encoder loss is defined as:
      $$\mathcal{L}_{\text{cross-redir}}^{\text{image}} = \frac{1}{N}\sum_{i=1}^N \log(1 + \exp(\cos(\mathcal{V}(v_i^*), \mathcal{T}_0(t_i^*)) - \cos(\mathcal{V}(v_i^*), \mathcal{T}_0(\hat{t}_i))))$$
    - **Design Motivation**: Standard InfoNCE treats all other safe descriptions within a batch as negatives; however, these descriptions may have legitimate semantic associations with the current unsafe input, and pushing them away disrupts the pre-trained semantic structure. The relative loss only requires the unsafe embedding to be "closer to the safe target than to the original unsafe description," leaving relationships among other safe concepts intact.

3. **Progressive Curriculum Training**

    - **Function**: Trains the model with unsafe–safe pairs ordered by increasing semantic difficulty.
    - Three stages: epoch 1 uses only easy pairs (high cosine similarity between safe and unsafe descriptions); epoch 2 adds medium-difficulty pairs; from epoch 3 onward, all difficulty levels are included.
    - **Design Motivation**: Training directly on hard pairs causes severe perturbation of the representation space; starting from easy pairs allows the model to adapt smoothly and reduces unnecessary representational drift.

4. **NSFWCaps Benchmark Dataset**

    - 1,000 quadruples constructed from NoCaps (non-COCO distribution) to test out-of-distribution safety generalization.
    - Safe–unsafe description JINA-CLIP similarity of 0.81 (vs. 0.62 for ViSU), indicating tighter semantic coupling.
    - Unsafe variants generated using LLaMA-3-70B; quality ensured via NudeNet + Q16 filtering.

### Loss & Training

- Total loss = proximity-based cross-modal redirection loss (image + text) + proximity-based unimodal redirection loss (image + text) + preservation loss (maintaining pre-trained representational structure).
- LoRA (r=16) adapters fine-tune both visual and text encoders.
- Adam optimizer, lr=1e-4, batch size 48, 9 training epochs.
- Backbone: ViT-L/14 (compatible with Stable Diffusion v1.4 and LLaVA).

## Key Experimental Results

### Main Results (Cross-Modal Retrieval + Zero-Shot Classification)

| Method | ViSU T*→V (R@1, ↑) | NSFWCaps T*→V (R@1, ↑) | Zero-Shot Avg. Acc. (11 datasets) |
|--------|---------------------|------------------------|----------------------------------|
| CLIP (original) | 2.8 | 3.8 | 74.3% |
| Safe-CLIP | 14.5 | 35.4 | 52.2% |
| SafeR-CLIP | **27.9** (+13.4%) | **79.5** (+44.1%) | **60.2%** (+8.0%) |

### Ablation Study (Safety on Real NSFW Data, Unsafe Retrieval Rate ↓ Lower is Better)

| Method | NSFW URLs V→T | NudeNet V→T | SMID V→T |
|--------|---------------|-------------|----------|
| CLIP | 91.6% | 94.1% | 96.3% |
| Safe-CLIP | 21.1% | 13.0% | 14.2% |
| SafeR-CLIP | **18.5%** | **10.7%** | **3.1%** |

### Text-to-Image Safety (I2P Benchmark, NSFW Score ↓ Lower is Better)

| Method | Avg. NSFW Score |
|--------|----------------|
| SD v1.4 (original) | 37.1 |
| + Safe-CLIP | 16.1 |
| + SafeR-CLIP | **16.0** |
| + SafeR-CLIP + SLD-Strong | **12.0** |

### Key Findings

- **44.1% improvement in unsafe→safe redirection on NSFWCaps**: Proximity-based alignment demonstrates a substantial advantage in out-of-distribution settings — semantically more coherent redirection directions generalize better.
- **8.0% recovery in zero-shot accuracy**: From Safe-CLIP's 52.2% to 60.2%, validating the effectiveness of the minimal-intervention principle in preserving pre-trained knowledge.
- **Unsafe retrieval rate on SMID reduced from 14.2% to 3.1%**: The most pronounced improvement is on non-pornographic NSFW categories including violence and discrimination.
- **Compatible with inference-time safety guidance**: Combining with SLD further reduces NSFW scores in text-to-image generation.

## Highlights & Insights

- **The "minimal intervention" design principle**: Respecting the geometric structure of the pre-trained representation space is critical for safety fine-tuning — redirecting along the shortest semantic path is preferable to forcing a mapping onto an arbitrarily chosen safe target. This principle generalizes to fine-tuning tasks beyond safety.
- **The pitfall of InfoNCE negative sampling**: In safety fine-tuning, random in-batch negatives under standard contrastive learning become "false negatives" — semantically valid safe concepts are incorrectly pushed away. This highlights the necessity of designing contrastive losses with task-specific awareness.

## Limitations & Future Work

- Proximity-based pairing relies on the quality of CLIP's own embedding space; if CLIP's semantic representations for certain NSFW categories are inherently biased, nearest-neighbor selection may be inaccurate.
- Training still depends on synthetic data from ViSU; distributional gaps between synthetic unsafe images and real NSFW content may limit safety generalization.
- Validation is limited to the CLIP architecture; SigLIP is briefly tested only in supplementary material, and applicability to broader VLM architectures remains to be verified.
- The safety–generalization trade-off, while improved, persists: 60.2% vs. the original CLIP's 74.3%, leaving a remaining gap of approximately 14%.

## Related Work & Insights

- **vs. Safe-CLIP**: Safe-CLIP uses fixed pairs + standard InfoNCE, incurring 22% generalization loss; SafeR-CLIP uses proximity-based pairs + relative loss, reducing the loss to 14% while achieving stronger safety performance.
- **vs. UWM (inference-time method)**: UWM achieves training-free safety by manipulating unsafe model weights, offering high efficiency but limited safety gains; SafeR-CLIP, as a training-based method, delivers stronger safety guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ Proximity-aware redirection and relative loss design are concise and effective, though the core contributions are primarily at the loss function level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers retrieval, zero-shot classification, text-to-image generation, and image-text generation; includes real NSFW data evaluation and the new NSFWCaps benchmark.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear (two explicit limitations of Safe-CLIP); method derivation follows naturally.
- Value: ⭐⭐⭐⭐ Directly applicable to the safe deployment of VLMs; the minimal-intervention principle is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[CVPR 2026\] Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality](../../CVPR2026/multimodal_vlm/benchmarking_vision-language_models_under_contradictory_virtual_content_attacks_.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)

</div>

<!-- RELATED:END -->
