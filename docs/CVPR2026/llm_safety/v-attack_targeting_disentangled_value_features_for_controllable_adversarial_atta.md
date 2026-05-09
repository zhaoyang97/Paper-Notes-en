---
title: >-
  [Paper Note] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs
description: >-
  [CVPR 2026][LLM Safety][adversarial attack] This work discovers that Value features in ViT exhibit more disentangled local semantic representations compared to Patch features, and proposes V-Attack, which achieves precise and controllable local semantic attacks on LVLMs via self-enhanced Value features and text-guided semantic manipulation, improving average ASR by 36%.
tags:
  - CVPR 2026
  - LLM Safety
  - adversarial attack
  - vision-language models
  - Value features
  - semantic manipulation
  - controllable attack
date: 2026-05-08
content_hash: 60cb7d3326450c29
---

# V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs

**Conference**: CVPR 2026
**arXiv**: [2511.20223](https://arxiv.org/abs/2511.20223)
**Code**: [GitHub](https://github.com/Summu77/V-Attack)
**Area**: AI Security
**Keywords**: adversarial attack, vision-language models, Value features, semantic manipulation, controllable attack

## TL;DR

This work discovers that Value features in ViT exhibit more disentangled local semantic representations compared to Patch features, and proposes V-Attack, which achieves precise and controllable local semantic attacks on LVLMs via self-enhanced Value features and text-guided semantic manipulation, improving average ASR by 36%.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Adversarial attacks have evolved from disrupting classification predictions to manipulating image semantics in LVLMs. However, existing methods achieve extremely low success rates when precisely manipulating specific concepts — simultaneously altering 3 concepts yields a success rate below 10%.

**Core Finding**: ViT self-attention causes semantic entanglement in Patch features (global context dominates, diluting local semantics), whereas Value features naturally suppress global-context channels and retain high-entropy, disentangled local semantics. Channel distribution analysis shows that Patch features are dominated by a small number of highly activated channels (correlated with the CLS token), while Value features exhibit a uniform distribution.

## Method

### Overall Architecture

Multi-surrogate-model Value feature extraction → Self-Value Enhancement → Text-Guided Value Manipulation → PGD iterative adversarial perturbation generation.

### Key Designs

1. **Disentanglement of Value Features**: Analysis of CLIP-L/14 reveals that the information entropy of Patch features drops sharply at intermediate layers, whereas Value features consistently maintain high entropy. Text-alignment analysis shows that cosine similarity maps between $\mathbf{V}$ and specific text exhibit clear spatial alignment (e.g., "dog" → 0.28 vs. $\mathbf{X}$'s 0.22), making $\mathbf{V}$ a more precise target for semantic manipulation.

2. **Self-Value Enhancement**: A self-attention operation is applied to the extracted Value features (with Q=K=V all derived from Value), reinforcing the internal consistency of local semantics:
$$\widetilde{\mathbf{V}}^{(k)} = \text{Attn}(\mathbf{V}^{(k)}, \mathbf{V}^{(k)}, \mathbf{V}^{(k)})$$

3. **Text-Guided Value Manipulation**:
   - Source and target concepts are encoded via the CLIP text encoder.
   - The cosine similarity between each enhanced Value token and the source text is computed.
   - An adaptive threshold $\tau^{(k)}$ selects the token set $\mathcal{I}_{\text{align}}^{(k)}$ aligned with the source concept.
   - Loss: $\mathcal{L} = \sum_{k} \sum_{i \in \mathcal{I}_{\text{align}}^{(k)}} [-s_i^{(k)}(t_s) + s_i^{(k)}(t_t)]$
   - PGD iterations combined with random resizing augmentation to enhance transferability.

### Loss & Training

Optimization is performed jointly across multiple surrogate models (CLIP variants). For the selected semantically aligned tokens, the objective simultaneously pushes representations away from the source concept and toward the target concept.

## Key Experimental Results

### Main Results (Local Semantic Attack, MS-COCO)

| Method | LLaVA CAP | InternVL CAP | DeepseekVL CAP | GPT-4o CAP | Avg |
|--------|-----------|--------------|----------------|------------|-----|
| MF-it | 0.051 | 0.040 | 0.040 | 0.028 | 0.040 |
| SSA-CWA | 0.262 | 0.304 | 0.241 | 0.285 | 0.273 |
| M-Attack | 0.370 | 0.405 | 0.483 | 0.544 | 0.450 |
| **V-Attack** | **Best** | **Best** | **Best** | **Best** | **+36%** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Attack Patch features $\mathbf{X}$ | Low | Semantic entanglement |
| Attack Value features $\mathbf{V}$ | Significant gain | Disentangled local semantics |
| + Self-Enhancement | Further improvement | Richer semantics |
| + Text-Guided | **Best** | Precise localization & manipulation |

### Key Findings

- Value features suppress the highly activated channels in Patch features that carry dominant global information.
- Attacking Value features improves average ASR by 36% over attacking Patch features.
- The method remains effective on closed-source models such as GPT-4o and GPT-o3.

## Highlights & Insights

- **Deep Insight**: This work is the first to reveal the disentangled nature of ViT Value features, offering a new perspective for adversarial attacks.
- **Precise Controllability**: Single-concept-level precise semantic substitution is achieved (e.g., "dog" → "cat").
- **Strong Transferability**: Perturbations generated using white-box surrogate models remain effective on the black-box GPT-4o.

## Limitations & Future Work

- The approach relies on white-box surrogate models such as CLIP; transferability may degrade when architectural differences are large.
- The success rate for simultaneously attacking multiple concepts still has room for improvement.
- The ethical risks associated with such tools warrant attention.

## Related Work & Insights

- AttackVLM was the first to leverage CLIP for adversarial attacks on LVLMs; V-Attack identifies a more precise attack target within the same paradigm.
- M-Attack employs crop augmentation and model ensembling; V-Attack approaches the problem from the perspective of feature selection.
- This work also provides insights for LVLM security defenses: protecting the Value feature space may be a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of Value feature disentanglement is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models and scenarios, including GPT-4o.
- Writing Quality: ⭐⭐⭐⭐ Analysis is thorough and visualizations are excellent.
- Value: ⭐⭐⭐⭐ Exposes security vulnerabilities in LVLMs.
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](../../NeurIPS2025/llm_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)

<!-- RELATED:END -->
