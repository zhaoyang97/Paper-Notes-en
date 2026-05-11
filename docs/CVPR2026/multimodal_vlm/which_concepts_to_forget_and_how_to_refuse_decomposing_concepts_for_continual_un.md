---
title: >-
  [Paper Note] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Continual unlearning] This paper proposes CORE (COncept-aware REfuser), a framework for continual unlearning in large vision-language models (LVLMs). It decomposes vision-language deletion targets into fine-grained visual attribute concepts and textual intent concepts, employs a concept modulator to identify concept combinations requiring refusal, and generates concept-aligned refusal responses via a mixture of refusal experts (refusers). CORE achieves state-of-the-art unlearning-retention trade-offs of 90.67% CRR and 88.02% AR across 16 sequential unlearning tasks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Continual unlearning
  - large vision-language models
  - concept decomposition
  - mixture of refusal experts
  - selective knowledge deletion
date: 2026-05-08
content_hash: 6248890e6725f46c
---

# Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.21484](https://arxiv.org/abs/2603.21484)
**Code**: None
**Area**: Multimodal VLM / Machine Unlearning
**Keywords**: Continual unlearning, large vision-language models, concept decomposition, mixture of refusal experts, selective knowledge deletion

## TL;DR
This paper proposes CORE (COncept-aware REfuser), a framework for continual unlearning in large vision-language models (LVLMs). It decomposes vision-language deletion targets into fine-grained visual attribute concepts and textual intent concepts, employs a concept modulator to identify concept combinations requiring refusal, and generates concept-aligned refusal responses via a mixture of refusal experts (refusers). CORE achieves state-of-the-art unlearning-retention trade-offs of 90.67% CRR and 88.02% AR across 16 sequential unlearning tasks.

## Background & Motivation
1. **Background**: Large vision-language models (e.g., MiniGPT, InstructBLIP), pretrained on large-scale multimodal data, have achieved remarkable performance across various vision-language tasks. However, pretraining corpora inevitably contain inappropriate or sensitive content, which may cause models to produce undesirable outputs.
2. **Limitations of Prior Work**: (a) Retraining from scratch is infeasible—pretraining data is often inaccessible and computational costs are prohibitive; (b) Deletion requests arrive sequentially over time (driven by user demands and AI regulations), necessitating *continual* rather than one-shot unlearning; (c) Existing unlearning methods (gradient ascent, random labeling, etc.) distort shared representations during sequential updates, introducing spurious correlations—the model mistakes surface-level visual-language cues for refusal signals—resulting in two failure modes: **irrelevant refusal** (semantically misaligned refusals for prior unlearning tasks) and **over-refusal** (erroneous refusals of benign queries).
3. **Key Challenge**: Visual and linguistic representations in LVLMs are highly entangled; editing specific knowledge inevitably perturbs other information. As sequential unlearning tasks accumulate, this entanglement-induced "representation distortion" compounds, making it increasingly difficult for the model to distinguish what should be refused from what should not.
4. **Goal**: (a) How to precisely identify concept combinations requiring refusal across multi-step unlearning (*which to forget*); (b) How to generate refusal responses that are semantically aligned with unlearning targets rather than generically refusing everything (*how to refuse*).
5. **Key Insight**: The authors' core insight is that concept-level approaches enable more precise and interpretable unlearning: explicitly extracting target concepts and refusing relevant concept combinations mitigates spurious correlations more effectively than direct parameter manipulation.
6. **Core Idea**: Decompose vision-language deletion targets into visual attribute concepts and textual intent concepts; employ a concept modulator to identify distinctive concept combinations per unlearning category; and route specialized refusal experts via a routing mechanism to generate concept-aware refusal responses.

## Method

### Overall Architecture
CORE operates on a frozen LVLM (visual encoder and language model are unchanged). The overall pipeline proceeds as follows: (1) For each unlearning category, an LLM generates concept description sets for visual attributes and textual intents. (2) Concept modules produce activation scores between the input and all learned concepts. (3) A concept modulator reweights concept activations via learned weights, suppressing irrelevant concepts and emphasizing category-specific ones. (4) A concept-similarity-based routing mechanism assigns refusal experts. (5) The mixture of refusal experts transforms visual features to guide the language model in generating concept-aligned refusals. (6) At inference, refusal intensity is calibrated according to the conceptual relevance between the input and unlearning tasks.

### Key Designs

1. **Concept Recognition & Modulation**:

    - **Function**: Decomposes inputs into interpretable concept activations and identifies distinctive concept combinations for each unlearning category.
    - **Mechanism**: For each unlearning category $k$, an LLM generates 20 visual attribute concepts (e.g., "protesters holding banners") and 20 textual intent concepts. Each concept module $\bm{\mathcal{E}}_{\text{q},k}$ produces alignment activation scores between the input and concept set $\mathcal{C}_{\text{q},k}$. To ensure semantic grounding, CLIP encoder similarity serves as a supervision target: $\mathcal{L}_{\text{con}} = -\sum \text{sim}(E^t_{\text{q},i}, \hat{E}_{\text{q},i})$. The key innovation lies in the concept modulator $\bm{\mathcal{M}}$: as tasks accumulate, concept semantics across different unlearning categories overlap, causing irrelevant concepts to exhibit high activations. The modulator identifies the unlearning category of the input via a learned classification head and outputs weights $\{m_k\}$ that reweight concept activations: $\bar{E}^t_{\text{q},i} = \bigoplus_{k} m_k \cdot \bm{\mathcal{E}}_{\text{q},k}(x^t_{\text{q},i})$.
    - **Design Motivation**: Without the modulator, large numbers of irrelevant concepts are activated (visualized as red-marked erroneous concepts in the paper), leading to imprecise refusal behavior. Ablation results show that removing the modulator (MOD) reduces CRR from 88.14% to 83.95% and AR from 86.74% to 74.31% (Avg).

2. **Concept-Aware Refusal Generation**:

    - **Function**: Guides the language model to generate concept-aligned refusal responses through a mixture of specialized refusal experts.
    - **Mechanism**: $N_R=20$ refusal experts (refusers) $\{\mathcal{V}_j\}$ are introduced, each implemented as a lightweight connector module. A router $\mathcal{R}$ computes contribution weights $\{\alpha_j\}$ for each refuser based on refined concept activations. The mixed output $\Delta\mathcal{P}(x^t_{\text{img}}) = \sum_j \alpha_j \cdot \mathcal{V}_j(x^t_{\text{img}})$ is added to the output of the pretrained connector module; the transformed visual features guide the language model to produce refusals. Only 2 refusers are activated per sample.
    - **Design Motivation**: Unlike direct modification of pretrained model parameters, dedicated connector module transformations preserve pretrained capabilities while enabling precise refusal behavior.

3. **Conceptual Relevance Guided Routing**:

    - **Function**: Efficiently manages a fixed number of refusers across continual tasks—reusing refusers for semantically similar tasks and adapting underutilized ones for novel concepts.
    - **Mechanism**: The conceptual relevance between the current task $t$ and a prior task $t'$ is computed as: $r^{t'} = \sigma(\text{sim}(\bar{E}^t_{\text{img}}, \bar{E}^{t'}_{\text{img}}) \cdot \text{sim}(\bar{E}^t_{\text{txt}}, \bar{E}^{t'}_{\text{txt}}))$. High relevance triggers a contrastive loss $\ell_+$ to bring routing outputs closer; low relevance triggers $\ell_-$ to push them apart: $\mathcal{L}_{\text{ref}} = \sum_{t'} [r^{t'} \cdot \ell_+(F^t, F^{t'}) + (1-r^{t'}) \cdot \ell_-(F^t, F^{t'})]$.
    - **Design Motivation**: A fixed number of refusers must balance reuse (avoiding parameter waste) and specialization (avoiding interference). Ablation results show that removing routing (ACT) causes CRR to drop sharply from 88.14% to 54.53%, demonstrating that unguided refuser reuse leads to irrelevant concept coverage.

### Inference-Time Refusal Calibration
- The maximum conceptual relevance $\beta \in [0,1]$ between the inference query and all unlearned tasks is computed, and the refuser mixture contribution is adjusted accordingly: $\mathcal{P}(\bar{x}_{\text{img}}) + \beta \cdot \Delta\mathcal{P}(\bar{x}_{\text{img}})$.
- Ablation results show that removing calibration (CAL) causes AR to drop catastrophically from 86.74% to 4.11% (Avg), with the model refusing nearly all inputs.

### Loss & Training
Training proceeds in two stages: (1) Concept modules and the modulator are first trained ($\mathcal{L}_{\text{con}} + \mathcal{L}_{\text{mod}}$) to establish reliable concept predictions; (2) The router and refusers are subsequently trained ($\mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{ref}}$) to generate concept-aware refusals. Feature prototypes from prior tasks are used to prevent catastrophic forgetting.

## Key Experimental Results

### Main Results (Vicuna-based LVLM, after 16 sequential unlearning tasks)

| Method | S↑ (General Capability) | AR↑ (Answer Retention Rate) | CRR↑ (Contextual Refusal Rate) | ΔRR↓ (Refusal Deviation) |
|------|-------------|-----------------|-------------------|-----------------|
| EWC | 76.22 | 24.90 | 51.01 | 35.38 |
| LwF | 72.09 | 43.12 | 41.01 | 33.13 |
| SCRUB | 63.38 | 8.84 | 57.69 | 36.95 |
| MoEAdapter | 94.46 | 54.25 | 52.82 | 31.98 |
| O3 | 92.85 | 81.76 | 73.03 | 9.03 |
| **CORE (Ours)** | **96.54** | **88.02** | **90.67** | **3.74** |

### Ablation Study (Avg metrics)

| MOD | ACT | CAL | S↑ | AR↑ | CRR↑ | ΔRR↓ |
|-----|-----|-----|-----|------|------|------|
| ✓ | ✓ | ✓ | **97.64** | **86.74** | **88.14** | 8.38 |
| ✗ | ✓ | ✓ | 93.10 | 74.31 | 83.95 | 8.17 |
| ✓ | ✗ | ✓ | 93.82 | 86.90 | 54.53 | 33.81 |
| ✓ | ✓ | ✗ | 37.71 | 4.11 | 86.09 | 10.79 |

### Key Findings
- **All three components are indispensable**: Removing MOD degrades concept recognition accuracy (AR drops 12.4%); removing ACT causes CRR to plummet by 33.6% due to erroneous refuser reuse; removing CAL is the most critical failure—AR drops to 4.11%, with the model refusing nearly all inputs.
- **CORE maintains stability throughout the unlearning sequence**: Figure 3 shows that conventional methods (EWC, LwF, etc.) exhibit sustained degradation in general capability and retained-data performance as unlearning steps increase, whereas CORE remains consistently stable.
- **Cross-LVLM generalization**: On a LLaMA-2-based LVLM, CORE also significantly outperforms O3 (AR: 84.41% vs. 66.73%; CRR: 84.54% vs. 76.74%).
- **Concept visualizations confirm modulator effectiveness**: With the modulator, activated concepts are highly focused (e.g., "protesters holding banners"); without it, large numbers of irrelevant concepts are activated.

## Highlights & Insights
- **The concept decomposition approach** is particularly elegant: it reframes "what to forget" as "where to localize in concept space," naturally supporting interpretability—one can directly inspect which visual attributes and textual intents triggered a refusal.
- **The inference-time calibration mechanism** is critical to practical utility: dynamically adjusting refusal intensity based on conceptual relevance between the input and unlearning tasks effectively resolves the over-refusal problem. This simple mechanism recovers AR from 4.11% to 86.74%.
- **The refuser routing design** draws on the MoE paradigm, reusing or adapting refusers based on concept similarity, enabling a fixed number of refusers to accommodate a growing stream of unlearning tasks.

## Limitations & Future Work
- Concept descriptions are generated by an LLM, so quality depends on LLM capability and lacks a validation mechanism—erroneous concept descriptions may induce incorrect refusal boundaries.
- Each unlearning category is assigned a fixed 20 visual + 20 textual concepts, which may be insufficiently flexible for categories with highly varying conceptual complexity.
- Refusers are initialized from pretrained connector modules; diversity among refusers is not explicitly enforced, potentially leading to functional redundancy.
- The experimental scope is relatively constrained (6 categories from a safety benchmark + 80 categories from ImageNet-R); more complex real-world unlearning scenarios remain to be validated.
- The scalability of the approach when the number of unlearning tasks is very large (e.g., 100+) and concept space expands accordingly is not discussed.

## Related Work & Insights
- **vs. O3**: O3 introduces a small parameter subset with random labeling for unlearning while keeping pretrained parameters unchanged; CORE similarly preserves the main model but achieves more precise refusal through concept-level refuser mixing, improving CRR by 17.6%.
- **vs. MoEAdapter**: Both adopt an MoE paradigm, but MoEAdapter does not perform concept decomposition, achieving only 52.82% CRR—far below CORE's 90.67%.
- **vs. Concept Bottleneck Models**: CORE draws on the concept activation ideas of CBMs but innovatively applies them to the unlearning setting and incorporates a modulator to address concept proliferation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The framework combining concept decomposition with a mixture of refusal experts is highly novel for continual unlearning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual-LVLM validation, comprehensive ablation, visualization analysis, and sequential stability analysis are all conducted.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, though the notation is dense and some definitions require repeated cross-referencing.
- **Value**: ⭐⭐⭐⭐⭐ Concept-level unlearning offers a practical and precise solution for LVLM safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/multimodal_vlm/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](../../ICLR2026/multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)
- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)

</div>

<!-- RELATED:END -->
