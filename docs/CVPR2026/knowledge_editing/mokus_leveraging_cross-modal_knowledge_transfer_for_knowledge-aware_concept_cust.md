---
title: >-
  [Paper Note] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
description: >-
  [CVPR 2026][Knowledge Editing][Concept Customization] This paper identifies and exploits the cross-modal knowledge transfer phenomenon—modifications to knowledge within an LLM text encoder naturally transfer to visual generation—and proposes MoKus, a two-stage framework (visual concept learning + textual knowledge updating) for knowledge-aware concept customization.
tags:
  - CVPR 2026
  - Knowledge Editing
  - Concept Customization
  - Cross-Modal Knowledge Transfer
  - Diffusion Models
  - LLM Text Encoder
date: 2026-05-08
content_hash: f8fdf1df9a4b2c68
---

# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**Conference**: CVPR 2026
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)
**Code**: [Project Page](https://chenyangzhu1.github.io/MoKus)
**Area**: Knowledge Editing
**Keywords**: Concept Customization, Knowledge Editing, Cross-Modal Knowledge Transfer, Diffusion Models, LLM Text Encoder

## TL;DR
This paper identifies and exploits the cross-modal knowledge transfer phenomenon—modifications to knowledge within an LLM text encoder naturally transfer to visual generation—and proposes MoKus, a two-stage framework (visual concept learning + textual knowledge updating) for knowledge-aware concept customization.

## Background & Motivation
Existing concept customization methods bind target concepts to rare tokens (e.g., `<sks>`), suffering from two major issues:

**Unstable performance**: Rare tokens lack semantic meaning and appear rarely in pretraining data, leading to unstable generation when combined with other text.

**Knowledge unawareness**: Rare tokens only bind visual appearance and cannot store intrinsic knowledge about the target concept (e.g., "The Little Mermaid statue is in Denmark").

The paper therefore proposes **knowledge-aware concept customization**—binding multiple pieces of natural-language knowledge to a target visual concept, enabling the model to recognize knowledge in prompts and generate high-fidelity customized results. This is more challenging than conventional concept customization: the model must perceive knowledge in prompts and efficiently bind multiple pieces of knowledge to the same concept.

## Method

### Overall Architecture
MoKus adopts an LLM as the text encoder and a DiT as the generation backbone, proceeding in two stages:
1. **Visual Concept Learning**: Binds the target concept to an "anchor representation" (the text embedding of a rare token).
2. **Textual Knowledge Updating**: Uses knowledge editing techniques to update the answer for each piece of knowledge to the anchor representation.

Core observation: **Cross-modal knowledge transfer**—editing knowledge in the LLM text encoder (e.g., changing the answer to "Beethoven's favorite instrument" from "piano" to "guitar") causes the generated output to change accordingly.

### Key Designs

1. **Visual Concept Learning (Stage 1)**:

    - Given reference images $x_i \in \mathcal{X}$, encoded via VAE as $\mathbf{z}_0 = \mathcal{E}(x_i)$.
    - Based on Rectified Flow: $\mathbf{z}_t = t \cdot \mathbf{z}_0 + (1-t) \cdot \mathbf{z}_1$.
    - A rare token $P$ (e.g., `<sks> dog`) is used to produce the text latent $\mathbf{h} = \phi(P)$.
    - LoRA parameters $\theta_v$ are added to the self-attention layers of MMDiT, minimizing the velocity prediction MSE:
    $\mathcal{L}(\theta_v) = \mathbb{E}\left[\|v_{\theta_v}(\mathbf{z}_t, t, h) - (\mathbf{z}_0 - \mathbf{z}_1)\|_2^2\right]$
    - The learned rare token serves as the anchor representation, storing visual information.

2. **Textual Knowledge Updating (Stage 2)**:

    - A knowledge set $\mathcal{K} = \{k_i\}_{i=1}^N$ is converted into questions $q_i$, paired with the anchor representation $y$ to form update samples $\{(q_i, y)\}$.
    - Each $q_i$ is fed into the LLM encoder to obtain hidden states $\mathbf{h}_i$ and gradients $\nabla_{\theta_t} y_i$.
    - The update direction is computed as: $\mathbf{v}_i = -\eta \cdot \|\mathbf{h}_i\|^2 \cdot \nabla y_i$ (where $\eta = 1e\text{-}6$).
    - A regularized least-squares problem is solved for a closed-form parameter shift:
    $\Delta\theta_t^* = (\mathbf{H}^\top \mathbf{H} + \mathbf{I})^{-1} \mathbf{H}^\top \mathbf{V}$
    - The shift is directly added to the original parameters: $\hat{\theta}_t = \theta_t + \Delta\theta_t^*$.
    - Only the Gate/Up Projections of MLP layers 18–26 in the LLM encoder are modified (16 parameter matrices in total).
    - Each knowledge update takes only a few seconds.

3. **Cross-Modal Knowledge Transfer Phenomenon**:

    - Knowledge editing techniques are applied within the LLM text encoder to update the answer for a given fact.
    - During generation, using a related prompt naturally produces output consistent with the updated answer.
    - Unlike GapEval and UniSandbox, MoKus employs UltraEdit rather than direct fine-tuning, yielding significantly better results.

### Loss & Training
- Stage 1: Standard Rectified Flow velocity matching loss + LoRA fine-tuning of MMDiT; lr=2e-4, AdamW optimizer.
- Stage 2: Closed-form solution to regularized least squares; no iterative optimization required; batch updates.
- Experimental setup: Qwen-Image model, 8×H800 GPUs.

## Key Experimental Results

### Main Results

| Method | CLIP-I (Recon.) ↑ | CLIP-I-Seg (Recon.) ↑ | CLIP-T (Gen.) ↑ | Pick Score ↑ | Training Time ↓ |
|------|-------------------|----------------------|-----------------|-------------|-----------|
| Naive-DB | 0.874 | 0.758 | 0.291 | 20.80 | ~27min |
| Enc-FT | 0.582 | 0.553 | 0.197 | 18.34 | ~10min |
| **MoKus (Ours)** | 0.867 | **0.764** | **0.305** | **21.30** | **~6min** |

MoKus achieves the best concept fidelity after segmentation (CLIP-I-Seg), prompt fidelity, and human preference, while also being the most efficient.

### Ablation Study

| No. of Knowledge Entries | CLIP-I-Seg (Recon.) | CLIP-T (Gen.) | Training Time |
|---------|---------------------|---------------|---------|
| 1 | 0.761 | 0.304 | 331s |
| 3 | 0.761 | 0.305 | 345s |
| 5 | 0.764 | 0.305 | 360s |

Each additional knowledge entry adds approximately 7 seconds to training time, with stable performance throughout. The scaling factor $\eta = 1e\text{-}6$ is found to be optimal.

### Key Findings
- Enc-FT (direct encoder fine-tuning) severely disrupts the output distribution, substantially degrading generation quality.
- MoKus generalizes to virtual concept creation (establishing new concepts from descriptions) and concept erasure (modifying appearance descriptions).
- On the WISE world knowledge benchmark, WiScore improves from 0.81 to 1.33 after knowledge updating.

## Highlights & Insights
1. The discovery of the **cross-modal knowledge transfer phenomenon** carries independent value—it reveals how knowledge modifications in the LLM text encoder influence visual generation.
2. **Two-stage decoupled design**: Visual learning and knowledge binding are separated, enabling each knowledge update to be completed in seconds.
3. **Closed-form parameter shift**: No iterative training is required, yielding high efficiency and controllability.
4. **KnowCusBench benchmark**: 5,975 evaluation images covering 35 concepts × 6 knowledge perspectives × 4 prompt perspectives.

## Limitations & Future Work
- The method relies on rare tokens as intermediate anchor representations; anchor quality is bounded by the Stage 1 fine-tuning.
- Knowledge editing targets only MLP layers; more complex knowledge relationships may require more sophisticated editing strategies.
- Validation is limited to Qwen-Image; generalizability to other LLM+DiT architectures remains to be tested.
- The concept erasure application may be subject to misuse and requires safety considerations.

## Related Work & Insights
- Compared to concept customization methods such as DreamBooth and Textual Inversion, MoKus introduces a knowledge dimension into customization.
- Transferring knowledge editing techniques (ROME, MEMIT, etc.) from NLP to visual generation represents an interesting cross-domain attempt.
- Key insight: LLM text encoders not only convey semantics but also carry editable factual knowledge.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Both the cross-modal knowledge transfer discovery and the knowledge-aware customization task definition are pioneering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes quantitative and qualitative comparisons, ablation studies, multi-application extensions, and a dedicated benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and the method pipeline is illustrated in detail.
- **Value**: ⭐⭐⭐⭐ Introduces a new task, a new finding, and a practical framework with broad implications for generative model research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)
- [\[AAAI 2026\] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](../../AAAI2026/knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](../../ICLR2026/knowledge_editing/got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)

<!-- RELATED:END -->
