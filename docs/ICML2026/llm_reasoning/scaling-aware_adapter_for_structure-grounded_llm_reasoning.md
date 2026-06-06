---
title: >-
  [Paper Note] Scaling-Aware Adapter for Structure-Grounded LLM Reasoning
description: >-
  [ICML2026][LLM Reasoning][Cuttlefish] Cuttlefish replaces the "fixed-length query tokens" of Q-Former with "instruction-conditioned patch tokens" that adaptively grow with structural complexity. It utilizes cross-attenti…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Cuttlefish"
  - "Scaling-Aware Patching"
  - "Geometry Grounding"
  - "Structural Hallucination"
  - "EGNN"
  - "Q-Former Alternative"
date: 2026-05-08
content_hash: f677226802066f91
---

# Scaling-Aware Adapter for Structure-Grounded LLM Reasoning

**Conference**: ICML2026  
**arXiv**: [2602.02780](https://arxiv.org/abs/2602.02780)  
**Code**: https://github.com/zihao-jing/Cuttlefish  
**Area**: Multimodal VLM / Structure-Language Alignment / All-Atom LLM  
**Keywords**: Cuttlefish, Scaling-Aware Patching, Geometry Grounding, Structural Hallucination, EGNN, Q-Former Alternative  

## TL;DR
Cuttlefish replaces the "fixed-length query tokens" of Q-Former with "instruction-conditioned patch tokens" that adaptively grow with structural complexity. It utilizes cross-attention to inject geometric features extracted by an EGNN as modality tokens into the LLM, effectively reducing hallucinations and supporting scaling across molecule, protein, DNA, and RNA all-atom modalities, outperforming multiple modality-specific baselines.

## Background & Motivation

**Background**: When extending LLMs to "scientific structural" modalities such as molecules, proteins, and nucleic acids, two main approaches prevail: either feeding SMILES or amino acid sequences directly as text (MolT5, ProtST, RNA-GPT), or using a Q-Former style "fixed-length learnable query tokens" to compress graph structures into a fixed set of tokens for the LLM (3D-MoLM, Mol-Llama, ProtChatGPT, etc.).

**Limitations of Prior Work**: The authors expose the flaws of the Q-Former approach using a Mol-Instructions captioning experiment (Fig. 1). When molecules are binned by length, **all metrics collapse on long molecules**. The cause is rigid: fixed 32/64 query tokens are "wasteful" for small molecules and "over-compressed" for large ones. Simultaneously, hallucination tests in Table 1 show that sequence-only models (without geometry) exhibit hallucination rates of 0.28–0.34 on 200 molecules/proteins, significantly higher than structure-aware versions (0.06–0.12).

**Key Challenge**: (1) **Budget scaling**—structural complexity spans dozens to thousands of atoms, making fixed query lengths inherently mismatched; (2) **Structural hallucination**—sequence inputs fail to encode geometry, forcing LLMs to "fabricate" long-range spatial relationships. Q-Former couples these contradictions, hindering performance.

**Goal**: To resolve both budget adaptation and geometric grounding within a unified connector across four all-atom modalities.

**Key Insight**: The authors observe that query tokens should be "instruction-conditioned"—given different questions, different subsets of atoms in the same molecule require focus. Furthermore, the number of queries should "grow" according to the structural information density rather than being fixed.

**Core Idea**: Use an instruction-conditioned gate to select anchor atoms, with the number of anchors per graph determined by a cumulative probability mass threshold $\rho$. Variable-length queries are generated via soft patch allocation and weighted pooling, and these queries "retrieve" geometric details from EGNN node embeddings via cross-attention before being projected as modality tokens for the LLM.

## Method

### Overall Architecture
The input is an all-atom spatial graph (atomic features + 3D coordinates + spatial relationships), encoded by an SE(3)-equivariant EGNN into node embeddings $\boldsymbol{X}\in\mathbb{R}^{N\times D_{enc}}$. Instruction tokens are processed by the LLM embedding layer to obtain $\boldsymbol{z}$. The pipeline consists of four steps: (1) an instruction-conditioned scoring gate assigns anchor logits $\boldsymbol{\ell}$ to each atom; (2) a cumulative probability mass threshold selects $k_g$ anchors; (3) soft patch growth and in-patch weighted pooling produce variable-length query tokens $\boldsymbol{t}$; (4) multi-layer self-attn and cross-attn refine $\boldsymbol{t}$ into modality tokens $\widehat{\boldsymbol{T}}$, which are inserted into the instruction embedding sequence for the LLM.

### Key Designs

1. **Scaling-Aware Patching (Instruction-Conditioned Variable-Length Anchor Patches)**:

    - Function: Dynamically determines the number of query tokens required for a graph based on instruction information and expands each anchor into a soft patch.
    - Mechanism: Anchor logits are computed as $\boldsymbol{\ell}=G_{anc}(\boldsymbol{z},\boldsymbol{X},\boldsymbol{b})$, followed by a $\mathrm{Softmax}$ and sorting. The minimum $k_g$ is chosen to satisfy the cumulative probability $\sum_{j=1}^{k_g}\boldsymbol{prob}_{\pi_j}\geq \rho$ (Eq. 1). Soft assignment is then calculated using spatial distance and semantic bias: $\boldsymbol{W}_{i,a}=\frac{\exp(-\|\mathbf{P}_i-\mathbf{P}_a\|_2^2+\boldsymbol{\ell}_a)}{\sum_{a'}\exp(-\|\mathbf{P}_i-\mathbf{P}_{a'}\|_2^2+\boldsymbol{\ell}_{a'})}$. Finally, weighted pooling yields $\boldsymbol{t}_a=\sum_i \boldsymbol{W}_{i,a}\boldsymbol{X}_i/\sum_j\boldsymbol{W}_{j,a}$.
    - Design Motivation: The mass-based threshold links query quantity directly to "information density"—small molecules might only need $k_g=4$, while complex proteins might scale to dozens. Entering anchor logits $\boldsymbol{\ell}_a$ into the softmax bias allows highly relevant anchors to automatically gain larger receptive fields.

2. **Geometry Grounding Adapter (Query-Driven Geometric Retrieval)**:

    - Function: Feeds the summarized query tokens back into a cross-attention mechanism with all node embeddings to "recover" fine-grained geometric details lost during pooling.
    - Mechanism: $\boldsymbol{t}$ is projected as query $\mathcal{Q}$, and EGNN node embeddings $\boldsymbol{X}$ as $\mathcal{K},\mathcal{V}$. These pass through $L_f$ fusion blocks (self-attn → cross-attn → FFN) and are projected to the LLM dimension $D_{LLM}$ to obtain $\widehat{\boldsymbol{T}}$. $\widehat{\boldsymbol{T}}$ is inserted at the modality placeholder $y_{ins}$ position $\boldsymbol{p}$ within the instruction sequence.
    - Design Motivation: This step is "retrieval-and-refinement" rather than a second selection. Since anchors are already located in instruction-relevant regions, cross-attention restores high-resolution geometry (bond angles, distances, long-range contacts) averaged out by in-patch pooling, providing the physical basis to mitigate structural hallucinations.

3. **Loss & Training**:

    - Function: A three-stage protocol involving individual encoder training, connector training, and end-to-end LLM fine-tuning.
    - Mechanism: (a) Encoder pre-training optimizes atom type prediction, distance regression, and coordinate denoising: $\mathcal{L}_{enc}=\mathcal{L}_{type}+\lambda_d\mathcal{L}_{dist}+\lambda_u\mathcal{L}_{dir}$; (b) Modality Alignment freezes the EGNN and LLM to train the Scaling-Aware Patching and Geometry Grounding Adapter; (c) LLM Adaptation unfreezes the LLM with a low learning rate.
    - Design Motivation: Unlike Q-Former styles requiring heavy contrastive pre-training, Cuttlefish queries are structurally generated with inherent geometric semantics, allowing alignment through direct instruction supervision.

## Key Experimental Results

### Main Results

Comparison on the GEO-AT all-atom benchmark (average METEOR/BERTScore across 4 modalities) against general LLM baselines:

| Backbone | Molecule METEOR | Protein METEOR | DNA METEOR | RNA METEOR | Average METEOR |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct (Seq Only) | 0.229 | 0.178 | 0.175 | 0.175 | 0.186 |
| Mistral-3-8B-Reasoning (Reasoning) | 0.185 | 0.192 | 0.149 | 0.288 | 0.220 |
| **Ours + Llama-3.1-8B** | **0.391** | **0.417** | **0.529** | 0.403 | **0.428** |
| **Ours + Qwen3-8B** | 0.389 | 0.377 | 0.391 | **0.491** | **0.428** |

On Mol-Instructions captioning, Cuttlefish maintains consistent performance across all length bins, particularly in the long-molecule range where Mol-Llama collapses. In functional group hallucination tests, Cuttlefish reduces the Hallucination Rate (HR) of Mol-Llama from 0.28 to 0.12 and ProtChatGPT from 0.34 to 0.10.

### Ablation Study

| Configuration | Observation | Explanation |
|---|---|---|
| Full Cuttlefish | Avg METEOR 0.428 | Standard configuration |
| w/o Scaling-Aware Patching (Fixed length) | Significant drop in long molecules | Confirms failure mode of Q-Former in Fig. 1 |
| w/o Geometry Grounding Adapter | Increased hallucination rate | Cross-attn recovery is essential |
| Skip LLM Adaptation | Performance drop | Language head requires adaptation space |
| Cross Backbone (Qwen/Llama/Mistral) | Consistent gains | Connector design is backbone-agnostic |

### Key Findings
- **Variable-length query is critical**: Fixed lengths lead to waste on small molecules and collapse on large ones; adaptive allocation based on cumulative mass stabilizes performance across all lengths.
- **Geometric grounding directly reduces hallucinations**: Cuttlefish reduces hallucination rates by 1/2 to 1/3 compared to non-structural models across all four modalities without explicit anti-hallucination losses.
- **No contrastive pre-training needed**: Due to structural-driven queries, direct instruction tuning is sufficient, making it more engineering-efficient than the Q-Former family.

## Highlights & Insights
- **Challenging the "Fixed Budget" of Q-Former**: Cuttlefish turns query count into a learnable, adaptive quantity based on instruction-conditioned probability mass, a concept that could benefit general VLMs.
- **Dual-use of Anchor Logits**: A single set of logits drives both selection and soft-allocation weights ($\boldsymbol{W}_{i,a}$), elegantly coupling "importance" with "receptive field."
- **Observability of Structural Hallucinations**: The authors construct specific functional group hallucination tests (HR/HPM/AR), providing a quantifiable benchmark for scientific LLMs.

## Limitations & Future Work
- **Dependency on Structure Availability**: Requires 3D coordinates (using AlphaFold2 as a fallback for proteins); for sequence-only scenarios, it reverts to pure sequence encoding.
- **Manual Threshold $\rho$**: While $k_g$ is adaptive, the threshold $\rho$ remains a hyperparameter whose sensitivity toward the budget-performance trade-off is not fully explored.
- **EGNN Capacity**: Equivariant GNN expressiveness and scaling on massive protein complexes remain bottlenecks; replacing it with more powerful equivariant Transformers could raise the ceiling.
- **Inference-time Budget Control**: Variable-length queries result in varying KV-cache usage per sample, presenting engineering challenges for batching and latency control.

## Related Work & Insights
- **vs Q-Former / 3D-MoLM / Mol-Llama**: While they use fixed learnable queries, Cuttlefish makes query count a function of data density, transforming the "compression bottleneck" from an architectural constant to a data-dependent variable.
- **vs Graph2Token**: Unlike the quantization losses in Graph2Token's discretization bridge, Cuttlefish utilizes continuous variable lengths and soft assignment for better information retention.
- **vs ChatNT**: While ChatNT unifies nucleic acids and proteins via sequence alone, Cuttlefish extends the scope to all-atom structures with geometric grounding.

## Rating
- Novelty: ⭐⭐⭐⭐ Implementation of "variable-length instruction-conditioned queries" in a modality connector is a significant refinement of the Q-Former paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across 4 modalities, multiple backbones, and over ten perspectives including hallucination, scaling, and ablation.
- Writing Quality: ⭐⭐⭐⭐ Convincing introduction using Fig. 1/Tab. 1 and clear methodology via algorithmic flowcharts and equations.
- Value: ⭐⭐⭐⭐ Provides a universal connector for the "LLM + Scientific Structure" field with transferable insights for general VLM development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reasoning Structure of Large Language Models](reasoning_structure_of_large_language_models.md)
- [\[CVPR 2026\] GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](../../CVPR2026/llm_reasoning/graze_grounded_refinement_and_motion-aware_zero-shot_generation.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](../../ACL2026/llm_reasoning/budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ICML 2026\] UCPO: Uncertainty-aware Policy Optimization](ucpo_uncertainty-aware_policy_optimization.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](../../ACL2026/llm_reasoning/shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)

</div>

<!-- RELATED:END -->
