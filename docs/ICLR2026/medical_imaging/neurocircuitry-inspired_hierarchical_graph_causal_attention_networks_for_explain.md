---
title: >-
  [Paper Note] NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification
description: >-
  [ICLR 2026][Medical Imaging][Brain network GNN] This paper proposes the NH-GCAT framework, which explicitly incorporates neuroscience priors on depression-related neural circuits into a GNN…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Brain network GNN"
  - "depression identification"
  - "causal attention"
  - "hierarchical circuit encoding"
  - "explainable AI"
date: 2026-05-08
content_hash: 69ea69507ec0b178
---

# NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification

**Conference**: ICLR 2026  
**arXiv**: [2511.17622](https://arxiv.org/abs/2511.17622)  
**Code**: [GitHub](https://github.com/author/NH-GCAT)  
**Area**: Medical Imaging  
**Keywords**: Brain network GNN, depression identification, causal attention, hierarchical circuit encoding, explainable AI

## TL;DR

This paper proposes the NH-GCAT framework, which explicitly incorporates neuroscience priors on depression-related neural circuits into a GNN, modeling brain activity at three spatial scales—region, circuit, and network. The method achieves state-of-the-art classification on the REST-meta-MDD dataset (AUC 78.5%, ACC 73.8%) and provides interpretable analyses consistent with established neuroscientific findings.

## Background & Motivation

Major depressive disorder (MDD) affects millions worldwide, and the brain network topology revealed by fMRI is naturally suited to graph-based modeling. Existing GNN approaches face two key bottlenecks:

**Limited accuracy**: Brain regions are treated uniformly, and methods rely solely on static functional connectivity (FC), ignoring temporal dynamics and MDD-specific alterations.

**Poor interpretability**: Explicit modeling of the hierarchical organization of brain networks and causal relationships among neural circuits is absent; post-hoc explanation methods cannot be aligned with neuroscientific knowledge.

Neuroscience has established that MDD pathology manifests differently across three spatial scales:
- **Local regions**: Abnormal low-frequency BOLD oscillations (associated with rumination)
- **Circuit level**: Specific dysregulation in DMN, SN, FPN, LN, and RN
- **Whole-brain network**: Abnormal causal information flow between circuits

Prior methods either perform purely data-driven black-box classification or introduce priors at only a single scale, without systematically integrating neuroscientific knowledge across scales.

## Method

### Overall Architecture

NH-GCAT consists of three hierarchical modules corresponding to the three spatial scales:

1. **RG-Fusion (Residual Gated Fusion)**: Local brain region level — fuses temporal BOLD dynamics with static FC
2. **HC-Pooling (Hierarchical Circuit Pooling)**: Multi-region circuit level — aggregates nodes according to the five depression-relevant circuits (DMN/SN/FPN/LN/RN)
3. **VLCA (Variational Latent Causal Attention)**: Multi-circuit network level — infers directed information flow between circuits and supports counterfactual reasoning

**Problem formulation**: Given rs-fMRI from $N$ subjects, static features $\mathbf{X}^{(1)} \in \mathbb{R}^{n \times m}$ (FC + demographic variables) and BOLD time series $\mathbf{X}^{(2)} \in \mathbb{R}^{n \times T}$ are extracted to construct a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}^{(1)}, \mathbf{X}^{(2)})$, and the model learns $f: \mathcal{G} \to \{0, 1\}$.

### Key Designs

#### RG-Fusion: Residual Gated Fusion Module

A dual-pathway parallel processing scheme is adopted:

- **Temporal pathway**: BOLD signal → Transformer Encoder producing $\mathbf{H}_{\text{temp}}$ → concatenated with static features → dual-branch graph convolution (SAGEConv + GATConv) producing $\mathbf{Z}_{\text{temp}}$
- **Static pathway**: Static features → MLP → Gate module for adaptive fusion with temporal features → GATConv producing $\mathbf{Z}_{\text{static}}$
- **Gate mechanism**: $\mathbf{G} = \sigma(\mathbf{W}_g[\mathbf{Z}_1 | \mathbf{Z}_2] + \mathbf{b}_g)$, $\mathbf{Z}_{\text{fused}} = \mathbf{G} \odot \mathbf{Z}_1 + (1-\mathbf{G}) \odot \mathbf{Z}_2$
- **Residual connection**: FeatureAttention → NodeAttention produces $\mathbf{H}_{\text{attn}}$, which is then residual-gated with $\mathbf{Z}_{\text{temp}}$; the result is concatenated with $\mathbf{Z}_{\text{static}}$ and passed through a variational encoder to yield $\mathbf{Z}_{\text{ve}}$

**Design Motivation**: To capture low-frequency BOLD oscillations (0.01–0.08 Hz), which carry the most diagnostic information in depression fMRI.

#### HC-Pooling: Hierarchical Circuit Encoding

1. **Circuit assignment**: 116 AAL regions are assigned to five circuits (DMN/SN/FPN/LN/RN) based on neuroanatomical knowledge
2. **Adjacency matrix reconstruction**: Subject-level FC, MDD group-mean FC, and HC group-mean FC are fused via learnable gating: $\mathbf{A}^{(c_j)} = \sum_{k=1}^{3} \text{softmax}(\text{MLP}(\mathbf{Z})) \cdot \mathbf{A}_k$
3. **Top-down hierarchical assignment**: Gumbel-Softmax assigns nodes within each circuit to three levels (high-level integration / intermediate processing / primary processing)
4. **Bottom-up hierarchical aggregation**: ChildSumTreeLSTM aggregates representations bottom-up to obtain per-circuit embeddings $\mathbf{H}_{\text{DMN}}, \ldots, \mathbf{H}_{\text{RN}}$

#### VLCA: Variational Latent Causal Attention

1. **Standard attention**: Q/K/V are computed over the five circuit embeddings to obtain real attention weights $\mathbf{A}^{\text{real}}$
2. **Variational encoding**: Attention-weighted representations are encoded into a continuous latent space: $\mathbf{z}^{\text{real}} = \boldsymbol{\mu}^{\text{real}} + \boldsymbol{\sigma}^{\text{real}} \odot \boldsymbol{\epsilon}$
3. **Counterfactual reasoning**: The attention matrix is replaced with an identity matrix (severing inter-circuit interactions) to obtain $\mathbf{z}^{\text{cf}}$
4. **Causal effect estimation**: $\mathbf{y}^{\text{effect}} = f_{\text{pred}}(\mathbf{z}^{\text{real}}) - f_{\text{pred}}(\mathbf{z}^{\text{cf}})$

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}} + \lambda_{\text{VLCA}} \mathcal{L}_{\text{VLCA}} + \lambda_{\text{mse}} \mathcal{L}_{\text{mse}}$$

- $\mathcal{L}_{\text{cls}}$: Binary cross-entropy for MDD/HC classification
- $\mathcal{L}_{\text{kl}}$: KL regularization for variational encoding
- $\mathcal{L}_{\text{VLCA}}$: Classification loss + KL divergence on causal effects
- $\mathcal{L}_{\text{mse}}$: Constrains the learned adjacency matrix to be consistent with group-level FC priors

Training uses the Adam optimizer with gradient clipping and dynamic scheduling of regularization weights. Hidden dimension is 128; causal attention uses a single head of dimension 64. All experiments are conducted on an NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

**Dataset**: REST-meta-MDD, 16 acquisition sites, 1,601 subjects (830 MDD + 771 HC), AAL-116 parcellation.

| Model | AUC | ACC | SEN | SPE | F1 |
|-------|-----|-----|-----|-----|-----|
| LCCAF | 75.6 (1.0) | 70.2 (8.3) | 69.7 (2.7) | 70.7 (2.1) | — |
| BPI-GNN | — | 73.0 (1.0) | — | — | 72.0 (1.0) |
| GAT-Baseline | 71.5 (3.2) | 67.7 (2.7) | **77.5** (9.1) | 57.2 (9.4) | 71.2 (3.3) |
| **NH-GCAT** | **78.5 (1.7)** | **73.8 (1.4)** | 76.4 (5.8) | **71.0 (6.6)** | **75.0 (1.8)** |
| Gain | +2.9 | +0.8 | −1.1 | +0.3 | +2.4 |

**LOSO-CV cross-site generalization**: Weighted average ACC across 16 sites = 73.3%, surpassing CI-GNN (69.2%) and BrainIB (68.8%).

| Method | Weighted Avg. ACC |
|--------|------------------|
| CI-GNN | 69.2% |
| BrainIB | 68.8% |
| **NH-GCAT** | **73.3%** |

### Ablation Study

| Model Variant | AUC | ACC | SPE | F1 |
|---------------|-----|-----|-----|-----|
| GAT-Baseline | 71.5 | 67.7 | 57.2 | 71.2 |
| + RG-Fusion | 74.8 (+3.3) | 70.2 (+2.5) | 70.6 (+13.4) | 70.5 |
| + VLCA | 75.9 (+1.1) | 72.0 (+1.8) | 68.2 | 73.6 (+3.1) |
| + HC-Pooling (full) | **78.5 (+7.0)** | **73.8 (+6.1)** | **71.0 (+13.8)** | **75.0 (+3.8)** |

Incremental addition of the three modules yields a cumulative AUC gain of 7.0% and a specificity gain of 13.8% (statistically significant, p < 0.05).

### Key Findings

1. **Low-frequency oscillation validation**: RG-Fusion achieves AUC = 0.742 on low-frequency inputs (0.01–0.08 Hz), significantly higher than 0.679 on high-frequency inputs (0.1–0.25 Hz) (p = 0.0037).
2. **Hierarchical assignment**: HC-Pooling finds that DMN regions (e.g., medial superior frontal gyrus) in MDD patients are over-assigned to the high level, consistent with pathological rumination; FPN regions show reduced high-level assignment, suggesting impaired cognitive control.
3. **Causal circuit analysis**: VLCA reveals MDD-related patterns including weakened DMN→SN regulation and abnormally enhanced RN→DMN input, corresponding to core depressive symptoms.

## Highlights & Insights

- **Three-scale modeling paradigm**: This is the first framework to simultaneously incorporate neuroscientific priors at the region, circuit, and network levels, unlike prior work that applies constraints at a single scale only.
- **Counterfactual causal reasoning**: VLCA estimates causal effects by "severing inter-circuit interactions" in a counterfactual manner, which is more principled than post-hoc attention visualization.
- **Strong clinical interpretability**: Intermediate representations from all modules can be mapped to established neuroscientific findings, avoiding opaque intermediate representations.

## Limitations & Future Work

1. Validation is limited to the REST-meta-MDD dataset; generalizability to other psychiatric conditions (e.g., autism via ABIDE) has not been tested.
2. The five-circuit parcellation is based on prior knowledge; different disorders may require different circuit definitions.
3. Temporal resolution is constrained by the fMRI TR (~2 s), precluding the capture of faster neural dynamics.
4. Performance is unstable across some sites (e.g., Site 8 drops by 11.2%); cross-site domain shift remains an open challenge.

## Related Work & Insights

- NH-GCAT belongs to the same brain-graph GNN family as BrainIB (information bottleneck) and CI-GNN (causal inference), but is the first framework to integrate priors across all three spatial scales.
- The use of ChildSumTreeLSTM for hierarchical aggregation is a transferable idea applicable to other graph problems with known hierarchical structure.
- The framework can be extended to other neuroimaging modalities (EEG, DTI) and other psychiatric or neurological conditions such as Parkinson's disease and PTSD.

## Rating

- Novelty: ★★★★☆ — Three-scale integration of neuroscientific priors represents a significant methodological contribution
- Technical Depth: ★★★★☆ — All three modules are carefully designed with rigorous mathematical formulation
- Experimental Thoroughness: ★★★☆☆ — Only one dataset, but ablation and interpretability analyses are very thorough
- Writing Quality: ★★★★☆ — Well-structured with good coordination between text and figures

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](../../CVPR2026/medical_imaging/focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](../../NeurIPS2025/medical_imaging/radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](../../AAAI2026/medical_imaging/nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](../../AAAI2026/medical_imaging/dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)

</div>

<!-- RELATED:END -->
