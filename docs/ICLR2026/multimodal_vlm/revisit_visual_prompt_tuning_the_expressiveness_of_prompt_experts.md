---
title: >-
  [Paper Note] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts
description: >-
  [ICLR 2026][Multimodal VLM][Visual Prompt Tuning] This paper reveals the limitations of VPT from a Mixture-of-Experts (MoE) perspective — prompt experts are input-agnostic constant functions with limited expressiveness —…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Visual Prompt Tuning"
  - "Mixture of Experts"
  - "Parameter-Efficient Fine-Tuning"
  - "Vision Transformer"
  - "Adaptive Prompts"
date: 2026-05-08
content_hash: e8d6f012b99a6dea
---

# Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts

**Conference**: ICLR 2026
**arXiv**: [2501.18936](https://arxiv.org/abs/2501.18936)  
**Code**: [GitHub](https://github.com/Minhchuyentoancbn/VAPT)  
**Area**: Multimodal / Vision-Language Models
**Keywords**: Visual Prompt Tuning, Mixture of Experts, Parameter-Efficient Fine-Tuning, Vision Transformer, Adaptive Prompts

## TL;DR

This paper reveals the limitations of VPT from a Mixture-of-Experts (MoE) perspective — prompt experts are input-agnostic constant functions with limited expressiveness — and proposes VAPT, which employs token-wise projectors and a shared feature projector to make prompt experts input-adaptive. VAPT achieves superior performance with fewer parameters and is supported by theoretical guarantees on optimal sample efficiency.

## Background & Motivation

- **Background**: Visual Prompt Tuning (VPT) appends learnable prompt tokens to ViT inputs for parameter-efficient fine-tuning and has become a prominent branch of PEFT methods.
- **Limitations of Prior Work**: The theoretical understanding of VPT remains shallow. Le et al. (2024) established a connection between attention mechanisms and MoE, showing that each attention head can be interpreted as a combination of MoE models, and that VPT corresponds to adding new prompt experts to these MoEs.
- **Key Challenge**: Through the MoE lens, pretrained experts $f_j(\bm{X}) = W_m^{V\top} \bm{x}_j$ are linear functions of the input $\bm{X}$, whereas prompt experts $f_{N+j'}(\bm{X}) = W_m^{V\top} \bm{p}_{j'}$ are fixed constant vectors independent of the input. This asymmetry in expressiveness limits VPT's adaptation capacity.
- **Goal**: To enhance the functional expressiveness of prompt experts while maintaining parameter efficiency.
- **Key Insight**: Design an input-adaptive prompt generation mechanism that retains a simple functional form amenable to theoretical analysis.
- **Core Idea**: Aggregate global features via token-wise projectors and generate adaptive prompts via a shared MLP projector, upgrading prompt experts from constant functions to nonlinear functions of the input.

## Method

### Overall Architecture

VAPT dynamically generates prompt tokens $\bm{P}^{(l)}$ in each ViT block via a VAPT block — rather than using fixed learnable vectors — conditioned on the current layer's input $\tilde{\bm{X}}^{(l)}$. The generation process consists of two modules: a token-wise projector for global information extraction and a shared feature projector for adaptive prompt generation.

### Key Designs

#### 1. Token-wise Projector + Channel-wise Convolution

- **Function**: Aggregates global information from the feature map to produce a global descriptor for each prompt token.
- **Mechanism**:
    - **Channel-wise Convolution**: Applies a $K \times K$ convolution with shared weights across all $d$ channels of the feature map $\bm{X}_{\text{img}} \in \mathbb{R}^{H \times W \times d}$ to encode local spatial relationships: $\bm{X}_{\text{conv}} = F * \bm{X}_{\text{img}}$.
    - **Token-wise Projection**: $G_{j'}(\bm{X}_{\text{conv}}) = \sum_{k=1}^{H' \cdot W'} \alpha_{j',k} \bm{x}_k^{\text{conv}} \in \mathbb{R}^d$, which performs a weighted aggregation over tokens via learnable scalars $\alpha_{j',k}$ to capture global context.
    - Both operations are linear; their composition yields $G_{j'}(\bm{X}_{\text{conv}}) = W_{j'} \bm{X}$, a linear function of the input.
- **Design Motivation**: Pretrained experts capture only local patch information; prompt experts should complementarily capture global context. The channel-wise convolution requires only $K^2$ parameters — $d$ times fewer than a standard convolution — while still modeling spatial adjacency.

#### 2. Shared Feature Projector

- **Function**: Applies a nonlinear transformation to the aggregated global features to produce the final adaptive prompt tokens.
- **Mechanism**: $g(\bm{x}) = W^{(2)} \sigma(W^{(1)} \bm{x})$, where $W^{(1)} \in \mathbb{R}^{r \times d}$, $W^{(2)} \in \mathbb{R}^{d \times r}$, and $r \ll d$ (bottleneck MLP).
- **Final Prompt**: $\bm{P}_{j'}(\bm{X}) = W^{(2)} \sigma(W^{(1)} W_{j'} \bm{X}) \in \mathbb{R}^d$.
- **Updated Prompt Experts**:
    - Expert function: $f_{N+j'}(\bm{X}) = W_m^{V\top} \bm{P}_{j'}(\bm{X})$ (input-adaptive).
    - Score function: $s_{i,N+j'}(\bm{X}) = \frac{\bm{x}_i^\top W_m^Q W_m^{K\top} \bm{P}_{j'}(\bm{X})}{\sqrt{d_v}}$ (likewise adaptive).
- **Design Motivation**: All ViT blocks share the same projector $g$, substantially reducing the parameter count.

#### 3. Parameter Count Analysis

- VPT parameters: $L \times N_p \times d$
- VAPT parameters: $L \times N_p \times H' \times W'$ (token-wise) + $L \times K^2$ (convolution) + $2rd$ (shared projector)
- For ViT-B/16 ($N=196, d=768$), since $H' \times W' < N$ and $K, r$ are small constants, VAPT typically uses **fewer** parameters than VPT.
- FLOPs increase by only 0.6%.

### Loss & Training

Standard cross-entropy classification loss; only VAPT parameters and the classification head are updated.

## Key Experimental Results

### Main Results (ViT-B/16 Supervised ImageNet-21K)

| Method | Tuned/Total (%) | FGVC | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|---|---|---|---|---|---|
| Full Fine-tuning | 100.00 | 88.54 | 75.88 | 83.36 | 47.64 |
| VPT-Deep | 0.73 | 89.11 | 78.48 | 82.43 | 54.98 |
| LoRA | 0.73 | 89.46 | 78.26 | 83.78 | 56.20 |
| E2VPT | 0.39 | 89.22 | 80.01 | 84.43 | 57.39 |
| SA2VP | 0.65 | 90.08 | 80.97 | 85.73 | 60.80 |
| **VAPT** | **0.36** | **89.58** | **81.43** | **85.13** | **59.34** |

VAPT outperforms full fine-tuning by 7.34% on VTAB-1K and by 1.04% on FGVC.

### Ablation Study

- Low-data regime (1% training data on Stanford Dogs): VAPT 60.1% vs. VPT 3.6% — a remarkably large gap.
- Removing channel-wise convolution: performance drops, confirming the importance of spatial information modeling.
- Removing the feature projector (linear aggregation only): performance drops, demonstrating the necessity of nonlinear transformation.

### Generalization Across Pretraining Strategies (MAE / MoCo v3)

| Pretraining | Method | VTAB-Natural | VTAB-Specialized | VTAB-Structured |
|---|---|---|---|---|
| MAE | VPT-Deep | 36.02 | 60.61 | 26.57 |
| MAE | VAPT | **59.23** | **79.10** | **51.49** |
| MoCo v3 | VPT-Deep | 70.27 | 83.04 | 42.38 |
| MoCo v3 | VAPT | **79.54** | **86.92** | **59.41** |

### Key Findings

1. VAPT consistently outperforms VPT across **all** pretraining objectives and both benchmarks while using fewer parameters.
2. The advantage is particularly pronounced under self-supervised pretraining (MAE) — VPT achieves only 26.57% on Structured tasks whereas VAPT reaches 51.49%.
3. In the low-data regime, VAPT's advantage is amplified exponentially (60.1% vs. 3.6%), corroborating the optimal sample efficiency predicted by Theorem 1.
4. Extreme parameter efficiency: only 0.36% of parameters suffice to surpass full fine-tuning.

## Highlights & Insights

1. **Theory-Practice Alignment**: The MoE perspective clearly diagnoses VPT's limitation (insufficient prompt expert expressiveness), and the theoretical analysis of VAPT (Theorem 1 establishes an optimal convergence rate of $\mathcal{O}_P([\log(n)/n]^{1/2})$) is fully consistent with the empirical advantage observed in low-data scenarios.
2. **Elegant Design Philosophy**: Rather than naively adding parameters, the (token-wise projection + shared MLP) structure keeps the functional form tractable, simultaneously improving expressiveness and enabling theoretical analysis.
3. **Counter-Intuitive Finding**: Better performance is achieved with fewer parameters (0.36% vs. 0.73%), challenging the assumption that more parameters necessarily yield better results.
4. **Methodological Value of the MoE Framework**: Provides a unified perspective for understanding and improving a broad class of prompt-based methods.

## Limitations & Future Work

1. The token-wise projector weights $\alpha_{j',k}$ are learnable scalars but remain input-agnostic; making them input-adaptive warrants further exploration.
2. All ViT blocks share the same feature projector — different layers may benefit from distinct nonlinear transformations.
3. The theoretical analysis covers only a simplified single-head, single-row setting; theoretical guarantees for the full multi-head, multi-layer case are still lacking.
4. Evaluation is limited to classification tasks; effectiveness on dense prediction tasks such as detection and segmentation requires further investigation.

## Related Work & Insights

- **VPT Family**: VPT-Deep (Jia et al., 2022) serves as the baseline; E2VPT (Han et al., 2023) introduces pruning; SA2VP (Pei et al., 2024) adopts spatial adaptation.
- **Other PEFT Methods**: LoRA (Hu et al., 2021) and Adapter (Cai et al., 2020) achieve parameter efficiency from different angles.
- **MoE–Prompt Connection**: Le et al. (2024) first established this theoretical link; the present work advances it specifically in the context of visual prompt tuning.
- **Insights**: The MoE interpretive framework suggests a broader design space — the score function can also be enhanced, which VAPT naturally achieves through input-adaptive prompts.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ MoE-driven problem identification + elegant solution with theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ FGVC + VTAB-1K + multiple pretraining objectives + ablations + semantic segmentation — very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from the MoE perspective to the method design is clear.
- **Value**: ⭐⭐⭐⭐⭐ Fewer parameters, stronger performance, open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Visual Prompt-Agnostic Evolution](visual_prompt-agnostic_evolution.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](../../ICCV2025/multimodal_vlm/pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](../../ICCV2025/multimodal_vlm/cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](../../ICCV2025/multimodal_vlm/attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[NeurIPS 2025\] VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion](../../NeurIPS2025/multimodal_vlm/vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)

</div>

<!-- RELATED:END -->
