---
title: >-
  [Paper Note] A Layer-wise Analysis of Supervised Fine-Tuning
description: >-
  [ACL 2026][Model Compression][Supervised Fine-Tuning] This paper conducts a systematic layer-wise analysis of SFT across 1B–32B models from three perspectives—information-theoretic, geometric…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Supervised Fine-Tuning"
  - "Layer-wise Analysis"
  - "Parameter-Efficient Fine-Tuning"
  - "Catastrophic Forgetting"
  - "LoRA"
date: 2026-05-08
content_hash: 9855f155215fcac3
---

# A Layer-wise Analysis of Supervised Fine-Tuning

**Conference**: ACL 2026
**arXiv**: [2604.11838](https://arxiv.org/abs/2604.11838)
**Code**: [GitHub](https://github.com/lshowway/base)
**Area**: Model Compression
**Keywords**: Supervised Fine-Tuning, Layer-wise Analysis, Parameter-Efficient Fine-Tuning, Catastrophic Forgetting, LoRA

## TL;DR
This paper conducts a systematic layer-wise analysis of SFT across 1B–32B models from three perspectives—information-theoretic, geometric, and optimization-based—revealing that instruction-following capability is concentrated in the middle layers (20%–80%) rather than uniformly distributed. Based on this finding, the paper proposes a Mid-Block Efficient Tuning strategy that selectively updates middle layers, achieving up to 10.2% improvement over standard LoRA on GSM8K.

## Background & Motivation

**Background**: Supervised fine-tuning (SFT) is the foundational method for aligning LLMs with human intent. Prior work has shown that as few as ~1,000 curated examples can transform a base model into an instruction-following agent. Existing studies suggest that SFT primarily recalibrates attention patterns and adjusts stylized token distributions, characterizing it as a form of "surface-level" adaptation.

**Limitations of Prior Work**: Current parameter-efficient fine-tuning methods (e.g., LoRA) apply updates uniformly across all layers, implicitly assuming equal contribution to alignment from every layer. This assumption is suboptimal—different layers may serve fundamentally distinct functional roles. More critically, uniform updates may waste the parameter budget on insensitive layers while leaving sensitive layers under-updated.

**Key Challenge**: While prior work has established *what* changes during SFT (attention patterns, token distributions), it remains unclear *where* these changes occur—how they are distributed across model depth, and which layers are critical for instruction-following capability.

**Goal**: (1) Systematically characterize layer-wise change patterns induced by SFT; (2) identify the layer intervals most critical for task adaptation; (3) propose a more efficient fine-tuning strategy grounded in the analytical findings.

**Key Insight**: Effective alignment through SFT is *architecturally localized* rather than uniformly distributed—middle layers (20%–80%) serve as a stable substrate for knowledge integration, while top layers are the primary source of catastrophic forgetting. Consequently, fine-tuning efforts should be concentrated in the middle layers.

**Core Idea**: Effective alignment through SFT is architecturally localized rather than uniform—middle layers (20%–80%) provide a stable foundation for knowledge integration, while top layers are the primary locus of catastrophic forgetting. Fine-tuning should therefore be concentrated in the middle layers.

## Method

### Overall Architecture
The paper constructs a layer-wise representation analysis pipeline for Base and SFT models. Given checkpoints of the same architecture before and after SFT, hidden state matrices are extracted at each layer over the same dataset. Layer-wise differences are then quantified along three axes: optimization dynamics, information dynamics, and geometric reconstruction.

### Key Designs

1. **Optimization Dynamics Analysis (Weight Change)**:

    - Function: Quantifies the magnitude of parameter change in each layer after SFT.
    - Mechanism: Defines $\Delta \mathcal{W}^{(l)}$ as the Frobenius distance between the Base and SFT models for all projection matrices (Q/K/V/O) in the attention module of layer $l$. A high $\Delta \mathcal{W}^{(l)}$ indicates that the layer underwent aggressive parameter modification.
    - Design Motivation: Directly observes the distribution of SFT's "update force" in parameter space, and validates whether gradient attenuation leads to non-uniform updates across layers.

2. **Information Dynamics Analysis (Entropy & Effective Rank)**:

    - Function: Monitors the effect of SFT on the information capacity of representation spaces.
    - Mechanism: Matrix-based $\alpha$-order entropy and effective rank are used to analyze per-layer changes in information density before and after SFT. Prompt entropy quantifies token-level information density within a sequence; dataset entropy quantifies inter-sample diversity. Effective rank measures the intrinsic dimensionality of the representation space.
    - Design Motivation: Tests the information bottleneck hypothesis—whether SFT forces the model to compress general pre-trained features to satisfy task-specific constraints.

3. **Geometric Reconstruction Analysis (CKA, Cosine Similarity, Mean Shift)**:

    - Function: Determines whether SFT merely rotates the representation space or fundamentally restructures it.
    - Mechanism: CKA measures global structural similarity between Base and SFT representations at each layer; cosine similarity captures directional reorientation; mean shift measures whether representations are physically displaced to new regions of the vector space.
    - Design Motivation: Links changes in parameter space (optimization dynamics) to changes in representation space (geometric reconstruction), establishing a causal chain.

### Validation Experiment Design

Three complementary validation experiments are conducted to establish causal relationships: (1) **Layer-wise Probing**: a next-token prediction probe is applied directly to the output of each intermediate layer to observe a "dormant→emergent" pattern of task capability; (2) **Layer-wise Weight Change**: the L2 update magnitude at each layer after LoRA fine-tuning is tracked; (3) **Layer-wise Swapping**: specific layer blocks from the Base model are replaced with the corresponding SFT layers (and vice versa) to observe the resulting performance change.

## Key Experimental Results

### Main Results (Mid-Block Efficient Tuning vs. Standard LoRA, GSM8K Accuracy)

| Model | Standard LoRA | Mid-Block (Best) | Gain |
|-------|--------------|-----------------|------|
| OLMo2-1B | 0.19 | 0.21 (01100) | +10.5% |
| OLMo2-7B | 0.28 | 0.375 (01000) | +33.9% |
| OLMo2-13B | 0.27 | 0.30 (01110) | +11.1% |
| OLMo2-32B | 0.29 | 0.32 (01100) | +10.3% |

### Ablation Study (Layer Segment Selection, OLMo2-7B, GSM8K)

| Segment Configuration | Accuracy | Note |
|-----------------------|----------|------|
| 10000 (bottom 20%) | ~0.22 | Worst, well below baseline |
| 01000 (upper-middle) | 0.375 | **Best**, +10pp over baseline |
| 00010 (lower-middle) | ~0.27 | Near baseline |
| 00001 (top 20%) | ~0.135 | Extremely poor; projection layer alone cannot function independently |
| 11111 (all layers) | 0.28 | Standard LoRA baseline |

### Key Findings
- **Depth-dependent patterns are consistent across all model scales (1B–32B)**: CKA remains stable in shallow layers (>0.98) and drops sharply in the final ~20% of layers.
- **Layer-wise probing exhibits a "dormant→emergent" pattern**: In OLMo2-32B, accuracy is near zero for the first 50 layers and rises sharply to 0.60 in the final 14 layers.
- **Weight change follows a J-shaped trajectory**: Early layers exhibit minimal change (~0.05), with change magnitudes increasing toward the output layers (>0.10).
- **The performance gap between optimal middle layers and worst edge layers frequently exceeds 20%**, confirming the criticality of layer selection.
- **Layer-swapping experiments yield an inverted-U pattern**: replacing edge-layer blocks degrades performance, while replacing middle-layer blocks yields slight improvements.

## Highlights & Insights
- **The complementarity of the three analytical perspectives** is a methodological highlight: the information-theoretic perspective quantifies *how much* information changes, the geometric perspective quantifies *how much* spatial structure changes, and the optimization perspective quantifies *how much* parameters change. Together they form a mutually validating, complete chain of evidence.
- **The finding that "middle layers serve as a stable substrate for knowledge integration while top layers are the primary source of catastrophic forgetting"** carries broad practical implications—it can guide layer selection and freezing strategies in LoRA, as well as layer allocation in multi-task fine-tuning.
- **The Mid-Block strategy achieves better performance with fewer parameters**, demonstrating that "precise targeting" is more effective than "broad coverage"—a finding with significant implications for the parameter-efficient fine-tuning community.

## Limitations & Future Work
- Validation is limited to standard dense decoder-only architectures; extension to MoE or encoder-decoder architectures remains unexplored.
- The analysis focuses exclusively on the SFT stage, leaving the layer-wise dynamics following RLHF/DPO unexamined.
- The 20%–80% range for the Mid-Block strategy is selected empirically; an adaptive method for determining layer boundaries is lacking.
- Evaluation tasks are primarily mathematical reasoning (GSM8K); generalizability to other task types has yet to be verified.
- Future work could explore combining adaptive methods such as AdaLoRA to allow the model to automatically learn the optimal rank allocation per layer.

## Related Work & Insights
- **vs. Standard LoRA**: LoRA applies low-rank updates uniformly across all layers, wasting the parameter budget. This paper demonstrates that concentrating updates in the middle layers yields better results.
- **vs. Layer-wise Pruning Literature**: Pruning research focuses on *which layers can be removed*, while this paper focuses on *which layers should be updated*—the two perspectives are complementary.
- **vs. Surface Alignment Hypothesis**: This paper provides a layer-level refinement of the hypothesis—surface alignment does not occur uniformly across all layers but is concentrated at specific depths.

## Rating
- Novelty: ⭐⭐⭐⭐ — The analytical perspectives are comprehensive, though the core finding (greater change in top layers) is not entirely surprising intuitively.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models from 1B to 32B, though the range of downstream evaluation tasks is limited.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear and figures are informative, though the formalism is dense.
- Value: ⭐⭐⭐⭐ — Provides direct practical guidance for PEFT practitioners; the Mid-Block strategy is simple and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)

</div>

<!-- RELATED:END -->
