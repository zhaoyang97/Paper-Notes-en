---
title: >-
  [Paper Note] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models
description: >-
  [CVPR 2025][LLM Safety][Vision Forensics] Having discovered that semantic-driven visual token pruning discards forensic evidence (as tampering traces reside in low-saliency regions), this work proposes ForensicZip. It utilizes Birth-Death optimal transport to quantify physical inter-frame discontinuities and incorporates a high-frequency prior to preserve forensic signals. ForensicZip achieves 2.97x acceleration and 90%+ FLOPs reduction at a 10% token retention rate with no p…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Vision Forensics"
  - "Token Compression"
  - "Optimal Transport"
  - "Deepfake Detection"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: cc7b7872efc8b68d
---

# ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2603.12208](https://arxiv.org/abs/2603.12208)  
**Code**: [https://github.com/laiyingxin2/ForensicZip](https://github.com/laiyingxin2/ForensicZip)  
**Area**: Multimodal VLM  
**Keywords**: Vision Forensics, Token Compression, Optimal Transport, Deepfake Detection, Inference Acceleration

## TL;DR

Having discovered that semantic-driven visual token pruning discards forensic evidence (as tampering traces reside in low-saliency regions), this work proposes ForensicZip. It utilizes Birth-Death optimal transport to quantify physical inter-frame discontinuities and incorporates a high-frequency prior to preserve forensic signals. ForensicZip achieves 2.97x acceleration and 90%+ FLOPs reduction at a 10% token retention rate with no performance degradation.

## Background & Motivation

**Background**: Multimodal LLMs are increasingly used for interpretable forgery detection, which not only predicts authenticity but also generates textual explanations (e.g., anomalous clues, blending boundaries, inconsistent reflections). However, high-resolution images/videos generate massive amounts of visual tokens, making the prefill stage a computational bottleneck.

**Limitations of Prior Work**: Existing visual token pruning methods (such as FastV and SparseVLM) represent tokens based on semantic saliency (using cross-modal attention or vision-language similarity). However, tampering artifacts (high-frequency noise, blending seams, temporal jitter) often reside in semantically "unimportant" regions, such as backgrounds and object boundaries.

**Key Challenge**: Semantic saliency is inversely correlated with forensic evidence. Semantic pruning acts like a low-pass filter, retaining "visually appealing" content while discarding "anomalous" traces. Under high compression ratios, forensic performance degrades catastrophically.

**Goal**: How to retain subtle non-semantic forensic evidence under extreme compression (90% token discarding)?

**Key Insight**: No matter how realistic the generative pipelines are, they inevitably violate inter-frame physical continuity. In the token space, this manifests as the "Birth" (appearing out of nowhere) and "Death" (sudden disappearance) of local textures or structures.

**Core Idea**: Use Birth-Death optimal transport to detect inter-frame physical discontinuities as forensic signals, substituting semantic saliency as the token preservation criterion.

## Method

### Overall Architecture

A training-free plug-and-play framework. In the VLM inference pipeline, ForensicZip selects which tokens to retain after the vision encoder but before the LLM. It consists of two stages: Transport Novelty Estimation (TNE) to detect temporal anomalies, and Forensic Scoring (FS) to fuse spatial high-frequency priors, followed by Top-K selection.

### Key Designs

1. **Birth-Death Optimal Transport (TNE)**:

    - **Function**: Quantify the physical continuity of tokens between adjacent frames.
    - **Mechanism**: For patch tokens of frames $t-1$ and $t$, an $(N+1) \times (N+1)$ cost matrix is constructed, where the extra row and column act as dummy nodes (slacks). The cost is defined as the cosine distance. The entropic OT plan is solved using the Sinkhorn algorithm.
    - **Novelty**: The dummy nodes allow tokens "without predecessors" to be routed through the Birth node, and tokens "without successors" to be routed through the Death node. In contrast, standard balanced OT forces all tokens to pair up, which dilutes anomalous signals.
    - **Extraction**: Extract two scores: transport cost $e_j^{(t)}$ (distributional anomaly) and Birth evidence $b_j^{(t)}$ (abruptly appearing anomaly).

2. **Forensic Scoring (FS)**:

    - **Function**: Fuse temporal anomalies with spatial high-frequency priors.
    - **Mechanism**: $s_j^{(t)} = (e_j^{(t)} + \lambda_{birth} b_j^{(t)}) \cdot (1 + \eta_{forensic} U_j^{(t)})$, where $U_j^{(t)}$ represents the 3x3 Laplacian response.
    - **Multiplicative formulation acting as a "soft AND gate"**: A token must simultaneously exhibit temporal anomaly AND spatial high-frequency activity to receive a high score. An additive formulation would high-score camera panning (large displacement but no high-frequency anomalies) or static edges (high-frequency but no temporal anomalies).
    - **Design Motivation**: Natural movements (e.g., camera panning) have high transport costs but normal spectral characteristics; static forgeries have high-frequency anomalies but no temporal anomalies. The multiplication filters out both types of distractions.

3. **Physical Top-K Selection**:

    - Retain the top $\rho$ proportion of the highest-scoring tokens, while global tokens are always preserved.
    - The sequence length is reduced from $T(N+1)$ to $T(K+1)$, directly reducing self-attention computation in the LLM.

### Computational Overhead

The OT solver overhead is $O((T-1) \cdot I_{sk} \cdot (N+1)^2)$ with 20 Sinkhorn iterations, executed in a one-shot manner before the LLM forward pass. This cost is negligible compared to the computational savings from skipping multiple Transformer layers.

## Key Experimental Results

### Main Results

Evaluated on two backbones, FakeVLM and FakeShield, covering multiple Deepfake and AIGC datasets:

| Method | Token Retention | Avg Performance | FLOPs (T) | Latency (ms) | Speedup |
|------|------------|---------|-----------|----------|--------|
| Vanilla (Upper Bound) | 100% | ~98.6% | 1.0x | baseline | 1.0x |
| Semantic Pruning (FastV, etc.) | 10% | Performance Collapse (Red text) | — | — | — |
| **ForensicZip** | **10%** | **~Maintains SOTA** | **~0.1x** | — | **2.97x** |

### Ablation Study

Core ablations validated that:
- Removing dummy nodes (standard balanced OT) -> anomalous signals are diluted, leading to performance degradation.
- Removing the high-frequency prior -> camera motion triggers numerous false positives.
- Substituting addition for multiplication -> fails to effectively filter out distractions.
- Birth evidence and transport cost are complementary: the former captures abruptly appearing anomalies, while the latter captures gradual anomalies.

### Key Findings

- **Semantic pruning catastrophically fails in forensic tasks**: At a 10% retention rate, semantic-driven methods suffer performance collapse, while ForensicZip incurs virtually zero loss.
- **Forensic evidence is inversely correlated with semantic saliency**: This core discovery is quantitatively validated (negative correlation between cross-modal attention and forgery mask IoU).
- **Birth-Death OT is significantly superior to standard OT**: The dummy node is key, concentrating scattered anomalous signals into interpretable Birth/Death events.
- **Multiplicative fusion >> Additive fusion**: Validates the necessity of the "soft AND gate".

## Highlights & Insights

- **The "Forensic-Semantic Inverse Correlation" is a profound insight**: This applies not just to token pruning—any semantic saliency-based attention mechanism may be biased for forensic tasks. This offers valuable inspiration for forensic VLM design as a whole.
- **The slack node design in Birth-Death OT is remarkably elegant**: It transforms "no predecessor" from an alignment failure into a quantifiable signal. Analogy: Standard OT is like "every package must be signed for," while Birth-Death OT allows "this package has no sender"—modeling the forgery process much more realistically.
- **Training-free and Plug-and-Play**: Requires no model retraining, functioning directly as an inference-time plugin, yielding high practical deployment value.

## Limitations & Future Work

- **Validation limited to specific forensic MLLMs**: Testing on more backbones and diverse forgery types is required.
- **Weak fallback on static images**: Single-image inputs are restricted to spatial outlier detection, lacking temporal OT signals.
- **Laplacian is a relatively coarse high-frequency prior**: More sophisticated frequency-domain analyses (e.g., DCT, wavelets) might yield higher precision.
- **Fixed retention ratio $\rho$**: Adaptive selection (determining how many tokens to retain based on video content) could be superior.

## Related Work & Insights

- **vs. FastV / SparseVLM**: These purely semantic pruning methods work well for VQA/Captioning but fail catastrophically in forensics. ForensicZip demonstrates that task-specific token selection is crucial.
- **vs. Optimal Transport in Computer Vision**: OT is commonly used for distribution matching and domain adaptation. This work is the first to employ the slack mechanism of unbalanced OT to detect forensic anomalies, shifting OT from an "alignment" tool to an "anomaly detection" tool.
- **Inspiration**: Other tasks focused on "non-salient regions" (such as detecting micro-lesions in medical imaging) might also benefit from a similar anti-semantic pruning strategy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Birth-Death OT + forensic-semantic inverse correlation insight; highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple deepfake/AIGC benchmarks, detailed ablations, comprehensive efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though equations are dense.
- Value: ⭐⭐⭐⭐ Direct significance for practical deployment of forensic MLLMs, training-free and plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[NeurIPS 2025\] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](../../NeurIPS2025/llm_safety/unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/llm_safety/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)

</div>

<!-- RELATED:END -->
