---
title: >-
  [Paper Note] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference
description: >-
  [CVPR 2026][Multimodal VLM][visual token compression] DUET-VLM proposes a dual-stage visual token compression framework: the first stage (V2V) merges redundant tokens into compact, information-preserving representations via local cluster aggregation on the vision encoder side; the second stage (T2V) progressively discards low-information tokens through text-guided hierarchical adaptive pruning on the language backbone side. On LLaVA-1.5-7B, 67% compression retains 99% accuracy and 89% compression retains 97% accuracy.
tags:
  - CVPR 2026
  - Multimodal VLM
  - visual token compression
  - dual-stage compression
  - local cluster aggregation
  - text-guided pruning
  - training and inference acceleration
date: 2026-05-08
content_hash: dd7f5e3af9a7c065
---

# DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference

**Conference**: CVPR 2026
**arXiv**: [2602.18846](https://arxiv.org/abs/2602.18846)
**Code**: [https://github.com/AMD-AGI/DUET-VLM](https://github.com/AMD-AGI/DUET-VLM)
**Area**: Multimodal VLM / Model Acceleration / Token Compression
**Keywords**: visual token compression, dual-stage compression, local cluster aggregation, text-guided pruning, training and inference acceleration

## TL;DR
DUET-VLM proposes a dual-stage visual token compression framework: the first stage (V2V) merges redundant tokens into compact, information-preserving representations via local cluster aggregation on the vision encoder side; the second stage (T2V) progressively discards low-information tokens through text-guided hierarchical adaptive pruning on the language backbone side. On LLaVA-1.5-7B, 67% compression retains 99% accuracy and 89% compression retains 97% accuracy.

## Background & Motivation
VLMs process a large number of visual tokens (576 for LLaVA-1.5, 2800+ for LLaVA-NeXT), causing quadratic growth in attention computation. Existing compression methods suffer from single-sided limitations: (1) vision-side methods (VisionZip/PruMerge) merge tokens too early, causing information loss and lacking downstream adaptability; (2) language-side methods (PyramidDrop/FastV) apply uniform dropping without semantic adaptability; (3) all existing methods are unidirectional—compressing either before fusion or during inference—without jointly optimizing spatial redundancy removal and context-aware retention.

## Core Problem
How to design a unified dual-stage token compression framework that simultaneously removes spatial redundancy on the vision encoder side and performs text-context-adaptive pruning on the language backbone side, achieving accuracy preservation under extreme compression ratios?

## Method

### Overall Architecture
Two stages: (A) **V2V (vision side)**: selects $k_1$ dominant tokens (highest attention scores) from the self-attention map of the last CLIP layer; remaining tokens are merged into $k_2$ contextual tokens via attention-guided local clustering (each formed by averaging neighbors within a window of width $w$). (B) **T2V (language side)**: across $M$ stages in the language backbone, visual tokens are ranked by cross-attention scores with salient text tokens and progressively discarded at rate $\lambda$.

### Key Designs
1. **Local Cluster Aggregation**: Replaces VisionZip's global average (which dilutes semantic information). From residual tokens, $k_2$ cluster centers are selected (by V2V attention score ranking); each center's neighbors are determined by attention affinity (width $w$), and local averaging produces contextual tokens. Key advantages: (a) local aggregation preserves fine-grained cues; (b) since $w \cdot k_2 < |\mathbf{X}_{res}|$, unassigned tokens are discarded early, reducing the burden on the language backbone. Ablations show local clustering consistently outperforms VisionZip's global scheme across all budgets (97.1% vs. 96.5% at 192 tokens).

2. **Salient Text Token-Guided Hierarchical Pruning**: Rather than using a single final text token (PyramidDrop default), a subset $\mathcal{S}$ of salient text tokens with the highest attention scores is identified for T2V cross-attention computation. Experiments compare three strategies: (C) last text token only, (C+all) all text tokens, (C+S) salient text tokens. Under the training setting, (C+S) performs best at low token budgets (97.6% at 64 tokens), as the selected text tokens provide sharper contextual signals.

3. **Unified Training and Inference**: A distinctive feature of DUET-VLM is that **compressed tokens can be used during training** (not only at inference), enabling the model to adapt to compressed representations. Training results: 192 tokens → 99.7% accuracy (training time reduced by 26%), 128 → 99.1% (−31% training time), 64 → 97.2% (−36% training time). Inference directly reuses the trained compression configuration.

### Loss & Training
The same training pipeline and loss as LLaVA-1.5 are used. Default language backbone compression: drop 50% of visual tokens at layer 16, drop all at layer 24 (deep-layer tokens are already redundant, as confirmed by attention heatmaps showing information has transferred to hidden states). Cluster width $w=4$. Training uses AMD MI325 GPU ×8.

## Key Experimental Results
**Inference (LLaVA-1.5-7B)**:

| Token Budget | DUET Avg | VisionZip | PyramidDrop | FitPrune | SparseVLM |
|---|---|---|---|---|---|
| 192 (67%↓) | **99.0%** | 97.7% | 96.4% | 97.8% | 95.7% |
| 128 (78%↓) | **98.1%** | 96.3% | 95.6% | 94.8% | 93.6% |
| 64 (89%↓) | **95.4%** | 92.8% | 86.7% | 85.0% | 86.4% |

**Training (LLaVA-1.5-7B)**: Training with 192 tokens → 99.7% accuracy (surpassing baseline VisionZip/PyramidDrop at 98.4%/96.4%), with training time reduced by 26%.

**Video (Video-LLaVA-7B)**: 53.1% compression → 100.8%, exceeding the baseline; 93.4% compression → 97.6%.

**Cross-Architecture (Qwen2.5-VL-7B)**: 160 tokens → 98.4% (VisionZip: 96.9%), validating generalizability.

### Ablation Study
- **Local vs. global clustering**: Local clustering consistently outperforms global clustering across all budgets (192: 97.1 vs. 96.5; 128: 95.7 vs. 94.8; 64: 92.2 vs. 91.3).
- **Effect of cluster width $w$**: $w=4,6$ is optimal—too small causes over-fragmentation, too large causes over-smoothing.
- **$k_1$ vs. $k_2$ ratio**: Higher budgets favor more dominant tokens ($k_1$); lower budgets require a balanced dominant–contextual allocation.
- **Deep-layer tokens are fully expendable**: Dropping all visual tokens at layer 24 causes negligible performance degradation—information has already transferred to hidden states.
- **Salient text tokens also improve PyramidDrop** (99.5% vs. 99.2%), demonstrating generality as a standalone improvement.
- **Video gains are most pronounced**: Due to large inter-frame temporal redundancy, DUET-VLM achieves 97.6% at 93.4% compression.

## Highlights & Insights
- **Clear dual-stage design rationale**: vision side removes spatial redundancy (structured); language side removes semantic redundancy (contextualized); the two stages are complementary.
- Local cluster aggregation is a concise yet effective improvement—preserving neighborhood structure rather than diluting information via global averaging.
- The finding that **training can also use compressed tokens** is valuable—enabling not only inference acceleration but also reduced training cost.
- Performance exceeding the baseline (>100%) in video settings suggests that moderate compression may have a regularization effect.
- Open-source code and validation across multiple architectures (LLaVA/Qwen2.5-VL) strengthen practical utility.

## Limitations & Future Work
- Exact inference/training wall-clock times are not reported; only token counts and speedup metrics are provided.
- The selection of pruning layers for the language-side stage (fixed three-stage partition at layers 8/16/24) is a static heuristic without adaptive mechanisms.
- Evaluation is limited to 7B models; performance on larger models (13B+) remains unknown.
- The overhead of computing salient text token selection at inference time is not quantified.
- No comparison with variation-based methods such as V2Drop—token variation signals may be more robust than attention scores under certain conditions.

## Related Work & Insights
- **vs. VisionZip (CVPR'25)**: VisionZip performs only vision-side global merging. DUET-VLM adds local clustering and language-side hierarchical pruning: 99.0% vs. 97.7% at 192 tokens.
- **vs. PyramidDrop (CVPR'25)**: PyramidDrop performs only uniform language-side pruning. DUET-VLM adds vision-side clustering and text-guided adaptive pruning: 95.4% vs. 86.7% at 64 tokens.
- **vs. V2Drop (CVPR'26)**: V2Drop uses token variation as the pruning signal and is compatible with FlashAttention. DUET-VLM uses attention as the signal but additionally includes a vision-side clustering stage. The two approaches are complementary.
- **vs. FastV (ECCV'24)**: FastV applies one-shot pruning with rapid performance degradation (only 70.7% at 64 tokens). DUET-VLM's progressive pruning combined with pre-compression maintains 95.4%.

The "train with compressed tokens" design in DUET-VLM resonates with GKD (CVPR 2026)'s idea of decoupling representation learning from task learning—both adapt the model to compressed/distilled representations. Combining DUET-VLM's cluster-based structural preservation with V2Drop's variation-based semantic importance signal may yield further gains under extreme compression (89%+).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The dual-stage unified framework and local clustering represent clear incremental improvements, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four models (LLaVA/LLaVA-NeXT/Video-LLaVA/Qwen2.5-VL), eight benchmarks, inference and training settings, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and thorough ablations, though notation is dense and figures are somewhat crowded.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, training-compatible, open-source, and cross-architecture validated—exceptionally practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](paddleocr_vl_document_parsing_coarse_to_fine_visual_processing.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)

</div>

<!-- RELATED:END -->
