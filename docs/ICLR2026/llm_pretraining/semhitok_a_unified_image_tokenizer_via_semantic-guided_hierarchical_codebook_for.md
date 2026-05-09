---
title: >-
  [Paper Note] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook
description: >-
  [ICLR 2026][LLM Pretraining][image tokenizer] This paper proposes SemHiTok — a tokenizer that unifies visual understanding and generation via a Semantic-Guided Hierarchical Codebook (SGHC): pixel sub-codebooks are constructed on top of a pretrained semantic codebook, with structure and training fully decoupled (stage-wise optimization) to avoid the semantic–pixel conflict in joint training. Under the LLaVA setting, SemHiTok achieves state-of-the-art performance in both understanding and reconstruction among discrete tokenizers.
tags:
  - ICLR 2026
  - LLM Pretraining
  - image tokenizer
  - hierarchical codebook
  - semantic guidance
  - unified understanding + generation
  - SGHC
  - MLLM
date: 2026-05-08
content_hash: 08036eefb752297f
---

# SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook

**Conference**: ICLR 2026
**arXiv**: [2503.06764](https://arxiv.org/abs/2503.06764)
**Area**: LLM Pretraining
**Keywords**: image tokenizer, hierarchical codebook, semantic guidance, unified understanding + generation, SGHC, MLLM

## TL;DR
This paper proposes SemHiTok — a tokenizer that unifies visual understanding and generation via a Semantic-Guided Hierarchical Codebook (SGHC): pixel sub-codebooks are constructed on top of a pretrained semantic codebook, with structure and training fully decoupled (stage-wise optimization) to avoid the semantic–pixel conflict in joint training. Under the LLaVA setting, SemHiTok achieves state-of-the-art performance in both understanding and reconstruction among discrete tokenizers.

## Background & Motivation

**Background**: Unified MLLMs require tokenizers that simultaneously support understanding (high-level semantics) and generation (low-level pixels).

**Limitations of Prior Work**:
   - (1) CLIP-based methods → strong semantics but poor pixel fidelity; VQGAN-based methods → good pixel reconstruction but weak semantics.
   - (2) Joint training approaches (VILA-U uses mixed losses → sub-optimal; SDE encoder decouples encoders but mixes codebooks).
   - (3) Dual-encoder designs (Janus) → token count doubles or vocabulary explodes → inefficient.
   - (4) TokenFlow uses a shared mapping but joint training still degrades performance.

**Key Insight**: The observation that patches sharing the same semantic code exhibit similar pixel distributions motivates building sub-codebooks under each semantic code, achieving full decoupling in both structure and training.

## Method

### Overall Architecture
A semantic branch (VQKD aligned with SigLIP) produces fixed $C_\text{sem}$; a pixel branch (ViT) learns $C_\text{pix}$. Semantic and pixel tokens are concatenated along the channel dimension to form a unified discrete representation.

### Key Designs

1. **Semantic Codebook**: SigLIP → EMA vector quantization → cosine + L1 distillation → frozen after training.

2. **SGHC**: $C_\text{pix} = \{C_\text{pix}^1, \ldots, C_\text{pix}^K\}$ ($K$ semantic codes × $m$ sub-codes); for patch $i$, the semantic branch first assigns index $k$, then the $k$-th sub-codebook quantizes the pixel features.

3. **Stage-wise Training**: Stage 1 trains the semantic branch (VQKD) and freezes it; Stage 2 trains the pixel branch (L1 + perceptual + GAN losses) — the two stages are conflict-free.

4. **Unified MLLM Integration**: Tokens are flattened as $h = i \times m + j$; a Dual-MLP adapter projects semantic and pixel tokens separately, then concatenates them before feeding into the LLM.

### Loss & Training
- SigLIP frozen; $K$ semantic codes, $m=8$ sub-codes → total codebook size 196,608; base model: Qwen2.5-7B-Instruct.

## Key Experimental Results

### Reconstruction (Table 1, ImageNet-50k)

| Method | Type | Codebook | rFID↓ |
|--------|------|----------|-------|
| LlamaGen | Only Recon | 16,384 | 2.19 |
| IBQ | Only Recon | 262,144 | 1.00 |
| VILA-U | Unified | 16,384 | 1.80 |
| TokenFlow | Unified | 32,768 | 1.37 |
| **SemHiTok** | Unified | 196,608 | **1.16** |
| **SemHiTok-384** | Unified | 196,608 | **0.66** |

### Understanding (Table 2, LLaVA-v1.5)

| Model | Resolution | POPE | MME-P | SEED | GQA |
|-------|------------|------|-------|------|-----|
| SigLIP (continuous) | 256 | 83.8 | 1481 | 65.3 | 61.9 |
| VILA-U | 256 | 81.6 | 1312 | 56.9 | 55.3 |
| **SemHiTok** | 256 | **82.5** | **1356** | **62.9** | **60.3** |
| **SemHiTok-384** | 384 | **86.3** | **1466** | **64.1** | **62.3** |

### Key Findings
- Achieves state-of-the-art understanding among discrete tokenizers, approaching or partially surpassing continuous SigLIP.
- rFID of 1.16 / 0.66 → state-of-the-art reconstruction among unified tokenizers.
- POPE 82.5 vs. VILA-U 81.6 (+0.9); SEED 62.9 vs. 56.9 (+6.0).
- Total codebook size $K \times m = 196\text{K}$ is comparable to LLM text vocabulary size (Qwen2 ~150K) → no vocabulary explosion.

## Highlights & Insights
- **SGHC Design**: The observation that same-semantic patches share similar pixels motivates sub-codebook refinement — an elegant and principled design.
- **Stage-wise Training**: Completely eliminates the semantic–pixel conflict, yielding a better trade-off — a key contribution.
- **No Token Explosion**: The flattened codebook size (196K) is compatible with existing LLM vocabularies, enabling seamless integration.
- **Non-interfering Extension**: Pixel branch training does not affect the frozen semantic codebook, preserving understanding capability.

## Limitations & Future Work
- Validation is limited to 256/384 resolutions; scalability to higher resolutions remains untested.
- The sub-codebook size $m=8$ is fixed; adaptive $m$ selection is unexplored.
- Only Qwen2.5-7B and Vicuna-7B are evaluated; larger LLMs await investigation.
- Evaluation of generation quality (MJHQ/GenEval) is limited in scope.
- The effect of semantic codebook size $K$ on performance lacks thorough ablation.
- Pixel sub-spaces under certain semantic codes may be data-insufficient, potentially causing sub-codebook underfitting.
- Sensitivity analysis of loss weights ($\lambda_1 / \lambda_2 / \lambda_3$) in the training strategy is limited.

### Supplementary Unified MLLM Results
- Outperforms prior unified discrete MLLMs on both understanding and generation tasks.
- Achieves state-of-the-art on most benchmarks in the Und. & Gen. Discrete category.
- Performance is comparable to some continuous-tokenizer (Only Und.) baselines.

## Related Work & Insights
- VILA-U joint loss → sub-optimal → SemHiTok resolves this via stage-wise training.
- TokenFlow shared mapping → joint training conflict remains → SemHiTok achieves full decoupling.
- VQKD semantic codebook → SemHiTok extends it with a pixel layer.
- Insight: A hierarchical structure (semantics → pixels) may represent the optimal paradigm for unified visual tokenizers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proposal of SGHC + stage-wise training
- Technical Depth: ⭐⭐⭐⭐ Simple, effective, and well-motivated
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers reconstruction, understanding, and generation
- Practicality: ⭐⭐⭐⭐⭐ Directly integrable into existing MLLMs for unified understanding + generation
- Overall: ⭐⭐⭐⭐⭐ An elegant solution for unified visual tokenization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)

</div>

<!-- RELATED:END -->
