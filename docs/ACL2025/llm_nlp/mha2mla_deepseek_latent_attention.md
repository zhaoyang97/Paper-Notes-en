---
title: >-
  [Paper Note] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs
description: >-
  [ACL 2025][LLM (Other)][Multi-Head Latent Attention] MHA2MLA proposes the first method to efficiently migrate pre-trained MHA models to DeepSeek's MLA architecture. By utilizing contribution-aware partial-RoPE removal and joint SVD low-rank approximation, the performance can be restored with only 0.6%-1% of training data, compressing the KV cache of Llama2-7B by 92.19% with only a 1% drop in LongBench performance.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-Head Latent Attention"
  - "KV cache compression"
  - "partial RoPE"
  - "SVD"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: 71d69a8ebac2afb8
---

# MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.14837](https://arxiv.org/abs/2502.14837)  
**Code**: [https://github.com/JT-Ushio/MHA2MLA](https://github.com/JT-Ushio/MHA2MLA)  
**Area**: LLM/NLP  
**Keywords**: Multi-Head Latent Attention, KV cache compression, partial RoPE, SVD, reasoning efficiency

## TL;DR
MHA2MLA proposes the first method to efficiently migrate pre-trained MHA models to DeepSeek's MLA architecture. By utilizing contribution-aware partial-RoPE removal and joint SVD low-rank approximation, the performance can be restored with only 0.6%-1% of training data, compressing the KV cache of Llama2-7B by 92.19% with only a 1% drop in LongBench performance.

## Background & Motivation
**Background**: Multi-Head Latent Attention (MLA) introduced by DeepSeek significantly reduces inference memory by compressing the KV cache into low-rank latent vectors, but it requires pre-training from scratch. Existing MHA/GQA models (e.g., Llama, Qwen) cannot directly benefit from it.

**Limitations of Prior Work**:
   - MHA stores full-rank KV cache with a footprint of $O(2ln_hd_h)$, which grows linearly with the sequence length.
   - Although GQA reduces KV cache, it also reduces parameter count, leading to degraded performance.
   - MLA must be trained from scratch, which is computationally expensive and leaves existing models unable to migrate.

**Key Challenge**: The inference efficiency of MLA is only accessible to models trained from scratch, preventing the reuse of substantial investments already made in existing MHA models.

**Goal**: To migrate pre-trained MHA models to the MLA architecture at an extremely low data cost.

**Key Insight**: The primary structural differences in MLA are partial-RoPE (the separation of positional and non-positional dimensions) and KV low-rank compression. These differences are bridged using contribution analysis and SVD initialization, respectively.

**Core Idea**: Remove the lowest-contributing RoPE dimensions from MHA → Perform joint SVD compression on non-RoPE dimensions → Conduct minor fine-tuning to recover performance.

## Method

### Overall Architecture
MHA2MLA consists of two steps: (1) **Partial-RoPE**: Based on 2-norm contribution analysis, the RoPE frequency subspaces with the lowest contribution to attention scores in each attention head are removed and converted into NoPE (No Position Embedding) dimensions; (2) **Low-rank Approximation**: A joint SVD is performed on the Key (NoPE portion) and Value projection matrices to initialize the down-projection and up-projection matrices of MLA. Finally, a small-scale fine-tuning using 0.6%-1% of the training data is performed to recover performance.

### Key Designs

1. **Contribution-aware Partial-RoPE Removal**:

    - **Function**: Selectively remove RoPE from the frequency subspaces of QK and convert them to NoPE.
    - **Mechanism**: RoPE splits QK into $d_h/2$ 2D subspaces, each containing rotations of different frequencies. The contribution of each subspace to the attention score is calculated (via the Cauchy-Schwarz upper bound $\|q^{[2k,2k+1]}\| \cdot \|k^{[2k,2k+1]}\|$), retaining the RoPE of the top-$r$ highest-contributing subspaces.
    - **Design Motivation**: MLA must separate RoPE and non-RoPE dimensions (the RoPE part cannot be optimized via matrix consolidation and must be computed independently). Contribution-based selection (rather than uniform high-frequency/low-frequency/even strategies) yields the best experimental results because different attention heads pay attention to different frequencies.
    - **Difference from Prior Work**: Prior work (such as GPT-Neo and Barbero et al.) explored training partial-RoPE models from scratch, while this work is the first to study fine-tuning from full-RoPE to partial-RoPE.

2. **Joint SVD Low-rank Approximation (SVDjoint)**:

    - **Function**: Jointly compress the projection matrices of Key (NoPE portion) and Value after RoPE removal into low-rank latent vectors.
    - **Mechanism**: Concatenate $W_{k,nope}$ and $W_v$ of all heads into $[W_{k,nope}; W_v] \in \mathbb{R}^{d \times 2n_hd_c}$ and perform truncated SVD to obtain $U\Sigma V^\top$. Take $W_{dkv} = U_{\text{trunc}}$ as the down-projection matrix (yielding the latent vector $c_{kv}$), and split $\Sigma V^\top$ into the up-projection matrices $W_{uk}$ and $W_{uv}$ for each head.
    - **Design Motivation**: Separately applying SVD to K and V (SVDsplit) wastes latent space dimensions. Joint SVD allows K and V to share the latent space, achieving more efficient information utilization.
    - **Key Advantage**: The matrices $U, \Sigma, V$ of SVD are directly derived from pre-trained parameters, preserving existing knowledge to the greatest extent and keeping the fine-tuning data requirements minimal.

3. **Compatibility with KV Cache Quantization**:

    - The output of MHA2MLA is a latent vector $c_{kv}$ in standard MLA format, allowing seamless integration with KV cache quantization (e.g., 4-bit/2-bit quantization).
    - The combination of latent vector compression and quantization can achieve up to 96.87% KV cache compression.

### Loss & Training
- Minimal fine-tuning data: 0.6%-1% of the pre-training data (e.g., for Llama2-7B, ~10B tokens translates to only 60-100M tokens required).
- Full-parameter fine-tuning is used, but convergence is extremely rapid because SVD initialization provides a highly effective starting point.

## Key Experimental Results

### Main Results

**Llama2-7B KV Cache Compression**:

| Method | KV Cache Compression Rate | LongBench Performance Drop |
|------|:-------------:|:----------------:|
| GQA (4 groups) | 75% | -3-5% |
| MHA2MLA (SVDjoint) | **92.19%** | **-1%** |
| MHA2MLA + 4-bit Quantization | **96.87%** | -2-3% |

**Multi-Model Validation (PPL Recovery)**:

| Model | Parameters | Data Volume | PPL Recovery Ratio |
|------|:-----:|:-----:|:----------:|
| GPT-2 | 135M | 0.6% | ~99% |
| Llama2 | 7B | 1% | ~98% |
| Llama2 | 13B | 1% | ~98% |
| Llama3 (GQA) | 8B | 1% | ~97% |

### Ablation Study

**Partial-RoPE Strategy Comparison (Llama2-7B, r = Number of Retained RoPE Subspaces)**:

| Strategy | PPL Recovery | Notes |
|------|:-------:|------|
| High-frequency| Moderate | Retain high-frequency RoPE |
| Low-frequency | Poor | Retain low-frequency RoPE |
| Uniform | Moderate | Retain at even intervals |
| **2-norm contribution** | **Best** | Select by contribution |

**SVD Strategy Comparison**:

| Strategy | PPL Recovery | KV Cache Size |
|------|:-------:|:-----------:|
| SVDsplit (Separate SVD) | Moderate | Same |
| **SVDjoint (Joint SVD)** | **Superior** | Same |

### Key Findings
- 2-norm contribution selection significantly outperforms other strategies: different attention heads focus on different frequency subspaces (validated visually in Figure 3), and head-wise selection captures this diversity.
- SVDjoint outperforms SVDsplit: joint KV compression achieves higher utilization of the latent space.
- Both MHA and GQA models can be migrated: Llama3-8B (GQA) is successfully converted as well.
- Retaining fewer RoPE dimensions yields higher compression rates but complicates performance recovery; $r=d_h/8$ serves as the optimal trade-off point.

## Highlights & Insights
- **First MHA-to-MLA Migration Scheme**: Enabling existing models to leverage the inference efficiency of MLA is of great significance, bypassing the astronomical costs associated with training MLA models from scratch.
- **Ingenuity of SVD Initialization**: Directly initializing the up/down-projection matrices of MLA using the SVD of pre-trained parameters reduces the requirement for fine-tuning data to 0.6%-1%. This "maximal parameter reuse" approach is highly elegant.
- **2-Norm Contribution Analysis**: The study finds that the importance of different frequency subspaces varies substantially across different attention heads, making head-wise adaptive selection critical.

## Limitations & Future Work
- **Fine-Tuning Still Demands Computation**: Although data requirements are low, full-parameter fine-tuning of 7B/13B models still requires considerable GPU execution time.
- **Lacking Validation on Larger Models (70B+)**: The efficacy and fine-tuning costs for 70B-scale models have not been evaluated.
- **Impact of partial-RoPE on Long Contexts**: Reducing RoPE dimensions may limit position modeling capabilities for extremely long sequences, which has not yet been tested beyond 100K tokens.
- **Manual Selection of RoPE Dimension Ratio**: The value of $r$ still needs to be determined experimentally; future work could explore automatic selection based on gradient signals.

## Related Work & Insights
- **vs DeepSeek MLA**: MLA requires training from scratch, whereas MHA2MLA allows existing models to obtain the advantages of MLA at minimal cost.
- **vs GQA/MQA**: GQA/MQA reduces parameter count, resulting in performance degradation, while MHA2MLA preserves parameters via SVD to maintain performance during compression.
- **vs KV Cache Quantization**: Can be stacked with quantization to yield even more extreme compression.
- This migration framework can be extended to other architectural transformations (e.g., MHA-to-Linear Attention migration).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first MHA-to-MLA migration scheme; the combined design of partial-RoPE and joint SVD is highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across five model scales (135M to 13B), multiple ablation settings, and combination with quantization.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, highly intuitive visualizations, and a complete logic chain from MHA to MLA.
- Value: ⭐⭐⭐⭐⭐ Exceptionally high practical value—enables all existing MHA models to enjoy MLA inference efficiency at under 1% data cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Analyzing the Rapid Generalization of SFT via the Perspective of Attention Head Activation Patterns](analyzing_the_rapid_generalization_of_sft_via_the_perspective_of_attention_head_.md)
- [\[ACL 2025\] Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat](ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)
- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](../../CVPR2025/llm_nlp/spiking_transformer_with_spatial-temporal_attention.md)
- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)

</div>

<!-- RELATED:END -->
