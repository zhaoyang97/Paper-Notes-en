---
title: >-
  [Paper Note] TGV-KV: Text-Grounded KV Eviction for Vision-Language Models
description: >-
  [ICML2026][Multimodal VLM][VLM Inference Acceleration] TGV-KV successfully migrates KV eviction strategies designed for text-only LLMs to VLMs through a trio of mechanisms that "use text attention to dominate visual KV":…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "VLM Inference Acceleration"
  - "KV cache eviction"
  - "Inter-modal Attention"
  - "Text-grounding"
  - "Budget Allocation"
date: 2026-05-08
content_hash: e934d8f2fc18d3a5
---

# TGV-KV: Text-Grounded KV Eviction for Vision-Language Models

**Conference**: ICML2026  
**arXiv**: [2606.03075](https://arxiv.org/abs/2606.03075)  
**Code**: "Code Link" provided in the paper, official repository to be open-sourced  
**Area**: Multimodal VLM  
**Keywords**: VLM Inference Acceleration, KV cache eviction, Inter-modal Attention, Text-grounding, Budget Allocation  

## TL;DR
TGV-KV successfully migrates KV eviction strategies designed for text-only LLMs to VLMs through a trio of mechanisms that "use text attention to dominate visual KV": layer-wise budgeting via text-vision attention, weighting visual importance using dominant text tokens, and prioritizing text KV during eviction. Under a 5% retention rate, it maintains accuracy close to full KV on LLaVA-NeXT/Qwen3-VL while increasing throughput by 52.6%.

## Background & Motivation

**Background**: VLMs adopt the autoregressive generation paradigm of LLMs, caching K/V tensors for all historical tokens to avoid re-computation. However, high-resolution images and long videos frequently occupy thousands or even tens of thousands of tokens. KV cache memory grows linearly with context length, serving as the primary bottleneck in VLM inference. A suite of KV cache eviction methods has been developed for LLMs, such as H2O, SnapKV, PyramidKV, and Ada-KV, which decide which KV pairs to discard based on attention scores or observation windows.

**Limitations of Prior Work**: Directly applying these LLM-validated eviction methods to VLMs leads to severe performance degradation. Experiments on LLaVA with a 5% retention rate show that SnapKV's performance on ChartQA drops from 18.0 to 0.4, rendering it nearly useless. This indicates that LLM-based KV importance metrics are inapplicable to VLMs.

**Key Challenge**: The authors attribute this collapse to the significant "modality gap" in VLMs and confirm this through three experimental observations: (1) visual tokens are highly homogeneous while text tokens are diverse; (2) text-vision cross-modal attention regions are "low-attention troughs," with intra-modal attention being much stronger; (3) when cumulative attention across all layers is summed, a sharp jump occurs at the text-visual boundary, causing most text KV pairs to be evicted first if sorted purely by "cumulative attention"—despite text KV being the most fragile component in VLMs.

**Goal**: To design a "VLM-native" KV eviction pipeline without fine-tuning the model, addressing three sub-problems: how to allocate budgets across layers, how to measure cross-modal KV importance, and which modality to discard when budgets are extremely tight.

**Key Insight**: Through systematic attention deconstruction, the authors derived three key observations: inter-layer budgets should be determined by text-vision (TV) cross-modal attention intensity; KV importance should be governed by TV+TT rather than VV; and text KV is highly sensitive while vision KV is highly redundant, necessitating the prioritization of text KV under tight budgets.

**Core Idea**: "Ground" the entire eviction process using text—text is not just an object to be evicted, but the anchor for determining "which visual KV pairs are important."

## Method

### Overall Architecture
TGV-KV is a plug-in KV cache controller deployed after the prefill stage and before the decoding stage; it does not modify model weights or require calibration datasets. The input to the inference pipeline consists of a unified sequence $\mathbf{X} \in \mathbb{R}^{(N_v+N_t) \times d}$ formed by concatenating $N_v$ visual tokens from the vision encoder and $N_t$ text tokens. After the VLM completes a prefill and generates the attention matrix $\mathbf{A}_l$ for each layer, TGV-KV sequentially triggers three sub-modules: (1) TVB slices the total budget $B$ into per-layer budgets $b_l$ based on the inter-layer distribution of TV attention; (2) TWR calculates a "text-weighted" importance score for all KV pairs within each layer; (3) TPR selects the retention set based on Top-K importance while mandating that all text KV pairs are ranked first. The retained KV pairs are accessed during the decoding stage, and new tokens are simply appended.

### Key Designs

1.  **Text-Vision Budgeting (TVB) — Using Cross-Modal Attention as an Inter-Layer Budget "Barometer"**:
    -   **Function**: Dynamically allocates the global budget $B$ across $L$ layers, determining how many KV pairs each layer should retain.
    -   **Mechanism**: First, the text-to-vision sub-block $\mathbf{A}_l^{(TV)} = \mathrm{softmax}(\mathbf{Q}_l^{(T)} [\mathbf{K}_l^{(V)}]^{\mathsf T}) \in \mathbb{R}^{N_t \times N_v}$ is extracted from the attention matrix of the $l$-th layer. The sum of this block represents the "total intensity of text requesting information from vision" for that layer. Budget proportions $b_l^{(TV)} = \sum_{i,j} [\mathbf{A}_l^{(TV)}]_{ij} / \sum_{l'} \sum_{i,j} [\mathbf{A}_{l'}^{(TV)}]_{ij}$ are normalized across layers and multiplied by $B$ to obtain the layer budget.
    -   **Design Motivation**: Comparative experiments show that using VV, TT, or uniform distribution as layer budget metrics results in significantly worse performance than TV at a 5% retention rate. TV intensity directly correlates with the "intensity of cross-modal information fusion"; layers with more intense fusion deserve more retained KV pairs.

2.  **Text-Weighted Ranking (TWR) — Weighting Visual KV by Dominant Text Tokens**:
    -   **Function**: Computes an importance score $s_{l,j}$ for each KV pair within a layer for Top-K selection.
    -   **Mechanism**: For the text side, the "attention received" for each text token is calculated based on the TT sub-block and divided by its occurrence in the lower triangular part $N_t - j + 1$ to get the average $w_{l,j} = \sum_{i \ge j} [\mathbf{A}_l^{(TT)}]_{ij} / (N_t - j + 1)$. This highlights "dominant text tokens" (vertical bright lines in the attention map) that are consistently attended to throughout subsequent text. After normalizing to $\tilde w_{l,i}$, each row of $\mathbf{A}_l^{(TV)}$ is re-weighted to produce the final visual token scores $s_{l,j}^{(V)} = \sum_i \tilde w_{l,i} [\mathbf{A}_l^{(TV)}]_{ij}$. On the text side, the score is simply the column sum $s_{l,j}^{(T)} = \sum_{i \ge j} [\mathbf{A}_l^{(TT)}]_{ij}$.
    -   **Design Motivation**: Experiments show that pure self-attention or VV+TT as importance metrics lead to near-collapse (scoring ~4 on ChartQA at 5% retention), while TV+TT is significantly more stable. The existence of "dominant text tokens" means that different instructions ("describe the image" vs. "is there a taxi near the streetlight") yield fundamentally different visual importance distributions; weighting ensures visual KV retention aligns with the user's specific query.

3.  **Text-Prioritised Retention (TPR) — Filling Budgets with Text Before Vision**:
    -   **Function**: Decides which KV pairs enter the retention set $\mathcal{I}_l$ given the budget $b_l$ and calculated importance scores.
    -   **Mechanism**: A piecewise strategy: if $b_l > N_t$, all text KV pairs are retained, and the remaining $b_l - N_t$ slots are filled by visual tokens via Top-K on $s_{l,j}^{(V)}$; if $b_l \le N_t$, visual tokens are entirely discarded, and Top-K is performed only within the text tokens based on $s_{l,j}^{(T)}$.
    -   **Design Motivation**: Random eviction experiments reveal that "text is extremely sensitive, while vision is highly redundant"—at 5% retention, prioritizing the eviction of vision still maintains scores of 10–46, while prioritizing text eviction drops scores to 0.2. This asymmetry implies that any form of "fair ranking by score" for text KV is unsafe; all text KV must be unconditionally protected as long as the budget allows.

### Loss & Training
TGV-KV is a pure inference-time algorithm that introduces no additional training or fine-tuning and requires no calibration datasets. All budget and importance calculations are based on the attention matrices produced during a single prefill pass, making it plug-and-play for any VLM based on standard self-attention.

## Key Experimental Results

### Main Results

The authors evaluated TGV-KV on LLaVA-1.5-7B / LLaVA-NeXT-7B / LLaVA-OV / Qwen3-VL-4B/8B across tasks including ChartQA, DocVQA, VizWiz, TextVQA, TextCaps, COCO-Caption, and Video-TT, comparing it with baselines like StreamingLLM, SnapKV, H2O, ElasticCache, and PrefixKV. The following table extracts representative results for LLaVA at a 5% extreme retention rate:

| Model / Task | Metric | Full KV (Vanilla) | TGV-KV (5%) | vs. Full KV |
| --- | --- | --- | --- | --- |
| LLaVA-NeXT / VizWiz-VQA | Acc. | 100% | 99.2% | -0.8 pt |
| Qwen3-VL-8B / DocVQA | ANLS | 100% | 92.5% | -7.5 pt |
| LLaVA-1.5 / ChartQA (vs best baseline) | Relaxed Acc. | 18.0 | Significantly leads (+33.0 pt relative to strongest baseline) | / |
| LLaVA-NeXT End-to-End | Throughput | 1.0× | 1.526× | +52.6% |
| All Models | KV Memory | 1.0× | 0.05× | -95% |

TGV-KV closely approaches full KV accuracy under extreme compression, particularly on the LLaVA series. On dense text OCR tasks like DocVQA, a 5% budget still retains over 90% ANLS.

### Ablation Study

The following table summarizes three key comparisons (LLaVA, 5% retention) verifying the necessity of each TGV-KV module:

| Setting | ChartQA ↑ | TextVQA-lite ↑ | Description |
| --- | --- | --- | --- |
| Vanilla | 18.0 | 47.9 | Full KV |
| Uniform layer budget + TV+TT Importance | 14.3 | 36.4 | Without TVB |
| TV layer budget + TV+TT Importance (≈TVB) | 14.3 | 36.4 | TVB provides superior inter-layer structure |
| Uniform + Observation Window | 0.4 | 8.7 | Importance metric swap leads to collapse |
| Uniform + Pure self-attention | 4.8 | 23.5 | LLM-style failure |
| Uniform + VV+TT Importance | 4.6 | 22.8 | Fails without TV |
| Uniform + TV+TT Importance | 11.0 | 37.3 | TWR core effective |
| Uniform + Prioritize evicting text | 0.2 | 4.4 | Verifying TPR's inverse upper bound |
| Uniform + Prioritize evicting vision | 10.0 | 31.0 | Text protection is critical |

### Key Findings
- The "importance measure," rather than "layer budget," is the primary factor in preventing collapse—observation windows or pure self-attention drop to single digits at 5% retention, whereas incorporating text-vision attention restores double-digit performance.
- "Text as anchor" is a mandatory conclusion for VLM KV eviction: random text eviction drops ChartQA to 0.2, while 99.2% retention on VizWiz tasks can only be maintained with TPR.
- TVB’s TV intensity signal naturally shifts budgets toward middle layers where information fusion is most intense. This is more robust than fixed rules like PyramidKV because it is model- and task-agnostic.
- The +52.6% throughput gain stems from 95% memory compression allowing for larger batches and longer sequences. TGV-KV’s own overhead is minimal as it reuses prefill attention matrices.

## Highlights & Insights
- **Translating Modality Gap from "Problem" to "Signal"**: Previous works viewed the "low attention trough" in TV regions as a flaw. This paper uses the relative intensity of these troughs across layers as the core signal for budget allocation.
- **"Dominant Text Token" Weighting** makes visual KV importance instruction-sensitive for the first time: given the same image, the retained visual KV will differ between prompts like "describe this" and "is there a taxi," a capability lacking in prompt-agnostic methods like H2O/SnapKV.
- **Asymmetric Protection Strategy**: Through simple experiments comparing random text vs. vision eviction, the authors established the "text cannot be lost" constraint and elegantly implemented it via TPR's piecewise formula.
- The method requires zero training, fine-tuning, or calibration data, making it highly engineering-friendly for any standard self-attention VLM.

## Limitations & Future Work
- To maintain parallelism, the paper explicitly avoids head-wise budget allocation. If future attention kernels (e.g., PagedAttention, FlashDecoding-v2) allow fine-grained head budgets without breaking parallelism, TGV-KV could be further refined.
- TVB/TWR rely on attention matrices from the prefill stage. For deployments like FlashAttention that do not explicitly materialize attention, an additional "light attention recomputation" pass is needed to obtain TV/TT blocks.
- While throughput increases by 52.6% at 5% budget, the latency breakdown on models like Qwen3-VL-8B remains to be detailed. Combination strategies with token pruning (FastV, VisionZip) for video contexts (tens of thousands of tokens) warrant further study.
- TPR currently uses hard rules (text prioritized first). This might over-protect text in "heavy-image, light-text" captioning scenarios; learned modality weights or task-adaptive priorities could provide improvements.

## Related Work & Insights
- **vs. H2O / SnapKV / StreamingLLM**: These are designed for LLMs, using self-attention or observation windows. This paper proves these fail on VLMs due to the modality gap causing erroneous text KV eviction.
- **vs. PyramidKV / Ada-KV**: These use fixed rules or extra calibration for layer budgets; TVB uses dynamic allocation based on prefill TV attention without needing calibration sets.
- **vs. AirCache**: Also recognizes text importance, but AirCache requires extra computation to identify key tokens and lacks inter-layer information flow analysis.
- **vs. FastV / VisionZip / CDPruner (token pruning)**: These discard visual tokens once before/during prefill. TGV-KV operates at the KV level and allows subsequent layers to retain more information, offering higher flexibility. They can be used in tandem: "prune tokens first, then compress KV."

## Rating
- Novelty: ⭐⭐⭐⭐ Reversing the "modality gap" from a source of failure into a signal for budget allocation is a refreshing perspective innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers five VLMs of varying scales/architectures, image + video tasks, and 5 baselines with multiple retention rates (5% to 50%).
- Writing Quality: ⭐⭐⭐⭐ The logical progression through three Observations is very clear, and notation is standard.
- Value: ⭐⭐⭐⭐⭐ Provides a zero-training, plug-and-play VLM inference compression solution with 95% memory savings and significant throughput gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](../../NeurIPS2025/multimodal_vlm/prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)
- [\[ACL 2026\] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](../../ACL2026/multimodal_vlm/vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/multimodal_vlm/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)

</div>

<!-- RELATED:END -->
