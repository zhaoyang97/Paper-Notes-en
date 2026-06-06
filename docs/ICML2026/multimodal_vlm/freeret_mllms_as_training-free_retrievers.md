---
title: >-
  [Paper Note] FreeRet: MLLMs as Training-Free Retrievers
description: >-
  [ICML 2026][Multimodal VLM][Training-free Retrieval] FreeRet proposes a fully training-free two-stage multimodal retrieval framework: the first stage bypasses the MLLM's final MLP layer and uses controlled generation pro…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Training-free Retrieval"
  - "MLLM Embedding"
  - "Lexicalization Pressure"
  - "LLM Framing Effect"
  - "Two-stage Retrieval"
date: 2026-05-08
content_hash: e6880c814f99955f
---

# FreeRet: MLLMs as Training-Free Retrievers

**Conference**: ICML 2026  
**arXiv**: [2509.24621](https://arxiv.org/abs/2509.24621)  
**Code**: None  
**Area**: Multimodal VLM / Multimodal Retrieval  
**Keywords**: Training-free Retrieval, MLLM Embedding, Lexicalization Pressure, LLM Framing Effect, Two-stage Retrieval

## TL;DR
FreeRet proposes a fully training-free two-stage multimodal retrieval framework: the first stage bypasses the MLLM's final MLP layer and uses controlled generation prompts to extract semantically faithful embeddings for candidate retrieval; the second stage reformulates reranking as a multiple-choice question (MCQ) to circumvent the LLM framing bias. It outperforms retrieval models trained on tens of millions of paired data on the MMEB benchmark.

## Background & Motivation
**Background**: CLIP-style dual-tower models are dominant in multimodal retrieval but struggle with long queries, compositional semantics, and interleaved modalities. Recent works treat MLLMs as universal encoders, followed by post-training using contrastive learning, RL, or data augmentation.

**Limitations of Prior Work**: Training-based approaches suffer from two major issues: first, changing a backbone or modal combination requires massive amounts of paired data for re-tuning; second, generalization is fragile (SOTA models on MMEB often show significant performance drops on MIEB). Existing training-free methods (e.g., E5-V, PromptEOL) focus solely on embeddings without reranking, resulting in performance far inferior to trained versions.

**Key Challenge**: While MLLMs possess strong multimodal semantics and reasoning capabilities, their final MLP layer is optimized for "next token prediction." This "lexicalization pressure" forces semantic vectors toward the vocabulary space, destroying the fine-grained semantics needed for retrieval. For reranking, a hidden bias exists: choosing between labels like "Yes/No," "True/False," or "Right/Wrong" can cause a 5–8% accuracy variance for the same semantic judgment.

**Goal**: Utilize a single MLLM to perform both embedding and reranking without updating any weights, while specifically identifying and mitigating the two aforementioned biases.

**Key Insight**: Treat the MLLM as a generator. Since intermediate layers are closer to semantics than the final layer, skip the last MLP. Since binary reranking suffers from lexical bias, frame it as an MCQ to allow the model to select "A/B" instead.

**Core Idea**: For the embedding stage, generate representations using "intermediate hidden states + three types of controlled prompts." For the reranking stage, convert discriminative tasks into multiple-choice questions, using the probability of option "A" from the LM head as the score.

## Method

### Overall Architecture
FreeRet decomposes retrieval into two stages, both handled by the same untrained MLLM. **Stage 1 Embedding**: Given an input $x$ (any combination of modalities), a control prompt is appended to generate a word $y$. Instead of taking the final MLP output, the hidden state $h_L^{\text{Attn}}(y)$ after the last attention layer and before the last MLP is used as the embedding $e(x)$. Top-$n$ candidates are retrieved via cosine similarity. **Stage 2 Reranking**: The query and each candidate are wrapped into an MCQ prompt ("A. Match / B. Not Match"). The probability $p(\text{`A'})$ is extracted from the LM head and normalized via softmax to serve as the relevance score. The entire pipeline requires no extra parameters or auxiliary models, enabling a seamless "single model for retrieve + rerank + generate" RAG flow.

### Key Designs

1. **Bypassing the Final MLP to Alleviate Lexicalization Pressure (§3.2)**:
    - **Function**: Extracts embeddings from a position where semantics are not yet "dragged toward the vocabulary" by the final MLP, without modifying parameters.
    - **Mechanism**: The authors conducted probing on Qwen2.5-VL (3B/7B/32B). By defining metrics like $\alpha_\ell^{\text{Attn}}=\cos(h^{\text{MLP}}_{\ell-1},h^{\text{Attn}}_{\ell})$ and $\beta_\ell^{\text{MLP}}=\cos(h^{\text{MLP}}_{\ell},\mathbf{w}_{y^*})$, they found that $\alpha$ drops sharply to $< 0.3$ after the last MLP, while $\beta$ jumps to $\sim 0.5$. This indicates that "lexicalization" is almost entirely concentrated in the final MLP. Consistently, the inter-layer cosine similarity of 250 synonym pairs drops from $\sim 94\%$ to $\sim 87\%$ after the last MLP. Thus, $h_L^{\text{Attn}}$ is taken as the embedding, skipping the MLP.
    - **Design Motivation**: Embeddings should capture semantics, whereas the final MLP of an MLLM serves generation. Skipping this single layer yields stable gains of 5.33% and 5.71% on 3B and 7B models, respectively (Tab. 3a).

2. **Controlled Generation Prompts Injecting Three Priors (§3.3)**:
    - **Function**: Replaces free-form word summarization (as in E5-V) with controlled generation under three constraints to ensure words are semantically focused and aligned with downstream tasks.
    - **Mechanism**: Three lightweight constraints are overlaid: (i) Task alignment: "You are required to assess if <A> is related to <B>"; (ii) Semantic grounding: "Capture the semantics of <X>"; (iii) Noise suppression: "Do not use function words, prepositions, or symbols." Tab. 3b shows these steps add 4.29, 1.49, and 2.47 percentage points on the 3B model.
    - **Design Motivation**: Without constraints, models often output drift words like "Self" or "Searching" or functional tokens. Task alignment ensures query and target summary words are systematically aligned, making their embeddings naturally closer in cosine space.

3. **MCQ Reranking to Alleviate the LLM Framing Effect (§3.4)**:
    - **Function**: Eliminates the "asymmetric bias of label words" during reranking, preventing systematic deviations caused by different phrasing of the same meaning.
    - **Mechanism**: The authors observed that although "Right/Wrong," "Yes/No," and "True/False" are logically equivalent, accuracy can vary by up to 5% on the same benchmark. Under context-free instructions, the output logits for these labels are skewed, and higher skew correlates with lower downstream accuracy—a phenomenon consistent with the LLM bias noted by Zhao et al. (2021), termed the "LLM framing effect." The solution is to use an MCQ format: "A. Match, B. Not Match," taking $p(\text{`A'})$ for the score.
    - **Design Motivation**: MCQ neutralizes semantic/emotional bias by mapping the problem to the "A/B format" heavily represented in LLM pre-training data, outperforming direct yes/no by 8.4% without any training (Fig. 4).

### Loss & Training
Completely training-free. Modifications only involve (i) extraction location, (ii) prompt templates, and (iii) reranking output format. It is model-agnostic and plug-and-play for MLLMs including the Qwen2-VL, Qwen2.5-VL, InternVL3, and LLaVA-OV series.

## Key Experimental Results

### Main Results (MMEB, Average Precision@1 across 36 datasets)

| Method | Backbone | Training Data (M) | Average |
|------|----------|--------------|------|
| MMRet (embed-only) | LLaVA-1.6-7B | 26.2 | 44.0 |
| GME (embed-only) | Qwen2-VL-7B | 8.0 | 56.0 |
| LamRA-Ret | Qwen2.5-VL-7B | 1.4 | 52.4 |
| E5-V (train-free reproduction) | Qwen2.5-VL-7B | – | 39.8 |
| **FreeRet-embed** | Qwen2.5-VL-7B | – | **53.7** |
| MM-Embed (top-10 rerank) | LLaVA-Next-7B | 1.1 + 0 | 54.9 |
| LamRA (top-10 rerank) | Qwen2.5-VL-7B | 1.4 + 1.1 | 55.0 |
| **FreeRet (top-10)** | Qwen2.5-VL-7B | – | **67.8** |
| **FreeRet (top-50)** | Qwen2.5-VL-7B | – | **70.7** |

### MMEB-V2 Video Subsets (None trained on video retrieval)

| Method | Backbone | Training Data (M) | Video Cls | Video Ret |
|------|----------|--------------|-----------|-----------|
| VLM2Vec-V2 | Qwen2-VL-2B | 1.7 | 39.3 | 28.8 |
| GME | Qwen2-VL-7B | 8.0 | 37.4 | 28.4 |
| **FreeRet-embed** | Qwen2-VL-2B | – | 47.7 | 31.7 |
| **FreeRet** | Qwen2-VL-7B | – | **63.2** | **39.3** |

### Ablation Study (Tab. 3)

| Setting | 3B | 7B | Description |
|------|----|----|------|
| Extract $h^{\text{MLP}}_L$ (baseline) | 45.34 | 47.97 | E5-V style extraction |
| Extract $h^{\text{Attn}}_L$ (FreeRet) | 50.67 | 53.68 | Skip one MLP layer |
| Extract $h^{\text{MLP}}_{L-2}$ | 50.64 | 48.78 | Performance drops if skipping 2+ layers |
| Yes/No reranking | 58.39 | 65.28 | Framing bias baseline |
| True/False | 60.06 | 66.71 | Slightly less bias |
| **MCQ reranking** | **60.31** | **70.72** | Eliminates framing effect |

### Key Findings
- The final MLP layer is a performance bottleneck; however, skipping more than one layer sacrifices semantic information. "Precisely skipping one layer" is most effective, especially for shallower models.
- Among prompt controls, "semantic grounding" provides the highest gain (~5pt), indicating that MLLMs default to over-generalized summary words that drift from input semantics.
- The 8% gap between "Yes/No" and "MCQ" is almost entirely due to pre-training distribution bias of labels rather than logic.
- On video tasks, FreeRet-2B outperforms VLM2Vec-V2 (trained on 1.7M video pairs), suggesting that untrained MLLMs already encode robust cross-modal information.

## Highlights & Insights
- Provides a systematic training-free retrieval manual by quantifying the gains of extraction location and prompt framing.
- The use of cosine similarity and LM-head projection to characterize "lexicalization pressure" is a clear mechanistic analysis tool for LLM representations.
- The mitigation of the "LLM framing effect" via MCQ has broad value for LLM-as-judge research, including rerankers and reward models.
- By not modifying weights, FreeRet preserves MLLM conversational and reasoning abilities, enabling retrieval, reranking, and generation within a single model.

## Limitations & Future Work
- The second stage requires an MLLM forward pass for each query-candidate pair; latency may be a bottleneck in large-scale real-world scenarios.
- The performance depends on the inherent strength of the MLLM; the "free lunch" might not hold for smaller or highly specialized vertical models (e.g., medical, code).
- MCQ templates and prompt controls are manually designed; systematic prompt searching or sensitivity analysis was not performed.

## Related Work & Insights
- **vs. E5-V**: E5-V extracts from the final hidden layer without considering lexicalization. FreeRet's layer skipping and prompt control improve performance by 13.9pt on MMEB.
- **vs. Trained Models (MM-Embed / LamRA / GME)**: These require 1M–26M multimodal pairs for training; FreeRet matches or exceeds them without training.
- **vs. PromptEOL / Echo-Embedding**: FreeRet extends the logic of these text-only training-free methods to the multimodal domain and adds the critical reranking stage.
- **vs. Zhao et al. (2021) framing bias**: FreeRet adapts LLM calibration research to the retrieval context, providing a lightweight solution for de-biasing LLM-as-judge tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic training-free multimodal retrieval with clear mechanistic explanations for lexicalization and framing effects.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 36 MMEB datasets + MMEB-V2 video across multiple MLLM families, though lacks efficiency/latency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, clean methodology narrative, and well-aligned probing and ablation studies.
- Value: ⭐⭐⭐⭐ High practical value for RAG and multimodal retrieval communities, with methodological insights for LLM-as-judge research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training](med-scout_curing_mllms_geometric_blindness_in_medical_perception_via_geometry-aw.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)

</div>

<!-- RELATED:END -->
