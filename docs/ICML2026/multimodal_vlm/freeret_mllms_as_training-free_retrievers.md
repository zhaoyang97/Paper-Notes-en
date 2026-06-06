---
title: >-
  [Paper Note] FreeRet: MLLMs as Training-Free Retrievers
description: >-
  [ICML 2026][Multimodal VLM][Training-free retrieval] FreeRet proposes a fully training-free two-stage multimodal retrieval framework: Stage 1 bypasses the MLLM's final MLP and uses controlled generation prompts to extrac…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Training-free retrieval"
  - "MLLM embedding"
  - "lexicalization pressure"
  - "LLM framing effect"
  - "two-stage retrieval"
date: 2026-05-08
content_hash: d7aa817df77cbe5c
---

# FreeRet: MLLMs as Training-Free Retrievers

**Conference**: ICML 2026  
**arXiv**: [2509.24621](https://arxiv.org/abs/2509.24621)  
**Code**: None  
**Area**: Multimodal VLM / Multimodal Retrieval  
**Keywords**: Training-free retrieval, MLLM embedding, lexicalization pressure, LLM framing effect, two-stage retrieval

## TL;DR
FreeRet proposes a fully training-free two-stage multimodal retrieval framework: Stage 1 bypasses the MLLM's final MLP and uses controlled generation prompts to extract semantically faithful embeddings for candidate retrieval; Stage 2 reformulates reranking as a multiple-choice question to avoid LLM framing bias. On MMEB, it outperforms retrieval models trained on tens of millions of paired data.

## Background & Motivation
**Background**: CLIP-style dual-tower models dominate multimodal retrieval, but struggle with long queries, compositional semantics, and interleaved modalities. Recent works treat MLLMs as general encoders, followed by contrastive learning/RL/data augmentation for further training.

**Limitations of Prior Work**: Training-based approaches have two major drawbacks: (1) Each change of backbone or modality combination requires extensive paired data and retraining; (2) Generalization is fragile (models SOTA on MMEB often drop significantly on MIEB). Existing training-free methods (E5-V, PromptEOL) focus only on embedding, lack reranking, and perform much worse than trained models.

**Key Challenge**: MLLMs inherently possess strong multimodal semantic and reasoning abilities, but their final MLP is designed for "next token prediction"—this "lexicalization pressure" pulls semantic vectors toward the vocabulary, undermining the fine-grained semantics needed for retrieval. Reranking suffers from another hidden bias: the choice of label pairs ("Yes/No", "True/False", "Right/Wrong") can cause 5–8% accuracy differences for the same judgment.

**Goal**: Without modifying any weights, use a single MLLM for both embedding and reranking; explicitly identify and address the two aforementioned biases.

**Key Insight**: Treat the MLLM as a generator—since its intermediate layers are closer to semantics than the final layer, skip the last MLP; since binary reranking is subject to lexical bias, reformulate it as an MCQ so the model selects "A/B" as the answer.

**Core Idea**: In the embedding stage, use "intermediate hidden states + three types of control prompts (task/semantic/denoising)" for generation; in reranking, reformulate discrimination as multiple-choice, using the LM head's probability for option A as the score.

## Method

### Overall Architecture
FreeRet decomposes retrieval into two stages, both handled by the same untrained MLLM. **Stage 1 Embedding**: Given input $x$ (any modality combination), concatenate a control prompt and have the model generate a word $y$; instead of the final MLP output, extract the hidden state after the last attention layer and before the final MLP, $h_L^{\text{Attn}}(y)$, as the embedding $e(x)$, and use cosine similarity to recall top-$n$ candidates. **Stage 2 Reranking**: Package the query and each candidate into an MCQ prompt ("A. Match / B. No match"), take $p(\text{`A'})$ from the LM head, then softmax as the relevance score. The entire pipeline introduces no extra parameters, does not rely on auxiliary models, and can be seamlessly integrated into RAG workflows for "single-model retrieve + rerank + generate".

### Key Designs

1. **Bypassing the Final MLP to Alleviate Lexicalization Pressure (§3.2)**:

    - **Function**: Without changing any parameters, move the embedding extraction point from the final MLP (where semantics are "pulled toward the vocabulary") to after the attention layer.
    - **Mechanism**: Using Qwen2.5-VL (3B/7B/32B) for probing, the authors define $\alpha_\ell^{\text{Attn}}=\cos(h^{\text{MLP}}_{\ell-1},h^{\text{Attn}}_{\ell})$, $\beta_\ell^{\text{MLP}}=\cos(h^{\text{MLP}}_{\ell},\mathbf{w}_{y^*})$, etc., and find that $\alpha$ drops sharply to <0.3 after the final MLP, while $\beta$ jumps to ~0.5 at the same point, indicating "lexicalization" is almost entirely concentrated in the last MLP. Additionally, the inter-layer cosine similarity for 250 synonym pairs drops from ~94% to ~87% after the final MLP. Conclusion: directly use $h_L^{\text{Attn}}$ as the embedding, skipping the final MLP.
    - **Design Motivation**: Embedding should capture semantics, while the MLLM's final MLP serves generation. This "skip one layer" step yields stable gains of 5.33%/5.71% on 3B/7B models (Tab. 3a), forming the foundation for all subsequent improvements.

2. **Controlled Generation Prompts Inject Three Types of Priors (§3.3)**:

    - **Function**: Replace E5-V's "Summary above content in one word" free-form word summarization with controlled generation incorporating three constraints, ensuring the generated "single word" is semantically focused, free from function word pollution, and aligned with downstream tasks.
    - **Mechanism**: Three lightweight constraints are added sequentially—(i) Task alignment: "You are required to assess if <A> is related to <B>"; (ii) Semantic grounding: "Capture the semantics of <X>"; (iii) Noise suppression: "Do not use function words, prepositions, or symbols". Tab. 3b shows these steps add 4.29, 1.49, 2.47 points on 3B, and 5.07, 0.9, 2.17 on 7B, respectively. All changes are prompt-only, with no weight modifications.
    - **Design Motivation**: Without constraints, the model often outputs semantically drifting or purely functional words like "Self", "Searching", "Growing", diluting the embedding space; "task prior" aligns the summary words of query and target, making their cosine similarity naturally closer.

3. **Multiple-Choice Reranking to Mitigate LLM Framing Effect (§3.4)**:

    - **Function**: In the reranking stage, eliminate "asymmetric bias of label words" so the model no longer exhibits systematic bias for semantically equivalent binary choices.
    - **Mechanism**: The authors find that "Right/Wrong", "Yes/No", "True/False" are logically equivalent but can differ by up to 5% in accuracy on the same benchmark. When the model freely chooses these labels under context-free instructions, the output logits are clearly skewed, and greater skew correlates with lower downstream accuracy—consistent with Zhao et al. 2021's LLM bias, termed here as the "LLM framing effect". The remedy is to reformulate reranking as MCQ: "A. Match, B. No match", and use $p(\text{`A'})$ from the LM head for SoftMax. MCQ neutralizes semantic/emotional bias and leverages the prevalence of "A/B question types" in LLM pretraining data.
    - **Design Motivation**: Intuitively, reranking seems to be a binary question, but the "formulation" itself is a confounding variable; MCQ maps the problem to the question-type space, outperforming direct yes/no by 8.4%, and this change also requires no training (Fig. 4).

### Loss & Training
No training is required. All modifications involve only (i) extraction position, (ii) prompt templates, and (iii) reranking output format. No new parameters are introduced, enabling "plug-and-play" model-agnosticism across MLLMs such as Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni, InternVL3, LLaVA-OV series, etc.

## Key Experimental Results

### Main Results (MMEB, 36 datasets, average Precision@1)

| Method | Backbone | Training Data (M) | Mean |
|--------|----------|-------------------|------|
| MMRet (embed-only) | LLaVA-1.6-7B | 26.2 | 44.0 |
| GME (embed-only) | Qwen2-VL-7B | 8.0 | 56.0 |
| LamRA-Ret | Qwen2.5-VL-7B | 1.4 | 52.4 |
| E5-V (train-free, reproduced) | Qwen2.5-VL-7B | – | 39.8 |
| **FreeRet-embed** | Qwen2.5-VL-7B | – | **53.7** |
| MM-Embed (top-10 rerank) | LLaVA-Next-7B | 1.1+0 | 54.9 |
| LamRA (top-10 rerank) | Qwen2.5-VL-7B | 1.4+1.1 | 55.0 |
| **FreeRet (top-10)** | Qwen2.5-VL-7B | – | **67.8** |
| **FreeRet (top-50)** | Qwen2.5-VL-7B | – | **70.7** |

### MMEB-V2 Video Subset (no video retrieval training)

| Method | Backbone | Training Data (M) | Video Cls | Video Ret |
|--------|----------|-------------------|-----------|-----------|
| VLM2Vec-V2 | Qwen2-VL-2B | 1.7 | 39.3 | 28.8 |
| GME | Qwen2-VL-7B | 8.0 | 37.4 | 28.4 |
| **FreeRet-embed** | Qwen2-VL-2B | – | 47.7 | 31.7 |
| **FreeRet** | Qwen2-VL-7B | – | **63.2** | **39.3** |

### Ablation Study (Tab. 3)

| Setting | 3B | 7B | Note |
|---------|----|----|------|
| Use $h^{\text{MLP}}_L$ (baseline) | 45.34 | 47.97 | E5-V extraction |
| Use $h^{\text{Attn}}_L$ (FreeRet) | 50.67 | 53.68 | Skip one MLP layer |
| Use $h^{\text{MLP}}_{L-2}$ | 50.64 | 48.78 | Skipping two transformer layers hurts |
| Yes/No reranking | 58.39 | 65.28 | Framing bias baseline |
| True/False | 60.06 | 66.71 | Slightly less bias |
| **MCQ reranking** | **60.31** | **70.72** | Eliminates framing effect |

### Key Findings
- The final MLP is the performance bottleneck, but skipping more layers sacrifices semantic retention, so "precisely skipping one layer" is optimal. This effect is more pronounced in shallower models.
- Among the three prompt controls, "semantic grounding" yields the highest single gain (~5pt), indicating that MLLMs by default output generalized but semantically drifting summary words, which is the main noise source for embeddings.
- The 8% gap between "Yes/No vs MCQ" is almost entirely due to label pretraining distribution bias, unrelated to logic—this is a severely underestimated detail and serves as a warning for all LLM judge or rerank tasks.
- On video tasks, FreeRet-2B outperforms VLM2Vec-V2 trained on 1.7M video pairs, indicating that "untrained MLLMs" already encode cross-modal information well; the key is how to extract it.

## Highlights & Insights
- Provides a systematic training-free retrieval manual by thoroughly analyzing and quantifying the benefits of "where to extract embeddings + how to prompt" and "how to formulate reranking"; many implicit steps in RAG are made explicit.
- Using cosine similarity and LM-head projection to characterize "lexicalization pressure" offers a clear mechanistic analysis, serving as a general tool for LLM representation research.
- Transferring the "LLM framing effect" to all LLM-as-judge research is directly valuable: rerankers, automatic evaluation, and reward models can all reference the MCQ-based debiasing design.
- Since no weights are modified, FreeRet naturally preserves the MLLM's dialogue/instruction-following/reasoning abilities, enabling retrieval, reranking, and generation to be run within a single model—highly conducive to minimalist RAG implementations.

## Limitations & Future Work
- The second stage requires a forward pass through the MLLM for each query–candidate pair; more candidates mean slower inference. The paper limits candidates to top-5/10/50, but latency may still be a bottleneck in large-scale real-world retrieval.
- Relies entirely on the assumption that "untrained MLLMs are already strong enough"; for small models or specialized domains (medical, code), this free-lunch may not hold. The paper does not analyze lower bounds for small models.
- MCQ templates and prompt controls are manually designed; there is no systematic study on "automatic prompt search" or "per-task prompt tuning". Stability depends on prompt quality, and prompt sensitivity is not fully explored.

## Related Work & Insights
- **vs E5-V**: E5-V directly uses the final hidden state for embedding, ignoring lexicalization; FreeRet's layer skipping + controlled prompts improve the same backbone by 13.9pt on MMEB.
- **vs Trained MM-Embed / LamRA / GME**: These methods require 1M–26M multimodal pairs for training; FreeRet matches or surpasses them without training, revealing the underestimated potential of training-free approaches.
- **vs PromptEOL / MetaEOL / Echo-Embedding**: These text-only training-free methods focus solely on embedding; FreeRet extends their philosophy to multimodal and adds the crucial reranking stage, representing a systematic inheritance.
- **vs Zhao et al. (2021) framing bias**: FreeRet directly applies LLM calibration research to retrieval, using MCQ to "formalize debiasing", providing a lightweight solution for subsequent LLM-as-judge work.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic training-free multimodal retrieval, with clear explanations of lexicalization and framing effect mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers MMEB 36 datasets + MMEB-V2 video, across multiple MLLM families; lacks efficiency/latency comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, concise three-step narrative, well-coordinated probing and ablation.
- Value: ⭐⭐⭐⭐ Directly applicable to RAG and multimodal retrieval communities, with methodological insights for LLM judge research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](../../CVPR2026/multimodal_vlm/gtr_turbo_merged_checkpoint_free_teacher.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](../../CVPR2026/multimodal_vlm/llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)

</div>

<!-- RELATED:END -->
