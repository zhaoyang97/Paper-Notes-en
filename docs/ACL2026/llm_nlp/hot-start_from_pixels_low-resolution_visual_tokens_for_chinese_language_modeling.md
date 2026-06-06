---
title: >-
  [Paper Note] Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling
description: >-
  [ACL 2026][LLM/NLP][pixel-based LM] By rendering Chinese characters into $8\times 8$ grayscale thumbnails for next-character prediction in a GPT-2 style decoder…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "pixel-based LM"
  - "Chinese characters"
  - "hot-start"
  - "low-resolution"
  - "visual tokens"
date: 2026-05-08
content_hash: eb233274bcc97d89
---

# Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling

**Conference**: ACL 2026  
**arXiv**: [2601.09566](https://arxiv.org/abs/2601.09566)  
**Code**: To be confirmed  
**Area**: NLP / Chinese Language Modeling / Visual Representation  
**Keywords**: pixel-based LM, Chinese characters, hot-start, low-resolution, visual tokens

## TL;DR
By rendering Chinese characters into $8\times 8$ grayscale thumbnails for next-character prediction in a GPT-2 style decoder, the final accuracy (39.21%) matches the index-based baseline (39.10%). Crucially, it more than doubles the baseline performance during early training (at 0.4% of the data), demonstrating that "visual structure" serves as a natural hot-start prior for Chinese character modeling.

## Background & Motivation

**Background**: Mainstream Chinese LMs still treat characters as discrete token IDs (e.g., Chinese versions of GPT/BERT). Existing glyph-augmented works (Glyce, ChineseBERT) use glyph features **as auxiliary inputs** coupled with token embeddings; pixel-based LMs like PIXEL directly perform MLM on rendered text but target pixel-to-pixel reconstruction.

**Limitations of Prior Work**:
- Index-based representations strip characters like "山" (mountain) into abstract IDs, losing visual information that is inherently meaningful to humans.
- Chinese is logographic; character shapes themselves encode semantic and phonetic information. Discrete IDs suffer from slow convergence when data is scarce.
- Existing glyph-augmented methods use glyphs as side info without a controlled comparison for "complete token ID replacement."

**Key Challenge**: Should representations be token IDs or pixels? While both may achieve similar final accuracy, the **training dynamics** might differ significantly—visual representations naturally possess geometric structures in embedding space, potentially providing a structural prior for early training.

**Goal**: To completely replace character representations from index to pixel and systematically answer four research questions: (RQ1) Is vision sufficient? (RQ2) How are the early learning dynamics? (RQ3) How low can the resolution go? (RQ4) Is it robust under partial occlusion?

**Key Insight**: Conduct a clean "visual-in, token-out" controlled experiment using the same GPT-2-small decoder. The input path uses a ResNet+Adapter to process grayscale images ranging from $4\times 4$ to $96\times 96$.

**Core Idea**: Visual structure itself is a ready-to-use prior. A **minimum resolution of $8\times 8$** is sufficient to match the index-based baseline and triggers a "hot-start" effect in early training.

## Method

### Overall Architecture
Two pipelines share the same GPT-2-small (117M parameters, 12 layers, 768 dimensions) decoder:

- **Index path**: Character → Discrete ID → Embedding → Decoder.
- **Visual path**: Character → Rendered as grayscale thumbnail (default $8\times 8$, 10% margin) → ResNet encoder → Vision Adapter → Decoder embedding space.

The training objective is standard next-character cross-entropy:

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \log P(c_{t+1}^{(i)}|I_1^{(i)}, \dots, I_t^{(i)})$$

(For clarity, the inline version can be read as $\mathcal{L}_{CE} = -\frac{1}{N}\sum_i \sum_t \log P(c_{t+1}|I_{\le t})$)

Dataset: THUCNews (740k news articles, 12.8M character instances, split into fixed sequences of length 128), utilizing a quadratic curriculum: the number of training sequences per epoch grows by $5000 + 918.37 \cdot \text{epoch} + 18.74 \cdot \text{epoch}^2$. The validation set is fixed at 5k sequences.

### Key Designs

1. **Learnable Visual Encoder for Extremely Low Resolution**:
    - **Function**: Encodes $8\times 8$ or even $4\times 4$ grayscale character images into 768-dimensional decoder-ready vectors.
    - **Mechanism**: Original ResNet is designed for $64\times 64$ and is heavily over-parameterized for $8\times 8$. The authors provide three implementations: (a) Original ResNet (26.45M, +16% FLOPs); (b) Minimal encoder + deep adapter optimized for $8\times 8$ (22.32M, +12%); (c) Minimal encoder + simple linear adapter (12.61M, +7%). The optimal (c) uses 33.5% fewer parameters than the index baseline (18.97M).
    - **Design Motivation**: To validate visual representation, it must be proven that performance does not come from sheer parameter count; the minimal encoder actually matched the final accuracy of larger encoders.

2. **Three Levels of Cropping: Vision-100% / Vision-80% / Vision-50%**:
    - **Function**: Tests spatial robustness and verifies the model relies on "structure" rather than "OCR reconstruction."
    - **Mechanism**: Vision-80% keeps the top 80% of pixels, and Vision-50% keeps the top 50%, with the rest filled by the background. At $8\times 8$, the signal-carrying pixels for Vision-100% are $6\times 6$, Vision-80% are $6\times 5$, and Vision-50% are reduced to $6\times 3$.
    - **Design Motivation**: If the model were merely performing reverse OCR to obtain IDs, performance should collapse under heavy occlusion. The fact that Vision-50% maintains 38.63% accuracy proves the model learns **distributed visual features**—the "toast-center" effect, where central strokes carry most discriminative information.

3. **Visual-in, Token-out Paradigm**:
    - **Function**: Uses vision as input while predicting in token space for fair comparison with index baselines.
    - **Mechanism**: Unlike the "pixel-in pixel-out" of PIXEL / PIXAR series, HotStart maintains a softmax over character IDs at the output. This ensures metrics like cross-entropy, perplexity, and accuracy are directly comparable; only the input switches from ID embedding to visual embedding.
    - **Design Motivation**: To answer whether "vision is useful as an input representation," keeping the output interface fixed is a necessary experimental control.

### Loss & Training
- AdamW, lr $2\times 10^{-4}$ (OneCycle max $1.5\times 10^{-3}$), batch 128, weight decay 0.01, FP16, early stopping patience 7.
- Joint training: End-to-end gradients for visual encoder + adapter + decoder. Ablations found that freezing the decoder while training the adapter is significantly worse than joint training.
- No OCR pre-training and no introduction of character ID signals—pure visual-based language modeling.

## Key Experimental Results

### Main Results: Final Accuracy (RQ1 + RQ3 + RQ4)
Accuracy / PPL across resolutions and cropping levels on THUCNews:

| Mode | $4\times 4$ | $8\times 8$ | $20\times 20$ | $30\times 30$ | $80\times 80$ |
|------|-------------|-------------|---------------|---------------|---------------|
| Vision-100% | 29.70 / 85.33 | **39.21 / 46.59** | 39.16 / 45.83 | 39.14 / 48.73 | 39.03 / 49.41 |
| Vision-80% | 18.28 / 195 | 39.18 / 46.23 | 39.15 / 46.33 | 39.07 / 48.83 | 39.08 / 48.74 |
| Vision-50% | 2.10 / 2249 | **38.63 / 47.95** | 38.70 / 48.04 | 38.66 / 49.81 | 38.57 / 50.33 |
| Index-based baseline | — | — | — | — | **39.10 / 47.58** |

**Key Observation**: $8\times 8$ nearly reaches the accuracy of $80\times 80$; Vision-50% with severe occlusion drops by less than 0.6 points.

### Ablation Study: Hot-Start and Efficiency (RQ2)

| Training Samples | Index baseline | $8\times 8$ Vision | $40\times 40$ Vision |
|------------|----------------|--------------------|---------------------|
| 4,096 | 4.30% | 4.19% (-0.11) | 13.06% (+8.76) |
| 6,152 | 4.61% | 5.57% (+0.96) | 14.7% (+10.09) |
| 8,200 | 5.84% | **12.34%** (+6.5) | 15.46% (+9.62) |
| 10,248 | 8.45% | 13.94% (+5.49) | 15.92% (+7.47) |

Efficiency analysis (zhwp + RTX 5090 Laptop GPU):

| Config | Params | FLOPs | samples/sec | Acc@8k | Final Acc |
|------|--------|-------|-------------|--------|-----------|
| Text (index) | 18.97M | 26.30G | 347.3 | 5.30% | 39.10% |
| Vision-100% (orig.) | 26.45M | 30.61G (+16%) | 314.3 | 8.88% | 39.21% |
| Vision-100% (opt.) | 22.32M | 29.56G (+12%) | 306.9 | 8.75% | 39.18% |
| **Vision-100% (simp.)** | **12.61M** | 28.14G (+7%) | 323.4 | 6.97% | 39.19% |

### Key Findings
- **Hot-start is real**: The $8\times 8$ visual model reaches 12.34% at 8.2k samples, 2.1x the index baseline's 5.84%; the $40\times 40$ model hits 13.06% at 4.1k samples, 3x the baseline's 4.30%.
- **Higher resolution triggers hot-start earlier**, but all eventually converge to ~39% accuracy, implying the "visual prior" mainly impacts early convergence speed rather than the asymptotic limit.
- **"Toast-center" effect**: Attention is concentrated on central strokes (≈30% of pixels carry most discriminative power), allowing edge pixels to be discarded—this explains the robustness of Vision-50%.
- **Vision (simp.) uses fewer parameters (12.61M) + only +7% FLOPs** to achieve full hot-start gains; its net training efficiency is better than the text baseline (8k vision samples at 6.97% > 10k text samples at 6.26%).
- Replicated on Chinese Wikipedia 2019: $8\times 8$ vision reaches 8.88% at 8k samples vs. 5.30% for text; final accuracy 32.4% vs. 32.1%.

## Highlights & Insights
- **Counter-intuitive evidence for "vision as a prior"**: While intuitively Chinese characters are visual, mainstream LMs discard glyphs. This paper provides hard evidence via a controlled study that "discarding them actually hurts."
- **"Hot-start" as a new metric**: Instead of comparing representations solely on final accuracy, it focuses on accuracy differences during the "early 1% of training," characterizing the true value of a prior.
- **Minimalist encoder is better**: Contrary to the "bigger is better" trend in NLP, $8\times 8$ inputs do not need large networks; over-engineering hinders efficiency. This suggests tasks with low resolution or low token counts can benefit from aggressive model sizing.

## Limitations & Future Work
- Only verified on GPT-2-small; whether larger LMs (some scaling analysis was included but focused on small models) benefit from the same hot-start remains to be seen.
- Only tested next-character prediction; whether visual representations match index-based ones in downstream tasks (QA, reasoning, translation) needs further study.
- Some conclusions (e.g., toast-center) are based on qualitative visualization and lack rigorous attribution analysis.
- Did not explore hybrid visual representations with modern tokenization like multi-character tokens or subwords.

## Related Work & Insights
- **vs Glyce / ChineseBERT**: They use glyphs as auxiliary info on top of IDs; this work replaces IDs entirely to enable clean attribution.
- **vs PIXEL / PIXAR**: They are visual-in visual-out; this work is visual-in token-out, facilitating easier comparison with index baselines.
- **vs DeepSeek-OCR / Pix2Struct**: They focus on OCR/transcription; this work focuses on "language modeling using vision," a completely different goal.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of the "Hot-start phenomenon" and "$8\times 8$ is enough" is novel; the paradigm (visual-in, token-out for LM) is also rare.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered resolution, cropping, efficiency, and scale, though downstream task coverage is weak.
- Writing Quality: ⭐⭐⭐⭐ Clear RQ-driven structure with high information density in tables.
- Value: ⭐⭐⭐⭐ Insights into representation design for Chinese LMs, low-resource training, and interpretable representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](../../AAAI2026/llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[ACL 2026\] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)

</div>

<!-- RELATED:END -->
