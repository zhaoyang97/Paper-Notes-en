---
title: >-
  [Paper Note] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis
description: >-
  [ACL 2026][LLM Evaluation][Video caption evaluation] This paper proposes VC-Inspector, a reference-free video caption evaluation metric based on lightweight open-source multimodal models (Qwen2.5-VL 3B/7B). By utilizing…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Video caption evaluation"
  - "reference-free evaluation"
  - "factual accuracy"
  - "Large Vision-Language Models"
  - "hallucination detection"
date: 2026-05-08
content_hash: da0310cb07502cf0
---

# VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis

**Conference**: ACL 2026  
**arXiv**: [2509.16538](https://arxiv.org/abs/2509.16538)  
**Code**: [https://dipta007.github.io/VC-Inspector](https://dipta007.github.io/VC-Inspector)  
**Area**: Video Understanding / Caption Evaluation  
**Keywords**: Video caption evaluation, reference-free evaluation, factual accuracy, Large Vision-Language Models, hallucination detection

## TL;DR

This paper proposes VC-Inspector, a reference-free video caption evaluation metric based on lightweight open-source multimodal models (Qwen2.5-VL 3B/7B). By utilizing a controllable factual error synthesis pipeline to generate training data, it achieves a Spearman correlation of $\tau_b$=42.58 on VATEX-Eval, surpassing the GPT-4o-reliant G-VEval ($\tau_b$=39.40), and reaches 99.6% accuracy on hallucination detection benchmarks.

## Background & Motivation

**Background**: Video caption evaluation primarily relies on text-matching metrics against reference captions (BLEU, ROUGE, CIDEr). However, these are costly to obtain and struggle to capture semantic equivalence. Reference-free evaluation is a more practical direction but remains underdeveloped.

**Limitations of Prior Work**: (1) Reference-free metrics based on pre-trained vision-language embeddings (e.g., EMScore, CLIPScore) are limited by the context length of text encoders and lack a consistent scoring scale—different captions for the same video yield very narrow score ranges, making quality differentiation difficult; (2) Methods using large proprietary models like GPT-4o (e.g., G-VEval) rely on prompt engineering and are non-reproducible; (3) Most existing methods are image-centric and fail to model the temporal dynamics of video.

**Key Challenge**: Reliable evaluation should prioritize factual accuracy—errors in objects and actions should linearly decrease scores according to severity. However, existing metrics fail to detect even basic factual inconsistencies (e.g., incorrect objects).

**Goal**: Construct a fact-based, interpretable, open-source, and lightweight reference-free evaluation metric for video captions.

**Key Insight**: The primary bottleneck in training a fact-aware evaluator is the lack of annotated captions across different factual quality levels—existing captions are either correct or incorrect without intermediate gradients. The authors designed an LLM-based controllable factual error synthesis pipeline to address this data scarcity.

**Core Idea**: Use an LLM to systematically replace objects and actions in ground truth captions to generate pseudo-captions with varying error levels, supplemented with deterministic scores and explanatory labels, to fine-tune a lightweight multimodal model as an evaluator.

## Method

### Overall Architecture

The workflow consists of two steps: (1) Data Generation—Starting from ground truth captions in ActivityNet-Captions, Llama-3.3-70B is used for controllable replacement of objects and actions to generate pseudo-captions with deterministic quality scores (1-5) and error explanations; (2) Model Training—Qwen2.5-VL (3B/7B) is fine-tuned using LoRA, with the visual encoder and projection layer frozen. The input is the video and a candidate caption, and the output consists of a quality score and a factual error explanation.

### Key Designs

1.  **Controllable Factual Error Synthesis Pipeline**:
    - **Function**: Addresses the bottleneck of lacking multi-gradient factual quality training data.
    - **Mechanism**: Given a ground truth caption $X$, an LLM extracts an object set $\mathcal{O}$ and an action set $\mathcal{A}$. Then, $K \sim \text{Unif}(0,M)$ objects and $L \sim \text{Unif}(0,N)$ actions are randomly sampled for replacement. Replacements must be within the same category but have different meanings (e.g., car→truck instead of car→building). Scores are calculated using a deterministic formula: $score = 1 - |\mathcal{R}|/(|\mathcal{O}|+|\mathcal{A}|)$, discretized into a 1-5 scale. 10 pseudo-captions are generated per ground truth, resulting in 44K training instances (ActivityNet-FG-It) after balanced sampling.
    - **Design Motivation**: Compared to PAC-S/FactVC, which only perform binary positive/negative contrast, this method generates captions with multi-gradient quality, allowing the evaluator to distinguish subtle quality differences. Deterministic scoring avoids the unreliability of LLMs in floating-point comparisons.

2.  **Joint Score-Explanation Training Paradigm**:
    - **Function**: Enhances evaluation interpretability and strengthens factual anchoring.
    - **Mechanism**: The model not only predicts a quality score $S \in \{1,...,5\}$ but also generates a text explanation $E$ specifying which objects/actions are incorrect. The explanation serves as an auxiliary supervision signal to help the model learn better factual anchoring. Ablation studies show that adding explanations improves $\tau_b$ on VATEX-Eval from 34.29 to 37.99 (+3.7 points).
    - **Design Motivation**: Existing metrics output only a single scalar score. Explanations provide a basis for judgment and can serve as feedback to guide caption improvement—experiments show that using VC-Inspector's explanations to guide iterative refinement with Qwen2.5-VL improves caption quality across multiple dimensions.

3.  **Video-native Fact Anchoring Architecture**:
    - **Function**: Utilizes video encoders to capture temporal dynamics and supports long-context reasoning.
    - **Mechanism**: Based on Qwen2.5-VL (32K context length) as the backbone, the visual encoder and projection layer are frozen, while the LLM part is fine-tuned via LoRA ($\alpha=r=32$, dropout=0.05). 32 frames are uniformly sampled per video at 224x224 resolution. Training follows standard language modeling loss, and inference uses temperature=0 to ensure reproducibility.
    - **Design Motivation**: Compared to image-encoder-based metrics (e.g., EMScore based on CLIP), video-native models capture temporal information like actions and event sequences. Unlike G-VEval (reliant on GPT-4o with only 3 concatenated frames), Qwen2.5-VL natively supports video input.

### Loss & Training

Standard language modeling loss (next-token prediction) is used with LoRA fine-tuning. Global batch size is 128, learning rate is 1e-4, and training takes approximately 32 GPU hours on 4×A100 GPUs.

## Key Experimental Results

### Main Results

**Human Correspondence Results on VATEX-Eval (Reference-free)**

| Method | $\tau_b$ | $\rho$ | Model Size | Open Source |
| :--- | :--- | :--- | :--- | :--- |
| VC-Inspector-7B | **42.58** | **45.99** | 7B | ✓ |
| G-VEval | 39.40 | - | GPT-4o | ✗ |
| VC-Inspector-3B | 37.99 | 42.45 | 3B | ✓ |
| Qwen2.5-VL-7B | 34.70 | 39.40 | 7B | ✓ |
| ViCLIPScore | 30.92 | 39.86 | - | ✓ |
| EMScore | 22.88 | 29.79 | - | ✓ |
| CLIPScore | 22.33 | 29.09 | - | ✓ |

**Flickr8K-Expert/CF Results (Reference-free, $\tau_b$)**

| Method | Expert | CF |
| :--- | :--- | :--- |
| VC-Inspector-7B | **63.4** | **46.0** |
| VC-Inspector-3B | 59.9 | 39.0 |
| HICE-S | 55.9 | 37.2 |
| PAC-S | 53.9 | 36.0 |
| CLIPScore | 51.1 | 34.4 |

### Ablation Study

| Config | $\tau_b$ (VATEX-Eval) | Description |
| :--- | :--- | :--- |
| Obj + Act (Full) | 37.99 | Best |
| Obj Only | 36.40 | -1.59 |
| Act Only | 33.23 | -4.76 |
| w/o Explanation | 34.29 | Explanation gives +3.7 gain |

**Hallucination Detection Accuracy**

| Method | FOIL-COCO | ActivityNet-FOIL |
| :--- | :--- | :--- |
| VC-Inspector-3B | **99.6** | **99.3** |
| FLEUR | 96.8 | - |
| PAC-S | 90.2 | 91.0 |

### Key Findings

- VC-Inspector-7B not only outperforms all reference-free methods but even exceeds most reference-based metrics in the reference-free setting.
- Both object and action errors are crucial, but object errors contribute more to evaluation quality ($\tau_b$=36.40 for objects-only vs. 33.23 for actions-only).
- Training with auxiliary explanations significantly boosts performance (+3.7 $\tau_b$ points) and enables iterative caption quality improvement.
- Computational efficiency is superior to existing methods: 0.30s/video vs. 0.42s for EMScore (single A100).

## Highlights & Insights

- The deterministic scoring mechanism (based on replacement ratio) is superior to model or human scoring—it avoids subjectivity and inconsistency while ensuring scores remain in a fixed 0-1 range with stable ranking.
- Explanations are not just for interpretability but serve as effective training signals—this "score + explanation" joint training paradigm is transferable to other evaluation tasks (e.g., text summarization, dialogue quality).
- Achieving SOTA results on Flickr8K by treating images as single-frame videos demonstrates that the model's factual anchoring ability generalizes across modalities.

## Limitations & Future Work

- Currently focuses only on two types of factual errors: objects and actions. It does not yet cover fine-grained errors like attributes (color, size), spatial relationships, or temporal ordering.
- Training data is limited to ActivityNet; generalization to highly specialized videos (medical, industrial) requires verification.
- Evaluation dimensions could be expanded to include temporal consistency, level of detail, and style adaptation.

## Related Work & Insights

- **vs EMScore**: EMScore relies on CLIP image encoder frame/video-level embedding matching, restricted by context length and lacking factual anchoring. VC-Inspector uses an LVLM to reason directly about factual correctness.
- **vs G-VEval**: G-VEval relies on GPT-4o with only 3 concatenated frames and is non-reproducible. VC-Inspector is open-source, lightweight (3B/7B), uses native video encoding, and performs better.
- **vs PAC-S/FactVC**: These only perform binary positive/negative synthesis. VC-Inspector allows for more nuanced evaluation through multi-gradient quality data synthesis.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of controllable factual error synthesis and joint score-explanation training is a clever design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five evaluation benchmarks, multiple settings, and detailed ablation/efficiency analyses are provided.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous experimental logic.
- Value: ⭐⭐⭐⭐⭐ Provides the first open-source tool for factual evaluation of video captions, directly usable as a reward model for RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/llm_evaluation/vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[ACL 2026\] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)

</div>

<!-- RELATED:END -->
