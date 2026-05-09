---
title: >-
  [Paper Note] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis
description: >-
  [ACL 2026][Video Understanding][Video Caption Evaluation] This paper proposes VC-Inspector, a reference-free video caption evaluation metric based on open-source lightweight multimodal models (Qwen2.5-VL 3B/7B), which generates training data through a controllable factual error synthesis pipeline and achieves $\tau_b$=42.58 human judgment correlation on VATEX-Eval, outperforming GPT-4o-based G-VEval ($\tau_b$=39.40), while reaching 99.6% accuracy on hallucination detection benchmarks.
tags:
  - ACL 2026
  - Video Understanding
  - Video Caption Evaluation
  - Reference-free Evaluation
  - Factual Accuracy
  - Large Multimodal Models
  - Hallucination Detection
date: 2026-05-08
content_hash: 40df38fb14e77d92
---

# VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis

**Conference**: ACL 2026  
**arXiv**: [2509.16538](https://arxiv.org/abs/2509.16538)  
**Code**: [https://dipta007.github.io/VC-Inspector](https://dipta007.github.io/VC-Inspector)  
**Area**: Video Understanding / Caption Evaluation  
**Keywords**: Video Caption Evaluation, Reference-free Evaluation, Factual Accuracy, Large Multimodal Models, Hallucination Detection

## TL;DR

This paper proposes VC-Inspector, a reference-free video caption evaluation metric based on open-source lightweight multimodal models (Qwen2.5-VL 3B/7B), which generates training data through a controllable factual error synthesis pipeline and achieves $\tau_b$=42.58 human judgment correlation on VATEX-Eval, outperforming GPT-4o-based G-VEval ($\tau_b$=39.40), while reaching 99.6% accuracy on hallucination detection benchmarks.

## Background & Motivation

**State of the Field**: Video caption evaluation primarily relies on reference caption-based text matching metrics (BLEU, ROUGE, CIDEr), but these metrics are costly and fail to capture semantic equivalence. Reference-free evaluation is a more practical direction, yet its development lags behind.

**Limitations of Prior Work**: (1) Reference-free metrics based on pretrained vision-language embeddings (e.g., EMScore, CLIPScore) are limited by text encoder context length and lack consistent scoring scales—scores for different captions of the same video show minimal variation, making quality differentiation difficult; (2) Methods using large proprietary models like GPT-4o for scoring (e.g., G-VEval) rely on prompt engineering and are not reproducible; (3) Most existing approaches are image-centric and cannot model temporal dynamics in videos.

**Root Cause**: Reliable caption evaluation should center on factual accuracy—errors in objects and actions should linearly reduce scores according to severity, but existing metrics fail to detect even basic factual inconsistencies (e.g., wrong objects).

**Paper Goals**: Build a factual accuracy-based, interpretable, open-source lightweight reference-free video caption evaluation metric.

**Starting Point**: The observation that the main bottleneck in training factual-aware evaluators is the lack of annotated captions with different factual quality levels—existing captions are either correct or wrong, with no intermediate gradients. The authors design an LLM-based controllable factual error synthesis pipeline to address this data bottleneck.

**Core Idea**: Use LLMs to systematically replace objects and actions in ground truth captions to generate pseudo-captions with varying error degrees, paired with deterministic scores and explanatory annotations, for fine-tuning lightweight multimodal models as evaluators.

## Method

### Overall Architecture

The process consists of two steps: (1) Data Generation—starting from ground truth captions in ActivityNet-Captions, using Llama-3.3-70B to controllably replace objects and actions to generate pseudo-captions, deterministically computing quality scores (1-5 scale), while generating error explanations; (2) Model Training—fine-tuning Qwen2.5-VL (3B/7B) with LoRA, freezing the visual encoder and projection layers, training only the LLM component. Input consists of video + candidate caption, output is quality score and factual error explanation.

### Key Designs

1. **Controllable Factual Error Synthesis Pipeline**:

    - Function: Addresses the bottleneck of lacking multi-gradient factual quality annotated data
    - Mechanism: Given ground truth caption $X$, use LLM to extract object set $\mathcal{O}$ and action set $\mathcal{A}$, randomly sample $K \sim \text{Unif}(0,M)$ objects and $L \sim \text{Unif}(0,N)$ actions for replacement. Replacements require same category but different meaning (e.g., car→truck not car→building). Score computed with deterministic formula: $score = 1 - |\mathcal{R}|/(|\mathcal{O}|+|\mathcal{A}|)$, discretized to 1-5 scale. Each ground truth generates 10 pseudo-captions, balanced sampling yields 44K training instances (ActivityNet-FG-It)
    - Design Motivation: Compared to PAC-S/FactVC which only perform binary positive-negative contrast, this method generates multi-gradient quality captions, enabling the evaluator to distinguish quality differences more finely. Deterministic scoring avoids unreliability of LLM floating-point comparisons

2. **Joint Score-Explanation Training Paradigm**:

    - Function: Enhances evaluation interpretability and strengthens factual grounding
    - Mechanism: The model not only predicts quality score $S \in \{1,...,5\}$, but also generates textual explanation $E$ describing which objects/actions have errors. Explanation serves as auxiliary supervision signal, helping the model learn better factual grounding. Ablation experiments show that adding explanations improves $\tau_b$ on VATEX-Eval from 34.29 to 37.99 (+3.7 points)
    - Design Motivation: Existing metrics only output single scalar scores without explaining reasoning. Explanations not only enhance interpretability but can serve as feedback signals to guide caption improvement—experiments show using VC-Inspector's explanations to guide iterative caption refinement with Qwen2.5-VL improves caption quality across multiple dimensions

3. **Video-Native Factual Grounding Architecture**:

    - Function: Leverages video encoder to capture temporal dynamics, supports long-context reasoning
    - Mechanism: Based on Qwen2.5-VL (32K context length) as backbone, freezing visual encoder and projection layers, fine-tuning only LLM component with LoRA ($\alpha=r=32$, dropout=0.05). Each video uniformly samples 32 frames at 224×224 resolution. Training uses standard language modeling loss, inference uses temperature=0 for reproducibility
    - Design Motivation: Compared to image encoder-based metrics (EMScore based on CLIP), video-native models can capture actions, event sequences and other temporal information. Compared to G-VEval (relying on GPT-4o, concatenating only 3 frames), Qwen2.5-VL natively supports video input

### Loss & Training

Standard language modeling loss (next-token prediction), using LoRA fine-tuning. Global batch size 128, learning rate 1e-4, training ~32 GPU hours on 4×A100 GPUs.

## Key Experimental Results

### Main Results

**Human Judgment Correlation on VATEX-Eval Reference-free Setting**

| Method | $\tau_b$ | $\rho$ | Model Size | Open Source |
|------|---------|--------|---------|------|
| VC-Inspector-7B | **42.58** | **45.99** | 7B | ✓ |
| G-VEval | 39.40 | - | GPT-4o | ✗ |
| VC-Inspector-3B | 37.99 | 42.45 | 3B | ✓ |
| Qwen2.5-VL-7B | 34.70 | 39.40 | 7B | ✓ |
| ViCLIPScore | 30.92 | 39.86 | - | ✓ |
| EMScore | 22.88 | 29.79 | - | ✓ |
| CLIPScore | 22.33 | 29.09 | - | ✓ |

**Flickr8K-Expert/CF Reference-free Setting ($\tau_b$)**

| Method | Expert | CF |
|------|--------|-----|
| VC-Inspector-7B | **63.4** | **46.0** |
| VC-Inspector-3B | 59.9 | 39.0 |
| HICE-S | 55.9 | 37.2 |
| PAC-S | 53.9 | 36.0 |
| CLIPScore | 51.1 | 34.4 |

### Ablation Study

| Config | $\tau_b$ (VATEX-Eval) | Note |
|------|----------------------|------|
| Modify objects+actions (full model) | 37.99 | Best |
| Modify objects only | 36.40 | -1.59 |
| Modify actions only | 33.23 | -4.76 |
| No explanation training | 34.29 | Explanations provide +3.7 gain |

**Hallucination Detection Accuracy**

| Method | FOIL-COCO | ActivityNet-FOIL |
|------|-----------|-----------------|
| VC-Inspector-3B | **99.6** | **99.3** |
| FLEUR | 96.8 | - |
| PAC-S | 90.2 | 91.0 |

### Key Findings

- VC-Inspector-7B in reference-free setting not only outperforms all reference-free methods, but even surpasses most metrics requiring reference captions
- Both object and action errors are important, but object errors contribute more to evaluation quality (objects-only $\tau_b$=36.40 vs actions-only 33.23)
- Explanation-assisted training shows significant improvement (+3.7 $\tau_b$ points), and explanations can be used for iterative caption quality improvement
- Computational efficiency superior to existing methods: 0.30s/video vs EMScore's 0.42s (single A100)

## Highlights & Insights

- Deterministic scoring mechanism (based on replacement ratio) outperforms model/human scoring—avoids subjectivity and inconsistency, while ensuring scores remain in fixed 0-1 range maintaining ordinal relationships
- Explanations serve not only as interpretability tools but also as effective training signals—this "score+explanation" joint training paradigm can transfer to other evaluation tasks (e.g., text summarization evaluation, dialogue quality assessment)
- Treating images as single-frame videos on Flickr8K still achieves best results, indicating the learned factual grounding capability has cross-modal generalization

## Limitations & Future Work

- Currently focuses only on two factual error types—objects and actions, not covering finer-grained errors like attributes (color, size), spatial relations, temporal ordering
- Training data from ActivityNet, generalization ability on highly specialized videos (medical, industrial) remains to be verified
- Evaluation dimensions can be further expanded to temporal consistency, level of detail, style adaptation, etc.

## Related Work & Insights

- **vs EMScore**: Based on CLIP image encoder frame-level/video-level embedding matching, limited by context length and lacking factual grounding. VC-Inspector directly reasons about factual correctness using LMMs
- **vs G-VEval**: Relies on GPT-4o, concatenates only 3 frames, not reproducible. VC-Inspector is open-source lightweight (3B/7B), native video encoding, with superior performance
- **vs PAC-S/FactVC**: Only perform binary positive/negative data synthesis, VC-Inspector generates multi-gradient quality data for finer evaluation

## Rating

- Novelty: ⭐⭐⭐⭐ Controllable factual error synthesis + joint score-explanation training is an elegant combinatorial design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five evaluation benchmarks, multiple settings, ablations and computational efficiency analysis comprehensively covered
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous experimental logic
- Value: ⭐⭐⭐⭐⭐ Provides first open-source video caption factual evaluation tool, directly usable as RL reward model

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking](../../ICLR2026/video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[AAAI 2026\] StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis](../../AAAI2026/video_understanding/stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](../../ICLR2026/video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)

<!-- RELATED:END -->
