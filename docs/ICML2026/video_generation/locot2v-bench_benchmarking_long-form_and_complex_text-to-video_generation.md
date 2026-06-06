---
title: >-
  [Paper Note] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation
description: >-
  [ICML 2026][Video Generation][Long-form Video Generation Benchmark] LocoT2V-Bench is a professional benchmark for **long-form + complex scene** generation…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Long-form Video Generation Benchmark"
  - "Complex Text Alignment"
  - "Hierarchical Metadata"
  - "Character Consistency"
date: 2026-05-08
content_hash: 7012868df7f5c161
---

# LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation

**Conference**: ICML 2026  
**arXiv**: [2510.26412](https://arxiv.org/abs/2510.26412)  
**Code**: TBD  
**Area**: Video Generation / Multimodal VLM / Evaluation Benchmarks  
**Keywords**: Long-form Video Generation Benchmark, Complex Text Alignment, Hierarchical Metadata, Character Consistency

## TL;DR
LocoT2V-Bench is a professional benchmark for **long-form + complex scene** generation, featuring 234 real-world videos across 18 themes with prompts averaging 249 words. It introduces the LoCoT2V-Eval framework with 5 dimensions and 17 sub-dimensions (including hierarchical VQA, conditional gating, and an Auditor-Evaluator dual-agent HERD). Systematic evaluation of 17 models reveals a common performance bottleneck: "Strong perceptual quality, weak fine-grained alignment, and poor character consistency."

## Background & Motivation

**Background**: Text-to-Video (T2V) has seen significant progress in short videos, but long-form generation (> 10 seconds, multi-scene, complex spatio-temporal dynamics) remains an open challenge. Existing benchmarks (e.g., VBench, EvalCrafter) target short videos with simplified prompts, making them unsuitable for evaluating complex scenes.

**Limitations of Prior Work**:
- Primarily focus on frame-level visual quality and global prompt consistency, ignoring fine-grained alignment (character attributes, specific actions).
- CLIP-Score and FID are inadequate for long videos and complex multi-scene prompts.
- Insufficient evaluation of character consistency, long-term temporal coherence, and high-level narrative expression.

**Key Challenge**: The gap between professional-grade control requirements for long videos (precise character settings, camera movements, multi-scene coherence) and current simplified evaluation frameworks.

**Goal**: 
- Construct a long-video benchmark oriented toward professional production workflows (234 real videos, 18 themes, structured multi-scene prompts).
- Design a comprehensive multi-dimensional evaluation framework covering Perceptual Quality, Text Alignment, Temporal Coherence, Dynamic Quality, and Human Expectation fulfillment.

**Key Insight**: Leveraging real-world videos to derive hierarchical metadata (scene, character, background, camera) and employing multi-round conditional VQA for more precise evaluation.

**Core Idea**: **Hierarchical VQA + Conditional Gating** combined with **Auditor-Evaluator Dual-Agent HERD** to systematically assess fine-grained alignment and high-level goal achievement in long-form generation.

## Method

### Overall Architecture
The framework consists of two core modules:
- **Data Construction**: 234 videos collected from YouTube, with complex prompts for multiple scenes and hierarchical metadata generated via MLLM, LLM, and manual verification.
- **Evaluation**: The LoCoT2V-Eval framework assesses generated videos across 5 major dimensions and 17 sub-dimensions.

### Key Designs

1. **Hierarchical VQA + Conditional Gating (Fine-grained Alignment)**:
    - **Function**: Transitions text-video alignment from coarse-grained (global CLIP) to fine-grained (scene-by-scene, character-by-character verification).
    - **Mechanism**: A tree-like multi-round QA framework performs hierarchical verification for each scene: Scene Existence Gating → Character Localization & Attribute Verification → Background & Camera Verification. A "Locate Query" ("Is there a man with a red hat?") anchors the character first, followed by a "Judge Query" ("Is this man tall?") to verify attributes, preventing hallucinatory scoring. Character attributes are calculated as $f^c_{\text{attr}} = \frac{1}{N_c} \sum_k y_k$, and actions as $f^c_{\text{action}} = a^c_s \cdot \frac{1}{M_c} \sum_q A(q \mid H_{N_c})$ (where $a^c_s$ is the anchor flag).
    - **Design Motivation**: Complex prompts require multi-level detail verification; conditional gating prevents models from receiving action scores if character localization fails (illusory scoring); multi-round dialogue history $H_k = H_{k-1} \cup \{(q^{c, k}, y_k)\}$ ensures subsequent queries are grounded in verified context.

2. **Multi-dimensional Evaluation Framework**:
    - **Function**: Systematically evaluates Perceptual Quality (PQ), Text-Video Alignment (TVA: Global OA + Fine-grained FGA), Temporal Quality (TQ: CC / BC / WE), Dynamic Quality (DQ), and Human Expectation Realization Degree (HERD).
    - **Mechanism**:
      - **PQ**: DeQA-Score with multi-scale frame sampling: $PQ(v) = \frac{1}{|W|} \sum_w \frac{1}{n_\alpha} \sum_{f \in w} \text{DeQA}(f)$.
      - **OA**: Qwen3-VL-8B replaces CLIP, scoring 0-100 to capture character/scene/interaction consistency.
      - **CC**: SAM3 tracking → MLLM verification → FG-CLIP2 embedding similarity.
      - **BC / WE**: Adjacent frame FG-CLIP2 + Optical Flow, computed via streaming to avoid OOM in long videos.
      - **DQ**: Aggregates frame-level (motion, smoothness) and high-level (segment/video-level non-periodicity + info flow).
    - **Design Motivation**: Current benchmarks lack fine-grained alignment and character identity consistency; this framework covers the full chain from low-level pixels to high-level semantics.

3. **Auditor-Evaluator Dual-Agent (HERD Evaluation)**:
    - **Function**: Reduces subjective bias in assessing how well the video satisfies human expectations implied by the prompt.
    - **Mechanism**: The Auditor agent independently analyzes the video (without seeing the reference expectations) to generate an objective content report. The Evaluator then combines this report with the video to score 6 dimensions (Sentiment, Narrative, Character Development, Visual Style, Thematic Expression, General Impression) from 1-5. Weighted aggregation yields $S_{\text{HERD}} = \frac{1}{|D|} \sum_d s_d$.
    - **Design Motivation**: Single-agent evaluation is prone to first-impression bias or hallucinations; decoupling roles (reporting vs. evaluating) enhances objectivity and auditability.

### Data Construction Comparison

| Benchmark | Samples | Avg. Word Count | Complexity | Features |
|------|------|---------|--------|------|
| EvalCrafter | 700 | 12.33 | 3.74 | Basic short video |
| VBench-Long | 946 | 7.64 | 2.54 | Simplified long video |
| VBench 2.0 | 90 | 125.46 | 8.13 | Complex single scene |
| **LocoT2V-Bench** | **234** | **248.85** | **8.70** | **Long + Complex + Hierarchical Metadata** |

## Key Experimental Results

### Main Results (Selected from 17 Long-form Video Models)

| Method | PQ | OA | FGA | TVA Mean | CC | BC | TQ Mean | HERD | DQ | Total |
|------|--------|--------|---------|--------|--------|--------|-------|------|--------|------|
| FreeNoise | 73.89 | 18.12 | 10.38 | 14.25 | 15.38 | 98.77 | 69.85 | 53.65 | 50.55 | 52.44 |
| DiTCtrl | 56.55 | 48.25 | 45.54 | 46.90 | 25.72 | 96.86 | 72.50 | 60.75 | 49.37 | 57.21 |
| LongLive | 80.51 | 55.50 | 36.15 | 45.83 | 54.92 | 99.18 | 83.66 | 81.30 | 61.52 | 70.56 |
| LongCat-Video | 77.75 | 65.59 | 51.01 | 58.30 | 42.08 | 98.31 | 78.45 | 84.80 | 59.29 | 71.72 |
| Sora2 | 66.59 | 69.64 | 54.09 | 61.87 | 45.40 | 99.10 | 80.97 | 86.42 | 64.78 | 72.13 |
| Kling 3.0 | 70.26 | 73.08 | 56.94 | 65.01 | 36.97 | 98.96 | 78.55 | 87.47 | 56.16 | 71.49 |

### Key Findings
- **Strong Perception, Weak Fine-grained Alignment**: PQ scores are 70-84%, but FGA scores are only 10-56% (a 2-7x difference). Models generate high-quality frames but struggle to follow complex textual constraints precisely.
- **Superior Background Stability, Poor Character Consistency**: BC is generally 95-99%, but CC is mostly < 50% (even the best, CausVid, reaches only 45.97%). Environmental stability is easier to maintain than character identity.
- **Huge Gap between Global and Fine-grained Alignment**: OA is 50-73%, but FGA drops by an average of 40 percentage points. MLLMs tend to give optimistic global ratings while overlooking missing finer details.
- **Kling 3.0 and Sora2 Lead**: HERD scores peak at 87.47% and 86.42%, respectively. Proprietary models show significantly stronger alignment with human expectations.
- **Multi-prompt vs. Direct Input**: Direct input methods (CausVid, SkyReels-V2) generally outperform multi-prompt decomposition methods (FreeNoise, MEVG) in FGA, suggesting end-to-end models handle complex text better.

## Highlights & Insights
- **Hierarchical Metadata Design**: Unlike previous methods where LLMs directly generate long descriptions, this work reverse-engineers 4D metadata (Scene-Character-Background-Camera) from real videos to provide a ground truth for fine-grained evaluation.
- **Conditional Gating VQA**: The "Locate → Judge" transition and multiplication gate $a^c_s$ prevent hallucinatory scoring; this approach is transferable to other multi-round reasoning evaluation tasks.
- **Auditor-Evaluator Decoupling**: Mimics the film review process (Content Analysis vs. Quality Rating) to eliminate single-agent hallucination/bias and improve the reliability of subjective metrics like HERD.
- **Streaming Evaluation**: Converts memory-intensive algorithms from EvalCrafter/VBench into streaming formats (multi-scale sampling, streaming CLIP/Optical Flow), enabling support for ultra-long videos.
- **Complex Prompt Library**: With 248.85 words and 8.70 complexity, this is the most challenging benchmark to date, reflecting the density of constraints in professional video production.

## Limitations & Future Work
- The sample size of 234 is relatively small and may not cover all extreme edge cases.
- The 6 dimensions of HERD are inherently subjective; discrepancies may exist between GPT-5's generated expectations and actual user expectations.
- Character consistency relies on SAM3 tracking, which may accumulate errors during complex motions, partial occlusions, or long trajectories.
- The evaluation pipeline depends on multiple models, making deployment complex and affecting historical comparability when tools are upgraded.
- Future Work: Expand sample size to 500-1000, incorporate real user evaluation to validate HERD, and improve character tracking robustness.

## Related Work & Insights
- **vs. VBench / EvalCrafter**: These use simplified prompts for short videos; Ours uses complex multi-scene prompts + hierarchical metadata + fine-grained evaluation.
- **vs. VBench 2.0**: VBench 2.0 uses complex prompts (125 words) but only 90 samples; Ours uses 249 words × 234 samples across 18 themes, with source-based prompts to reduce hallucination.
- **vs. Multi-prompt Methods** (Vlogger, StoryAdapter): These decompose long prompts using LLMs; Ours shows that direct input methods currently perform better, as decomposition may lose context.
- **Insight**: The fine-grained evaluation framework (Conditional VQA) is transferable to 3D generation and image editing; hierarchical metadata serves as a blueprint for systematically building complex prompt benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically introduce hierarchical metadata, conditional gating VQA, and HERD for long-form video.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 17 representative models (both multi-prompt and direct input), exposing common bottlenecks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, precise methodology, well-organized experiments, and actionable conclusions.
- Value: ⭐⭐⭐⭐⭐ Provides the most comprehensive benchmark for long-form video; guides model improvements; the design logic is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](../../ACL2026/video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[ICML 2026\] Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)

</div>

<!-- RELATED:END -->
