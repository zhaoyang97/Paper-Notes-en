---
title: >-
  [Paper Note] From Evaluation to Defense: Advancing Safety in Video Large Language Models
description: >-
  [ICLR2026][Recommender Systems][video LLM safety] This work constructs VideoSafetyEval (11.4k video-query pairs covering 19 risk categories), revealing that the video modality degrades safety performance by 34.2%, and proposes VideoSafety-R1, a three-stage framework (Alarm Token + SFT + Safety-guided GRPO) that improves defense success rate by 71.1% on VSE-HH.
tags:
  - ICLR2026
  - Recommender Systems
  - video LLM safety
  - benchmark
  - alarm token
  - GRPO
  - safety alignment
date: 2026-05-08
content_hash: bd6d3d75e8d2f0f6
---

# From Evaluation to Defense: Advancing Safety in Video Large Language Models

**Conference**: ICLR2026  
**arXiv**: [2505.16643](https://arxiv.org/abs/2505.16643)  
**Code**: To be confirmed  
**Area**: Recommender Systems  
**Keywords**: video LLM safety, benchmark, alarm token, GRPO, safety alignment

## TL;DR
This work constructs VideoSafetyEval (11.4k video-query pairs covering 19 risk categories), revealing that the video modality degrades safety performance by 34.2%, and proposes VideoSafety-R1, a three-stage framework (Alarm Token + SFT + Safety-guided GRPO) that improves defense success rate by 71.1% on VSE-HH.

## Background & Motivation
**Background**: Safety risks in image LLMs have been extensively studied (MMBench, SIUO, SafeVLM, etc.), but safety alignment in video LLMs remains severely underexplored. The temporal dynamics, visual cues, and evolving context in videos introduce subtler and more effective risks than static images.

**Limitations of Prior Work**: Systematic evaluation of 21 mainstream video LLMs reveals that introducing the video modality causes an average drop of 34.2% in defense success rate (DSR), exposing systemic vulnerabilities in multimodal attack exploitation. VideoLLaMA3-2B suffers a DSR drop as large as 79.4%.

**Research Gap in Safety**: Existing defense methods (SafeVLM, SPA-VL, MM-RLHF) focus exclusively on static images, neglecting video safety. Video anomaly detection (VAD) is related but distinct — VAD targets abnormal event detection, whereas safety alignment governs model behavioral responses to harmful inputs.

**Core Design Philosophy**: Safety alignment should evolve from passive "harm perception" to active "safety reasoning" — models should not only recognize harmful content but also analyze the harmfulness of video-text pairs through a reasoning chain and generate helpful, safe responses.

## Method

### Overall Architecture
VideoSafety-R1 is a post-training framework comprising three innovative components: the VideoSafetyThinking dataset → AT-SFT (Alarm Token-guided safety fine-tuning) → Safety-guided GRPO (safety-guided reinforcement learning).

### Key Designs

1. **VideoSafetyEval (VSE) Benchmark**

    - 11.4k video-query pairs covering 6 major risk categories (violence, controlled substances, pornography, etc.), 19 subcategories, and 10 linguistic communities.
    - Three subsets: VSE-HH (harmful video + harmful query, strongest adversarial setting), VSE-SH (safe video + harmful query), and VSE-SafeQ (safe queries, evaluating over-refusal rate).
    - Data sourced from YouTube, processed via DINOv2 static filtering → commercial video understanding model annotation → template-driven query generation.

2. **Alarm Token-Guided Safety Fine-Tuning (AT-SFT)**

    - Learnable alarm tokens $\mathbf{h}_v^{\text{alarm}}$ are injected at the end of the visual sequence, and $\mathbf{h}_t^{\text{alarm}}$ at the end of the text sequence.
    - Multi-task training objective: $\mathcal{L}_{\text{AT-SFT}} = \mathcal{L}_{\text{base}} + \lambda_1 \mathcal{L}_{\text{ATC}}^v + \lambda_2 \mathcal{L}_{\text{ATC}}^t$
    - Alarm Token Classification (ATC) performs binary classification (harmful/safe) independently for visual and textual modalities, aligning the hidden states of alarm tokens with safety signals.
    - Serves as a "pre-activation" step for the safety mechanism, providing a foundation for subsequent GRPO training.

3. **Safety-guided GRPO**

    - Cold-start phase: 15k samples are used to train structured chain-of-thought reasoning (`<think>` safety reasoning + `<answer>` response + `<vidType>`/`<textType>` dual-modality labels).
    - Composite reward function: $r = r_{\text{format}} + \alpha \cdot r_{\text{ROUGE}} + \gamma_1 \cdot r_v + \gamma_2 \cdot r_t$
    - Dynamic Reward Adaptation (DRA): when both modality classifications are correct, the ROUGE weight is reduced (encouraging diversity); when classifications are incorrect, the ROUGE weight is increased (enforcing alignment with safety references).
    - $\alpha = \alpha_{\min} + (1 - \text{Correct}_v \cdot \text{Correct}_t)(\alpha_{\max} - \alpha_{\min})$

### VideoSafetyThinking Dataset
46k video-query-chain-of-thought triplets: 6k for AT-SFT, 15k for cold-start SFT, and 25k for GRPO training.

## Key Experimental Results

### Main Results: 21 Video LLMs on VSE-HH

| Model | DSR (w/ video)↑ | DSR (w/o video) | DSR Drop↓ | Helpfulness↑ |
|------|------------|-----------|----------|--------|
| Gemini-2.5-Pro | 86.7% | 99.5% | 12.8% | 1.6 |
| GPT-4o | 73.0% | 98.4% | 25.9% | 2.2 |
| VideoLLaMA3-2B | 18.4% | 89.3% | **79.4%** | 2.3 |
| InternVideo2.5-8B | 16.5% | 53.5% | 69.2% | 1.0 |

### VideoSafety-R1 Performance

| Metric | Baseline (VideoLLaMA3-2B) | VideoSafety-R1 | Gain |
|------|------|------|------|
| VSE-HH DSR | 18.4% | — | **+71.1%** |
| MMBench DSR | — | — | **+59.1%** |
| VLGuard | — | — | **+44.3%** |
| FigStep | — | — | **+15.0%** |

### Key Findings
- The video modality causes significant safety degradation across all models — even GPT-4o drops by 25.9%.
- Models that rely more heavily on efficient video encoding (1fps) suffer greater degradation (VideoLLaMA3: −79.4% vs. VideoLLaMA2: −7.3%).
- VideoSafety-R1 achieves the highest DSR on 18 out of 19 subcategories.
- Safety improvements do not substantially harm general capabilities — helpfulness scores remain at reasonable levels.
- The model generalizes to image safety benchmarks (MMBench/VLGuard/FigStep), indicating that safety reasoning ability is transferable across modalities.

## Highlights & Insights
- This is the first large-scale real-world video LLM safety benchmark — grounded in YouTube community guidelines and closely aligned with practical scenarios.
- The progressive safety alignment design — from perception (AT-SFT alarm tokens) to reasoning (Safety-guided GRPO chain-of-thought) — generates helpful safety responses rather than simple refusals.
- The Dynamic Reward Adaptation mechanism elegantly balances safety and response quality — relaxing ROUGE constraints when classification is correct to encourage natural replies.
- The independent dual-modality annotation design (video harmfulness vs. text harmfulness) enables the model to distinguish risks originating from different sources.

## Limitations & Future Work
- Binary harmfulness labels (harmful/safe) may be overly coarse; fine-grained risk levels are not considered.
- Over-refusal (false rejection rate) requires trade-off analysis against safety — the VSE-SafeQ subset enables such evaluation but is not thoroughly analyzed in the paper.
- The base model is VideoLLaMA3-2B (2B parameters); the effectiveness on larger models (7B+) is not sufficiently validated.
- Annotation quality for the 46k training samples depends on commercial LLMs, introducing potential annotation bias.
- Evaluation relies on the Qwen-Long API as a judge, which may introduce evaluation bias.

## Related Work & Insights
- **vs. SafeVLM/SPA-VL**: Both focus on static image safety; this paper is the first to systematically address video safety.
- **vs. Video Anomaly Detection (UCF-Crime/XD-Violence)**: VAD detects anomalous events; this paper controls model behavioral responses — the objectives are fundamentally different.
- **vs. MM-RLHF**: Uses DPO for visual safety alignment; this paper employs GRPO with rule-based rewards — offering greater controllability.
- **vs. SafeWatch-Bench**: Focuses on video content safety understanding; this paper targets model response safety alignment — the two represent complementary directions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic work on video LLM safety, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 21 models across 4 safety benchmarks with multi-component ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with three components presented in progressive layers.
- Value: ⭐⭐⭐⭐⭐ Establishes benchmark and methodological foundation for video LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](../../NeurIPS2025/recommender/r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)

</div>

<!-- RELATED:END -->
