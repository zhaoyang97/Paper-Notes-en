---
title: >-
  [Paper Note] Asking like Socrates: Socrates helps VLMs understand remote sensing images
description: >-
  [CVPR 2026][Multimodal VLM][Remote sensing image understanding] This paper identifies the "pseudo-reasoning" phenomenon in remote sensing VLMs—where explicit reasoning chains actually degrade performance—attributing it t…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Remote sensing image understanding"
  - "evidence-chain reasoning"
  - "pseudo-reasoning"
  - "Socratic method"
  - "two-stage reinforcement learning"
date: 2026-05-08
content_hash: d9822168776fd042
---

# Asking like Socrates: Socrates helps VLMs understand remote sensing images

**Conference**: CVPR 2026
**arXiv**: [2511.22396](https://arxiv.org/abs/2511.22396)  
**Code**: [https://geox-lab.github.io/Asking_like_Socrates](https://geox-lab.github.io/Asking_like_Socrates)  
**Area**: Remote Sensing / Multimodal Reasoning
**Keywords**: Remote sensing image understanding, evidence-chain reasoning, pseudo-reasoning, Socratic method, two-stage reinforcement learning

## TL;DR
This paper identifies the "pseudo-reasoning" phenomenon in remote sensing VLMs—where explicit reasoning chains actually degrade performance—attributing it to the "Glance Effect" (insufficient single-pass perception). It proposes RS-EoT (Evidence-of-Thought), an iterative evidence search paradigm. A SocraticAgent self-play mechanism synthesizes reasoning trajectories for SFT cold-start, followed by two-stage progressive RL (grounding → VQA) to enhance and generalize reasoning. RS-EoT-7B achieves state-of-the-art performance across multiple remote sensing VQA and grounding benchmarks.

## Background & Motivation

**Background**: Deep reasoning models (DeepSeek-R1-style SFT-RL paradigm) have achieved breakthroughs in mathematics and code, and have been extended to multimodal settings (Vision-R1, WeThink, R1-OneVision, etc.). However, anomalous behavior emerges in remote sensing tasks.

**Pseudo-Reasoning Problem**: Remote sensing VLMs generate explicit reasoning chains, yet performance shows **no improvement or even degrades**. Models merely "narrate reasoning processes" rather than "genuinely reason."

**Glance Effect**: Remote sensing images cover large spatial extents with significant scale variation and sparse, subtle visual cues. Models perform a single shallow perception pass ("one glance") before reasoning, leading to reasoning based on incomplete visual evidence—reasoning degenerates into linguistically self-consistent narration rather than evidence-grounded logic.

**Key Challenge**: Remote sensing reasoning requires iterative, non-static evidence acquisition, whereas existing models adopt a "glance-then-reason" paradigm. Human remote sensing analysts employ repeated inspection-refinement cycles.

**Core Idea**: RS-EoT — reasoning guides perception, dynamically searching for new visual evidence during the reasoning process (reasoning → perception → reasoning → perception... loop), rather than relying on a fixed initial view.

## Method

### Overall Architecture
**SFT Cold-Start** (SocraticAgent synthesizes RS-EoT-4K dataset) → **Stage 1 RL: Grounding** (IoU reward enhances evidence search capability) → **Stage 2 RL: VQA** (multiple-choice reconstruction + graded reward generalizes reasoning) → RS-EoT-7B.

### Key Designs

1. **SocraticAgent (RS-EoT Reasoning Trajectory Synthesis)**:

    - Function: Synthesizes reasoning trajectories with iterative evidence-search characteristics from scratch.
    - **Reasoner** (GPT-5-mini): Pure text reasoning without image access. Responsible for reasoning, querying the Perceiver, and integrating feedback.
    - **Perceiver** (Gemini-2.5-flash): Has image access but no original task query. Only answers questions posed by the Reasoner.
    - **Verifier** (doubao-seed-1.6-thinking): Validates the final answer — if the image-blind Reasoner still arrives at the correct answer, the dialogue constitutes a reliable reasoning trajectory.
    - **Self-Play Prompting Mechanism (Core Ingenuity)**: The Reasoner is told "the Perceiver is weak and cannot handle complex questions," forcing it to decompose problems and pose simple, incremental queries; the Perceiver is told "the Reasoner has weak reasoning ability," forcing it to provide concise, accurate answers. This mutual "capability deprecation" strategy ensures detailed, progressive reasoning trajectories.
    - Output: RS-EoT-4K dataset (covering RGB, infrared, and SAR modalities), with a maximum of 6 dialogue turns.

2. **Two-Stage Progressive RL**:

    - **Stage 1: Fine-grained Grounding RL**
        - Function: Reinforces the model's evidence search capability through precise localization tasks.
        - Mechanism: "Sharpening iron with iron" — grounding tasks inherently require progressively refined visual evidence search, most directly reinforcing RS-EoT behavior.
        - Reward: IoU score + format reward.
        - Data: DIOR-RSVG + VRSBench.
    - **Stage 2: General RS VQA RL**
        - Function: Generalizes RS-EoT capability to broad remote sensing understanding scenarios.
        - Problem: Existing RS VQA data predominantly consists of simple Yes/No questions, making reward hacking trivially easy.
        - **Multiple-Choice Reconstruction Strategy**: Exploits the property of multiple QA pairs per image by randomly inverting $n$ answers into incorrect options, constructing multiple-choice questions that require the model to verify each option individually.
        - **Graded Reward**: $r_{qa} = 1 - \frac{1}{N}\sum|y_i - \hat{y}_i|$ — correct selections and correct rejections both yield positive rewards, producing a stable training signal.
        - Design Motivation: Symmetric penalties and equally weighted options force multi-step reasoning and evidence aggregation.

3. **Two Core Principles of the RS-EoT Reasoning Paradigm**:

    - Reasoning is driven by natural language — language serves not only as a descriptive tool but as a controller for perceptual operations.
    - Visual information functions as on-demand evidence — rather than relying on a single global perception pass, the model progressively searches, validates, and integrates local visual evidence according to reasoning demands.

### Loss & Training
SFT uses RS-EoT-4K (5 epochs, lr=3e-5). Two-stage RL uses GRPO (2 epochs each, lr=1e-6, batch=512). Based on Qwen2.5-VL-7B.

## Key Experimental Results

### Main Results (Remote Sensing VQA + Grounding)

| Benchmark | Metric | RS-EoT-7B | Qwen2.5VL | WeThink | VL-Rethinker | Geo-R1 |
|-----------|--------|-----------|-----------|---------|-------------|--------|
| RSFG-VQA | Avg@5 | **67.85** | 62.45 | 55.04 | 58.80 | 45.03 |
| RSFG-SC | Object@F1 | **56.52** | 36.78 | 38.35 | 34.84 | 20.82 |
| VRSBench | Avg@5 | **63.09** | 62.45 | 62.17 | 55.04 | 57.00 |
| RSVQA | Avg@5 | **75.16** | 67.20 | 40.74 | 65.57 | 34.50 |
| DIOR-RSVG | mIoU | **45.29** | 35.64 | 33.96 | 25.48 | 20.97 |
| VRSBench-Ref | mIoU | **48.04** | 21.99 | 34.07 | 25.29 | 4.51 |

RS-EoT-7B achieves **consistent state-of-the-art performance across all VQA and grounding tasks**, with particularly notable gains in Object@F1 (36.78 → 56.52, +53.7%) and grounding mIoU (35.64 → 45.29, +27.1%).

### Ablation Study (Per-Stage Contributions)

| Stage | RSFG-VQA | DIOR mIoU | Notes |
|-------|----------|-----------|-------|
| Qwen2.5-VL Baseline | 62.45 | 35.64 | No reasoning |
| + SFT Cold-Start | +gain | +gain | RS-EoT pattern injection |
| + RL-Grounding | +further | **Large gain** | Evidence search enhancement |
| + RL-VQA | **Best** | Maintained | Generalization to broad VQA |

### Key Findings
- **Quantitative Validation of Pseudo-Reasoning**: Reasoning models such as WeThink perform worse than non-reasoning baselines on RS tasks (Fig. 1a), confirming that pseudo-reasoning is a genuine and widespread problem.
- Attention map analysis of RS-EoT reveals clear alternating "reasoning → evidence search → reasoning" cycles, demonstrating genuine evidence-driven reasoning rather than pseudo-reasoning.
- Grounding RL exhibits positive transfer to VQA tasks — fine-grained localization capability enhances global understanding.
- The multiple-choice reconstruction strategy successfully prevents reward hacking, as evidenced by steadily increasing rather than oscillating reward curves.

## Highlights & Insights
- **Diagnosis of Pseudo-Reasoning**: This work is the first to systematically identify and explain the anomalous phenomenon of "reasoning reducing performance" in remote sensing VLMs; the attribution to the Glance Effect is precise and compelling.
- **Elegance of the Self-Play Prompting Mechanism**: Informing each agent that the other is "weak" forces both to fulfill their respective roles. This is an exceptionally concise and effective prompt engineering technique with broad applicability to other multi-agent data synthesis scenarios.
- **"Sharpening Iron with Iron" Training Philosophy**: First refining the model on grounding tasks—which most directly demand fine-grained evidence search—before generalizing to VQA represents an intuitive curriculum that mirrors skill acquisition.
- **Practical Multiple-Choice Reconstruction Strategy**: Transforming simple Yes/No VQA into RL-friendly formats addresses the reward hacking problem inherent in remote sensing RL training.

## Limitations & Future Work
- RS-EoT currently operates as an in-language loop (alternating reasoning and "self-questioning" in text) without explicitly retrieving image sub-regions; integration with visual grounding tools could enable genuine region retrieval.
- SocraticAgent relies on expensive APIs such as GPT-5-mini and Gemini-2.5-flash for data synthesis.
- Built upon Qwen2.5-VL-7B; effectiveness at larger scales remains unverified.
- Coverage is currently limited to RGB, infrared, and SAR modalities; other remote sensing modalities such as hyperspectral imagery remain to be explored.

## Related Work & Insights
- **vs. Geo-R1/VHM-RL**: These methods adopt SFT-RL but rely on single-pass global perception, leading to pseudo-reasoning in RS tasks. RS-EoT addresses this through iterative evidence search.
- **vs. Vision-R1/WeThink/R1-OneVision**: General-purpose multimodal reasoning models whose performance on RS tasks falls below even the baseline.
- **vs. EagleVision**: The latter actively acquires new viewpoints for spatial reasoning in video; RS-EoT iteratively searches for local evidence within single remote sensing images. Both share the core philosophy of "reasoning-driven perception."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pseudo-reasoning diagnosis + RS-EoT paradigm + SocraticAgent are all entirely original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple VQA and grounding benchmarks, attention visualizations, reward curves, and per-stage ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation (pseudo-reasoning + Glance Effect) is exceptionally clear and compelling.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both remote sensing AI and the broader multimodal reasoning field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)

</div>

<!-- RELATED:END -->
