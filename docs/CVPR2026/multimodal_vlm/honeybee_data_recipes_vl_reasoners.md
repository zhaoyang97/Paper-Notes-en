---
title: >-
  [Paper Note] HoneyBee: Data Recipes for Vision-Language Reasoners
description: >-
  [CVPR 2026][Multimodal VLM][VLM Reasoning] This paper systematically investigates the design space of VL reasoning training data—covering data source selection, intervention strategy filtering, and three-dimensional scaling across images, questions, and CoTs. Based on the resulting insights, the authors construct the HoneyBee dataset with 2.5M samples. A 3B VLM trained on HoneyBee surpasses the previous SOTA on MathVerse by 7.8pp, and a shared caption decoding strategy for test-time scaling reduces token consumption by 73%.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM Reasoning
  - CoT Data Curation
  - Data Scaling
  - Visual Reasoning
  - Test-Time Scaling
date: 2026-05-08
content_hash: 83d7a6ac9b0463c0
---

# HoneyBee: Data Recipes for Vision-Language Reasoners

**Conference**: CVPR 2026
**arXiv**: [2510.12225](https://arxiv.org/abs/2510.12225)
**Code**: [Dataset](https://huggingface.co/datasets/facebook/HoneyBee)
**Area**: Multimodal VLM / Data Engineering
**Keywords**: VLM Reasoning, CoT Data Curation, Data Scaling, Visual Reasoning, Test-Time Scaling

## TL;DR

This paper systematically investigates the design space of VL reasoning training data—covering data source selection, intervention strategy filtering, and three-dimensional scaling across images, questions, and CoTs. Based on the resulting insights, the authors construct the HoneyBee dataset with 2.5M samples. A 3B VLM trained on HoneyBee surpasses the previous SOTA on MathVerse by 7.8pp, and a shared caption decoding strategy for test-time scaling reduces token consumption by 73%.

## Background & Motivation

**State of the Field**: VLMs increasingly rely on high-quality CoT training data for reasoning tasks. The community has produced multiple VL reasoning datasets (Math-LLaVA, LLaVA-CoT, R1-OneVision, etc.).

**Limitations of Prior Work**:

1. There is no systematic understanding of what constitutes good VL reasoning data—different works use different sources, filtering strategies, and generation models, making it impossible to fairly compare the independent contribution of each factor.
2. The scaling behavior of VL reasoning data (should one scale the number of images, questions, or CoTs?) has never been systematically explored.
3. Many seemingly reasonable data augmentation strategies have not been rigorously validated.

**Root Cause**: The community has been blindly assembling and expanding datasets without understanding the true contribution of each design decision to final reasoning performance.

**Paper Goals**: To systematically understand and optimize the construction process of VL reasoning CoT datasets and provide reproducible "data recipes."

**Starting Point**: A three-stage controlled experiment—context curation (source selection) → data intervention (augmentation/filtering) → large-scale scaling (three dimensions: images, questions, CoTs).

**Core Idea**: Through strictly controlled experiments, the paper reveals three major findings: data source selection dominates over data augmentation strategies; most seemingly reasonable interventions are in fact harmful; and all three scaling dimensions remain unsaturated. These findings guide the construction of the optimal dataset.

## Method

### Overall Architecture

A three-stage data curation pipeline: **context curation** (rank 6 data sources with a fixed CoT generator) → **data intervention** (test 8 perceptual/reasoning augmentation strategies) → **large-scale scaling** (expand along image/question/CoT axes). The final result is the HoneyBee dataset (2.5M samples = 1.5M VL + 1.04M text-only reasoning), used for SFT training of the PLM series VLMs.

### Key Designs

1. **Context Source Ranking**

    - Function: Fairly compare the impact of different data sources on VL reasoning performance within a unified experimental framework.
    - Mechanism: Fix the CoT generator (Llama-4-Scout) and training pipeline; limit each source to 50K samples. Six sources are compared: ViRL, Math-LLaVA, R1-OneVision, Cauldron, PixMo, and MMK12.
    - Key Findings: Performance gaps across sources reach 11.4pp (ViRL best vs. MMK12 worst). **Mixing sources underperforms the single best source**—Top-2/Top-4 mixtures lead to performance degradation.
    - Design Motivation: The choice of data source has a far greater impact than any subsequent intervention strategy.

2. **Data Intervention Strategy Filtering**

    - Function: Rigorously test 8 perceptual/reasoning augmentation strategies to identify genuinely effective interventions.
    - Mechanism: Strategies tested include visual perturbation, text-rich image inclusion, perceptual redundancy filtering, shallow perceptual filtering, Caption-and-Solve (perceptual augmentation), text-only reasoning mixture, distractor augmentation, and length/difficulty filtering (reasoning augmentation).
    - Key Findings: Most interventions **degrade performance** (visual perturbation −1.6pp, text-rich images −1.3pp, perceptual redundancy filtering −3.6pp, difficulty balancing −5.5pp). Only two strategies consistently help: (a) **Caption-and-Solve** (generate an image caption before solving at training time, +3.3pp); (b) **mixing in text-only reasoning data** (+7.5pp).
    - Design Motivation: Data augmentation should not be applied intuitively—rigorous A/B testing is essential.

3. **Three-Dimensional Data Scaling**

    - Function: Independently study the effects of scaling the number of images, questions per image (via synthetic question generation), and CoTs per question.
    - Mechanism: For 39K real images, 16 CoTs per question and 14 synthetic new questions per image are generated; after majority-voting filtering, 1.5M VL samples and 1.04M text-only reasoning samples are obtained.
    - Key Findings: Scaling along all three dimensions simultaneously yields continuous performance improvements, with no saturation observed at 2.5M samples.

### Loss & Training

- Standard SFT: maximize $\log p(C_j \mid I_j, Q_j)$; full-parameter fine-tuning of PLM (including the visual encoder and LLM backbone).
- Train for 5 epochs and select the best checkpoint.
- An additional round of GRPO reinforcement learning yields a further +1.9pp improvement.
- Caption-and-Solve training format: the model first outputs an image caption $C$, then outputs the solution process $S$.

## Key Experimental Results

### Main Results

| Model | Average (5 tasks) | MathVerse | MathVista | MathVision | We-Math |
|------|-----------------|-----------|-----------|------------|---------|
| Qwen2.5-VL-3B | 42.6 | 35.0 | 58.9 | 23.7 | 49.2 |
| PLM-HoneyBee-3B | **46.2** | **42.8** | **61.2** | **29.9** | **59.3** |
| Gain | +8.4% | +22.3% | +3.9% | +26.2% | +20.5% |

| Model Scale | Average | Comparison |
|---------|---------|------|
| PLM-HoneyBee-1B | Outperforms InternVL-3-1B by 28pp | Smallest models also benefit |
| PLM-HoneyBee-8B | 49.8 | Outperforms Qwen2.5-VL-7B (48.5) by +2.7% |

### Ablation Study

| Factor | Finding | Impact |
|------|------|---------|
| Data source selection | ViRL best vs. MMK12 worst | 11.4pp gap |
| Mixing data sources | Top-2/Top-4 mixture underperforms single best source | −0.5~−1.5pp |
| Caption-and-Solve | Independent generation ($I \to C$, $(I,Q) \to S$) works best | +3.3pp |
| Text-only reasoning mixture | OpenThoughts3 re-annotated | +7.5pp |
| Visual perturbation | Harmful | −1.6pp |
| Difficulty balancing | Harmful | −5.5pp |
| 50K→250K→2.5M | Continuous improvement, unsaturated | +4.8pp |
| Shared caption decoding | One caption shared across 64 samples | Token count −73%, no performance drop |

### Key Findings

- **Data source selection has a far greater impact than any intervention strategy**: source gap of 11.4pp vs. best intervention of +7.5pp.
- **Most seemingly reasonable data augmentation strategies are actually harmful**: visual perturbation, text-rich image inclusion, and perceptual redundancy filtering all degrade performance.
- **Key to the success of Caption-and-Solve**: decoupling perception (image captioning) and reasoning (problem solving) into two independent generation processes.
- **All three scaling dimensions remain unsaturated**: further investment in dataset expansion continues to yield gains.
- **Findings from 3B and 8B model experiments are highly correlated**: data selection conclusions from small models generalize to larger models.
- **Shared caption decoding** saves 73% of tokens with no performance degradation.

## Highlights & Insights

1. This is a rigorous data engineering study with excellent experimental controls, similar in spirit to OpenThoughts but focused on multimodal settings.
2. The finding that "most seemingly reasonable data augmentations are actually ineffective" carries strong cautionary value for the community.
3. **Shared caption decoding** is highly practical: the model generates an image description before solving during training, and the description is reused across multiple sampling runs at inference time.
4. The finding that 2.5M samples remain unsaturated points to a clear direction for continued investment.

## Limitations & Future Work

1. The study focuses exclusively on mathematical reasoning tasks; data curation principles for general VL tasks such as VQA and image understanding remain unexplored.
2. Only a single teacher model (Llama-4-Scout) is used; whether different teacher models lead to different optimal recipes is unknown.
3. The work is limited to single-image settings; data curation for multi-image reasoning and video reasoning is not addressed.
4. Source rankings may be benchmark-dependent—a different evaluation suite could yield different rankings.

## Related Work & Insights

- **vs. OpenThoughts**: OpenThoughts systematically studies the design space of text reasoning data; HoneyBee extends the same philosophy to the multimodal domain, adding image-dimension scaling and perceptual augmentation strategies.
- **vs. LLaVA-CoT / R1-OneVision**: These works each propose VL reasoning datasets but lack fair comparisons; HoneyBee evaluates them under a unified framework and identifies ViRL as the best source.
- **vs. Math-LLaVA / MAVIS**: These focus on constructing data for specific mathematical visual scenarios; HoneyBee more comprehensively covers diverse sources and intervention strategies.
- **Insight**: The Caption-and-Solve "describe then reason" strategy may prove effective in other VL domains such as navigation and embodied intelligence.

## Rating

- Novelty: ⭐⭐⭐⭐ The methodology itself is not novel (data engineering + scaling), but the systematic experimental design and counter-intuitive findings are of high value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough; comprehensive comparisons across multiple models, datasets, and strategies with strict variable control.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, high-quality figures and tables, transparent experimental setup—an exemplary data engineering paper.
- Value: ⭐⭐⭐⭐⭐ Provides strong guidance for VL reasoning data research; the dataset is open-sourced at large scale.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

<!-- RELATED:END -->
