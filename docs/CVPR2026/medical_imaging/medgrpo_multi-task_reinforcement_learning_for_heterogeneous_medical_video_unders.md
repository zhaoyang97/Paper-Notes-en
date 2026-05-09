---
title: >-
  [Paper Note] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding
description: >-
  [CVPR 2026][Medical Imaging][Medical video understanding] MedGRPO introduces two key innovations to address training collapse in multi-dataset reinforcement learning for medical video understanding: cross-dataset reward normalization (mapping median performance across datasets of varying difficulty to a uniform reward value via a logistic function) and a medical LLM judge (comparative scoring across five clinical dimensions). Built on Qwen2.5-VL-7B and trained on MedVidBench (532K video instruction pairs), the method surpasses GPT-4.1 and Gemini-2.5-Flash.
tags:
  - CVPR 2026
  - Medical Imaging
  - Medical video understanding
  - reinforcement learning
  - cross-dataset reward normalization
  - VLM fine-tuning
  - multi-task learning
date: 2026-05-08
content_hash: c4518d86040eb825
---

# MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2512.06581](https://arxiv.org/abs/2512.06581)
**Code**: [https://uii-america.github.io/MedGRPO/](https://uii-america.github.io/MedGRPO/)
**Area**: Medical Imaging / Video Understanding
**Keywords**: Medical video understanding, reinforcement learning, cross-dataset reward normalization, VLM fine-tuning, multi-task learning

## TL;DR

MedGRPO introduces two key innovations to address training collapse in multi-dataset reinforcement learning for medical video understanding: cross-dataset reward normalization (mapping median performance across datasets of varying difficulty to a uniform reward value via a logistic function) and a medical LLM judge (comparative scoring across five clinical dimensions). Built on Qwen2.5-VL-7B and trained on MedVidBench (532K video instruction pairs), the method surpasses GPT-4.1 and Gemini-2.5-Flash.

## Background & Motivation

1. **Background**: Large vision-language models have achieved notable progress in general video understanding, but their performance degrades substantially on medical video understanding. Medical video understanding demands fine-grained surgical action interpretation, domain-specific terminology (e.g., distinguishing "grasper" from "tool"), surgical safety assessment, and multi-stage temporal reasoning.

2. **Limitations of Prior Work**:
   - **Lack of instruction-following training data**: Existing medical video datasets (CholecT50, EgoSurgery, etc.) contain rich annotations but not in QA dialogue format.
   - **Standard RL collapses on heterogeneous datasets**: Difficulty varies dramatically across datasets (e.g., median mIoU ≈ 0.5 for CoPESD spatio-temporal grounding vs. ≈ 0.12 for EgoSurgery); raw rewards in standard GRPO cause the model to overfit easy datasets and abandon difficult ones.
   - **General semantic similarity metrics fail to capture clinical distinctions**: "The tool grasps tissue" vs. "The grasper dissects the cystic duct" yields a cosine similarity ≈ 0.82, yet the clinical meanings are entirely different.

3. **Key Challenge**: How to conduct balanced multi-task reinforcement learning across heterogeneous medical video datasets with drastically different difficulty levels?

4. **Key Insight**: Median fairness — median-level performance receives the same normalized reward across all dataset–task pairs, eliminating bias in gradient updates.

5. **Core Idea**: Logistic reward normalization enables fair cross-dataset optimization, while a medical LLM judge replaces general semantic similarity to capture fine-grained clinical correctness.

## Method

### Overall Architecture

A two-stage training paradigm:
1. **SFT stage**: Supervised fine-tuning of Qwen2.5-VL-7B on MedVidBench to inject domain knowledge and establish baseline performance and percentile statistics.
2. **GRPO stage**: Reinforcement learning with MedGRPO — sampling 8 response groups, computing cross-dataset normalized rewards, and updating the policy via GRPO's group-relative advantage estimation.

Input: Medical video frames (adaptive sampling at 0.1–3 FPS) + instruction → Output: Text/localization results across 8 task types.

### Key Designs

1. **MedVidBench Data Construction and Quality Assurance Pipeline**:

   - **Function**: Systematically converts existing expert-annotated data into large-scale instruction-following QA pairs.
   - **Mechanism**: A three-stage pipeline — (1) *Expert annotation prompting*: source-specific strategies are applied; for frame-annotated datasets (CholecT50, etc.), bounding boxes and labels are overlaid on frames; for web-sourced datasets (AVOS, etc.), audio transcripts are extracted via Whisper-X and video metadata is incorporated. (2) *Dual-model generation*: GPT-4.1 and Gemini-2.5-Flash independently generate descriptions to prevent single-model bias. (3) *Quality validation*: sentence similarity between the two model outputs is computed, and low-quality pairs (similarity < 0.3) are filtered; train/test splits are partitioned by source video (ratio 0.85/0.15). The final dataset contains 532K samples spanning 8 data sources × 8 task types (video-level / segment-level / frame-level).
   - **Design Motivation**: Converting medical annotations to QA format requires expert-level understanding, making manual annotation costly; VLM-based conversion with dual-model validation offers a scalable, high-quality alternative.

2. **Cross-Dataset Reward Normalization**:

   - **Function**: Ensures fair gradient contribution from dataset–task pairs of varying difficulty during optimization.
   - **Mechanism**: For each dataset–task pair $(d,t)$, a logistic transformation is applied: $r_{norm}^{(d,t)}(x) = \frac{1}{1 + \exp(-k \cdot \frac{x - p_{50}^{(d,t)}}{IQR^{(d,t)}})}$, where $p_{50}$ is the median, $IQR = p_{75} - p_{25}$ is the interquartile range, and $k=3.0$ controls the slope. Percentile statistics are computed from SFT baseline predictions. Key properties: when $x = p_{50}$, the normalized reward is always 0.5 (median fairness); the logistic function provides non-zero gradients everywhere (no dead zones); IQR scaling is robust to outliers.
   - **Design Motivation**: Without normalization, training collapses immediately — CVS drops from 0.894 to 0.020, STG from 0.177 to 0.010, and TAG from 0.142 to 0.004, with highly unstable training entropy. The root cause is that high-magnitude rewards from easy datasets dominate gradient updates.

3. **Medical LLM Judge**:

   - **Function**: Evaluates fine-grained clinical correctness of medical descriptions.
   - **Mechanism**: GPT-4.1 is used for comparative similarity scoring ("How closely does the generated description match the reference?" rather than absolute quality rating) to avoid score inflation. Five clinical dimensions are assessed (1–5 points each): medical terminology precision, instrument–anatomy identification, specificity vs. vagueness, surgical workflow context, and action accuracy. A hybrid design is adopted: final reward = 50% normalized semantic similarity + 50% normalized LLM judge score, balancing passage-level semantic consistency with detail-level clinical correctness.
   - **Design Motivation**: Standard embedding metrics cannot distinguish clinically critical differences such as "tool" vs. "grasper," "grasps" vs. "dissects," or "tissue" vs. "cystic duct." Comparative scoring discriminates model quality differences more effectively than absolute scoring.

### Loss & Training

The GRPO objective uses asymmetric clipping ($\epsilon_{low}=0.2$, $\epsilon_{high}=0.3$), allowing larger positive updates while constraining negative ones. The standard GRPO KL penalty term is removed. SFT training: 3 epochs, LR $5 \times 10^{-7}$; GRPO training: 5000 steps, LR $5 \times 10^{-7}$, group size $G=8$. Localization tasks use multiplicative composite rewards (format penalty). All experiments are conducted on 8×H100 GPUs.

## Key Experimental Results

### Main Results

| Model | CVS acc | STG mIoU | TAG@0.3 | TAG@0.5 | VS llm | RC llm |
|-------|---------|----------|---------|---------|--------|--------|
| GPT-4.1 | 0.018 | 0.014 | 0.096 | 0.005 | 2.490 | 2.080 |
| Gemini-2.5-Flash | 0.101 | 0.047 | 0.045 | 0.021 | 2.352 | 1.912 |
| Qwen2.5VL-7B (off-shelf) | 0.105 | 0.020 | 0.006 | 0.068 | 2.452 | 2.090 |
| Qwen2.5VL-7B SFT | 0.894 | 0.177 | 0.142 | 0.091 | 3.596 | 2.757 |
| **Qwen2.5VL-7B MedGRPO** | **0.896** | **0.202** | **0.216** | **0.156** | **4.184** | **3.442** |

### Ablation Study

| Configuration | CVS | STG | TAG@0.3 | VS llm | RC llm |
|---------------|-----|-----|---------|--------|--------|
| A: Full MedGRPO | 0.896 | 0.202 | 0.216 | 4.184 | 3.442 |
| B: w/o reward normalization | 0.020 | 0.010 | 0.004 | 1.061 | 3.469 |
| C: TAG+STG only | 0.914 | 0.193 | 0.202 | 3.776 | 3.425 |
| D: VS+RC w/ LLM judge | 0.894 | 0.183 | 0.149 | 3.824 | 3.235 |
| E: VS+RC w/o LLM judge | 0.894 | 0.183 | 0.140 | 3.733 | 2.984 |

### Key Findings

- **Reward normalization is critical**: Removing normalization causes all metrics to collapse (Row B), with CVS dropping from 0.896 to 0.020, demonstrating that this is a necessary condition rather than an optional enhancement.
- **Multi-task synergy is substantial**: Including description tasks (VS+RC) in the reward actually improves localization performance: STG +4.7%, TAG@0.3 +6.9% (Row A vs. C).
- **LLM judge contribution is clear**: VS with LLM judge exceeds VS without by 0.091 (3.824 vs. 3.733); RC improves by 0.251 (3.235 vs. 2.984).
- **SFT already far surpasses closed-source models**: Qwen2.5VL-7B SFT achieves CVS 0.894 vs. GPT-4.1's 0.018, a gap of 50×.
- **Cross-model generalization**: The same pipeline applied to Qwen3-VL-4B yields consistent improvements (STG +0.043, TAG@0.3 +0.039).
- **2026 models are still insufficient**: GPT-5.4 achieves only 0.004 on STG, indicating that medical video understanding still requires domain adaptation.

## Highlights & Insights

- **Generality of logistic reward normalization**: The median fairness principle can be directly applied to any multi-dataset/multi-task RL training scenario beyond medicine. The key design choice is IQR scaling rather than min-max scaling, which provides greater robustness to outliers. This technique has direct reference value for multi-task RLHF.
- **Comparative scoring over absolute scoring**: The LLM judge frames evaluation as "how closely does this match the reference" rather than "how good is this in absolute terms," avoiding score inflation. This evaluation strategy is worth adopting in other LLM-as-judge settings.
- **Data-driven domain adaptation remains indispensable**: Even GPT-5.4 achieves near-zero performance on medical video localization. The paradigm of domain-specific data combined with fine-tuning remains irreplaceable in the medical domain.

## Limitations & Future Work

- **High cost of LLM judge**: Each training sample requires a GPT-4.1 evaluation call, limiting training scale and speed.
- **Static nature of percentile statistics**: The $p_{25}, p_{50}, p_{75}$ values used for normalization are derived from the SFT baseline, yet the distribution shifts during RL training, potentially requiring dynamic updates.
- **Only 4 task types included in GRPO**: Tasks such as CVS, NAP, and SA (accuracy-based tasks) are not directly incorporated into RL training.
- **GRPO training limited to 5000 steps**: The RL phase involves substantially less training than SFT and may not have fully converged.
- **Coverage bias from dual-model validation**: Shared blind spots between GPT-4.1 and Gemini-2.5-Flash may go undetected.
- **Open-source LLM judges not explored**: Dependence on GPT-4.1 increases cost and introduces external uncontrollability.

## Related Work & Insights

- **vs. SurgLLM/SurgLaVi**: These methods train on single surgical datasets and lack generalization across surgical types; MedVidBench covers 8 data sources to enable cross-domain training.
- **vs. VideoChat-R1.5**: General video RL models fail entirely on medical tasks (CVS = 0.000), demonstrating that medical RL requires domain-specialized reward design.
- **vs. DAPO**: MedGRPO adopts DAPO's asymmetric clipping and KL penalty removal, while adding cross-dataset normalization as a critical component.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combined design of cross-dataset reward normalization and a medical LLM judge exhibits practical innovation, though the technical complexity of individual components is moderate.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Eight task types, multiple baselines (including state-of-the-art 2026 models), cross-model validation, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Technical descriptions are clear, problem motivation is well-articulated, and qualitative analysis is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Establishes foundational infrastructure for medical video understanding (dataset + training paradigm + evaluation methodology) with lasting impact on the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)

</div>

<!-- RELATED:END -->
