---
title: >-
  [Paper Note] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference
description: >-
  [ACL 2025][Multimodal VLM][MLLM alignment] This work constructs OmniAlign-V (a 200K high-quality multimodal SFT dataset) and the MM-AlignBench evaluation benchmark. By utilizing diverse image sources, open-ended question designs, and varied response formats, it significantly enhances the human preference alignment capability of open-source MLLMs, enabling LLaVA-Next-32B to surpass Qwen2VL-72B after SFT+DPO.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "MLLM alignment"
  - "human preference"
  - "instruction tuning"
  - "DPO"
  - "multi-modal dataset"
  - "benchmark"
date: 2026-05-08
content_hash: a64b9795e5452587
---

# OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference

**Conference**: ACL 2025  
**Code**: [https://github.com/PhoenixZ810/OmniAlign-V](https://github.com/PhoenixZ810/OmniAlign-V)  
**Area**: Multimodal VLM  
**Keywords**: MLLM alignment, human preference, instruction tuning, DPO, multi-modal dataset, benchmark  

## TL;DR

This work constructs OmniAlign-V (a 200K high-quality multimodal SFT dataset) and the MM-AlignBench evaluation benchmark. By utilizing diverse image sources, open-ended question designs, and varied response formats, it significantly enhances the human preference alignment capability of open-source MLLMs, enabling LLaVA-Next-32B to surpass Qwen2VL-72B after SFT+DPO.

## Background & Motivation

### Problem Discovery: Alignment Degradation in MLLMs

- Open-source MLLMs approach commercial models on standard VQA benchmarks, yet exhibit a significant gap in human preference alignment.
- **Key Experiment** (Table 1): After multimodal SFT, MLLMs experience severe degradation on text-only alignment benchmarks.
    - InternLM2.5-7B $\rightarrow$ InternVL2-8B: AlpacaEval-V2 drops from 27.58 to 3.35 (**-87.9%**)
    - Qwen2-7B $\rightarrow$ Qwen2VL-7B: ArenaHard drops from 32.84 to 6.46 (**-80.3%**)

### Simply Adding High-Quality Text Data Does Not Help

- Replacing the text data in LLaVA-Next-778K with high-quality Magpie/Condor data.
- Results (Table 2): While text-only alignment improves, **multimodal alignment actually decreases**.
    - Multimodal metrics such as WildVision, MMVet, and MMBench deteriorate across the board.
- **Conclusion**: Language alignment capability cannot be directly transferred to multimodal alignment; dedicated multimodal human alignment data is required.

### Problems with Existing Multimodal Data

- Dominated by VQA formats: short questions and answers, factual responses.
- Lacks open-ended questions, creative tasks, and diverse response styles.
- Fails to meet the requirements of human preference alignment.

## Method

### OmniAlign-V Dataset Construction

#### 4.1 Task Classification

**Natural Images** (3 categories of tasks):
- Knowledge (VQA with background knowledge): Requires understanding of background knowledge.
- Inferential (Reasoning tasks): Requires logical reasoning and analysis.
- Creation (Creative tasks): Open-ended creative Q&A.

**Infographics** (4 categories of images):
- Arts, Charts, Diagrams, Posters.

#### 4.2 Image Filtering Strategy (Natural Images)

A two-step filtering approach ensures semantic richness:
1. **IC9600 Image Complexity Model**: Filters out images with low semantic content.
2. **Recognize Anything Model**: Filters out images with high complexity but meaningless content (e.g., repeating patterns of tents).

#### 4.3 Data Generation Pipeline

**Knowledge & Inferential**: Generated directly by GPT-4o with carefully designed few-shot prompts.

**Creative**: A more complex pipeline inspired by Condor:
1. Create a seed creative question set $Q_s = \{Q_1, Q_2, ..., Q_N\}$.
2. Generate the image caption $C$ using a lightweight MLLM.
3. The LLM selects a relevant subset $Q_s'$ from the seed set based on the caption.
4. Randomly select 3 question types as few-shot exemplars for GPT-4o.

**Infographic**: Specialized prompts designed for different image types generate questions that require comprehensive background knowledge.

#### 4.4 Post-refinement

1. **Instruction Augmented Knowledge QAs**: Adds complex instructions and constraints to knowledge-based QA.
2. **Enriched Inferential QAs**: Complements responses with detailed explanations and reasoning logic using a knowledge-rich LLM.
3. **Quality Improved Infographic QAs**:
    - GPT-4o excels at background knowledge explanation but exhibits inaccurate OCR.
    - Open-source MLLMs have accurate OCR but lack sufficient explanation.
    - **Fuses** responses from both models, followed by manual review.

#### Data Scale

| Subset | Quantity |
|------|------|
| Knowledge QAs | 39K |
| Inferential QAs | 37K |
| Creative QAs | 10K |
| Instruction-Following QAs | 38K |
| Infographic QAs | 44K |
| Detail QAs | 35K |
| **Total** | **~205K** |

### DPO Data Generation (OmniAlign-V-DPO)

- The high-quality responses from OmniAlign-V serve as positive samples.
- A LLaVA-Next baseline (generator G) is used to sample N responses with high temperature.
- An LLM Judger selects the response that deviates most from the original intent as the negative sample.

### MM-AlignBench Evaluation Benchmark

- 252 high-quality samples, annotated by humans.
- Diverse image sources (SAM-1B, CC-3M, AI2D, ChartQA, InfographicVQA).
- Filters 2,000 natural images and 1,000 infographics using IC and RAM filtering.
- GPT-4o generates diverse questions, followed by human review and refinement.
- Evaluation metric: Judged by GPT-4o against Claude3V-Sonnet reference answers.

## Experiments

### SFT Stage Evaluation

OmniAlign-V is merged with LLaVA-Next-778k (with text samples removed) to form OmniAlign-Vmix (946K).

**LLaVA-Next using InternLM2.5-7B as the LLM**:

| Metric | LLaVA-Next-778k | OmniAlign-Vmix | Change |
|------|-----------------|----------------|------|
| MM-AlignBench | 20.6 / -42.7 | 57.1 / +11.1 | **+36.5** |
| WildVision | 23.4 / -45.0 | 29.6 / -31.3 | **+6.2** |
| MIA-Bench | 76.9 | 86.7 | +9.8 |
| MMVet | 41.8 | 47.7 | +5.9 |
| MMMU | 44.1 | 46.8 | +2.7 |
| OCRBench | 56.2 | 58.9 | +2.7 |

- Human preference alignment improves substantially (MM-AlignBench +36.5 win rate).
- Standard VQA benchmarks witness unexpected improvements instead of degradation.

**Using Qwen2.5-32B as the LLM**:
- MM-AlignBench: 26.6 $\rightarrow$ 62.3 (+35.7)
- MMMU: 55.2 $\rightarrow$ 60.7 (+5.5)

### Text-only Alignment also Improves

Even without pure text samples in training data, OmniAlign-V improves text-only alignment:
- AlpacaEval-V2 (vs GPT-3.5): 29.8 $\rightarrow$ 50.1
- ArenaHard: 21.4 $\rightarrow$ 30.4
- **Insight**: High-quality multimodal data can boost original language capabilities.

### DPO Stage Evaluation

| Model | Stage | MM-AlignBench | WildVision |
|------|------|---------------|------------|
| LLaVANext-778k | SFT | 9.5 / -69.2 | 30.4 / -34.2 |
| LLaVANext-778k | SFT+DPO | 11.1 / -64.5 | 35.5 / -23.4 |
| LLaVANext-OA | SFT | 57.1 / +11.1 | 29.6 / -31.3 |
| LLaVANext-OA | **SFT+DPO** | **64.3 / +22.4** | **41.8 / -10.1** |
| InternVL2-8B | SFT+DPO | 64.7 / +19.4 | 51.4 / +1.9 |

- DPO yields further improvements on top of OmniAlign-V SFT.
- DPO yields limited effects when applied to the model trained solely on 778k SFT data—demonstrating that **the quality of SFT alignment data is a prerequisite for effective DPO**.

### MM-AlignBench Leaderboard

| Model | Win Rate↑ | Reward↑ |
|------|-----------|---------|
| Claude3.5V-Sonnet | 84.9 | +51.4 |
| GPT-4o | 81.3 | +49.0 |
| **LLaVA-OA-32B-DPO** | **74.2** | **+36.9** |
| Qwen2VL-72B | 61.5 | +21.6 |
| InternVL2-72B | 44.4 | -6.9 |

- LLaVA-OA-32B-DPO (32B) surpasses Qwen2VL-72B (72B), ranking only below Claude and GPT-4o.

### Ablation Study

Effects of incrementally adding OmniAlign-V subsets:
- +Knowledge/Inferential/Detail: Slight improvements.
- +**Instruction Following**: MM-AlignBench jumps from 23.4 to 36.5 (a critical subset).
- +Creation: MM-AlignBench further increases to 43.7.
- +Chart/Diagram/Poster: Reaches the final score of 57.1.

## Highlights & Insights

1. **Identified and quantified the MLLM alignment degradation problem**: Multimodal SFT causes language alignment capabilities to drop by 60-90%.
2. **Revealed a counterintuitive phenomenon**: Adding high-quality text data does not improve and may even harm multimodal alignment—specialized multimodal alignment data is indispensable.
3. **Highly systematic data engineering**: Image filtering (IC+RAM) $\rightarrow$ Task classification $\rightarrow$ Diverse generation strategies $\rightarrow$ Post-refinement $\rightarrow$ Human review.
4. **Synergy of SFT + DPO**: The quality of SFT alignment determines whether DPO can take effect.
5. **32B model outperforms 72B**: Demands attention that data quality > model scale.
6. **MM-AlignBench** fills the gap in multimodal preference alignment evaluation.

## Limitations & Future Work

- Data generation heavily relies on GPT-4o, resulting in high costs.
- The OCR fusion strategy for infographics requires human review and verification.
- MM-AlignBench consists of only 252 samples, which is relatively small in scale.
- Using GPT-4o as a judge for evaluation may introduce judgment bias.
- Does not discuss safety alignment (e.g., rejecting harmful requests), focusing primarily on preference and helpfulness alignment.

## Related Work & Insights

- **LLM Alignment**: High-quality SFT data from Magpie (Xu et al., 2024) and Condor (Cao et al., 2025).
- **Visual QA Data**: LLaVA (Liu et al., 2023b) converts traditional VQA into instruction formats; ShareGPT4V, etc.
- **Multimodal Alignment Evaluation**: WildVision (Lu et al., 2024), MIA-Bench (Qian et al., 2024), which feature repetitive and simple questions.
- **DPO**: Rafailov et al., 2024; application in the vision domain remains under-explored.

## Rating ⭐⭐⭐⭐⭐

Clear motivation (identifying and resolving MLLM alignment degradation), highly systematic data engineering, and comprehensive, robust experiments (32B outperforming 72B). The work provides a complete dataset + benchmark + code, serving as a landmark contribution to the multimodal alignment field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)

</div>

<!-- RELATED:END -->
