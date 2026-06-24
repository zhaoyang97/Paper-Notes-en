---
title: >-
  [Paper Note] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
description: >-
  [CVPR 2026][Multimodal VLM][Reward model] MMRB2 is the first benchmark for evaluating reward models of "omni models" (capable of reading/writing interleaved text and images in a single sequence). Spanning four tasks—text-to-image, image editing, interleaved generation, and multimodal reasoning—with 1,000 expert-annotated preference pairs per task, it reveals a significant gap between the strongest current judge (Gemini 3 Pro at 76.3% average consistency) and human experts (>9…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Reward model"
  - "omni model"
  - "preference benchmark"
  - "interleaved text-image generation"
  - "MLLM-as-a-judge"
date: 2026-05-08
content_hash: 2912176bc2a79bef
---

# Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_Multimodal_RewardBench_2_Evaluating_Omni_Reward_Models_for_Interleaved_Text_CVPR_2026_paper.html)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Reward model, omni model, preference benchmark, interleaved text-image generation, MLLM-as-a-judge

## TL;DR
MMRB2 is the first benchmark for evaluating reward models of "omni models" (capable of reading/writing interleaved text and images in a single sequence). Spanning four tasks—text-to-image, image editing, interleaved generation, and multimodal reasoning—with 1,000 expert-annotated preference pairs per task, it reveals a significant gap between the strongest current judge (Gemini 3 Pro at 76.3% average consistency) and human experts (>90%).

## Background & Motivation

**Background**: Reward Models (RM) are central components for LLM post-training, quantifying human preferences into scalar scores to guide policies in RLHF/RL or serve as filters for data quality and Best-of-N sampling. Recent "omni models" (e.g., Chameleon, GPT-Image, Gemini 2.5 Flash Image) have pushed multimodal generation to new heights by understanding and generating images and text in arbitrary sequences.

**Limitations of Prior Work**: Reward modeling for omni models remains largely unexplored, primarily due to the **lack of evaluation benchmarks**. Existing multimodal preference benchmarks (VL-RewardBench, Multimodal RewardBench 1) only cover "image-text-to-text" comprehension tasks and cannot evaluate interleaved outputs. On the text-to-image side, automated metrics (CLIPScore, VQAScore) and preference-trained RMs (ImageReward, HPSv2) were mostly trained on older data (SD 2.1 era), losing accuracy on the outputs of cutting-edge models.

**Key Challenge**: The versatility of omni models introduces unique difficulties for reward modeling: ① Images cannot be automatically verified as easily as math or code; ② Sequences can contain arbitrary numbers and orders of images and text, causing a judged dimension explosion; ③ High-quality preference data requires carefully designed annotation protocols; ④ Omni models are not yet powerful enough to easily sample high-quality responses compared to pure text RM benchmarks.

**Goal**: To build a **comprehensive, reliable, and highly predictive** reward model benchmark for omni models that covers their primary application surfaces and systematically characterizes where current judges (whether MLLM-as-a-judge or preference-trained RMs) fall short.

**Key Insight**: The authors decompose the capability boundaries of omni models into four representative sub-tasks and utilize "ensemble filtering" to remove easy pairs before applying expert annotation. This focuses human effort on the most informative hard samples, ensuring label reliability with over 90% expert consensus.

**Core Idea**: By combining "four tasks $\times$ 1,000 expert preference pairs" with a "positional consistent dual evaluation protocol," the authors establish MMRB2, the first benchmark allowing omni reward modeling to be measurable, comparable, and predictive of downstream performance.

## Method

### Overall Architecture
MMRB2 (Multimodal RewardBench 2) is essentially a **data benchmark and evaluation protocol** rather than a new model. It consists of four sub-tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning. Each task contains 1,000 data points, where each point is a triplet of "task prompt + chosen response + rejected response." To evaluate an RM, it must output a preference given the prompt and two responses; the benchmark scores the RM based on its consistency with human preference.

The data pipeline consists of four steps: ① **Response Generation**: Sampling prompts from 21 public/new tasks across difficulty levels and generating candidate responses using 7–11 SOTA models (including APIs, open-source models, and agents for interleaved/reasoning tasks); ② **Ensemble Filtering**: Using 9 MLLM judges to vote and remove "unanimous" easy pairs; ③ **Human Annotation**: Professional annotators score samples using task-specific frameworks, followed by filtering for high disagreement, ambiguity, or ties; ④ **Evaluation**: Calculating consistency between the RM and humans using a "positional consistent dual evaluation."

### Key Designs

**1. Four Sub-tasks Covering the Full Capability Spectrum of Omni Models**

To address the gap where existing benchmarks only test image-text-to-text, MMRB2 defines four complementary sub-tasks: **Text-to-Image** (judging object composition, spatial relations, attribute binding, and text rendering), **Image Editing** (judging editing faithfulness vs. preservation of irrelevant regions given 1–3 input images), **Interleaved Generation** (judging global planning and image-text coordination for stories/how-to guides), and **Multimodal Reasoning** ("thinking with images," where models produce intermediate sketches, judging reasoning trajectories and final answers). The 4,000 total pairs from 21 sources (e.g., WISE, Emu-Edit, ISG-Bench, BLINK) ensure the benchmark is both broad and aligned with real-world difficulty boundaries.

**2. Ensemble Filtering + Expert Annotation: Focus Human Effort on Hard Samples**

Because omni preference data is expensive and many easy pairs provide little discriminative power, MMRB2 uses **ensemble filtering**: 9 multimodal judges (GPT-5/4.1/4o, Gemini 2.5 Flash/Pro, Gemma-3, Qwen2.5-VL) evaluate each pair twice (A vs B and B vs A) to cancel positional bias. Pairs where the majority label accounts for $\ge 90\%$ of judge evaluations are discarded as "easy." The remaining hard samples are sent to Surge AI professional annotators. Each pair is independently scored by 3 annotators on a 7-point Likert scale, with three quality controls (removing high-variance, ambiguous, or tie pairs). This resulted in 1,000 pairs per task (retaining ~50% of the initial set) with an **expert consensus rate of 95.6%**.

**3. "Response-wise Annotation to Pair Construction" for Multimodal Reasoning**

Given that reasoning tasks have ground-truth answers but varying process correctness, these are not directly annotated in pairs. Instead, **responses are annotated individually before constructing pairs**. Responses containing both correct answers and reasoning processes (balanced between text-only and mixed-modality reasoning) are collected. Each response receives 3 human annotations to identify major reasoning errors. Pairs are constructed by taking a "preferred" response (unanimously judged as having no major errors and a correct answer) and a "non-preferred" response (either a correct answer with reasoning errors or an incorrect answer with reasoning). This decouples answer correctness from reasoning quality.

**4. Positional Consistent Dual Evaluation Protocol**

To mitigate the common judge bias toward the first option, MMRB2 requires each pair to be evaluated twice: original order (A vs B) and swapped order (B vs A). Both judgments are kept as independent data points. This doubles the sample size for comparison and **penalizes judges with severe positional bias**, as inconsistency across flips lowers their overall accuracy. Consistency is binary—1.0 if the judge preference matches the human majority vote (or reasoning/answer labels for Task 4), otherwise 0.0.

### Loss & Training
This work is an evaluation benchmark and does not involve model training; hence, there is no loss function. Evaluated entities include MLLM-as-a-judge (prompted LLMs) and preference-trained RMs (ImageReward, HPSv2/v3, PickScore, EditReward, UnifiedReward, VQAScore, etc.).

## Key Experimental Results

### Main Results
The consistency (accuracy, %) of MLLMs-as-a-judge on the four tasks of MMRB2. The strongest, Gemini 3 Pro, averaged only 76.3%, still below the >90% human level. GPT-4o dropped to 51.9% on Reasoning, suggesting it is no longer suitable as a SOTA multimodal evaluator.

| Judge | T2I | Image Edit | Interleaved | Reasoning | Average |
|-------|-----|------------|-------------|-----------|---------|
| Gemini 3 Pro | 74.4 | 74.9 | 76.4 | 79.5 | **76.3** |
| GPT-5 | 70.5 | 73.8 | 74.4 | 70.2 | 72.2 |
| Gemini 2.5 Pro | 70.5 | 71.3 | 75.1 | 66.6 | 70.9 |
| Qwen3-VL-32B (Best Open) | 64.1 | 67.3 | 70.5 | 56.6 | 64.6 |
| Gemini 2.5 Flash | 63.1 | 66.5 | 69.4 | 57.5 | 64.1 |
| GPT-4o | 60.3 | 65.0 | 61.5 | 51.9 | 59.7 |
| Human Expert | — | — | — | — | >90 |

Preference-trained RM vs. MLLM Judges (selected examples, %; "Edit" uses single-image subset, "Reasoning" uses text-only subset for fair comparison). While preference training improves scores relative to base models, most still trail the larger open-source judge Qwen3-VL-32B:

| Evaluator | T2I | Image Edit* | Reasoning* |
|-----------|-----|-------------|------------|
| Qwen2.5-VL-7B (Base Judge) | 50.4 | 57.8 | 53.7 |
| CLIPScore | 51.0 | — | — |
| ImageReward | 54.0 | — | — |
| PickScore | 58.6 | — | — |
| VQAScore (Qwen2.5-VL) | 58.3 | — | — |
| EditReward | — | 67.2 | — |
| UnifiedReward | 59.8 | — | 55.1 |
| Qwen3-VL-32B | 64.1 | 66.4 | 69.9 |
| GPT-5 | 70.5 | 74.3 | 83.8 |

### Ablation Study
The benchmark itself has no ablation modules; the table below organizes key analytical conclusions:

| Analysis Dimension | Key Value | Description |
|--------------------|-----------|-------------|
| Downstream Correlation | Pearson $r > 0.8$ | MMRB2 scores correlate strongly with Best-of-N downstream performance. |
| Intra vs. Inter-model Pairs | Gap of 5–13 pts | Consistency is higher for inter-model pairs (e.g., Gemini 3 Pro T2I 79.7% vs 70.4%). |
| Mixed Modality Bias | GPT-5 gap: 49.3 pts | Judges strongly prefer reasoning responses that contain images over text-only ones. |
| Test-time Scaling | K=9: +0.8–1.2% | Majority voting provides negligible gains for multimodal RMs. |

### Key Findings
- **Multimodal Reasoning is the hardest task**: Excluding Gemini 3 Pro, top API models achieve only 52-70% consistency on Reasoning (vs. 63-75% on generation). This likely stems from diverse solution paths and the need to judge both correctness and quality simultaneously.
- **Judges exhibit a "presence of image" bias**: In mixed-modality reasoning pairs, GPT-5 has 88.2% accuracy when the preferred response contains an image, but only 38.9% when it is text-only. The judge mistakes "having an image" for higher quality.
- **Preference training is effective but dates quickly**: EditReward and UnifiedReward show significant gains (+9.4%) on their specific domains, but most RMs trained on older data struggle with the distribution shift of modern SOTA outputs.

## Highlights & Insights
- **Turning "Position Bias" into an Evaluation Metric**: Using dual evaluation not only cancels bias but also naturally penalizes biased judges through lower consistency—an elegant design applicable to any pairwise benchmark.
- **"Filter first" Paradigm**: Using 9 diverse judges to eliminate easy samples concentrates human labor on high-information pairs, providing a scalable template for building discriminative preference datasets.
- **Quantifying the "Image-Favoring" Bias**: Even the strongest judges show nearly an 18-point bias, indicating that current multimodal RMs have not yet learned to judge *when* an image should be generated.
- **High Downstream Correlation ($r > 0.8$)**: Since MMRB2 scores predict Best-of-N gains (e.g., GPT-5 as a selector improved FLUX on GenAI-Bench by 6%), the benchmark serves as a practical tool for model selection.

## Limitations & Future Work
- **Lack of Dedicated Interleaved Evaluators**: Currently, no RM is specifically trained for interleaved image-text output; this sub-task relies solely on MLLM-as-a-judge.
- **Ensemble Filtering Bias**: While mitigated by judge diversity, removing pairs that 9 judges agree on may systematically exclude certain types of samples.
- **Limited Gains from Test-time Scaling**: Majority voting is largely ineffective for multimodal RMs, suggesting a need for different scaling mechanisms compared to text-only LLMs.
- **Scale Constraints**: 4,000 pairs across four tasks remains a relatively small sample for the near-infinite combinations of interleaved omni model outputs.

## Related Work & Insights
- **vs. Multimodal RewardBench 1 (MMRB1)**: MMRB1 focused only on MLLM comprehension (image-text-to-text), while MMRB2 expands to generation and reasoning for omni models.
- **vs. RewardBench (Text-only)**: MMRB2 adapts the "systematic library of judges" approach to the multimodal domain, adding the unique challenge of image verification.
- **vs. ImageReward / HPSv2 / PickScore**: These are evaluated baselines. MMRB2 demonstrates they are losing accuracy on modern outputs (e.g., ImageReward at 54.0% on T2I), highlighting the need for this new benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark for omni models, filling a critical gap in interleaved evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 15+ judges across 4 tasks with deep dives into bias and downstream correlation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and protocol, though some quantitative details are shifted to appendices.
- Value: ⭐⭐⭐⭐⭐ High correlation with Best-of-N performance makes this a direct utility for post-training and judge selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[CVPR 2026\] Text-Printed Image: Bridging the Image-Text Modality Gap by "Printing" Text into Images](text-printed_image_bridging_the_image-text_modality_gap_for_text-centric_trainin.md)
- [\[CVPR 2026\] DuoGen: Towards Autonomous Interleaved Multimodal Generation](duogen_towards_autonomous_interleaved_multimodal_generation.md)
- [\[ICLR 2026\] A High Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation](../../ICLR2026/multimodal_vlm/a_high_quality_dataset_and_reliable_evaluation_for_interleaved_image-text_genera.md)
- [\[CVPR 2026\] Camouflage-aware Image-Text Retrieval via Expert Collaboration](camouflage-aware_image-text_retrieval_via_expert_collaboration.md)

</div>

<!-- RELATED:END -->
