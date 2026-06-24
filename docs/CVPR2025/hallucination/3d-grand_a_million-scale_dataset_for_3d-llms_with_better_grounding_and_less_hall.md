---
title: >-
  [Paper Note] 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination
description: >-
  [CVPR 2025][Hallucination Detection][3D-LLM] This work constructs 3D-GRAND, the first million-scale **densely grounded** 3D scene-language dataset (40K scenes, 6.2M instructions), and proposes the 3D-POPE hallucination evaluation benchmark. It demonstrates that densely grounded instruction tuning significantly enhances the grounding capability of 3D-LLMs and reduces hallucinations, while also showcasing effective sim-to-real transfer.
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "3D-LLM"
  - "Dense Grounding"
  - "Hallucination"
  - "3D-POPE"
  - "sim-to-real"
date: 2026-05-08
content_hash: 19b50ec2d9c56dbf
---

# 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination

**Conference**: CVPR 2025  
**arXiv**: [2406.05132](https://arxiv.org/abs/2406.05132)  
**Code**: [https://3d-grand.github.io/](https://3d-grand.github.io/)  
**Area**: Hallucination Detection  
**Keywords**: 3D-LLM, Dense Grounding, Hallucination, 3D-POPE, sim-to-real

## TL;DR
This work constructs 3D-GRAND, the first million-scale **densely grounded** 3D scene-language dataset (40K scenes, 6.2M instructions), and proposes the 3D-POPE hallucination evaluation benchmark. It demonstrates that densely grounded instruction tuning significantly enhances the grounding capability of 3D-LLMs and reduces hallucinations, while also showcasing effective sim-to-real transfer.

## Background & Motivation
The success of 2D-LLMs largely stems from training on large-scale vision-language datasets (e.g., LAION-5B). However, the 3D-LLM field severely lacks large-scale 3D scene-language paired data. Existing datasets (e.g., ScanRefer, ScanQA) are built on only ~700 scenes from ScanNet, which is too small in scale. Although SceneVerse scales up to 62K scenes, its language annotations only feature **sparse grounding** (associated with only a single object) rather than **dense grounding** (every noun phrase is linked to a 3D object). While research in the 2D domain has demonstrated that dense grounding can mitigate VLM hallucinations, no systematic study on hallucination issues in 3D-LLMs has yet been conducted in the 3D field.

## Core Problem
1. Do 3D-LLMs fall victim to hallucinations? If so, how severe is it?
2. Can large-scale dense grounding data alleviate hallucinations in 3D-LLMs?
3. Can models trained on synthetic 3D data transfer successfully to real-world 3D scans?

## Method

### Overall Architecture
Two major contributions: the **3D-GRAND dataset** and the **3D-POPE evaluation benchmark**.
- Dataset: Collect 40K synthetic indoor scenes from 3D-FRONT and Structured3D $\rightarrow$ Annotate object attributes using GPT-4V $\rightarrow$ Construct scene graphs $\rightarrow$ Generate densely grounded language annotations via GPT-4 $\rightarrow$ Perform hallucination filtering + human quality control.
- Model: LoRA fine-tuning based on Llama-2, taking object-centric scene graphs (categories, centroids, extents) + textual instructions as input, and outputting densely grounded responses.

### Key Designs
1. **Dense Grounding Annotation Pipeline**: The core innovation lies in achieving **noun-level one-to-one grounding**—each noun phrase in a sentence referring to an object is explicitly linked to the corresponding object in the 3D scene. Compared with paragraph-level collective grounding or dialogue-level many-to-one grounding in SceneVerse, the granularity is substantially improved. Pipeline: 3D model $\rightarrow$ 2D image $\rightarrow$ GPT-4V attribute annotation $\rightarrow$ Scene graph $\rightarrow$ GPT-4 grounded text generation $\rightarrow$ Hallucination filtering. The error rate of LLM annotations (5.6% - 8.2%) is on par with professional human annotations (5%), but with a $178\times$ cost reduction (\$3,030 vs \$539,000) and a $1051\times$ time saving.

2. **3D-POPE Evaluation Benchmark**: Inspired by 2D POPE, this benchmark is designed to evaluate hallucinations in 3D-LLMs. It asks the model "Is there a [object] in the scene?" and constructs negative samples under three difficulty levels: random sampling, popular sampling (selecting high-frequency absent objects), and adversarial sampling (selecting absent objects that frequently co-occur with present ones). Evaluation metrics include Precision, Recall, F1-score, Accuracy, and Yes-ratio.

3. **Hallucination Filter**: Automatically checks for hallucinated object IDs produced by GPT-4 after data generation. As structured scene graphs are used as input, the system can verify whether each generated object ID actually exists in the scene, thereby filtering out annotations containing hallucinated IDs.

4. **Multi-task Design**: The dataset covers 8 types of 3D-text tasks, including grounded object captioning, grounded scene description, grounded QA, etc.

### Loss & Training
- LoRA fine-tuning of Llama-2 using AdamW (lr=2e-4, wd=0.01) with a cosine learning rate scheduler.
- BF16 precision, $12 \times \text{A}40$ GPUs, batch size of 96, trained for 10K steps (~48 hours).
- DeepSpeed ZeRO-2 + FlashAttention.
- Grounding tokens adopt a special format `<obj>ID</obj>`, weighted equally with ordinary tokens in the generation loss.
- During training, 50% of the objects in the scene are randomly sampled as input context to prevent overfitting to the complete scene graph.
- The dataset is stratified by scene source (60% from 3D-FRONT, 40% from Structured3D) and mixed-sampled during training.

## Key Experimental Results

### 3D-POPE Hallucination Evaluation (ScanNet Validation Set)

| Model | Random F1 | Popular F1 | Adversarial F1 |
|------|-----------|------------|----------------|
| 3D-LLM | 66.67 | 66.61 | 66.61 |
| 3D-VisTA | 51.79 | 49.54 | 51.15 |
| LEO | 62.25 | 59.55 | 59.78 |
| **3D-GRAND (zero-shot)** | **88.56** | **78.26** | **76.37** |

3D-LLM responds with "yes" to almost all questions (Yes-ratio of 99.8%), indicating severe hallucinations. The precision of 3D-GRAND (Random: 93.3%) vastly outperforms other methods.

### ScanRefer Grounding (Zero-shot, Never Trained on ScanNet)

| Method | Acc@0.25 | Acc@0.5 |
|------|----------|---------|
| 3D-LLM | 30.3 | - |
| 3D-GRAND (zero-shot) | **38.0** | 27.4 |

3D-GRAND outperforms 3D-LLM by 7.7% (@0.25 IoU) despite never being trained on ScanNet, demonstrating strong sim-to-real transfer capability.

### Ablation Study
- **Removing grounding tokens**: The precision on 3D-POPE drops by 1.4% - 2.7%, proving that dense grounding training reduces hallucinations.
- **Ground First vs. Ground Later**: Ground-then-answer (similar to CoT reasoning) performs better than answer-then-ground.
- **Data scaling**: Grounding accuracy and hallucination rates consistently improve as the data volume scales up, and grounded data exhibits better scaling effects than non-grounded data.
- **Human quality control**: Text accuracy is 85% - 88%, and grounding accuracy is 92% - 96%.

## Highlights & Insights
- **First million-scale densely grounded 3D dataset**: Features 40K scenes and 6.2M annotations, where every noun phrase is grounded to a 3D object.
- **Cost-efficiency of GPT-4 annotations**: Achieves a $178\times$ cost reduction and $1051\times$ time savings, while maintaining quality comparable to professional human annotation.
- **First hallucination benchmark for 3D-LLMs**: 3D-POPE systematically exposes the severe hallucination issues in existing 3D-LLMs.
- **Sim-to-real transfer**: Trained solely on synthetic data, yet outperforms models trained on real-world data in a zero-shot setting.

## Limitations & Future Work
- The model input consists of structured scene graphs (categories + coordinates) instead of raw point cloud features, which limits visual representation capabilities.
- The diversity of synthetic scenes (3D-FRONT/Structured3D) falls short of real scanners, possibly limiting generalization.
- Inference relies on the detection quality of Mask3D; undetected objects cannot be grounded.
- Validation is limited to indoor scenes; outdoor/driving scenarios are not yet covered.
- Performance still has significant room for drop/degradation under the Popular and Adversarial settings, where performance drops considerably.
- Language annotations only cover English; multilingual grounding capabilities remain unexplored.

## Related Work & Insights
- **SceneVerse**: The most similar work, which is also million-scale but only features sparse grounding. 3D-GRAND completely outperforms it in grounding granularity (noun-level vs. paragraph/session-level), and includes hallucination filtering and human quality control.
- **3D-LLMs**: Almost consistently answer "yes" on 3D-POPE, exposing severe hallucinations. 3D-GRAND outperforms 3D-LLM by 7.7% in a zero-shot setting on ScanRefer.
- **LEO**: Although better than 3D-LLM and 3D-VisTA on 3D-POPE, its F1-score is still significantly lower than that of 3D-GRAND (62% vs 89%).

## Inspirations & Connections
- The idea of using dense grounding to mitigate hallucinations can be extended to other modalities (video-language, audio-language).
- Leveraging LLMs for large-scale high-quality annotation generation along with automatic quality control is an effective path to address the scarcity of 3D data.
- The success of sim-to-real transfer provides empirical support for training embodied AI using synthetic data.
- The three-tier difficulty design of 3D-POPE can be adapted to evaluate hallucinations in other modal LLMs (e.g., Video-LLMs, Medical Image-LLMs).
- The Ground-First strategy is consistent with the Chain-of-Thought (CoT) concept: explicit localization prior to reasoning is crucial for 3D spatial reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First million-scale densely grounded 3D dataset + first 3D hallucination benchmark; the task definitions are pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive data scaling analysis, ablations, and human quality reviews, though evaluations on downstream tasks are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, well-defined problems, and thorough comparisons.
- Value: ⭐⭐⭐⭐⭐ The dataset and benchmark pose highly significant driving value for the 3D-LLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](../../AAAI2026/hallucination/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[ICLR 2026\] High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning](../../ICLR2026/hallucination/high_accuracy_less_talk_halt_reliable_llms_through_capability-aligned_finetuning.md)
- [\[ACL 2025\] TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation](../../ACL2025/hallucination/treecut_a_synthetic_unanswerable_math_word_problem_dataset_for_llm_hallucination.md)
- [\[ACL 2025\] Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](../../ACL2025/hallucination/aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)

</div>

<!-- RELATED:END -->
