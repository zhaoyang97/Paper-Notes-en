---
title: >-
  [Paper Note] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition
description: >-
  [CVPR 2026][Multimodal VLM][Table Recognition] This paper proposes TRivia, a self-supervised fine-tuning framework that leverages QA-driven GRPO reinforcement learning to enable VLMs to learn table recognition directly f…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Table Recognition"
  - "Self-supervised Fine-tuning"
  - "GRPO"
  - "Vision-Language Models"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 5fe40cb7239d8dd3
---

# TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition

**Conference**: CVPR 2026
**arXiv**: [2512.01248](https://arxiv.org/abs/2512.01248)  
**Code**: [https://github.com/HKU-TASR/TRivia](https://github.com/HKU-TASR/TRivia)  
**Area**: Multimodal VLM
**Keywords**: Table Recognition, Self-supervised Fine-tuning, GRPO, Vision-Language Models, Reinforcement Learning

## TL;DR
This paper proposes TRivia, a self-supervised fine-tuning framework that leverages QA-driven GRPO reinforcement learning to enable VLMs to learn table recognition directly from unannotated table images. The resulting TRivia-3B surpasses proprietary models such as Gemini 2.5 Pro and GPT-5 on multiple benchmarks.

## Background & Motivation

**Background**: Table Recognition (TR) aims to convert table images into semi-structured representations such as HTML or Markdown. Recent advances in VLMs have significantly improved TR performance, with proprietary models like Gemini 2.5 Pro demonstrating strong TR capabilities. Open-source VLMs, however, remain notably behind due to limited annotated data.

**Limitations of Prior Work**: TR data acquisition faces a trilemma: (1) synthetic data is scalable but lacks real-world visual diversity; (2) annotating real data is expensive and time-consuming; (3) distilling pseudo-labels from proprietary models is costly, is bounded by the teacher model's performance ceiling, and may violate terms of service. MinerU2.5, despite employing millions of samples with human annotations and Gemini distillation, still fails to surpass its teacher model.

**Key Challenge**: Open-source TR models are constrained by limited annotated data, and their performance ceiling is dictated by the teacher model — creating a bottleneck between annotation availability and achievable performance. Meanwhile, vast quantities of unannotated table images are readily available yet cannot be directly exploited.

**Goal**: (1) How to extract effective supervision signals from unannotated table images; (2) how to select samples with the highest training value; (3) how to generate diverse and verifiable QA pairs as reward signals.

**Key Insight**: QA is a downstream task of TR — if a model can correctly answer questions about a table, it implicitly indicates that the model's recognition of the table's structure and content is accurate. This is substantially easier than directly predicting HTML annotations, and the correctness of QA pairs can be verified through cross-validation without human labeling.

**Core Idea**: Use "ability to correctly answer table questions" as a proxy reward, enabling VLMs to learn table recognition in a self-supervised manner from unannotated table images via GRPO.

## Method

### Overall Architecture
TRivia consists of two phases: (1) a data preparation phase that selects the most informative samples from unannotated table images and automatically generates diverse QA pairs for each image; and (2) a training phase that fine-tunes the VLM using the GRPO reinforcement learning framework with QA accuracy as the reward function. The overall training proceeds in three stages: OTSL warm-up (700K synthetic data) → supervised fine-tuning (50K real data) → TRivia self-supervised RL (50K unannotated data).

### Key Designs

1. **QA-Driven GRPO Self-supervised Fine-tuning**:

    - **Function**: Reformulates TR as an RL problem that can be optimized with unannotated data.
    - **Mechanism**: For each table image, the TR model (policy) generates $R$ recognition outputs $\{o_j\}$. Each output is fed to an LLM (Qwen3-8B) to answer pre-generated QA pairs. The reward is defined as $\text{Reward}(o_j) = \frac{1}{|QA|}\sum_{(q,a)} F1(M_{LLM}(q;o_j), a)$, i.e., the mean F1 score over all answers. GRPO optimizes the policy based on relative reward differences within a group. An additional illegal-sample filtering step discards invalid or repetitive outputs (samples with zero reward) to prevent reward distribution compression and training instability.
    - **Design Motivation**: QA is considerably simpler than directly predicting HTML annotations — it does not require inferring complex structural attributes such as colspan/rowspan, but only requires understanding the content of specific regions. Moreover, QA correctness can be verified automatically without human annotation.

2. **Response-Consistency Sampling**:

    - **Function**: Automatically identifies the most training-valuable samples from the unannotated data pool.
    - **Mechanism**: For each image, the TR model generates $K$ recognition outputs, and the mean pairwise TEDS (Tree Edit Distance-based Similarity) across all output pairs is computed as the consistency score: $\text{Consistency}(I) = \frac{2}{K^2-K}\sum_{i<j} \text{TEDS}(o_i, o_j)$. Lower consistency indicates greater model uncertainty on the sample and thus higher training value for GRPO (which benefits from diverse responses). In practice, samples with consistency below 0.4 are filtered as noise, and the remaining range (0.4–1.0) is sampled uniformly.
    - **Design Motivation**: Not all unannotated samples contribute equally. Clustering-based methods can only measure data diversity but cannot assess training value for a specific model, while manual filtering is not scalable. Response-consistency sampling directly aligns with GRPO's training mechanism.

3. **Attention-Guided Diverse QA Generation**:

    - **Function**: Generates diverse QA pairs covering different regions of each table image.
    - **Mechanism**: The visual grounding property of VLM attention mechanisms is exploited — the visual tokens attended to by answer tokens define the "visual source" of each QA pair: $VS((q,a)) = \{v | \mathcal{A}(v|a) > \tau_\mathcal{A}\}$. The three-step pipeline proceeds as follows: (1) Qwen2.5-VL-72B is used to generate a candidate QA pool via repeated sampling; (2) InternVL3-78B cross-validates each QA for correctness and visual dependency (answerable with the image, unanswerable without); (3) a greedy selection retains the QA subset with minimum pairwise visual source IOU, ensuring coverage of different table regions. Each image ultimately retains approximately 30 diverse QA pairs.
    - **Design Motivation**: A single QA generation pass covers only part of the table, while repeated sampling tends to produce paraphrases. Attention guidance precisely quantifies the information source of each QA pair, enabling explicit maximization of spatial coverage.

### Loss & Training
Three-stage training: Stage 1 uses 700K synthetic and public data for OTSL format warm-up (visual encoder frozen); Stage 2 applies full-parameter supervised fine-tuning on 50K real tables; Stage 3 applies GRPO RL fine-tuning under the TRivia framework on 50K unannotated data.

## Key Experimental Results

### Main Results

| Model | OmniDocBench TEDS | CC-OCR TEDS | OCRBench TEDS | Overall TEDS |
|------|-------------------|-------------|---------------|--------------|
| UniTable | 82.76 | 57.84 | 67.73 | 70.86 |
| Qwen2.5-VL-72B | 87.85 | 81.22 | 81.33 | 83.52 |
| Gemini 2.5 Pro | 90.90 | **85.56** | 88.94 | 88.93 |
| GPT-5 | 84.91 | 63.25 | 79.91 | 78.30 |
| MinerU2.5 | 90.85 | 79.76 | 87.13 | 86.82 |
| PaddleOCR-VL | 91.12 | 79.62 | 79.29 | 83.36 |
| **TRivia-3B** | **91.60** | 84.90 | **90.76** | **89.88** |

### Ablation Study

| Configuration | OmniDocBench | CC-OCR | OCRBench | Overall | Note |
|------|-------------|--------|----------|---------|------|
| Stage-2 (SFT baseline) | 90.08 | 82.48 | 90.08 | 88.57 | Supervised fine-tuning ceiling |
| + 72B pseudo-label SFT | 84.41 | 70.54 | 80.87 | 80.02 | Poor pseudo-label quality, −8.55 TEDS |
| + 72B pseudo-label GRPO | 86.19 | 78.12 | 84.16 | 83.65 | GRPO mitigates but still −4.92 |
| TRivia-3B | 91.60 | 84.90 | 90.76 | 89.88 | QA reward surpasses supervised ceiling +1.31 |
| w/o Attention-guided QA | — | — | — | Significant degradation | Especially on complex tables |
| w/o Response-consistency | — | — | — | TEDS 52→63.5 | Slower convergence with random sampling |
| w/o Illegal filtering | — | — | — | Training instability | +25% convergence steps, −3 TEDS |

### Key Findings
- **QA proxy reward breaks the supervised learning ceiling**: TRivia-3B (89.88 TEDS) surpasses the Stage-2 supervised ceiling (88.57) by 1.31 TEDS, whereas directly generating pseudo-labels from the same teacher model (72B) causes a drop of 8+ TEDS.
- **A 3B model outperforms 72B+ proprietary models**: TRivia-3B exceeds Gemini 2.5 Pro (>100B parameters) and GPT-5 with only 3B parameters, demonstrating that self-supervised RL can compensate for parameter scale gaps.
- **Response-consistency sampling accelerates convergence**: Compared to random sampling, TEDS improves from 52 to 63.5 under equal training steps, by selecting samples that are most challenging for the current model.
- **Illegal-sample filtering is critical for training stability**: Without filtering invalid outputs, severe oscillations occur in late training; filtering reduces convergence steps by 25%.
- **As a data annotator**: Pseudo-labels generated by TRivia-3B used for SFT distillation achieve 89.99 TEDS, nearly matching TRivia-3B itself.

## Highlights & Insights
- **Elegant design of QA as proxy supervision**: Rather than directly predicting hard-to-verify HTML annotations, the downstream task (QA) correctness serves as indirect supervision. This paradigm generalizes to other structured output tasks — self-supervised RL becomes feasible whenever a downstream verification task can be designed.
- **Creative use of attention distributions**: The attention distribution during answer generation is used to localize the visual source of each QA pair, achieving spatial grounding without additional annotation and explicitly addressing QA diversity.
- **Breaking the teacher model ceiling**: Conventional distillation is bounded by teacher model quality. TRivia circumvents this limitation via RL — the teacher model is not used to produce training labels directly, but only to generate QA pairs as a verification tool, enabling the student model to surpass the teacher.

## Limitations & Future Work
- The framework has only been validated for **table recognition**; extending it to other document parsing tasks (charts, formulas, layout) requires redesigning the QA proxy.
- Response-consistency sampling is performed offline; as model capability evolves during training, the sampling distribution may become suboptimal — online updates could yield further improvements.
- The pipeline depends on multiple external models (Qwen2.5-VL-72B for QA generation, InternVL3-78B for verification, Qwen3-8B for answering), resulting in high deployment complexity.
- Validation is limited to the OTSL format; applicability to more general formats such as Markdown and HTML remains untested.
- S-TEDS on PubTabNet is slightly below dedicated expert models, suggesting that domain-specific fine-tuning still retains value.

## Related Work & Insights
- **vs. MinerU2.5**: Relies on large-scale human annotation and Gemini distillation, with performance bounded by the teacher model ceiling (86.82 TEDS). TRivia surpasses this ceiling via RL, reaching 89.88, with no human annotation required.
- **vs. UniTable**: A conventional image-to-markup approach constrained by resolution and context window limitations (448×448, 512 tokens), yielding poor performance on complex tables. TRivia is built on the Qwen2.5-VL architecture and supports higher resolution.
- **vs. DeepSeek-R1's GRPO application**: DeepSeek-R1 applies GRPO to enhance LLM reasoning; TRivia transfers this approach to visual document understanding, validating the effectiveness of GRPO in vision tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The self-supervised RL paradigm that breaks the annotation data ceiling is highly novel, and the QA proxy reward design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, 12 baselines, comprehensive ablation studies, and validation as an annotation generator.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear, though the overall length is considerable and some content could be condensed.
- **Value**: ⭐⭐⭐⭐⭐ A 3B model surpassing Gemini 2.5 Pro, pointing the way toward self-supervised RL for open-source TR with high practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[CVPR 2026\] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->
