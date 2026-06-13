---
title: >-
  [Paper Note] CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product
description: >-
  [AAAI 2026][Multimodal VLM][creativity evaluation] This paper constructs the CreBench creativity evaluation benchmark and the CreMIT multimodal instruction tuning dataset (2.2K samples, 79.2K human feedback annotations…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "creativity evaluation"
  - "multimodal large language models"
  - "human alignment"
  - "instruction tuning"
  - "benchmark"
date: 2026-05-08
content_hash: 45e65e560bbcdf3d
---

# CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product

**Conference**: AAAI 2026
**arXiv**: [2511.13626](https://arxiv.org/abs/2511.13626)  
**Code**: [Project Page](https://kaixuewen.github.io/Crebench)  
**Area**: Multimodal VLM
**Keywords**: creativity evaluation, benchmark, multimodal instruction tuning, human alignment, LLaVA

## TL;DR
This paper proposes CreBench, a multimodal creativity evaluation benchmark covering three dimensions—creative idea → creative process → creative product—with 12 fine-grained metrics. It additionally constructs CreMIT (2.2K samples, 79.2K human annotations, 4.7M instructions) and fine-tunes CreExpert, which significantly outperforms GPT-4V and Gemini-Pro-Vision on creativity evaluation.

## Background & Motivation

**Background**: MLLMs have achieved substantial progress on objective tasks such as visual question answering and image captioning. However, creativity is a highly abstract, subjective, and multi-dimensional human cognitive ability, and existing MLLMs remain far from aligned with human judgment in this regard.

**Limitations of Prior Work**:
- No benchmark specifically targeting creativity evaluation exists.
- Existing metrics (BLEU, CIDEr, CLIPScore) fail to capture creativity dimensions such as novelty and utility.
- Prior creativity-related datasets have limited coverage (no multimodal data, no process data, no human feedback).

**Key Challenge**: The open-ended and subjective nature of creativity makes automated evaluation extremely difficult, yet this is precisely the core capability AI systems need to develop.

**Goal**: To construct a multi-dimensional, human-aligned creativity evaluation benchmark along with a corresponding dataset and expert model.

**Key Insight**: Drawing from cognitive science and design theory, creativity is decomposed into three dimensions—idea → process → product—with 12 fine-grained metrics: originality, appropriateness, immersion, divergence, structuring, evaluation, elaboration, effectiveness, aesthetics, novelty, manufacturability, and systemic complexity.

**Core Idea**: Construct a human-aligned multi-dimensional creativity evaluation benchmark, and train CreExpert via expert annotation combined with GPT-4o-based instruction generation.

## Method

### Overall Architecture
A three-stage pipeline:
1. **Data Collection**: 512 middle school students and AI systems complete 4 open-ended creative tasks, producing textual ideas, behavioral logs, and visual artifacts.
2. **Expert Evaluation**: 3 experts score submissions on 12 metrics, yielding 79.2K annotations (Fleiss' κ = 0.71, ICC = 0.78).
3. **Instruction Generation**: GPT-4o converts expert feedback into 4.7M instruction–response pairs across 6 QA formats.

### Key Designs

1. **12-Dimensional Creativity Evaluation Framework**:

    - Creative Idea (2 metrics): Originality, Appropriateness
    - Creative Process (5 metrics): Immersion/Preparation, Divergence, Structuring, Evaluation, Elaboration
    - Creative Product (5 metrics): Effectiveness, Aesthetics, Novelty, Manufacturability, Systemic Complexity
    - Each metric is scored on a 5-point behaviorally anchored rating scale.

2. **CreMIT Instruction Dataset**:

    - 6 QA formats: Reasoning, What, How, Why, Y/N, MCQ
    - Covers human and AI creative outputs; multimodal inputs (text + process logs + images).

3. **CreExpert Model**:

    - Fine-tuned from LLaVA-1.5
    - Retains general knowledge while acquiring creativity understanding capability.

## Key Experimental Results

### Main Results
CreExpert significantly outperforms GPT-4V and Gemini-Pro-Vision across all 12 creativity dimensions, with particularly pronounced advantages in creative process evaluation and creative product evaluation.

### Ablation Study
- Multi-dimensional evaluation is more stable and more consistent with human judgment than single-dimensional evaluation.
- Diversity across the 6 QA formats substantially improves the model's creativity perception capability.
- The quality of human annotations (κ = 0.71) ensures the reliability of the training data.

### Key Findings
- Existing MLLMs (including GPT-4V) exhibit a significant gap from human judgment in creativity evaluation.
- This gap can be effectively bridged through expert annotation combined with instruction fine-tuning.
- Evaluating the creative process is the weakest aspect of existing methods, as it requires understanding sequential behavioral data.

## Highlights & Insights
- The **"idea → process → product" three-dimensional framework** represents a systematic operationalization of creativity research, offering a more comprehensive perspective than solely assessing output novelty.
- **High dataset quality**: authentic creative work from 512 students combined with multi-round standardized expert annotation, far exceeding the quality of crowdsourced labeling.
- **Six QA formats** ensure the model can handle diverse creativity-related queries.

## Limitations & Future Work
- The dataset scale is relatively small (2.2K instances), which may be insufficient to capture the full complexity of creativity.
- Only visual design tasks are considered; other creative domains such as textual composition and music creation are not included.
- The model is fine-tuned from LLaVA-1.5; a stronger backbone may yield further improvements.
- Expert annotation is costly and difficult to scale to larger data volumes.

## Related Work & Insights
- **vs. AesBench**: AesBench evaluates aesthetics, whereas this work evaluates creativity—creativity places greater emphasis on originality and depth of imagination rather than visual appeal.
- **vs. Creativity Evaluation for DALL-E/SD**: Prior work approximates creativity through diversity/novelty metrics; this paper employs a more comprehensive 12-dimensional human-aligned evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multi-dimensional human-aligned creativity evaluation benchmark, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons, but constrained by dataset scale.
- Writing Quality: ⭐⭐⭐⭐ Framework design is systematic, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ Makes important contributions to both creativity research and MLLM evaluation.

**Area**: Multimodal VLM
**Keywords**: creativity evaluation, multimodal large language models, human alignment, instruction tuning, benchmark

## TL;DR

This paper constructs the CreBench creativity evaluation benchmark and the CreMIT multimodal instruction tuning dataset (2.2K samples, 79.2K human feedback annotations, 4.7M instructions), evaluating MLLM creativity alignment across three dimensions—creative idea → process → product—with 12 metrics. The resulting fine-tuned model, CreExpert, substantially outperforms GPT-4V.

## Background & Motivation

Creativity is a core human cognitive capability. With the rapid development of MLLMs, a critical question arises: **does the "creativity" understood by these models align with human judgment?**

Existing problems:
1. **Creativity is inherently abstract, subjective, and multi-dimensional**, and existing MLLMs perform poorly on creativity evaluation.
2. **Lack of systematic evaluation benchmarks**: existing vision-language benchmarks (VQA, Captioning) focus on objective tasks with clear ground-truth answers, whereas creativity is open-ended.
3. **Inadequacy of traditional metrics**: BLEU, CIDEr, and CLIPScore cannot capture creativity dimensions such as novelty, utility, and imaginative depth.
4. **Limitations of existing creativity-related datasets**: absence of human feedback, instruction tuning data, process-level data, and AIGC-generated content.

**Core Contribution**: The first systematic multimodal creativity evaluation benchmark covering the complete creativity chain from "idea" to "process" to "product."

## Method

### Overall Architecture

CreBench consists of two components:

**1. Evaluation Dimension Framework (12 metrics across 3 dimensions)**

**Creative Idea**:
- Originality: the degree of deviation from conventional approaches
- Appropriateness: relevance and feasibility with respect to task requirements

**Creative Process**:
- Immersion/Preparation: initial reflection, observation, and strategic planning
- Divergence: diversity and experimental ideation in open-ended exploration
- Structuring: integration of visual elements into a coherent composition
- Evaluation: ongoing assessment and refinement of ideas
- Elaboration: detail and expressiveness in the final visual output

**Creative Product**:
- Effectiveness: clarity and coherence in conveying the solution
- Aesthetic: visual appeal, compositional balance, and expressiveness
- Novelty: originality in form, content, or symbolic expression
- Manufacturability: feasibility and functionality in the real world
- Systemic Complexity: degree of integration across multi-functional components

Each dimension is scored on a 5-point behaviorally anchored rating scale.

**2. CreMIT Dataset Construction Pipeline**

### Key Designs

#### Data Collection (Stage 1)

- **Task Design**: 4 open-ended real-world problem-solving scenarios (e.g., "cargo river crossing") designed to elicit unconventional creative responses.
- **Participants**: 512 middle school students from 5 schools, recruited via stratified cluster sampling to ensure demographic and cognitive diversity.
- **Data Modalities**: Each participant completes 3 creative tasks, contributing textual ideas, behavioral logs, and visual outputs.
- **AIGC Content**: AI-generated creative solutions are also collected for comparative analysis.

#### Expert Annotation (Stage 2)

- Follows the **Consensual Assessment Technique (CAT)**: 3 creativity education experts.
- Independent annotation after **two rounds of calibration training** to ensure inter-rater consistency.
- **Quality Control**: continuous consistency monitoring, periodic calibration meetings, automated checks combined with manual review.
- **Outcome**: 79.2K human feedback annotations covering 2.2K multi-dimensional evaluation instances.
- **Reliability Metrics**: Fleiss's $\kappa = 0.71$, ICC(2,1) = 0.78, reaching "substantial agreement."

#### Instruction Data Generation (Stage 3)

GPT-4o is used to transform expert feedback into 6 types of instruction–response pairs:
- **Reasoning**: analyzes the logic underlying expert scores
- **What**: investigates key characteristics and expressive elements of creative ideas
- **How**: explains how a creative idea is realized or executed
- **Why**: reveals the reasons behind evaluation outcomes
- **Yes/No**: binary judgments on novelty, relevance, etc.
- **MCQ**: converts evaluation scenarios into multiple-choice scoring (excellent to poor)

A total of **4.7M multi-type instructions** are generated.

#### CreExpert Model

Based on the LLaVA-1.5-7B architecture:
- Visual encoder: CLIP-ViT-L14 (336×336, 576 visual tokens)
- Modality bridge: two-layer MLP
- Language decoder: Vicuna-v1.5

Training strategy: the visual encoder is frozen; only the projection module and language model are fine-tuned (LoRA), trained within the LLaMA-Factory framework.

### Loss & Training

- Supervised instruction fine-tuning (SFT) is adopted.
- General knowledge is preserved while creativity evaluation capability is acquired.
- Trained on 8 × NVIDIA A40 48GB GPUs.
- Dataset split 50/50 into fine-tuning and evaluation sets.
- Evaluation metric: Pearson correlation coefficient (alignment between model predictions and human feedback).

## Key Experimental Results

### Main Results

**Table 2: Comparison of CreExpert with 11 MLLMs (Pearson Correlation Coefficient %)**

| Model | Creative Idea | Creative Process | Creative Product | Overall | Rank |
|---|---|---|---|---|---|
| **CreExpert** | **84.14%** | **72.19%** | **40.18%** | **65.50%** | 1 |
| GPT-4V | 15.16% | 45.01% | 27.64% | 29.27% | 2 |
| Gemini-Pro-Vision | 11.47% | 54.39% | 17.50% | 27.78% | 3 |
| mPLUG-Owl2 | 14.34% | 29.31% | 23.76% | 22.47% | 4 |
| LLaVA-1.5-7B | 13.06% | 28.78% | 19.87% | 20.57% | 5 |
| Qwen2.5-VL | 12.36% | 23.34% | 22.66% | 19.45% | 8 |
| TinyGPT | 3.29% | 8.15% | 7.89% | 6.44% | 12 |

**CreExpert surpasses GPT-4V by over 35% (Overall) and the baseline LLaVA-1.5-7B by nearly 45%.**

**Table 3: Cross-Task Ablation (Creative Idea Dimension)**

| Task | Baseline Ori. | CreExpert Ori. | Gain |
|---|---|---|---|
| Transport | 12.80% | 72.42% | +59.62% |
| Parking | 14.98% | 69.08% | +54.10% |
| Reach | 14.12% | 83.91% | +69.79% |
| Fence | 11.90% | 80.28% | +68.38% |

### Ablation Study

- **Consistent improvement across tasks**: the Originality dimension shows the largest gains (+54%~+70%), as creative ideas are primarily expressed textually, making LLM alignment more tractable.
- **Most stable improvement in Creative Process**: behavioral log data provides rich training signals.
- **Creative Product remains the most challenging**: the subjectivity of visual evaluation is highest, and substantial room for improvement remains.

### Key Findings

1. **Existing MLLMs perform extremely poorly on creativity evaluation**: GPT-4V achieves only 29.27% Overall, indicating that general-purpose large models are far from understanding human creativity.
2. **Domain-specific fine-tuning yields remarkable results**: fine-tuning LLaVA-1.5-7B with CreMIT alone improves performance from 20.57% to 65.50%.
3. **Creative Product is the most difficult dimension**: the highest score reaches only 40.18%, reflecting the challenge of holistic subjective judgment on visual design.
4. **Immersion/Preparation is the easiest metric to align**: CreExpert achieves 92.24%, as this is the most structured process metric.
5. **Open-source models show little variation among themselves** but differ dramatically from CreExpert, suggesting that general-purpose pretraining provides almost no benefit for creativity evaluation.

## Highlights & Insights

- **First systematic multimodal creativity evaluation benchmark**: fills an important gap, with a dimension framework grounded in cognitive science and design theory.
- **The three-dimensional evaluation of "idea → process → product"** breaks away from the tradition of focusing solely on final outputs, better reflecting theoretical frameworks in creativity research (e.g., Wallas's four-stage model).
- **Substantial dataset scale**: 79.2K human feedback annotations + 4.7M instructions represent rare large-scale annotated data in the creativity domain.
- **Six instruction types** ensure the model can handle diverse creativity-related query scenarios.
- **Inter-rater reliability** ($\kappa = 0.71$, ICC = 0.78) reaches an acceptable level, enhancing the credibility of the benchmark.

## Limitations & Future Work

1. **Participants are primarily middle school students**: the range and diversity of creativity may be limited and may not represent professional designers or artists.
2. **Only 4 task scenarios**: coverage is limited; extension to broader creativity domains (music, writing, programming, etc.) is desirable.
3. **Creative Product evaluation remains difficult** (peak score only 40%), necessitating stronger visual understanding capabilities.
4. **Subjectivity in human annotation**: despite $\kappa = 0.71$, annotation noise is inevitable given the inherently subjective nature of creativity.
5. **Quality of GPT-4o-generated instruction data**: may introduce bias or be unfaithful to the original expert feedback.
6. **Integration with creativity generation (rather than evaluation alone) warrants exploration**: enabling models to not only evaluate but also assist in creative production.

## Related Work & Insights

- **LLaVA series**: the foundational architecture for CreExpert, demonstrating the powerful adaptability of instruction fine-tuning.
- **AesBench**: an aesthetics evaluation benchmark; however, aesthetics ≠ creativity—creativity places greater emphasis on originality and imagination.
- **DALL-E/Stable Diffusion**: creativity in image generation models is often reduced to diversity metrics; this paper proposes a more systematic evaluation framework.
- **Consensual Assessment Technique (CAT)**: the gold standard for creativity evaluation proposed by Amabile.
- Insight: human alignment encompasses not only preference alignment but also alignment of higher-order cognitive abilities; creativity evaluation is an important direction.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4.0 |
| Technical Depth | 3.0 |
| Experimental Thoroughness | 4.0 |
| Writing Quality | 3.5 |
| Practical Value | 3.5 |
| **Overall** | **3.6** |

## Related Work & Insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Recursive Think-Answer Process for LLMs and VLMs](../../CVPR2026/multimodal_vlm/recursive_think-answer_process_for_llms_and_vlms.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](../../CVPR2026/multimodal_vlm/dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](../../CVPR2026/multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)

</div>

<!-- RELATED:END -->
