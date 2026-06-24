---
title: >-
  [Paper Note] Sightation Counts: Leveraging Sighted User Feedback in Building a BLV-aligned Dataset of Diagram Descriptions
description: >-
  [ACL 2025][Multimodal VLM][Visual Accessibility] This work proposes leveraging sighted users to "evaluate" rather than "generate" VLM diagram descriptions. This approach builds Sightation, the first multi-task dataset of 5k diagrams and 137k samples validated by BLV expert educators. After preference fine-tuning, a 2B model achieved an average improvement of `$1.67\sigma$` in BLV usefulness ratings.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Visual Accessibility"
  - "Diagram Description"
  - "BLV Preference Alignment"
  - "DPO"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 3e0ee7ba4fd33f05
---

# Sightation Counts: Leveraging Sighted User Feedback in Building a BLV-aligned Dataset of Diagram Descriptions

**Conference**: ACL 2025  
**arXiv**: [2503.13369](https://arxiv.org/abs/2503.13369)  
**Code**: [HuggingFace](https://hf.co/Sightation)  
**Area**: Others  
**Keywords**: Visual Accessibility, Diagram Description, BLV Preference Alignment, DPO, Vision-Language Models

## TL;DR

This work proposes leveraging sighted users to "evaluate" rather than "generate" VLM diagram descriptions. This approach builds Sightation, the first multi-task dataset of 5k diagrams and 137k samples validated by BLV expert educators. After preference fine-tuning, a 2B model achieved an average improvement of `$1.67\sigma$` in BLV usefulness ratings.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have made rapid progress in image understanding and description generation, with models like Qwen2-VL and GPT-4o capable of generating high-quality image descriptions. However, their outputs are predominantly optimized for sighted users, neglecting the actual accessibility needs of blind and low-vision (BLV) individuals. In educational contexts, textbook diagrams (e.g., scientific schematics, data charts) require textual descriptions to help BLV students understand, but existing VLMs often provide overly verbose or poorly targeted information.

**Limitations of Prior Work**: Existing datasets exhibit three main deficiencies. First, datasets like VisText and MathVista cover diagram descriptions but have never been validated by BLV users, making it impossible to determine if the description quality aligns with actual BLV needs. Second, while VizWiz-VQA and VizWiz-LF involve BLV participation, they focus solely on Visual Question Answering (VQA) and cannot support diverse training objectives such as completions, preference alignment, or retrieval. Third, the traditional practice of hiring sighted annotators to write diagram descriptions is flawed. Lundgard & Satyanarayan (2022) demonstrated that sighted individuals and BLV users have significantly misaligned preferences—the former tend to describe low-level numerical details, while the latter focus more on high-level insights and utility.

**Key Challenge**: A fundamental preference misalignment exists between the annotators (sighted individuals) and the end users (BLV group). Having sighted people directly generate descriptions is not only costly and prone to annotator bias (Geva et al., 2019), but also yields outputs that fail to meet BLV standards. Additionally, public reward models for general VLMs are extremely scarce, with none specialized for BLV, rendering traditional RLHF pipelines unviable.

**Goal**: (1) How to scalably collect BLV-aligned annotation data at a low cost? (2) How to perform preference alignment fine-tuning without a BLV-specialized reward model? (3) How to construct a BLV dataset supporting multi-task objectives such as completion, preference, retrieval, VQA, and reasoning?

**Key Insight**: The authors observe that the cognitive load of "evaluation" tasks is significantly lower than that of "generation" tasks—it is much easier for sighted annotators to judge which of two descriptions is better and rate them along various dimensions than to write a BLV-friendly description from scratch. Kreiss et al. (2022) demonstrated that sighted individuals can serve as effective proxies for BLV preferences, albeit limited to a few specific dimensions. Scaling this proxy approach to multi-dimensional evaluations on a dataset-wide level, validated by BLV expert educators, can simultaneously address annotation cost and preference alignment.

**Core Idea**: To construct the first large-scale BLV-aligned multi-task diagram description dataset using VLM-guided generation, multi-dimensional assessment by sighted annotators, and validation by BLV educators.

## Method

### Overall Architecture

The overall workflow consists of four stages: (1) generating BLV-friendly diagram descriptions using two-step VLM reasoning (guided generation); (2) employing 30 sighted annotators to perform preference selection, multi-dimensional scoring, and key sentence highlighting; (3) processing the annotations into multiple task formats such as completion, preference alignment, and retrieval; and (4) fine-tuning Qwen2-VL 2B/7B and BLIP-2 using SFT, DPO, and contrastive learning, followed by evaluations by 17 BLV and sighted educators.

The base data is sourced from the AI2D dataset (5k elementary school science diagrams), chosen because it requires no specialized domain knowledge while challenging VLM comprehension. Using GPT-4o mini and Qwen2-VL 72B, both guided and unguided descriptions are generated for each diagram, totaling 20k candidates.

### Key Designs

1. **Two-step Guided Generation (Guided Generation with Latent Supervision)**:

    - **Function**: Generates descriptions better aligned with BLV needs through two-step reasoning.
    - **Mechanism**: In the first round of reasoning, the VLM generates a set of question-answer pairs (guides) for the input diagram, which help the model distinguish primary information from secondary details. In the second round, the VLM takes both the diagram and the guides as input, producing the description under implicit guidance. This design ensures that the VLM selectively focuses on information of high value to BLV users rather than enumerating every pixel. For each diagram, two models (GPT-4o mini and Qwen2-VL 72B) generate both guided and unguided versions, resulting in 4 candidate descriptions.
    - **Design Motivation**: Direct prompting has limited effectiveness because VLMs lack inherent understanding of BLV needs. By employing QA pairs as intermediate latent representations, the model gains prior knowledge of "what is worth describing" in the second round without requiring explicit BLV training data.

2. **Three-layer Collaborative Annotation System (Multi-role Assessment Design)**:

    - **Function**: Enables complementary evaluations across 9 quality dimensions by leveraging groups with diverse visual abilities and professional backgrounds.
    - **Mechanism**: 30 sighted annotators assess visually grounded dimensions—**Factuality** (consistency with the diagram) and **Information Quality** (coverage of core info)—along with preference selection and best-sentence highlights. 9 sighted educators assess **General Usefulness** (estimated utility for BLV users). 8 BLV educators (teaching at schools for the blind) assess four fine-grained utility dimensions (**Summary Usefulness**, **Multiple Choice Usefulness**, **Open-ended Usefulness**, and **General Usefulness**) as well as **Explanatoriness** (whether descriptions state facts or interpret concepts). **Conciseness** and **Diversity** are assessed by all groups since they solely depend on textual comprehension.
    - **Design Motivation**: Traditional methods employ a single annotator group for all evaluation metrics, yet different dimensions require distinct expertise. Factuality requires visual access to the diagrams, whereas usefulness requires a deep understanding of BLV needs. Dividing evaluation tasks based on competency ensures high-quality annotations while reducing the workload on BLV annotators.

3. **Multi-task Dataset Construction Pipeline (Multi-task Dataset Construction)**:

    - **Function**: Systematically converts annotations into training data across five distinct task formats.
    - **Mechanism**: **SightationCompletions** (8k samples) organizes all 4k human-annotated descriptions into (diagram, instruction, description) triplets, while augmenting another 4k samples with dimension-specific suffixes from the top 25% highest-scoring descriptions. **SightationPreference** (16k pairs) constructs chosen-rejected pairs from three sources: within-model comparisons (2k pairs from direct preference annotations), cross-model comparisons (4k pairs sorted by average ratings), and synthetic comparisons (10k pairs constructed by stripping the primary sentences to form the rejected version). **SightationRetrieval** (1k rows) contains image retrieval data with top-1/5/10 positive targets and 10 hard negative distractors. VQA and reasoning subsets are also included.
    - **Design Motivation**: Diverse downstream applications of BLV needs cannot be served by a single format. By designing various processing pipelines, a single raw annotation dataset can simultaneously support SFT, DPO, contrastive learning, and VQA.

### Loss & Training

- **SFT Fine-tuning**: Full parameter SFT is applied to Qwen2-VL 2B, while Parameter-Efficient Fine-Tuning (PEFT) is used for the 7B model, both trained on SightationCompletions.
- **DPO Preference Alignment**: A key design choice is utilizing non-overlapping diagrams between the SFT and DPO phases. 4k descriptions from 1k randomly sampled diagrams in the unannotated pool are used for SFT prior to DPO training on SightationPreference, preventing overfitting caused by shared data.
- **Contrastive Learning**: InfoNCE loss is utilized to fine-tune partial parameters of BLIP-2, using only the top-1 positive sample and a single random negative sample to optimize computational efficiency.

## Key Experimental Results

### Main Results

**BLV Educator Evaluation — Overall Effect Size of the Proposed Approach** (Cohen's d in units of $\sigma$):

| Dimension | 2B Model | 7B Model | Best Model |
|------|---------|---------|---------|
| Conciseness | -0.09 | **1.69** | 7B |
| Diversity | **0.90** | 0.46 | 2B |
| Summary Usefulness | 0.39 | **0.53** | 7B |
| MC Usefulness | -0.18 | **0.20** | 7B |
| Open-ended Usefulness | **0.76** | 0.00 | 2B |
| **Average** | 0.36 | **0.58** | 7B |
| Explanatoriness | **1.08** | -2.38 | 2B |

After preference fine-tuning, the 2B model's user usefulness rating improved by an average of `$1.67\sigma$`; the SFT-tuned 2B model outperformed ChartGemma (3B) on 8 out of 11 automatic metrics; contrastive fine-tuning on BLIP-2 improved Precision@1 by 65 percentage points compared to the COCO fine-tuned baseline.

### Ablation Study

| Configuration | Average Effect Size | Explanatoriness | Description |
|------|-----------|--------|------|
| Fine-tuning only, without guided gen (2B) | `$0.49\sigma$` | `$1.49\sigma$` | Fine-tuning itself yields significant effects |
| Guided gen + DPO fine-tuning (2B) | `$0.52\sigma$` | `$1.06\sigma$` | Fine-tuning is further amplified with guided generation |
| Guided gen only, GPT-4o baseline | `$0.28\sigma$` | `$0.33\sigma$` | Guided generation also benefits un-tuned GPT |
| Guided gen only, un-tuned 2B | `$-0.15\sigma$` | `$0.08\sigma$` | Un-tuned model cannot exploit guided generation and even degrades |
| Guided gen only, DPO-tuned 2B | **`$0.58\sigma$`** | **`$3.17\sigma$`** | Fine-tuning acts as a prerequisite for guided generation to work |

### Key Findings

- **Guided generation requires fine-tuning as a prerequisite**: The un-tuned 2B model using guided generation yields an average effect size of `$-0.15\sigma$` (representing degradation), whereas after DPO fine-tuning, the effect size jumps to `$0.58\sigma$` when using guided generation. This suggests that models must learn BLV preferences via Sightation first to effectively utilize inference-time guided prompts.
- **2B and 7B models benefit in vastly different directions**: The 2B model improves significantly in explanatoriness (`$+1.08\sigma$`), creating highly interpretative descriptions. Conversely, the 7B model gains mostly in conciseness (`$+1.69\sigma$`), leading to more refined outputs. This implies that small and large models experience distinct alignment bottlenecks.
- **BLV educators independently verified the importance of conciseness**: Three blind educators independently stressed during interviews that "descriptions must be accurate, consistent, concise, and contain key elements," closely aligning with the 7B model's massive boost in conciseness.
- **Efficacy of synthetic contrastive data**: The strategy of removing the best sentence from a description to construct a rejected sample, and removing random non-best sentences to form a chosen sample, produced 10k preference pairs (making up 62.5% of SightationPreference), serving as a crucial data source for DPO training.

## Highlights & Insights

- **"Evaluation over Generation" Annotation Paradigm**: Transforming the role of annotators from creators to evaluators simultaneously addresses high annotation costs, annotator bias, and preference misalignment. The cognitive load of evaluation is significantly lower than that of text generation, enabling large-scale annotations.
- **Multi-Role Collaborative Assessment Design**: Recruiting three groups of evaluators based on visual ability and professional background to partition 9 distinct evaluation dimensions achieves higher efficiency and accuracy than traditional single-group setups. This "division of labor by expertise" scheme extends easily to other annotation tasks addressing heterogeneous user populations.
- **SFT-DPO Data Isolation Strategy**: Deliberately training the SFT phase and DPO phase on descriptions of disjoint diagram sets avoids overfitting caused by repeating on identical samples—a highly practical tip for preference alignment training.

## Limitations & Future Work

- **Monolithic Supervision Signals**: Guided generation solely employs QA formats, exploring no other latent supervision forms (such as heading generation or listing key elements), potentially capping the effectiveness of guided generation.
- **Restricted Diagram Scope**: Evaluated on AI2D (primary school science diagrams), the generalizability to complex diagrams in academic papers, flowcharts, or non-standard layouts remains unverified.
- **Limited Evaluator Cohort**: Only 17 expert educators participated in the validation (8 BLV + 9 sighted), leading to a small sample size that may limit statistical power.
- **No Segmentation Utilization**: Advanced image segmentation methods are not utilized to localize and parse detail hotspots in intricate diagrams.
- **Data Generation Bias**: Base candidates are generated by GPT-4o mini and Qwen2-VL, which may introduce inherent model-specific biases.

## Related Work & Insights

- **vs. VizWiz-VQA/LF**: The VizWiz series involves BLV populations but is confined to VQA, whereas Sightation spans five tasks (completions, preference, retrieval, VQA, reasoning), presenting broader applicability.
- **vs. VisText / ChartGemma**: Existing datasets feature significantly shorter average lengths than Sightation (37.5-74.6 words vs. 188.3 words) and lack BLV validation. Sightation's long-form descriptions are highly structured for BLV users to capture info-dense diagrams.
- **vs. Kreiss et al. (2022)**: Prior work verified sighted individuals as BLV preference proxies only on selected dimensions. This work scales this concept to dataset dimensions (137k samples) across 9 evaluation perspectives.

## Rating

- Novelty: ⭐⭐⭐⭐ The "evaluation over generation" labeling strategy is novel, and the three-layer collaborative assessment design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive four-way evaluation involving BLV educators, sighted individuals, VLM judges, and automatic metrics, coupled with rigorous ablation designs.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and logically sound with rigorous effect size analysis and comprehensive tables, though some symbolic notation is slightly complex.
- Value: ⭐⭐⭐⭐⭐ Fills a crucial gap in BLV-aligned datasets; the dataset is openly accessible and supports diverse multi-task training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interactive Episodic Memory with User Feedback](../../CVPR2026/multimodal_vlm/interactive_episodic_memory_with_user_feedback.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ACL 2025\] VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](vf_eval_aigc_video_feedback.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ACL 2025\] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)

</div>

<!-- RELATED:END -->
