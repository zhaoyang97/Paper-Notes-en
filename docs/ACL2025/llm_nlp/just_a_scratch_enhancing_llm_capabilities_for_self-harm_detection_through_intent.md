---
title: >-
  [Paper Note] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement
description: >-
  [ACL 2025][LLM (Other)][self-harm detection] Proposed the SHINES dataset and the CESM-100 emoji matrix to distinguish between "casual mention" and "serious intent" in self-harm expressions on social media. Combining contextual emoji interpretation and multi-task fine-tuning improved the LLM F1 score for self-harm detection from 0.74 (zero-shot) to 0.88 (multi-task + CESM-100), while generating interpretable reasons for predictions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "self-harm detection"
  - "intent classification"
  - "emoji interpretation"
  - "mental health"
  - "multitask learning"
date: 2026-05-08
content_hash: 87d42db8ab7f91a9
---

# Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement

**Conference**: ACL 2025  
**arXiv**: [2506.05073](https://arxiv.org/abs/2506.05073)  
**Code**: [Resource Page](https://www.iitp.ac.in/%7eai-nlp-ml/resources.html#SHINES)  
**Area**: LLM/NLP  
**Keywords**: self-harm detection, intent classification, emoji interpretation, mental health, multitask learning

## TL;DR

Proposed the SHINES dataset and the CESM-100 emoji matrix to distinguish between "casual mention" and "serious intent" in self-harm expressions on social media. Combining contextual emoji interpretation and multi-task fine-tuning improved the LLM F1 score for self-harm detection from 0.74 (zero-shot) to 0.88 (multi-task + CESM-100), while generating interpretable reasons for predictions.

## Background & Motivation

**Background**: Self-harm detection on social media is a crucial component of mental health intervention. While LLMs have been applied to tasks like stress detection and depression recognition, their performance on more fine-grained tasks like self-harm remains suboptimal.

**Limitations of Prior Work**: (1) Existing LLMs struggle to distinguish between "Casual Mention" (CM) and "Serious Intent" (SI) in self-harm expressions—e.g., "might as well electrocute myself" (exaggerated rhetoric) vs. "I don't even want to see tomorrow" (genuine cry for help); (2) Semantic shifts of emojis in self-harm contexts are largely ignored—a knife emoji 🔪 might imply self-harm, but a face with tears of joy 😂 might mask pain with laughter; (3) Existing LLMs (such as ChatGPT-4o and Gemini) exhibit inconsistent classification results on the same post.

**Key Challenge**: Self-harm signals are highly implicit and multimodal (text + emojis), making simple binary classification inadequate for capturing differences in intent levels.

**Goal**: Build datasets and emoji resources that distinguish CM/SI intent, and leverage a multi-task learning framework to enhance LLM capabilities in self-harm post detection and explanation.

**Key Insight**: Decompose self-harm detection into three sub-tasks (classification + CM/SI span extraction + explanation generation), utilizing the CESM-100 emoji matrix to provide additional contextual cues.

**Core Idea**: Enhance LLM self-harm detection via intent stratification (CM vs. SI) and a contextualized emoji matrix (CESM-100), achieving integrated detection and explanation.

## Method

### Overall Architecture

The overall pipeline consists of three phases: (1) Input Enhancement—integrating post text with CESM-100 emoji interpretations into a unified input; (2) Multi-task Fine-tuning—jointly training self-harm classification (main task) and CM/SI span extraction (auxiliary task); (3) Explanation Generation—producing interpretable reasoning for predictions based on the classification and span results.

### Key Designs

1. **SHINES Dataset**:

    - **Function**: Constructs an annotated dataset of 5,206 social media posts, providing self-harm labels, CM/SI span annotations, and emoji explanations.
    - **Mechanism**: Collected 4,206 posts from Reddit mental health subreddits (e.g., SuicideWatch, selfharm), using Presidio to remove PII. Three annotators independently labeled self-harm/non-self-harm (Fleiss' $\kappa = 0.78$). Up to 3 CM and SI spans were annotated per post (macro-F1: CM=0.66, SI=0.69). Additionally, 1,000 synthetic posts were generated using ChatGPT-3.5 (quality verified at F1=58% after manual emoji revision).
    - **Design Motivation**: Existing datasets lack CM/SI intent-level annotations, making it impossible to train models to distinguish rhetorical exaggeration from genuine cries for help.

2. **CESM-100 (Centennial Emoji Sensitivity Matrix)**:

    - **Function**: Constructs a self-harm contextual explanation matrix for 100 emojis.
    - **Mechanism**: Each emoji is annotated with its general meaning, self-harm contextual meaning, CM association level (Low/Medium/High), and SI association level. For example: coffin ⚰️ (CM: Low, SI: High), face with tears of joy 😂 (CM: High, SI: Medium), broken heart 💔 (CM: Medium, SI: High).
    - **Design Motivation**: The semantics of emojis in self-harm contexts differ significantly from everyday usage; models require this domain-specific semantic mapping to reduce misclassifications. Inter-annotator agreement reached Fleiss' $\kappa = 0.71$ (CM) / $0.75$ (SI).

3. **Multi-task Fine-tuning Framework**:

    - **Function**: Simultaneously optimizes three tasks: self-harm classification, CM/SI span extraction, and explanation generation.
    - **Mechanism**: Uses a unified input representation for decoder-only LLMs (Llama 3, Mental-Alpaca, MentalLlama). Binary cross-entropy loss is used for self-harm classification, sparse categorical cross-entropy loss (predicting start and end positions) for span extraction, and prompting is used for explanation generation to guide the model to reference CM/SI spans and CESM-100 interpretations.
    - **Design Motivation**: Multi-task learning allows models to share semantic understanding of CM/SI, thereby improving performance on the main task (self-harm detection).

4. **Emoji Usage Pattern Analysis**:

    - **Function**: Analyzes differences in emoji usage patterns between self-harm and non-self-harm posts.
    - **Mechanism**: Emoji combinations are more complex in self-harm posts (multi-emoji combinations are far more prevalent in SH than NSH), SI posts favor metaphorical usage (Metaphorical > Direct), and CM posts favor direct usage (Direct > Metaphorical).
    - **Design Motivation**: To validate the feasibility of emojis as signals for self-harm detection and the rationality of the CESM-100 design.

## Key Experimental Results

### Main Results (Llama 3 F1 across configurations)

| Setting | SHC F1 | CMSE F1 | SISE F1 | RG SemSim |
|------|--------|---------|---------|-----------|
| Zero-shot | 0.74 | - | - | 0.75 |
| Few-shot | 0.79 | - | - | 0.80 |
| Fine-tuning (Single-task) | 0.83 | - | - | 0.84 |
| MT Fine-tuning + CESM-100 | **0.88** | **0.85** | **0.84** | **0.88** |

### Ablation Study (Llama 3)

| Configuration | SHC F1 | Description |
|------|--------|------|
| MT + CESM-100 (Full) | 0.88 | Best |
| MT w/o CESM-100 | 0.84 | Removes emoji matrix, -0.04 |
| Single-task + CESM-100 | 0.86 | Removes multi-task, -0.02 |
| Single-task w/o CESM-100 | 0.83 | Baseline fine-tuning |

### Key Findings
- Multi-task fine-tuning yields a +4% F1 improvement over single-task, +7% over few-shot, and +12% over zero-shot (average across models).
- CESM-100 contributes approximately +4% F1 improvement, which is statistically significant ($p=0.0198$ for Llama, $p=0.017$ for Mental-Alpaca).
- Domain-specific models (Mental-Alpaca: 0.86, MentalLlama: 0.85) perform slightly below the general-purpose Llama 3 (0.88), possibly due to the larger parameter size of Llama 3.
- Emoji combinations in self-harm posts show a "more complex implies more likely SH" pattern: single emoji SH:NSH = 7815:5359, 4+ emojis SH:NSH = 82:22.

## Highlights & Insights
- The CESM-100 emoji matrix is a unique contribution—elevating emojis from simple sentiment signals to contextualized self-harm indicators with CM/SI dimensions, which can be reused for other mental health NLP tasks.
- The multi-task framework is well-designed. CM/SI span extraction, as an auxiliary task, not only improves detection performance but also provides structured material for explanation generation, achieving "explainable self-harm detection".

## Limitations & Future Work
- Due to hardware limitations (NVIDIA K80, 24GB), only 7-8B parameter models were evaluated, and larger models like GPT-4 were not tested.
- The emoji usage patterns in synthetic data (1,000 posts) required manual revision, indicating that LLM-generated emojis are still not natural enough.
- The dataset source is singular (Reddit), which may not represent the expressive styles of other platforms like Twitter or Instagram.
- The coverage of 100 emojis in CESM-100 is limited; emerging emojis and cross-cultural differences were not addressed.

## Related Work & Insights
- **vs McBain et al. 2025**: Found that models like GPT-4o exhibit an upward bias when assessing the severity of suicidal ideation. This paper directly addresses this issue through CM/SI intent stratification.
- **vs Grabb et al. 2024**: Pointed out that LLMs can potentially cause harm in mental health emergencies. The explainable reasoning generation in this work serves to assist rather than replace professional judgment.
- **vs Yang et al. 2024 (MentalLlama)**: Although domain-specific models possess mental health knowledge, they perform worse than general Llama 3 + multi-task fine-tuning on this task.

## Rating
- Novelty: ⭐⭐⭐⭐ CESM-100 and CM/SI intent stratification are creative contributions, but the overall framework (multi-task fine-tuning) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models $\times$ four settings + two ablation groups + significance testing, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐ Complete structure but average readability; the organization of the dataset description and methodology sections is somewhat loose.
- Value: ⭐⭐⭐⭐ Practical application value for social media mental health monitoring; SHINES and CESM-100 can serve as community resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
