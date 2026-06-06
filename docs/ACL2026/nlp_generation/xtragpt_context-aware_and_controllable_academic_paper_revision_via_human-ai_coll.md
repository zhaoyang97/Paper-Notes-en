---
title: >-
  [Paper Note] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration
description: >-
  [ACL 2026][Text Generation][Paper revision] This paper proposes XtraGPT—the first open-source LLM suite (1.5B–14B) specifically designed for academic paper revision. By fine-tuning on 7…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Paper revision"
  - "human-AI collaboration"
  - "context-awareness"
  - "controllable generation"
  - "academic writing"
date: 2026-05-08
content_hash: e5f71ac17f30ac33
---

# XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration

**Conference**: ACL 2026  
**arXiv**: [2505.11336](https://arxiv.org/abs/2505.11336)  
**Code**: [GitHub](https://github.com/Xtra-Computing/XtraGPT)  
**Area**: Text Generation  
**Keywords**: Paper revision, human-AI collaboration, context-awareness, controllable generation, academic writing

## TL;DR

This paper proposes XtraGPT—the first open-source LLM suite (1.5B–14B) specifically designed for academic paper revision. By fine-tuning on 7,000 top-tier conference papers and 140,000 criteria-guided instruction-revision pairs, it achieves context-aware paragraph-level controllable revisions. The 7B version matches GPT-4o-mini, while the 14B version outperforms it. Human evaluation indicates that the predicted scores of revised papers increase by an average of 0.65 points.

## Background & Motivation

**Background**: The application of LLMs in academic workflows is increasingly widespread, but it remains primarily limited to surface-level polishing via general-purpose models like ChatGPT. Existing AI writing tools either generate entire papers from scratch (raising originality and ethical concerns) or focus solely on grammatical corrections.

**Limitations of Prior Work**: (1) General LLM revisions of academic papers are often superficial—improving fluency without addressing core argumentative issues (e.g., unclear motivation or vague contributions); (2) Academic writing is inherently iterative, but current LLM workflows treat each prompt as an independent interaction, lacking context tracking across revision rounds; (3) Existing systems lack three critical types of controllability: following contextual examples, following user instructions, and following explicit writing criteria.

**Key Challenge**: Academic paper revision requires an understanding of the full-text context and adherence to domain-specific writing standards, yet general LLMs lack both full-text comprehension capabilities and the internalization of academic writing norms.

**Goal**: To build a human-AI collaborative framework for paper revision where the model acts as an "assistant" providing context-aware, targeted revisions, while humans retain creative control.

**Key Insight**: Modeling the revision task as criteria-guided conditional generation—given the full text $T$, the target paragraph $p$, and user instruction $q$, the model generates the revised paragraph $\hat{p} = \text{Model}_\theta(p, q, T)$. Revision intent is regularized through 20 writing criteria extracted from top-tier conference review guidelines.

**Core Idea**: By utilizing criteria-guided intent alignment and context-aware modeling, academic paper revision is elevated from "general polishing" to "precise structural improvement."

## Method

### Overall Architecture

The post-training framework of XtraGPT consists of three core components: (1) Criteria-guided intent alignment—20 writing criteria covering six sections: Title, Abstract, Introduction, Background, Experiments, and Conclusion; (2) Context-aware modeling—using the full text $T$ as an explicit input; (3) Controllable Post-Training (CPT)—maximizing $\log P_\theta(\hat{p} | q, T, p)$ on the ReviseQA dataset. During inference, the system follows the HAC protocol: users select a paragraph and issue instructions, the model returns revisions, and the user reviews and integrates them.

### Key Designs

1.  **Criteria-Guided Intent Alignment**:
    - **Function**: Maps vague user instructions to specific, executable revision strategies.
    - **Mechanism**: Extracts 20 paragraph-level writing criteria $C$ from ICLR review guidelines and expert experience, covering six core sections (e.g., "Consistency between Title and Content," "Strength and Clarity of Motivation in Introduction," "Support for Main Innovations in Experiments"). Each instruction-revision pair in the training data is explicitly associated with a criterion $c \in C$, enabling the model to learn to associate specific types of requests with corresponding revision strategies.
    - **Design Motivation**: Author instructions are often high-level and vague (e.g., "strengthen the contribution"), requiring a set of structured criteria to serve as a "bridge" to transform abstract intent into concrete textual operations. These criteria originate from authoritative writing guides, ensuring revisions comply with academic norms.

2.  **Context-Aware Modeling**:
    - **Function**: Ensures paragraph revisions are consistent with the global narrative of the paper.
    - **Mechanism**: The complete paper body $T$ (excluding acknowledgments and references, controlled within 16,384 tokens) is used as explicit input for the model. The training objective $\mathcal{L}_{CPT}(\theta) = -\mathbb{E}[\log P_\theta(\hat{p} | q, T, p)]$ forces the model to learn representations conditioned on global narrative, structure, and terminology.
    - **Design Motivation**: Revisions to the "Introduction Motivation" require entirely different considerations than those for "Experimental Analysis." Revisions lacking full-text context lead to inconsistencies—ablation studies show that removing context causes the LC win rate in the Conclusion section to plummet from 50% to 11.76%.

3.  **ReviseQA Dataset Construction**:
    - **Function**: Provides large-scale, high-quality criteria-guided revision training data.
    - **Mechanism**: Paragraphs were sampled from the six core sections of 7,000 submissions to ICLR 2024, and instruction-revision pairs were generated based on 20 criteria. GPT-4o-mini was used to generate revisions (with a hallucination rate of only 1.7%), and human quality verification was performed by three PhD students. The dataset comprises 140,000 high-quality instruction-revision pairs, with 5% reserved as a test set.
    - **Design Motivation**: Existing datasets either focus on grammatical correction or cover end-to-end generation, lacking large-scale training resources for paragraph-level structural revision.

### Loss & Training

The model uses a standard conditional language model loss $\mathcal{L}_{CPT}(\theta) = -\mathbb{E}[\log P_\theta(\hat{p} | q, T, p)]$. It employs full-parameter fine-tuning (outperforming LoRA). Evaluation uses Length-Controlled Win Rate (LC Win Rate), judged automatically via `alpaca_eval_gpt4_turbo_fn` to eliminate length bias.

## Key Experimental Results

### Main Results

**Length-Controlled Win Rate (vs. XtraGPT-7B as anchor)**

| Model | Title | Abstract | Intro | Background | Exp | Conclusion | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| QwQ-32B | 46.58 | 85.34 | 81.99 | 83.82 | 82.64 | 95.69 | **80.86** |
| DeepSeek-v3-671B | 56.42 | 65.71 | 68.32 | 74.12 | 72.11 | 64.83 | **67.70** |
| XtraGPT-14B | 55.29 | 59.43 | 50.90 | 59.43 | 57.87 | 52.11 | **55.49** |
| GPT-4o-Mini | 48.80 | 47.43 | 55.73 | 66.07 | 45.67 | 39.03 | **51.75** |
| **XtraGPT-7B (anchor)** | — | — | — | — | — | — | **50.00** |
| Qwen2.5-7B-Instruct | 39.93 | 45.14 | 45.64 | 39.28 | 33.87 | 31.17 | **40.80** |

### Ablation Study

| Configuration | Overall LC Win Rate | Description |
| :--- | :--- | :--- |
| XtraGPT-7B (Full CPT) | 50.00 | Anchor |
| w/o Writing Criteria | 44.65 | Removed criteria guidance |
| Qwen2.5-7B (Base) | 40.80 | Without fine-tuning |
| w/o Context $T$ | 34.71 | Removed full-text context |

### Key Findings

- XtraGPT-7B outperforms all open-source models of the same scale and exceeds GPT-4o-mini in the Abstract, Experiments, and Conclusion sections.
- Context $T$ is the most critical component: after its removal, the LC win rate for the Conclusion section dropped from 50% to 11.76%, and the overall rate fell to 34.71%.
- Criteria guidance contributes significantly but is secondary to context (44.65 vs. 50.00); it is particularly important in structured sections like the Introduction and Abstract.
- AI-SCIENTIST full-text evaluation shows: after revision, Contribution scores increased by +7.89%, Presentation by +12.50%, and Soundness by +6.41%, with the overall rating rising from 6.08 to 6.73 ($p < 0.001$).
- In human evaluation, the revision acceptance rate was 3.23/5.0, and instruction following was 3.78/5.0.

## Highlights & Insights

- The design philosophy of the HAC protocol is noteworthy: humans are responsible for creativity and decision-making, while AI handles execution and improvement, avoiding the originality and ethical risks associated with full automation.
- The extraction of 20 writing criteria is a valuable resource in itself—serving as a self-check list or a review guide for papers.
- Using AI-SCIENTIST as a paper quality evaluator is a clever experimental design—transforming the subjective question "has the paper improved?" into quantifiable changes in predicted scores.

## Limitations & Future Work

- ReviseQA only includes data from ICLR 2024, which may bias the results toward ML/AI writing styles; generalization to other disciplines (e.g., NLP, biomedicine) is unknown.
- Revisions generated by GPT-4o-mini used as training targets may introduce the preferences and stylistic biases of that model.
- Currently, only single-round revision evaluation is supported; the cumulative effect of multi-round iterative revisions hasn't been systematically measured.
- The 16K token context window limits the ability to process extremely long papers.
- The possibility of aligning with human revision history (e.g., revision records on OpenReview) has not yet been explored.

## Related Work & Insights

- **vs. AI Scientist**: AI Scientist pursues fully automated paper generation and review, which raises concerns about originality; XtraGPT is explicitly positioned as an "assistant tool" that maintains human control.
- **vs. STORM/CO-STORM**: STORM generates articles from scratch, facing issues with factual hallucinations and consistency; XtraGPT revises based on human drafts, naturally reducing hallucination risks.
- **vs. CycleResearcher**: CycleResearcher uses a paper generation-review loop for self-improvement but is prone to reward hacking; XtraGPT uses criteria-guided data validated by human annotation.

## Rating

- Novelty: ⭐⭐⭐⭐ First open-source LLM suite for academic paper revision; sound HAC framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ LC win rate + human evaluation + AI-SCIENTIST evaluation + ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework and clear positioning relative to existing work.
- Value: ⭐⭐⭐⭐⭐ Addresses daily pain points for researchers; high utility with open-source models + datasets + Overleaf plugin.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[ACL 2026\] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)

</div>

<!-- RELATED:END -->
