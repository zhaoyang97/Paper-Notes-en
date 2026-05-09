---
title: >-
  [Paper Note] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration
description: >-
  [ACL 2026][LLM/NLP][paper revision] This paper presents XtraGPT—the first open-source LLM suite (1.5B–14B) for academic paper revision. By fine-tuning on 7,000 top-venue papers and 140,000 criteria-guided instruction–revision pairs, it enables context-aware, paragraph-level controllable revision. The 7B variant matches GPT-4o-mini, the 14B variant surpasses it, and human evaluation shows an average predicted score improvement of 0.65 points after revision.
tags:
  - ACL 2026
  - LLM/NLP
  - paper revision
  - human-AI collaboration
  - context-aware
  - controllable generation
  - academic writing
date: 2026-05-08
content_hash: ac1e1d916df1e208
---

# XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration

**Conference**: ACL 2026
**arXiv**: [2505.11336](https://arxiv.org/abs/2505.11336)
**Code**: [GitHub](https://github.com/Xtra-Computing/XtraGPT)
**Area**: Text Generation
**Keywords**: paper revision, human-AI collaboration, context-aware, controllable generation, academic writing

## TL;DR

This paper presents XtraGPT—the first open-source LLM suite (1.5B–14B) for academic paper revision. By fine-tuning on 7,000 top-venue papers and 140,000 criteria-guided instruction–revision pairs, it enables context-aware, paragraph-level controllable revision. The 7B variant matches GPT-4o-mini, the 14B variant surpasses it, and human evaluation shows an average predicted score improvement of 0.65 points after revision.

## Background & Motivation

**Background**: LLMs are increasingly integrated into academic workflows, but their use is largely limited to surface-level polishing via general-purpose models such as ChatGPT. Existing AI writing tools either generate entire papers from scratch (raising concerns about originality and ethics) or perform only grammatical corrections.

**Limitations of Prior Work**: (1) General-purpose LLMs tend to revise academic papers superficially—improving fluency without addressing core argumentative issues such as unclear motivation or vague contributions; (2) Academic writing is inherently iterative, yet current LLM workflows treat each prompt as an independent interaction, lacking cross-revision context tracking; (3) Existing systems lack three critical controllability dimensions: adherence to in-context examples, user instructions, and explicit writing criteria.

**Key Challenge**: Academic paper revision requires understanding the full-document context and conforming to domain-specific writing standards, yet general-purpose LLMs lack both holistic comprehension and internalized academic writing norms.

**Goal**: To construct a human-AI collaborative paper revision framework in which the model serves as an "assistant" providing context-aware, targeted revisions while humans retain creative control.

**Key Insight**: The revision task is formulated as criteria-guided conditional generation—given the full paper $T$, target paragraph $p$, and user instruction $q$, the model generates a revised paragraph $\hat{p} = \text{Model}_\theta(p, q, T)$. Revision intent is normalized through 20 writing criteria distilled from top-venue reviewer guidelines.

**Core Idea**: Through criteria-guided intent alignment and context-aware modeling, XtraGPT elevates academic paper revision from "generic polishing" to "precise, structured improvement."

## Method

### Overall Architecture

The post-training framework of XtraGPT comprises three core components: (1) **criteria-guided intent alignment**—20 writing criteria covering six sections: title, abstract, introduction, background, experiments, and conclusion; (2) **context-aware modeling**—the full paper $T$ is provided as explicit input; (3) **controllable post-training (CPT)**—maximizing $\log P_\theta(\hat{p} | q, T, p)$ on the ReviseQA dataset. At inference time, the Human-AI Collaboration (HAC) protocol is followed: the user selects a paragraph and issues an instruction, the model returns a revision, and the user reviews and integrates the result.

### Key Designs

1. **Criteria-Guided Intent Alignment**:

    - **Function**: Maps vague user instructions to concrete, actionable revision strategies.
    - **Mechanism**: Twenty paragraph-level writing criteria $C$ are distilled from ICLR reviewer guidelines and expert experience, covering six core sections (e.g., "consistency between title and content," "strength and clarity of motivation in the introduction," "experimental support for key contributions"). Each instruction–revision pair in the training data is explicitly associated with a criterion $c \in C$, enabling the model to learn associations between specific request types and corresponding revision strategies.
    - **Design Motivation**: Author instructions are often high-level and vague (e.g., "strengthen the contributions"). A structured set of criteria serves as a "bridge" that translates abstract intent into concrete textual operations. Grounding these criteria in authoritative writing guidelines ensures that revisions conform to academic norms.

2. **Context-Aware Modeling**:

    - **Function**: Ensures paragraph-level revisions remain consistent with the full-document narrative.
    - **Mechanism**: The complete paper body $T$ (excluding acknowledgments and references, capped at 16,384 tokens) is provided as explicit model input. The training objective $\mathcal{L}_{CPT}(\theta) = -\mathbb{E}[\log P_\theta(\hat{p} | q, T, p)]$ forces the model to learn representations conditioned on the global narrative, structure, and terminology.
    - **Design Motivation**: Revising "introduction motivation" requires fundamentally different considerations than revising "experimental analysis." Revisions made without full-document context produce paragraphs that are inconsistent with the rest of the paper—ablation results show that removing context causes the LC win rate on the conclusion section to plummet from 50% to 11.76%.

3. **ReviseQA Dataset Construction**:

    - **Function**: Provides large-scale, high-quality criteria-guided revision training data.
    - **Mechanism**: From 7,000 submissions to ICLR 2024, paragraphs from six core sections of each paper are sampled, and instruction–revision pairs are generated according to the 20 criteria. Revisions are generated using GPT-4o-mini (hallucination rate of only 1.7%) and validated by three doctoral students. The dataset totals 140,000 high-quality instruction–revision pairs, with 5% held out as a test set.
    - **Design Motivation**: Existing datasets focus either on grammatical correction or end-to-end generation, leaving a gap in large-scale training resources for paragraph-level structured revision.

### Loss & Training

The standard conditional language modeling objective is used: $\mathcal{L}_{CPT}(\theta) = -\mathbb{E}[\log P_\theta(\hat{p} | q, T, p)]$. Full-parameter fine-tuning is adopted (found to outperform LoRA). Evaluation employs the Length-Controlled Win Rate (LC Win Rate), automatically judged via `alpaca_eval_gpt4_turbo_fn` to eliminate length bias.

## Key Experimental Results

### Main Results

**Length-Controlled Win Rate (vs. XtraGPT-7B as anchor)**

| Model | Title | Abstract | Introduction | Background | Experiments | Conclusion | Overall |
|-------|-------|----------|--------------|------------|-------------|------------|---------|
| QwQ-32B | 46.58 | 85.34 | 81.99 | 83.82 | 82.64 | 95.69 | **80.86** |
| DeepSeek-v3-671B | 56.42 | 65.71 | 68.32 | 74.12 | 72.11 | 64.83 | **67.70** |
| XtraGPT-14B | 55.29 | 59.43 | 50.90 | 59.43 | 57.87 | 52.11 | **55.49** |
| GPT-4o-Mini | 48.80 | 47.43 | 55.73 | 66.07 | 45.67 | 39.03 | **51.75** |
| **XtraGPT-7B (anchor)** | — | — | — | — | — | — | **50.00** |
| Qwen2.5-7B-Instruct | 39.93 | 45.14 | 45.64 | 39.28 | 33.87 | 31.17 | **40.80** |

### Ablation Study

| Configuration | Overall LC Win Rate | Notes |
|---------------|---------------------|-------|
| XtraGPT-7B (full CPT) | 50.00 | Anchor |
| w/o writing criteria | 44.65 | Criteria guidance removed |
| Qwen2.5-7B (base) | 40.80 | No fine-tuning |
| w/o context $T$ | 34.71 | Full-document context removed |

### Key Findings

- XtraGPT-7B outperforms all open-source models of comparable scale and surpasses GPT-4o-mini on the abstract, experiments, and conclusion sections.
- Full-document context $T$ is the most critical component: its removal causes the LC win rate on the conclusion to drop from 50% to 11.76% and the overall rate to fall to 34.71%.
- Criteria guidance contributes significantly but less than context (44.65 vs. 50.00), with particularly pronounced effects on structured sections such as the introduction and abstract.
- AI-SCIENTIST full-paper evaluation shows: contribution score +7.89%, presentation score +12.50%, rigor +6.41%, with the overall predicted score rising from 6.08 to 6.73 ($p < 0.001$).
- In human evaluation, revision acceptance rate is 3.23/5.0 and instruction-following is rated 3.78/5.0.

## Highlights & Insights

- The HAC protocol embodies a principled design philosophy: humans retain responsibility for creativity and decision-making while AI handles execution and improvement, avoiding the originality and ethical risks associated with full automation.
- The distillation of 20 writing criteria is itself a valuable resource—it can serve as a paper self-review checklist or reviewer guideline.
- Using AI-SCIENTIST as a paper quality evaluator is an elegant experimental design, transforming the subjective question of "did the paper improve?" into quantifiable predicted score changes.

## Limitations & Future Work

- ReviseQA is sourced exclusively from ICLR 2024, which may bias the model toward ML/AI writing styles; generalization to other disciplines (e.g., NLP, biomedicine) remains untested.
- Using GPT-4o-mini-generated revisions as training targets may introduce stylistic biases and preferences from that model.
- The current evaluation covers only single-round revision; the cumulative effect of multi-round iterative revision has not been systematically measured.
- The 16K token context window limits the handling of exceptionally long papers.
- Alignment with human revision histories (e.g., revision records on OpenReview) has not been explored.

## Related Work & Insights

- **vs. AI Scientist**: AI Scientist pursues fully automated paper generation and reviewing, raising concerns about originality; XtraGPT is explicitly positioned as an "assistive tool" that preserves human agency.
- **vs. STORM/CO-STORM**: STORM generates articles from scratch and is prone to factual hallucination and consistency issues; XtraGPT revises human drafts, which inherently reduces hallucination risk.
- **vs. CycleResearcher**: CycleResearcher employs a paper generation–review cycle for self-improvement but is susceptible to reward hacking; XtraGPT relies on human-annotated, criteria-guided training data for quality assurance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First open-source LLM suite targeting academic paper revision; the HAC framework is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ LC win rate + human evaluation + AI-SCIENTIST full-paper evaluation + ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly described with well-articulated differentiation from prior work.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a daily pain point for researchers; open-source models, dataset, and Overleaf plugin ensure high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](../../NeurIPS2025/llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](../../AAAI2026/llm_nlp/irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)
- [\[ACL 2026\] AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)
- [\[ICLR 2026\] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](../../ICLR2026/llm_nlp/optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)

<!-- RELATED:END -->
