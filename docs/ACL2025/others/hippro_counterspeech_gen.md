---
title: >-
  [Paper Note] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning
description: >-
  [ACL 2025][Counterspeech] Proposed the HiPPrO two-stage framework for multi-conditioned counterspeech generation. The first stage optimizes counterspeech generation in multiple attribute (strategy + emotion) spaces through hierarchical prefix learning, and the second stage enhances constructiveness using reference-free and reward-free preference optimization. Strategy consistency increases by ~38%, and ROUGE metrics improve by 2-3%.
tags:
  - "ACL 2025"
  - "Counterspeech"
  - "Prefix Learning"
  - "Multi-attribute Conditional Generation"
  - "Preference Optimization"
  - "Harmful Content Countering"
date: 2026-05-08
content_hash: 5ceed7baa60db319
---

# Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning

**Conference**: ACL 2025  
**arXiv**: [2505.11958](https://arxiv.org/abs/2505.11958)  
**Code**: Open-sourced on GitHub  
**Area**: Others  
**Keywords**: Counterspeech, Prefix Learning, Multi-attribute Conditional Generation, Preference Optimization, Harmful Content Countering  

## TL;DR
Proposed the HiPPrO two-stage framework for multi-conditioned counterspeech generation. The first stage optimizes counterspeech generation in multiple attribute (strategy + emotion) spaces through hierarchical prefix learning, and the second stage enhances constructiveness using reference-free and reward-free preference optimization. Strategy consistency increases by ~38%, and ROUGE metrics improve by 2-3%.

## Background & Motivation

### Background

**Background**: Counterspeech is a powerful tool to combat online hate speech. Prior studies mainly generate counterspeech under a single condition (e.g., a specific strategy).

**Limitations of Prior Work**: (a) Generating counterspeech based solely on a single strategy (e.g., "humor" or "fact-correction") is not fine-grained enough—effective real-world counterspeech considers both strategy and emotional tone; (b) Simultaneous conditioning on multiple attributes remains understudied; (c) The generated counterspeech may lack "constructiveness"—requiring preference optimization.

**Key Challenge**: Multi-attribute conditioning increases control difficulty—how to simultaneously satisfy both conditions of "using a fact-correction strategy" and "expressing an empathetic emotion"?

**Goal** Generate more effective counterspeech based simultaneously on both strategy and emotion dimensions.

**Key Insight**: Code different attributes using a hierarchical prefix embedding space—each attribute has an independent prefix parameter space, achieving joint multi-attribute control through hierarchical combination.

**Core Idea**: Hierarchical prefix learning for fusing multi-attributes + preference optimization to enhance constructiveness.

## Method

### Overall Architecture
Two-stage pipeline: (1) **Hierarchical Prefix Learning**—learns prefix embeddings for strategy and emotion attributes separately, which are hierarchically combined to guide generation; (2) **Preference Optimization**—further optimizes generation quality and constructiveness using reference-free and reward-free DPO variants.

### Key Designs

1. **Hierarchical Prefix Optimization**:

    - **Function**: Learns dedicated prefix embedding spaces for different attributes
    - **Mechanism**: Strategy prefix $P_{strategy}$ encodes counterspeech strategies (e.g., humor/fact/empathy), and emotion prefix $P_{emotion}$ encodes emotional tones (e.g., calm/resolute/gentle). Through hierarchical fusion—the strategy prefix first generates the framework, and then the emotion prefix adjusts the tone
    - **Design Motivation**: Attribute-independent prefix spaces prevent interference between different attributes, and hierarchical combination allows flexible multi-attribute control

2. **Reference-Free Preference Optimization**:

    - **Function**: Further enhances the constructiveness of generation on top of prefix learning
    - **Mechanism**: Uses DPO variants (such as SimPO or ORPO) that do not require a reference model or a reward model, reducing computational overhead
    - **Design Motivation**: Prefix learning guarantees attribute consistency, while preference optimization guarantees generation quality—the two are complementary

3. **Dataset Expansion**:

    - **Function**: Adds emotion annotations to existing counterspeech datasets
    - **Mechanism**: Emotion labels for 13,973 counterspeeches in IntentCONANv2 are annotated by 5 annotators
    - **Design Motivation**: Prior datasets only have strategy labels without emotion labels, making it impossible to train multi-attribute models

### Loss & Training
- Stage 1: Conditional language modeling loss + prefix embedding optimization
- Stage 2: SimPO/ORPO preference loss
- Based on LLMs such as LLaMA/Mistral

## Key Experimental Results

### Main Results

| Method | Strategy Consistency (↑) | Rouge-1 | Rouge-L | Constructiveness (Human Eval ↑) |
|------|-------------|---------|---------|-------------|
| Single-attribute Baseline | Baseline | Baseline | Baseline | Medium |
| Multi-attribute (Non-hierarchical) | Medium | Medium | Medium | Medium |
| **HiPPrO** | **+38%** | **+3%** | **+3%** | **Highest** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Emotion Prefix | Strategy consistent but emotion uncontrollable | Single attribute is insufficient |
| w/o Hierarchical Fusion | Two attributes interfere | Hierarchical structure is necessary |
| w/o Preference Optimization | Constructiveness decreases | Preference optimization improves quality |

### Key Findings
- The 38% increase in strategy consistency is significant—hierarchical prefixes allow the model to precisely follow specified strategies
- Human evaluation confirms the advantages in relevance and appropriateness of the generated counterspeech
- Multi-attribute conditional generation is more effective than single-attribute—counterspeech indeed requires simultaneous consideration of strategy and tone
- Newly annotated emotion labels provide a valuable resource to the community

## Highlights & Insights
- **Hierarchical prefix space** is an elegant solution for multi-attribute controllable generation—each attribute is optimized independently to avoid interference, and hierarchical combination allows flexible control.
- The insight that **"counterspeech needs to consider both strategy and emotion"** has practical value—solely "denouncing with facts" is insufficient; appropriate emotional expression is also needed to be effective.
- Preference optimization increases constructiveness—it is not just about saying the right things, but also "saying them well."
- The framework can be transferred to other multi-attribute controllable generation scenarios (e.g., joint control of style + topic + audience).

## Limitations & Future Work
- Only validated on dual attributes (strategy + emotion); extensibility to more attributes remains unknown
- The subjectivity of emotion annotation may introduce noise
- Validated only on English datasets
- The performance in real-world social media environments has not been evaluated

## Related Work & Insights
- **vs Single-conditioned counterspeech generation**: Previous work was conditioned only on strategy; HiPPrO adds the emotion dimension
- **vs Contrastive Perplexity (detoxification)**: CP removes toxic attributes; HiPPrO adds constructive attributes—opposite directions
- **vs Prefix-Tuning**: Traditional prefix-tuning uses a single prefix; HiPPrO uses hierarchical multi-prefixes

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical prefixes, multi-attributes, and preference optimization is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Automatic + human evaluation + ablation + dataset expansion
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear
- Value: ⭐⭐⭐⭐ Practical value for online safety and countering hate speech

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)
- [\[ACL 2025\] Interlocking-free Selective Rationalization Through Genetic-based Learning](interlocking-free_selective_rationalization_through_genetic-based_learning.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)

</div>

<!-- RELATED:END -->
