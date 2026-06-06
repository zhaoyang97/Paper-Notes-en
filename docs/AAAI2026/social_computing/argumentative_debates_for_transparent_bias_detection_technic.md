---
title: >-
  [Paper Note] Argumentative Debates for Transparent Bias Detection
description: >-
  [AAAI 2026][Social Computing][Bias Detection] This paper proposes ABIDE (Argumentative BIas Detection by DEbate), which constructs Quantitative Bipolar Argumentation Frameworks (QBAFs) via neighborhood-based argument sch…
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Bias Detection"
  - "Argumentation Framework"
  - "QBAF"
  - "Transparency"
  - "Fairness"
  - "Debate"
date: 2026-05-08
content_hash: 0bc57a9141efb16c
---

# Argumentative Debates for Transparent Bias Detection

**Conference**: AAAI 2026
**arXiv**: [2508.04511](https://arxiv.org/abs/2508.04511)  
**Code**: [ABIDE](https://github.com/hamed-ayoobi/ABIDE)  
**Area**: Social Computing
**Keywords**: Bias Detection, Argumentation Framework, QBAF, Transparency, Fairness, Debate

## TL;DR

This paper proposes ABIDE (Argumentative BIas Detection by DEbate), which constructs Quantitative Bipolar Argumentation Frameworks (QBAFs) via neighborhood-based argument schemes, models the bias detection process as a structured debate, enables transparent bias reasoning from individual neighborhoods to the global level, and formally proves the correspondence between QBAF semantics and the expected behavior of bias detection.

## Background & Motivation

**Background**: As AI becomes increasingly pervasive in society, fairness concerns have grown in importance. Numerous bias detection methods have been proposed, yet most overlook the need for transparency. Explainability is a core requirement for algorithmic fairness.

**Limitations of Prior Work**: (1) Most existing bias detection methods operate as black boxes, providing no explanation of the sources or reasoning behind detected biases; (2) the few interpretable methods that exist offer explanations that are insufficiently structured to support in-depth debate; (3) existing debate-based approaches are unstructured—debates merely supply information to other entities, and the outcomes are not faithfully explained by the debate itself.

**Key Challenge**: There is a need for a method that can both accurately detect bias and transparently expose the underlying reasoning process.

**Goal**: To design an argumentation-centric bias detection framework in which the detection process itself constitutes an interpretable debate.

**Key Insight**: Leveraging argument schemes from formal argumentation and the gradual semantics of QBAFs to map bias evidence onto attack/support relations in an argumentation graph.

**Core Idea**: Model bias detection as neighborhood-level debate combined with cross-neighborhood evidence aggregation, with results computed automatically via QBAF semantics, achieving full transparency.

## Method

### Overall Architecture

ABIDE operates in three stages: (1) constructing a local bias-QBAF for each neighborhood; (2) adding critical questions; (3) constructing a global bias-QBAF across multiple neighborhoods. Argument strengths are computed using DF-QuAD or quadratic energy semantics.

### Key Designs

1. **Argument Schemes for Neighborhood Bias**
    - **Function**: Define how to extract bias evidence from a single neighborhood.
    - **Mechanism**: For individuals with protected feature $X_p = g$, the scheme examines the difference in local success probabilities between the protected and non-protected groups within the neighborhood.
    - **QBAF Mapping**: Four argument nodes — $\text{Disadv}_g$ and $\text{Adv}_g$ have base scores of 0; $\text{Pos}_{=g}$ and $\text{Pos}_{\neq g}$ have base scores equal to the local success probabilities of each group. Attack/support relations encode the direction of bias.
    - **Theoretical Guarantee**: Under DF-QuAD, $\sigma(\text{Disadv}_g)$ equals exactly the difference in success probabilities between the two groups (Proposition 5).

2. **Critical Questions**
    - **Function**: Provide structured challenges to the quality of neighborhood evidence.
    - **Three Dimensions**: CQ1 significance (neighborhood size), CQ2 objectivity (neighborhood convexity to prevent adversarial selection), CQ3 diversity (distributional entropy of protected/non-protected groups).
    - **Mechanism**: Critical questions act as attackers; the smaller, less objective, or less diverse the neighborhood, the stronger the attack.
    - **Design Motivation**: Prevent unreliable neighborhoods from misleading bias detection.

3. **Global Bias Aggregation**
    - **Function**: Combine local evidence from multiple neighborhoods into a global bias judgment.
    - **Mechanism**: $\text{Disadv}_g$ nodes from multiple neighborhoods support, and $\text{Adv}_g$ nodes attack, a global $\text{bias}_g$ node (base score 0, defaulting to no bias).
    - **Guarantee**: It is formally proven that $\sigma(\text{bias}_g) > 0$ when neighborhood evidence supporting bias outweighs evidence against it.

### Computational Strategy

ABIDE is a rule-based framework rather than a learned model. It employs KNN neighborhoods and modular gradual semantics to compute argument strengths, requiring no training.

## Key Experimental Results

### Synthetic Bias Model — Single-Neighborhood Detection

| Method | Model | K | Accuracy | F1 | Runtime (s) |
|--------|-------|---|----------|----|-------------|
| ABIDE | Global 1 | 100 | **1.00** | **1.00** | **3.88** |
| IRB | Global 1 | 100 | 1.00 | 1.00 | 48.47 |
| ABIDE | Global 2 | 100 | **1.00** | **1.00** | **0.35** |
| IRB | Global 2 | 100 | 0.00 | 0.00 | 8.99 |
| ABIDE | Local 1 | 100 | **1.00** | **1.00** | **4.96** |
| IRB | Local 1 | 100 | 0.74 | 0.48 | 49.45 |

### Multi-Neighborhood Aggregation Detection

| Method | Model | Accuracy | F1 |
|--------|-------|----------|----|
| ABIDE | Global 2 | **1.00** | **1.00** |
| IRB | Global 2 | 0.00 | 0.00 |
| ABIDE | Local 1 | **1.00** | **1.00** |
| IRB | Local 1 | 0.70 | 0.36 |

### ChatGPT-4o Bias Detection

On the COMPAS dataset, ABIDE identifies 77 biased neighborhoods in ChatGPT-4o against African-American individuals, compared to only 2 detected by IRB, revealing the problem of LLMs inheriting societal biases.

### Key Findings

- ABIDE decisively outperforms IRB on Global 2 (intersectional bias: Black + female), where IRB scores zero across all metrics, as IRB only considers single-attribute bias.
- ABIDE runs 5–25× faster than IRB due to a more compact QBAF structure.
- Multi-neighborhood aggregation substantially improves detection performance (single-neighborhood Recall: 0.90 → multi-neighborhood: 1.00).

## Highlights & Insights

1. **Integration of formal theory and practice**: Each design decision is supported by a corresponding mathematical proposition (Propositions 2–6).
2. **Elegant critical question mechanism**: The three-dimensional questioning makes the method robust against adversarial examples.
3. **Natural interpretation under DF-QuAD**: Bias strength equals exactly the difference in success probabilities between the two groups.
4. **Extensible to human-machine debate**: The framework naturally supports multi-party debate scenarios.

## Limitations & Future Work

1. Neighborhood construction relies on distance metrics and the choice of $K$, which may be unreliable for high-dimensional sparse data.
2. Fairness is defined solely via statistical parity, without covering alternative fairness definitions.
3. Computational complexity scales linearly with the number of neighborhoods.
4. Integration with causal inference methods remains unexplored.
5. Evaluation is currently limited to binary classifiers.

## Related Work & Insights

- The gradual semantics of QBAFs offer a general framework for transparent reasoning in AI systems.
- The combination of argument schemes and critical questions is transferable to other AI safety problems.
- Bias detection applied to LLMs demonstrates that large models inherit societal biases present in their training data.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: Introducing argumentation theory into bias detection is a novel perspective.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Three-tier evaluation covering synthetic data, trained classifiers, and LLMs.
- **Writing Quality** ⭐⭐⭐⭐⭐: Theoretical derivations are rigorous and well-presented.
- **Value** ⭐⭐⭐⭐: Provides a transparent, theoretically grounded bias detection paradigm for AI fairness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](../../ACL2026/social_computing/confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](../../ACL2026/social_computing/gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)

</div>

<!-- RELATED:END -->
