---
title: >-
  [Paper Note] Interpretability Can Be Actionable
description: >-
  [ICML 2026 (Position Paper)][Interpretability][actionability] This position paper argues that "what interpretability research lacks is not new methods…
tags:
  - "ICML 2026 (Position Paper)"
  - "Interpretability"
  - "actionability"
  - "interpretability evaluation"
  - "position paper"
  - "deployment standards"
  - "evaluation framework"
date: 2026-05-08
content_hash: 08a585c26645f144
---

# Interpretability Can Be Actionable

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.11161](https://arxiv.org/abs/2605.11161)  
**Code**: None (Position Paper)  
**Area**: Interpretability / Position Paper  
**Keywords**: actionability, interpretability evaluation, position paper, deployment standards, evaluation framework

## TL;DR
This position paper argues that "what interpretability research lacks is not new methods, but evaluation criteria": research should take actionability (the ability of insights to drive specific decisions/interventions outside the interpretability field) as a core evaluation dimension. The authors define actionability across two dimensions—concreteness and validation—analyze current obstacles, identify 5 high-leverage application domains, and provide a 6-step checklist for researchers.

## Background & Motivation

**Background**: Interpretability has matured into a massive subfield of ML, with a rapid growth in paper volume and conference scale, spanning saliency maps, influence functions, feature visualization, SAEs, circuit discovery, and mechanistic interpretability. The underlying assumption is that "understanding models" automatically leads to more reliable, controllable, and safe systems.

**Limitations of Prior Work**: Criticisms are mounting—Krishnan, Greenblatt, Potts, and others have noted that most interpretability work does not influence training practices, deployment decisions, or policy. Mosbach et al. (2024) empirically found that while NLP interpretability papers are cited, the overwhelming majority are "conceptual citations" that rarely drive changes in training, architecture, or evaluation. At the ICML 2025 Actionable Interpretability Workshop, 22% of submissions were explicitly flagged by reviewers as "not actionable enough."

**Key Challenge**: The interpretability community **rewards methodological novelty** but **does not require demonstrating applications**. This combination of "low requirement + low reward" discourages actionable work. Furthermore, unlike mainstream ML, interpretability lacks a "forcing function" such as benchmark improvements, leaving the standard for success ambiguous.

**Goal**: (1) Provide a precise definition and dimensional breakdown of actionability; (2) diagnose root causes hindering actionability; (3) identify high-leverage domains for actionability; (4) provide evaluation metrics for different action types; (5) offer a 6-step checklist for researchers' self-assessment.

**Key Insight**: The authors do not oppose exploratory research. Their argument is that "actionability should be integrated into evaluation dimensions," placing application demonstration alongside methodological novelty. The former serves to constrain the latter—an explanation that is actionable indicates it captures true model behavior rather than analysis artifacts.

**Core Idea**: Formally establish actionability (whether an insight triggers specific decisions outside interpretability) as an evaluation criterion, supported by a two-dimensional classification, five domain levers, and three categories of evaluation metrics.

## Method

### Overall Architecture
As a position paper, there is no traditional technical pipeline. The structure is as follows: Section 2 defines actionability via two dimensions (concreteness × validation), mapping all interpretability work into this space; Section 3 diagnoses three major obstacles (incentive / methodology / deployment); Section 4 identifies five leverage domains; Section 5 categorizes actions by audience (Developers/Engineers/Domain Experts/End Users/Policy Makers) and impact level (Modifying Outputs/Deployment Use/Shaping Future Practice); Section 6 provides evaluation criteria for each action category; Section 9 concludes with a 6-step checklist.

### Key Designs

1.  **Actionability Two-Dimensional Coordinate System**:
    - **Function**: Maps any interpretability work onto a (concreteness, validation) plane to avoid binary "actionable / not actionable" misjudgments.
    - **Mechanism**: Concreteness measures if an action is precisely stated (ranging from "could inform safety" to precise specifications with implementation details). Validation measures if the action has empirical support (ranging from pure hypothesis to systematic quantitative evaluation). The Four Quadrants: Low-Low (foundational exploration, e.g., Geva et al.'s MLP key-value view), High-Low (concrete proposals without validation, e.g., certain Sci-AI trust work), High-High (typical success cases: ROME editing, SAE-based unlearning, Schut et al.'s AlphaZero→human player concept transfer).
    - **Design Motivation**: Spreads actionability across a continuous spectrum rather than a binary judgment, preserving space for exploratory research while encouraging high-quadrant objectives.

2.  **Five Domain Levers + Three Action Frameworks**:
    - **Function**: Guides researchers on where "actionable work" provides the highest return and whom it should impact.
    - **Mechanism**: Five levers include (a) problems scaling cannot solve (hallucinations, catastrophic forgetting, bias, adversarial fragility, requiring "why"-level explanations); (b) alignment (black-box testing cannot falsify deception); (c) surgical interventions (model editing / activation steering / concept bottleneck, where retraining is too costly); (d) architectural design (induction heads inspiring Mamba’s selective state); (e) translating explanations into domain terms (doctors requiring clinically relevant rather than pixel-level explanations). Three action categories by impact layer: Modifying outputs (data curation, training decisions, direct control, safe unlearning); Deployment use (end-user decisions like uncertainty estimation, deployment routing like FrugalGPT’s 98% cost savings); Shaping future practice (policy compliance, knowledge transfer from superhuman models, future architecture design).
    - **Design Motivation**: Recognizes that actionability is multi-layered—datapoint-level influence functions serve developers, while system-level fairness summaries serve policy makers. No one-size-fits-all exists.

3.  **Evaluation Metrics for Three Action Types**:
    - **Function**: Provides quantifiable evaluation dimensions to escape the "grading-on-curve" trap of internal comparisons.
    - **Mechanism**: (a) Output-modifying actions use 4 metrics: comparative utility (benchmarking against non-interpretability methods like prompting/fine-tuning/LoRA), mechanistic faithfulness (intervening in identified components produces predicted changes), generalization (stability across seeds/perturbations/architectures/scales), and specificity (intervention only affects target behavior). (b) Deployment-use actions look at task-enhancement (speed/accuracy of human decisions), understandability (alignment with user conceptual frameworks, e.g., FIX/T-FIX benchmarks), and reliability. (c) Shaping future practice evaluates the expansion of feasible governance tools, legibility to non-experts, and reduction of regulatory costs.
    - **Design Motivation**: Forces comparison against external baselines to reveal real-world value, addressing the lack of forcing functions in interpretability.

### Loss & Training
Not applicable (Position Paper). However, Section 9 provides a 6-step researcher checklist: Define goal → Lock audience → Propose specific action → Empirical validation → Test in real scenarios → Evaluate with actionable metrics.

## Key Experimental Results

### Main Results
The paper cites numerous "actionable success cases" as evidence. The following table summarizes representative actionable work prioritized in the paper:

| Category | Representative Work | Actionable Result |
| -------- | ------------------- | ----------------- |
| Data Curation | Koh & Liang 2017 (Influence Functions) | Detection of mislabeled samples, improving accuracy |
| Data Curation | Agia et al. 2025 (CUPID) | Robot learning reaching SOTA with only 33% of data |
| Model Editing | Meng et al. 2022 (ROME) | Factual editing based on MLP key-value perspective |
| Training Strategy | Casper et al. 2024a (Latent Adversarial Training) | Backdoor removal and robustness enhancement |
| Deployment Routing | Chen et al. 2024 (FrugalGPT) | Uncertainty routing matching GPT-4 with 98% lower cost |
| Knowledge Transfer | Schut et al. 2025 | AlphaZero concept vectors teaching human players new moves |
| Safety Audit | Anthropic 2025 (Claude Sonnet 4.5) | Internal activation analysis as a basis for safety audits |

### Ablation Study

| Dimension Comparison | Examples | Evaluation |
| -------------------- | -------- | ---------- |
| Low concreteness + Low validation | Geva et al. 2021 (MLP=key-value) | Exploratory; foundational for later model editing |
| High concreteness + Low validation | Certain Sci-AI verification work | Specifically proposed but implementation unvalidated |
| High concreteness + High validation | ROME / UCE / REVS / AlphaSteer | Precise specifications + empirical proof of utility |

### Key Findings
- **Asymmetry between Reward and Requirement**: Publication standards do not mandate actionability, while application demonstration summarized as "engineering" discourages investment from rational researchers.
- **Missing Forcing Function**: Mainstream ML utilizes benchmark improvements to force utility; interpretability lacks this metric, leading to "pseudo-prosperity" based on internal comparisons.
- **Two Deployment Obstacles**: Technical complexity (requiring deep internal model knowledge and specialized libraries like TransformerLens/NNsight) and the Open-Weight assumption (frontier models are increasingly closed-source).
- **The AxBench Wake-up Call**: Wu et al. (2025) empirically found that prompting and fine-tuning often outperform interpretability methods like SAEs in LLM steering, highlighting the need for comparison with non-interpretability baselines.
- **Understandability ≠ Faithfulness**: An explanation may be 100% technically faithful to model behavior but useless if the user cannot understand it; these must be evaluated separately.

## Highlights & Insights
- The breakdown of actionability into concreteness and validation axes is both rigorous and inclusive—criticizing intellectual "castles in the air" while maintaining an ecological niche for exploratory work.
- The audience × action table in Section 5 provides a self-check tool that every interpretability paper should adopt.
- The "policy-actionable" section uniquely incorporates the EU AI Act and GDPR Article 22, reminding researchers that explanations are governance tools, not merely engineering aids.
- The 6-step checklist provides a concrete rubric for reviewers which, combined with the 22% "not actionable" flag from ICML 2025, could exert immediate pressure on community culture.

## Limitations & Future Work
- As a position paper, it does not provide a new methodology; the practical utility of the proposed metrics depends on future benchmark developments.
- Defining success by application/practical metrics might exacerbate short-termism, potentially marginalizing foundational breakthroughs that lack immediate payoffs—the authors acknowledge this but offer insufficient mitigation.
- The audience layering is difficult to segment precisely in practice; a single SAE work might target both developers and regulators, creating open questions on how to satisfy both in a single manuscript.
- Technical actionability for closed-source frontier models is largely restricted to policy/auditing layers; technical actionability remains constrained by the open-weight assumption.
- Treating interpretability as a "service tool" may conflict with the traditional "interpretability as basic science" perspective; Section 7 debates this but does not fully reconcile the two.

## Related Work & Insights
- **vs Lipton 2018 (Mythos of Model Interpretability)**: Lipton focused on terminological confusion; this paper shifts focus to evaluation criteria rather than definitions.
- **vs Miller 2019 / Jacovi & Goldberg 2021**: Those works emphasized social and user context; this paper absorbs that into a comprehensive audience framework.
- **vs Rudin 2019**: Rudin argued for inherently interpretable models in high-stakes scenarios; this paper remains neutral but recognizes inherent interpretability as a natural path for actionability.
- **vs Nanda et al. 2025 (pragmatic vision)**: That work advocated for proxy tasks to drive iteration; this paper serves as a broader sister piece providing a full evaluation framework for the pragmatic direction.
- **vs Bau 2025 (curiosity-driven defense)**: Bau defended exploratory research; this paper does not negate it but requires actionability as an additional yardstick.

## Rating
- Novelty: ⭐⭐⭐⭐ Captures community pain points through a precise framework rather than methodological novelty.
- Experimental Thoroughness: ⭐⭐⭐ Position paper lacks traditional experiments but provides extensive success cases; future benchmarks are recommended.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure; the Figure 1 checklist is highly effective.
- Value: ⭐⭐⭐⭐⭐ Directly addresses ICML/NeurIPS/ICLR reviewer culture; likely to impact community evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](../../ACL2026/interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../../ICLR2026/interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] Beyond Additive Decompositions: Interpretability Through Separability](beyond_additive_decompositions_interpretability_through_separability.md)

</div>

<!-- RELATED:END -->
