---
title: >-
  [Paper Note] Interpretability Can Be Actionable
description: >-
  [ICML 2026 (Position Paper)][Interpretability][actionability] This position paper argues that "interpretability research lacks evaluation criteria rather than new methods." It advocates for actionability—the ability of insights to drive specific decisions or interventions outside the interpretability field—as the core evaluative dimension. The authors define actionability via two dimensions (concreteness and validation), analyze systemic barriers…
tags:
  - "ICML 2026 (Position Paper)"
  - "Interpretability"
  - "actionability"
  - "interpretability evaluation"
  - "position paper"
  - "deployment standards"
  - "evaluation framework"
date: 2026-05-08
content_hash: 2ab8a3a93a19b132
---

# Interpretability Can Be Actionable

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.11161](https://arxiv.org/abs/2605.11161)  
**Code**: None (Position Paper)  
**Area**: Interpretability / Position Paper  
**Keywords**: actionability, interpretability evaluation, position paper, deployment standards, evaluation framework

## TL;DR
This position paper argues that "interpretability research lacks evaluation criteria rather than new methods." It advocates for actionability—the ability of insights to drive specific decisions or interventions outside the interpretability field—as the core evaluative dimension. The authors define actionability via two dimensions (concreteness and validation), analyze systemic barriers, identify five high-leverage application domains, and provide a 6-step checklist for researchers.

## Background & Motivation

**Background**: Interpretability has evolved into a massive subfield of ML, with rapid growth in publications and conference scale, spanning saliency maps, influence functions, and feature visualization to SAEs, circuit discovery, and mechanistic interpretability. The underlying assumption is that "understanding models" automatically leads to more reliable, controllable, and secure systems.

**Limitations of Prior Work**: Increasing criticism from scholars like Krishnan, Greenblatt, and Potts suggests that most interpretability work fails to alter training practices, deployment decisions, or policy. Mosbach et al. (2024) empirically found that while NLP interpretability papers are cited, the vast majority are "conceptual citations" that rarely drive changes in architecture, training, or evaluation. At the ICML 2025 actionable interpretability workshop, $22\%$ of submissions were explicitly flagged by reviewers as "insufficiently actionable."

**Key Challenge**: The interpretability community **rewards methodological novelty** but **does not require demonstrations of application**. This "low requirements + low rewards" combination discourages actionable work. Furthermore, unlike mainstream ML, interpretability lacks a "forcing function" like benchmark performance improvements, leaving the standard for success ambiguous.

**Goal**: (1) Provide a precise definition and dimensions for actionability; (2) Diagnose the root causes hindering actionability; (3) Identify high-leverage domains for actionability; (4) Provide evaluation metrics for different action types; (5) Offer a 6-step checklist for researcher self-assessment.

**Key Insight**: The authors do not oppose exploratory research but argue that "actionability should be integrated into evaluation dimensions," placing application demonstrations alongside methodological novelty. The latter further constrains the former—an explanation that can be implemented indicates it captures true model behavior rather than artifacts.

**Core Idea**: Formally establish actionability (whether an insight triggers specific decisions outside interpretability) as an evaluation criterion, supported by a 2D taxonomy, five domain levers, and three categories of evaluation metrics.

## Method

### Overall Architecture
As a position paper, this work does not propose a pipeline but constructs an argumentative framework of "Definition—Diagnosis—Guidance—Evaluation." It formalizes actionability into a 2-dimensional space (concreteness $\times$ validation) to locate any interpretability work, diagnoses the scarcity of actionable research, identifies high-reward domains and audiences, and defines quantitative metrics for various action types along with a 6-step checklist.

### Key Designs

**1. Actionability 2D Coordinate System: From Binary Label to Continuous Spectrum**
The community often labels work as "actionable" or "not actionable." This binary fails to recognize exploratory research that lays the foundation for future deployment and ignores "detailed but unvalidated" theoretical constructs. The authors decouple this via two orthogonal axes: *concreteness* (how precisely the action is described, ranging from vague "could inform safety" to precise specifications with implementation details) and *validation* (the level of empirical support, ranging from hypotheses to systematic quantitative evaluation). These axes create four quadrants: Low-Low (foundational exploration, e.g., Geva et al.'s view of MLPs as key-value pairs), High-Low (specific proposals without validation), and High-High (successful cases like ROME for fact editing, SAE-based unlearning, or Schut et al.'s concept transfer in AlphaZero).

**2. Five Leverage Domains + Three Action Types: Guidance for High Impact**
Researchers need to know where effort yields the highest returns. The authors identify five high-leverage domains: (a) scaling problems unsolvable by compute alone (hallucination, catastrophic forgetting, bias); (b) alignment (black-box testing cannot falsify deception); (c) surgical interventions (model editing, activation steering, where retraining is too costly); (d) architectural design (e.g., induction heads inspiring selective states in Mamba); (e) translating explanations into domain terms (e.g., clinically relevant vs. pixel-level explanations for doctors). Actions are categorized by their "impact layer": modifying outputs (data curation, training decisions, direct control), deployment usage (uncertainty estimation, routing like FrugalGPT), and shaping future practices (policy compliance, knowledge transfer, future architectures).

**3. Evaluation Metrics for Three Action Types: Escaping the "Method Comparison" Trap**
Interpretability suffers from a lack of "benchmark gains" as a forcing function, leading to "grading-on-a-curve" against other interpretability methods. The authors propose metrics for each action type. *Modifying outputs* requires comparative utility (benchmarking against non-interpretability baselines like prompting/fine-tuning), mechanistic faithfulness, generalization across seeds/architectures, and specificity. *Deployment usage* focuses on task-enhancement (speed/accuracy of human decisions), understandability (alignment with user conceptual frameworks), and reliability. *Shaping future practice* looks at the expansion of governance tools and legibility to non-experts.

## Loss & Training
N/A (Position Paper). The authors provide a 6-step researcher checklist in Section 9 as an operational workflow: Define Goal $\rightarrow$ Identify Audience $\rightarrow$ Propose Specific Action $\rightarrow$ Empirical Validation $\rightarrow$ Real-world Testing $\rightarrow$ Evaluate using actionable metrics.

## Key Experimental Results

### Main Results
While containing no original experiments, the paper cites numerous "actionable success cases" as evidence:

| Category | Representative Work | Actionable Outcome |
| :--- | :--- | :--- |
| Data Curation | Koh & Liang 2017 (Influence Functions) | Detecting mislabeled samples to improve accuracy |
| Data Curation | Agia et al. 2025 (CUPID) | SOTA robot learning using only $33\%$ of data |
| Model Editing | Meng et al. 2022 (ROME) | Fact editing based on MLP key-value view |
| Training Strategy | Casper et al. 2024a (Latent Adv. Training) | Backdoor removal and robustness improvement |
| Deployment Routing | Chen et al. 2024 (FrugalGPT) | $98\%$ cost reduction via uncertainty routing |
| Knowledge Transfer | Schut et al. 2025 | AlphaZero concept vectors teaching humans new moves |
| Safety Auditing | Anthropic 2025 (Claude Sonnet 4.5) | Internal activation analysis as evidence for auditing |

### Ablation Study

| Dimensional Comparison | Example | Evaluation |
| :--- | :--- | :--- |
| Low Concreteness + Low Validation | Geva et al. 2021 (MLP=key-value) | Exploratory; foundational for model editing |
| High Concreteness + Low Validation | Various Sci-AI verification works | Concrete proposals lacking deployment validation |
| High Concreteness + High Validation | ROME / UCE / REVS / AlphaSteer | Precise specifications with empirical utility |

### Key Findings
- **Reward/Requirement Asymmetry**: Publication standards do not mandate actionability, while application demonstrations are often dismissed as "engineering."
- **Missing Forcing Function**: Mainstream ML is forced toward utility by benchmarks; interpretability lacks this, leading to "insular method comparisons."
- **Deployment Barriers**: High technical complexity (requiring deep internal knowledge + libraries like `TransformerLens`/`NNsight`) and the "open-weight assumption" (frontier models are often closed-source).
- **AxBench Insight**: Wu et al. (2025) found that prompting/fine-tuning often outperform interpretability methods like SAEs in LLM steering, highlighting the need for external baselines.
- **Understandability $\neq$ Faithfulness**: A technically faithful explanation is useless if the user cannot interpret it; they must be evaluated separately.

## Highlights & Insights
- The 2D coordinate system for actionability is both rigorous and inclusive, acknowledging exploratory work while penalizing "castles in the air."
- The Audience $\times$ Action framework in Section 5 provides a powerful self-assessment tool for researchers.
- The discussion on "policy-actionable" research links technical work to the EU AI Act/GDPR, positioning interpretability as a governance tool.
- The 6-step checklist serves as a practical rubric for reviewers to exert cultural pressure on the community.

## Limitations & Future Work
- As a position paper, it lacks its own methodology; the operationality of the proposed metrics depends on future benchmarks.
- Defining success via applications might trigger "short-termism," potentially marginalizing foundational research with long-term payoffs.
- The audience hierarchy is difficult to strictly partition in practice (e.g., an SAE paper targeting both developers and regulators).
- Actionability for closed-source frontier models is largely restricted to the policy/auditing layer due to the open-weight dependency of most technical actions.

## Related Work & Insights
- **vs. Lipton (2018)**: While Lipton focused on taxonomic confusion, this work shifts toward evaluation criteria.
- **vs. Miller (2019) / Jacovi & Goldberg (2021)**: This paper extends their focus on social/user context into a broader generalized audience framework.
- **vs. Rudin (2019)**: While Rudin advocates for inherently interpretable models, this paper remains neutral but acknowledges they are a natural path to actionability.
- **vs. Nanda et al. (2025)**: This work provides a broader evaluative framework for the "pragmatic vision" of driving iteration via proxy tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (Conceptual framework effectively addresses community pain points).
- Experimental Thoroughness: ⭐⭐⭐ (Position paper; relies on strong external case studies).
- Writing Quality: ⭐⭐⭐⭐⭐ (Extremely clear structure; checklists and diagrams are highly effective).
- Value: ⭐⭐⭐⭐⭐ (Potential to significantly shift evaluation standards in major ML conferences).

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
## Related Papers

- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](../../ACL2026/interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../../ICLR2026/interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] Beyond Additive Decompositions: Interpretability Through Separability](beyond_additive_decompositions_interpretability_through_separability.md)

</div>

<!-- RELATED:END -->
