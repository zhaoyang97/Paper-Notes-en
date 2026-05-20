---
title: >-
  [Paper Note] Interpretability Can Be Actionable
description: >-
  [ICML 2026 (Position Paper)][Interpretability][actionability] This is a position paper arguing that "what interpretability research lacks is not new methods…
tags:
  - "ICML 2026 (Position Paper)"
  - "Interpretability"
  - "actionability"
  - "interpretability evaluation"
  - "position paper"
  - "deployment standards"
  - "evaluation framework"
date: 2026-05-08
content_hash: 048d20ec999df44a
---

# Interpretability Can Be Actionable

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.11161](https://arxiv.org/abs/2605.11161)  
**Code**: None (position paper)  
**Area**: Interpretability / Position Paper  
**Keywords**: actionability, interpretability evaluation, position paper, deployment standards, evaluation framework

## TL;DR
This is a position paper arguing that "what interpretability research lacks is not new methods, but evaluation criteria": research should use actionability (whether insights can drive concrete decisions/interventions outside the interpretability domain) as a core evaluation dimension. The authors define actionability along the axes of concreteness and validation, analyze obstacles, list five high-leverage application domains, and provide a six-step checklist for researchers.

## Background & Motivation

**Background**: Interpretability has become a major subfield of ML, with rapidly growing numbers of papers and conferences, covering everything from saliency maps, influence functions, feature visualization to SAE, circuit discovery, and mechanistic interpretability. The underlying assumption is that "understanding the model" will automatically lead to more reliable, controllable, and safer systems.

**Limitations of Prior Work**: Criticism is mounting—Krishnan, Greenblatt, Potts, and others have pointed out that most interpretability work has not changed training practices, deployment decisions, or policy. Mosbach et al. (2024) empirically found that although NLP interpretability papers are cited, the vast majority of citations are "conceptual," rarely driving changes in training, architecture, or evaluation. At the ICML 2025 actionable interpretability workshop, 22% of submissions were explicitly marked by reviewers as "not actionable enough."

**Key Challenge**: The interpretability community **rewards methodological novelty** but **does not require demonstration of application**—this "low requirement + low reward" combination leads to a lack of actionable work. Unlike mainstream ML, interpretability lacks a "benchmark improvement" forcing function, so the definition of success remains vague.

**Goal**: (1) Provide a precise, multi-dimensional definition of actionability; (2) Diagnose root causes hindering actionability; (3) List domains where actionability has high leverage; (4) Propose evaluation metrics for different action types; (5) Offer a six-step checklist for researchers to self-assess.

**Key Insight**: The authors do not oppose exploratory research; their argument is that "actionability should be included as an evaluation dimension," placing methodological novelty and demonstration of application on equal footing. The latter constrains the former—an explanation that can be deployed indicates it captures real model behavior, not just artifacts.

**Core Idea**: Formally establish actionability (whether insights trigger concrete decisions outside the interpretability field) as an evaluation criterion, with a supporting two-dimensional classification, five leverage domains, and three types of evaluation metrics.

## Method

### Overall Architecture
As a position paper, this work does not present a traditional pipeline. The structure is: Section 2 defines actionability and its two axes (concreteness × validation), mapping all interpretability work into this 2D space; Section 3 diagnoses three main obstacles (incentive / methodology / deployment); Section 4 lists five domains where actionability has leverage; Section 5 presents an action framework by audience (developer/deployment engineer/domain expert/end user/policymaker) and action impact layer (output modification/deployment use/shaping future practice); Section 6 provides evaluation criteria for each of the three action types; Section 9 concludes with a six-step checklist.

### Key Designs

1. **Actionability 2D Coordinate System**:

    - **Function**: Places any interpretability work in the (concreteness, validation) 2D plane, avoiding the binary "actionable / not actionable" misjudgment.
    - **Mechanism**: Concreteness measures whether the action is precisely articulated (from "could inform safety" to detailed implementation specifications); validation measures whether the action is empirically supported (from pure hypothesis to systematic quantitative evaluation). Four quadrants: low-low (foundational exploratory work, e.g., Geva et al.'s MLP key-value perspective), high-low (concrete proposals but unvalidated, e.g., some sci-AI trust work), high-high (typical success cases: ROME editing, SAE-based unlearning, Schut et al.'s AlphaZero→human chess concept transfer).
    - **Design Motivation**: Lays out actionability as a continuous spectrum rather than a binary judgment, leaving room for exploratory research while encouraging the high-high quadrant as a goal.

2. **Five Leverage Domains + Three Types of Action Framework**:

    - **Function**: Informs researchers "where actionable work yields the highest return," "who it targets, and at what impact layer."
    - **Mechanism**: The five leverage domains include (a) problems unsolvable by scaling (hallucination, catastrophic forgetting, bias, adversarial vulnerability, requiring why-level explanations); (b) alignment (black-box testing cannot falsify deception); (c) surgical intervention (model editing / activation steering / concept bottleneck, retraining is too costly); (d) architecture design (induction head inspired Mamba's selective state); (e) translating explanations into domain terminology (clinicians need clinically relevant, not pixel-level, explanations). The three action types by impact layer: output modification (data curation, training decisions, direct control, safe unlearning); deployment use (end-user decisions such as uncertainty estimation, deployment routing such as FrugalGPT's uncertainty routing saving 98% cost); shaping future practice (policy compliance, superhuman model knowledge transfer, future architecture design). Each type corresponds to a different audience (developer/deployment engineer/domain expert/end user/policymaker), each requiring different forms of explanation.
    - **Design Motivation**: Actionability is not hierarchical—"data point-level influence functions" are useful for developers, "system-level fairness summaries" for policymakers; there is no one-size-fits-all.

3. **Evaluation Metrics for Three Types of Action**:

    - **Function**: Provides researchers with quantifiable actionability evaluation dimensions, avoiding the "grading-on-curve" trap of only comparing with other interpretability methods.
    - **Mechanism**: (a) Output modification actions should be evaluated on four metrics—comparative utility (compare with non-interpretability baselines like prompting/fine-tuning/LoRA to see if there is real marginal leverage), mechanistic faithfulness (whether intervening on identified components produces the predicted change), generalization (whether effects hold across seeds/input perturbations/architectures/scales), specificity (whether intervention only affects the target behavior without harming unrelated capabilities); (b) Deployment use actions should be evaluated on task-enhancement (whether human decisions become faster/more accurate), understandability (whether explanations fit users' existing conceptual frameworks, e.g., FIX/T-FIX benchmarks aligning with astrophysics or clinical SOFA scores), reliability (stability to small perturbations within the same task); (c) Shaping future practice actions should be evaluated on "whether feasible governance tools are expanded," whether explanations are legible to non-experts, and whether regulatory costs are reduced.
    - **Design Motivation**: Interpretability has long compared only within its own methods, lacking the mainstream ML "benchmark improvement" forcing function; forcing comparison with external baselines reveals true value.

### Loss & Training
Not applicable (position paper). However, Section 9 provides a six-step researcher checklist: clarify goal → identify audience → propose concrete action → empirical validation → test in real scenarios → evaluate using the above actionable metrics.

## Key Experimental Results

### Main Results
This paper presents no experiments but cites numerous "actionable success cases" as evidence. The table below summarizes representative actionable works highlighted in the paper:

| Category | Representative Work | Actionable Outcome |
|--------|------|------|
| Data curation | Koh & Liang 2017 (Influence Functions) | Detects mislabeled samples, improves accuracy |
| Data curation | Agia et al. 2025 (CUPID) | Robot learning achieves SOTA with only 33% data |
| Model editing | Meng et al. 2022 (ROME) | Fact editing based on MLP key-value perspective |
| Training strategy | Casper et al. 2024a (latent adversarial training) | Removes backdoor, improves robustness |
| Deployment routing | Chen et al. 2024 (FrugalGPT) | Uncertainty routing matches GPT-4 performance, reduces cost by 98% |
| Knowledge transfer | Schut et al. 2025 | AlphaZero concept vectors teach human chess players new moves |
| Safety audit | Anthropic 2025 (Claude Sonnet 4.5) | Internal activation analysis used as safety audit evidence |

### Ablation Study

| Dimension Comparison | Example | Evaluation |
|------|------|------|
| Low concreteness + low validation | Geva et al. 2021 (MLP=key-value) | Exploratory, foundational for later model editing |
| High concreteness + low validation | Some sci-AI verification work | Concrete proposals but unvalidated in practice |
| High concreteness + high validation | ROME / UCE / REVS / AlphaSteer | Precise specification + empirical demonstration of usability |

### Key Findings
- **Severe asymmetry between reward and requirement**: Publication standards do not enforce actionability, and demonstration of application is often dismissed as "engineering," so rational researchers do not invest in it.
- **Lack of forcing function**: Mainstream ML uses benchmark improvements to drive practicality, but interpretability lacks this yardstick, leading to a "false prosperity" of intra-method comparison.
- **Two major deployment obstacles**: Technical complexity (requires deep understanding of model internals + specialized libraries like TransformerLens/NNsight), open-weight assumption (frontier models are mostly closed-source, missing the most urgent targets for actionability).
- **AxBench wake-up call**: Wu et al. 2025 empirically found with AxBench that prompting and fine-tuning often outperform interpretability methods like SAE for LLM steering—highlighting the urgent need for comparison with non-interpretability baselines.
- **Understandability ≠ faithfulness**: An explanation may be 100% technically faithful to model behavior, but if users cannot understand it, it is useless; these must be evaluated separately.

## Highlights & Insights
- Decomposing actionability into concreteness and validation is both rigorous and inclusive—it can criticize empty theorizing while preserving a niche for exploratory work.
- The audience × action 2D table in Section 5 (developer/engineer/domain expert/end user/policymaker) is worth copying as a self-check for every interpretability paper.
- The "policy-actionable" section uniquely brings in the EU AI Act / GDPR Article 22, reminding researchers that explanations are also governance tools, not just for engineers.
- The six-step checklist can directly serve as a review rubric; combined with ICML 2025's 22% "not actionable enough" mark, it can exert immediate cultural pressure on the community.

## Limitations & Future Work
- As a position paper, no methods are proposed; the operationalizability of all actionable evaluation metrics depends on follow-up benchmarks.
- "Defining success by application/practical metrics" may exacerbate short-termism, marginalizing research that lacks immediate payoff but could yield foundational breakthroughs in the long term—the authors acknowledge this but do not fully mitigate it.
- The audience-layered framework is not easy to delineate in practice: an SAE work may target both developers and regulators, and how to satisfy both in writing remains an open question.
- For closed-source frontier models, actionability is almost limited to policy/audit layers; technical actionability at the frontier is still constrained by the open-weight assumption.
- Treating interpretability as a "service tool" may conflict with the traditional view of "interpretability as basic science"; although Section 7 debates this, it does not fully reconcile the perspectives.

## Related Work & Insights
- **vs Lipton 2018 (Mythos of Model Interpretability)**: Lipton emphasized terminological confusion and distinguished transparency from post-hoc explanation; this paper skips definitional debates and directly provides evaluation criteria.
- **vs Miller 2019 / Jacovi & Goldberg 2021**: Those works emphasized that explanations should fit social and user contexts; this paper adopts and extends this to a broader audience framework.
- **vs Rudin 2019**: Rudin argued that high-risk scenarios should use inherently interpretable models rather than post-hoc explanations; this paper takes no stance but acknowledges that inherent interpretability is a natural path to actionability.
- **vs Nanda et al. 2025 (pragmatic vision)**: That work advocated "using proxy tasks for rapid iteration"; this paper is a broader companion, providing a complete evaluation framework for the pragmatic direction.
- **vs Bau 2025 (curiosity-driven defense)**: Bau defended exploratory research; this paper does not deny it, but requires actionability as an additional yardstick.

## Rating
- Novelty: ⭐⭐⭐⭐ Not novel in method, but precisely addresses community pain points with a structured framework; the combination of two axes + five domains + three evaluation types is highly organized
- Experimental Thoroughness: ⭐⭐⭐ No traditional experiments, but cites many actionable success cases as evidence; recommends future benchmarks
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure, Figure 1 checklist is highly illustrative; broad coverage of cases
- Value: ⭐⭐⭐⭐⭐ Directly targets ICML / NeurIPS / ICLR review culture, with the potential to truly change interpretability community evaluation standards

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](../../ACL2026/interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../../ICLR2026/interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)
- [\[ICML 2026\] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)
- [\[ACL 2026\] Interpretability from the Ground Up](../../ACL2026/interpretability/interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)

</div>

<!-- RELATED:END -->
