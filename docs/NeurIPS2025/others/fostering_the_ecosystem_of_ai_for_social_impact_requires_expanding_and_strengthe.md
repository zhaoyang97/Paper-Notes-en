---
title: >-
  [Paper Note] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards
description: >-
  [NeurIPS 2025][AI for Social Impact] This paper argues that the academic ecosystem of AI for Social Impact (AISI) requires a dual-track reform: broadening the definition of "impact" to recognize contributions beyond deployment or methodological novelty, while simultaneously demanding causal-inference-level rigor in evaluating deployed systems.
tags:
  - "NeurIPS 2025"
  - "AI for Social Impact"
  - "evaluation standards"
  - "deployment"
  - "field experiments"
  - "research ecosystem"
  - "causal inference"
date: 2026-05-08
content_hash: ce318f60e2bf4dcd
---

# Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards

**Conference**: NeurIPS 2025
**arXiv**: [2510.18238](https://arxiv.org/abs/2510.18238)  
**Authors**: Bryan Wilder (Carnegie Mellon University), Angela Zhou (University of Southern California)
**Code**: None (Position Paper)  
**Area**: Other
**Keywords**: AI for Social Impact, evaluation standards, deployment, field experiments, research ecosystem, causal inference

## TL;DR

This paper argues that the academic ecosystem of AI for Social Impact (AISI) requires a dual-track reform: broadening the definition of "impact" to recognize contributions beyond deployment or methodological novelty, while simultaneously demanding causal-inference-level rigor in evaluating deployed systems.

## Background & Motivation

### State of the Field

The field of AI/ML for Social Impact (AISI) has grown rapidly over the past decade, giving rise to dedicated venues such as the AAAI Special Track on AI for Social Impact and the IJCAI Multi-Year Track on AI and Social Good, as well as specialized courses and summer programs at multiple universities. Nevertheless, the current academic ecosystem exhibits two structural deficiencies:

**Deficiency 1: A monolithic template for the "ideal project"**

Existing review criteria treat "methodological novelty + real-world deployment" as the gold standard for AISI papers. The AAAI AISI Track explicitly prioritizes deployed or near-deployment projects ("Scope and promise for social impact") and methodological novelty ("Novelty of approach"). IJCAI evaluation criteria include "contribution to state-of-the-art AI" and "collaboration with stakeholders/partners," with "potential deployment opportunities" as a key expectation.

This monolithic template produces three problems:
1. Researchers are compelled to fit every project into a "new method → deployment" framework, even when partners' actual needs differ.
2. purely applied contributions—helping organizations use existing tools correctly—are rarely recognized academically.
3. Methodological research with practical potential but no deployment is systematically undervalued.

**Deficiency 2: Insufficient rigor in deployment evaluation**

Even within the deployment framework, current evaluation practices fall far short of the standards established in economics and medicine. Deployment is treated as a "finish line," yet simple before-after comparisons can be confounded by temporal trends, population shifts, and other factors. The ML community largely lacks training in causal inference methodology.

### Core Position

Researchers and reviewers must simultaneously: (1) adopt a broader conception of social impact that is not limited to deployment; and (2) apply more rigorous standards when evaluating deployed systems.

## Method

### Reform Track 1: Recognizing Non-method Contributions

When partners—nonprofits, government agencies—lack ML expertise, researchers can generate substantial impact by helping them use existing tools correctly. Such work can answer questions of broad scientific relevance to the ML community:

- **How does ML integrate into organizational decision-making?** How do modeling choices alter the way a tool is used?
- **What is the actual value of ML in a given domain?** Does improved prediction genuinely improve outcomes, and if so, how?
- **Which formulations are inappropriate?** Negative results carry scientific value—other researchers can learn from prior formulation explorations.
- **At what level of complexity is "good enough" achieved?** Is a simpler formulation already sufficient?

The authors also draw a boundary: if an ML model contributes scientifically to the application domain itself (e.g., a clinical prediction model), such papers are better suited for domain-specific journals. ML venues should welcome papers that derive scientific conclusions about ML use and impact through applied projects.

### Reform Track 2: Recognizing Non-deployment Contributions

Methodological research can generate social impact without deployment through the following pathways:

**Changing practitioner cognition**: The most impactful methodological work shifts how applied researchers and data scientists approach problems. This requires understanding the diverse disciplinary backgrounds of partners' data analysts—social science, health science, political science—and translating core methodological ideas into those fields.

**Design for Maintenance**: Mattern (2018) observes that maintenance is systematically undervalued. Partners typically lack an ML PhD team to maintain complex pipelines. DellaVigna et al. (2024) find that projects leveraging existing infrastructure are more likely to scale from pilot to formal deployment—simpler tools are more readily adopted. The medical nomogram (a bedside calculator for logistic regression coefficients) is a canonical example.

**Single-variable Benchmarks**: Perdomo et al. (2023), Stoddard et al. (2024), and Salganik et al. (2020) demonstrate that single-variable predictors often match complex ML methods on many social prediction tasks. Reporting such benchmarks helps: (a) quantify the marginal gain of complex ML; (b) help organizations with varying resource levels select appropriate solutions; (c) support transferability of findings—different organizations may have different data columns and schemas, but single-variable benchmarks can be directly reproduced for comparison.

### Reform Track 3: Raising the Standards for Deployment Evaluation

The authors propose a differentiated evaluation framework according to deployment type:

**Pilot Test (proof-of-concept)**: The goal is to assess feasibility and acceptability, not effectiveness. Authors should explicitly label work as a pilot, restrict conclusions to feasibility findings, and refrain from claiming method efficacy.

**Randomized Controlled Trial (RCT)**: The following best practices should be adopted:
- **Preregistration**: Publicly register the trial protocol and analysis strategy before the trial begins to prevent p-hacking. ML venues should require authors to disclose whether preregistration was performed.
- **Power Analysis**: Determine the minimum detectable effect size given the sample size, avoiding wasteful use of partner resources.
- **Outcome Validity**: Justify that outcome metrics are meaningfully linked to real-world welfare, rather than relying solely on indicators automatically recorded by platforms.
- **Pre-specified Strategies for Heterogeneity Analysis**: Preregister subgroup analyses or algorithmic procedures and control for false positives due to multiple comparisons.
- **Choice of Randomization Unit**: Individual-level vs. cluster-level (service center/hospital/school), depending on the level at which the algorithm operates.
- **Resource Allocation Dilemma**: Algorithms often serve as the assignment mechanism for another intervention; the ideal RCT should randomize at the cluster level, but implementing partners may have only a single "cluster."

**Non-randomized Deployment (Event Study)**: Should be understood as an event study:
- Core idea: construct a counterfactual—what would have happened absent the deployed system?
- **Interrupted Time Series**: Use pre-deployment temporal trends to extrapolate the counterfactual. Figure 1 in the paper illustrates that if outcomes were already improving before deployment, a simple before-after comparison will overestimate—or even mask—a negative causal effect.
- More advanced designs: Differences-in-Differences (requires a control group and parallel trends assumption); Synthetic Control (requires multiple comparable control units).
- Minimum requirement: report pre-deployment outcome trends and changes in population composition before and after deployment.

## Key Experimental Results

This paper is a position paper and contains no traditional experiments. The core argumentative structure is summarized in the tables below.

**Table 1: Three contribution dimensions and their current recognition in peer review**

| Contribution Type | Specific Form | Current Recognition | Authors' Recommendation |
|---|---|---|---|
| Non-method contribution | Helping partners use existing ML tools correctly | Low: ML venues do not accept papers without new methods | Recognize scientific conclusions about ML use and impact derived from applied projects |
| Non-method contribution | Studying how ML enters organizational decision workflows | Low: mainly appears at HCI/interdisciplinary venues | ML venues should welcome usage studies that engage with methodological details |
| Non-deployment contribution | Changing practitioners' views on estimation strategies | Medium: some recognition but lacks fine-grained criteria | Add "maintainability" and "transferability" as review dimensions |
| Non-deployment contribution | Simple tools / single-variable benchmarks | Low: simple methods struggle to demonstrate "methodological novelty" | Normalize simple baselines; they should not be viewed as diminishing innovation |
| Deployment contribution | Deployed system + effect evaluation | High: the most recognized contribution type | Require higher evaluation rigor (RCT / event study) |

**Table 2: Comparison of three deployment evaluation methods**

| Evaluation Method | Applicable Setting | Core Requirements | Key Threats / Limitations |
|---|---|---|---|
| Pilot Test | Initial small-scale testing | Clearly labeled as pilot; conclusions restricted to feasibility | Not suitable for claiming effectiveness |
| RCT | Randomization is feasible | Preregistration, power analysis, outcome validity, heterogeneity strategy | Choice of randomization unit, resource allocation dilemma |
| Non-randomized (Event Study) | Randomization is infeasible | Construct counterfactual, report pre-trends, control for population changes | Relies on time-series extrapolation assumptions when no control group is available |

## Highlights & Insights

1. **Ecosystem perspective**: Rather than evaluating any single method or paper, this work systematically analyzes the incentive structures and coordination failures of the AISI field—a form of "meta-research" that is exceedingly rare in the ML community.

2. **"Design for Maintenance" philosophy**: Simpler tools are more readily adopted and maintained, yet perverse academic incentives make simple methods difficult to publish. The nomogram case (bedside manual computation of logistic regression coefficients) illustrates that the technologies actually adopted in practice are far simpler than those discussed in the literature.

3. **Introducing a causal inference perspective into ML evaluation**: The paper explicitly argues that ML deployment evaluation is fundamentally a causal inference task requiring counterfactual reasoning, rather than conventional accuracy or loss metrics. The interrupted time series diagram in Figure 1 is visually compelling.

4. **Portfolio thinking**: Encouraging researchers to build portfolios across contribution types—sometimes methodological, sometimes applied, sometimes deployment-focused—reduces incentive distortions at the individual project level and improves the long-term sustainability of the ecosystem.

5. **Deeper implications of single-variable benchmarks**: Many social prediction problems exhibit high Bayes error (low signal-to-noise ratio), leaving limited marginal gains for complex ML. This parallels the popularity-baseline phenomenon in recommender systems.

6. **Ethical responsibility toward partners**: Requiring methodological novelty in every project pushes researchers to steer collaborations toward directions that are "paper-friendly" rather than "partner-necessary"—an injustice to partner organizations operating with limited resources.

## Limitations & Future Work

1. **Lack of quantitative evidence**: The argument relies primarily on logical reasoning and case analysis, without systematic meta-analysis or statistical analysis of review decisions.

2. **High implementation barrier**: The proposed evaluation standards—RCT preregistration, power analysis, event study—impose a high barrier for most ML researchers; cross-disciplinary causal inference training is not standard.

3. **Feasibility of review reform is uncertain**: ML venue reviewers are predominantly methods-oriented researchers; ensuring they can assess the quality of event studies or RCTs is a practical challenge.

4. **North American academic context bias**: The incentive structures discussed—tenure, promotion, collaboration patterns—are more specific to the North American academic environment.

5. **Internal consistency of cited cases**: Whether the AISI projects cited as examples (e.g., the Greek COVID-19 RL system) themselves meet the rigorous evaluation standards proposed in the paper is not examined.

6. **Underspecified relationship with ML+X venues**: Specialized venues such as ML+Health and ML+Science are growing rapidly; how AISI should complement rather than compete with them is insufficiently discussed.

## Related Work & Insights

### Related Work

- **Foundational AISI literature**: Tomašev et al. (2020), Tambe et al. (2022), and Rolnick et al. (2024) emphasize that AISI requires deep collaboration with partners and confronting last-mile challenges.
- **Representative AISI projects**: Bastani et al. (2021) on the Greek COVID-19 RL traveler-testing system; Shi et al. (2021) on food rescue volunteer scheduling.
- **Causal inference methodology**: Cunningham (2021) and Angrist & Pischke (2009) provide standard references for RCTs and event studies; Freyaldenhoven et al. (2021) discuss panel event-study design.
- **ML reproducibility**: Beam et al. (2020) and Pineau et al. (2021) address reproducibility in ML evaluation; this paper extends that concern to deployment evaluation.
- **Validity research**: Coston et al. (2023) and Jacobs & Wallach (2021) import social science validity concepts into the evaluation of ML decision systems.
- **Prediction difficulty baselines**: Perdomo et al. (2023) and Salganik et al. (2020) demonstrate the competitiveness of simple baselines in social prediction.

### Insights

- **Direct recommendations for ML venues**: (1) Require field experiments to disclose whether preregistration was performed; (2) showcase examples of AISI-specific contribution types; (3) add fine-grained dimensions such as maintainability and single-variable baselines to review guidelines.
- **Implications for engineering practice**: Plan evaluation schemes (RCT or event study) during the ML system deployment design phase, rather than conducting retrospective analysis after deployment.
- **Resonance with AI Safety/Alignment**: Both communities ask "How do we know an AI system truly has a positive effect?"—but this paper approaches the question from a pragmatic causal inference perspective.
- **The KISS principle's tension with academia**: Engineering culture champions KISS, but academic incentives make simple methods difficult to publish; systemic, institutional-level solutions are needed.

## Rating

- Novelty: ⭐⭐⭐⭐ (As a position paper, it presents a clear structural critique and a three-track reform proposal)
- Experimental Thoroughness: ⭐⭐ (No experiments; argumentation is based on case studies and logical reasoning)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous argumentation, specific and actionable recommendations)
- Recommendation: ⭐⭐⭐⭐ (Offers practical guidance for the AISI ecosystem; the event study discussion is relevant to all ML deployment contexts)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](../../ACL2025/others/sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](../../ICML2026/others/comprehensive_ai_governance_requires_addressing_non-model_gains.md)
- [\[ICML 2025\] Position: AI Evaluation Should Learn from How We Test Humans](../../ICML2025/others/position_ai_evaluation_should_learn_from_how_we_test_humans.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)

</div>

<!-- RELATED:END -->
