---
title: >-
  [Paper Note] Position: AI Evaluations Should be Grounded on a Theory of Capability
description: >-
  [ICML 2026][Image Generation][AI Evaluation] The authors argue that "benchmark score = capability" is an **implicit inference** rather than a direct measurement. They advocate for explicitly modeling AI evaluation as a statistical inference task and suggest utilizing four psychometric capability theories (CTT/IRT/CDM/BNSM) as templates, introducing an "Evaluation Card" for evaluators to justify their modeling assumptions.
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AI Evaluation"
  - "Capability Theory"
  - "Psychometrics"
  - "IRT"
  - "Perturbation Robustness"
date: 2026-05-08
content_hash: 6c6dfe2d3ffd693d
---

# Position: AI Evaluations Should be Grounded on a Theory of Capability

**Conference**: ICML 2026  
**arXiv**: [2509.19590](https://arxiv.org/abs/2509.19590)  
**Code**: https://github.com/nathanaj99/ai_stat_test (Available)  
**Area**: LLM Evaluation / Psychometrics / Position Paper  
**Keywords**: AI Evaluation, Capability Theory, Psychometrics, IRT, Perturbation Robustness

## TL;DR
The authors argue that "benchmark score = capability" is an **implicit inference** rather than a direct measurement. They advocate for explicitly modeling AI evaluation as a statistical inference task and suggest utilizing four psychometric capability theories (CTT/IRT/CDM/BNSM) as templates, introducing an "Evaluation Card" for evaluators to justify their modeling assumptions.

## Background & Motivation

**Background**: Currently, LLM evaluation almost exclusively adopts the paradigm of "running on a benchmark and reporting average accuracy." Frameworks such as MMLU, BBH, and HELM default to the assumption that *score = capability*, and leaderboards (e.g., HuggingFace, Vellum) rank models directly using aggregated scores.

**Limitations of Prior Work**: It is well-recognized that evaluation results are fragile and non-reproducible. The score of a single model can fluctuate by over ten points depending on prompt phrasing, temperature, or system messages, and definitions of "capability" across different benchmarks are often incompatible. Furthermore, IRT-based methods (e.g., adaptive testing, tinyBenchmarks) are quietly redefining "capability" without explicit declaration.

**Key Challenge**: An **undeclared statistical model** exists between scores and capabilities. In classical machine learning (e.g., disease detection), capability definitions are clear (recall/precision), and metric selection is grounded. However, generative models are general-purpose; a single benchmark item may simultaneously test factual memory, linguistic ability, and reasoning. What *latent ability* the average accuracy is estimating remains an unstated modeling choice.

**Goal**: (1) Reveal that current evaluations implicitly commit to specific capability theories; (2) Provide a menu of alternative capability theories (CTT, IRT, CDM, BNSM, RT) and identify how to adapt them for AI systems; (3) Demonstrate via perturbation experiments that different capability theories yield **systematically different** conclusions; (4) Propose an Evaluation Card to standardize the disclosure of modeling decisions by evaluators.

**Key Insight**: Psychometrics has accumulated 60 years of methodology regarding "how to infer latent ability from limited observations"—the work of Lord and Rasch was specifically designed for human intelligence testing. AI evaluation is essentially the same problem and should inherit this methodology.

**Core Idea**: Reposition AI evaluation as an **inference task**: first define a generative model of capability $\phi_i = f(\theta, \text{item}_i) + \text{noise}$, and then discuss which assumptions in this model are untenable in AI scenarios and how they should be corrected.

## Method

As a position paper, this work does not propose a new algorithm but rather addresses the neglected modeling problem of what "latent ability" benchmark average scores actually estimate. The approach frames evaluation as statistical inference, compares four capability theories (CTT/IRT/CDM/BNSM) within a unified formula family, and uses prompt perturbation experiments to prove that the choice of theory systematically alters conclusions.

### Overall Architecture

The paper progresses through three layers. The first layer (Section 2) points out that current evaluations default to Classical Test Theory (CTT): $\phi_i = \theta_i + \epsilon_i$ and $\theta = \mathbb{E}_i[\theta_i]$, which assumes all items are equally informative and errors are i.i.d. IRT-based methods like tinyBenchmarks change the definition of ability to a latent variable, where two models with the same accuracy can have different abilities if their errors occur on items with different discrimination. The second layer (Section 3) integrates five theories into a unified formula $\phi_i = \theta_i + s(x_i) + r(h) + g(c) + \epsilon_i$, where $s(x_i)$ represents phrasing perturbations, $r(h)$ represents hyperparameter effects (temperature/top-$p$), and $g(c)$ represents context effects. It notes that psychometric assumptions like conditional independence and mean-zero noise are systematically violated in AI systems due to Potemkin understanding and prompt sensitivity. The third layer (Section 4) uses prompt perturbation as a specific case study to implement four theories as executable inference algorithms, comparing the resulting "capability estimates" across 7 open-source LLMs and 8 benchmark subtasks.

### Key Designs

**1. Unified "capability + perturbation" formula family: Making theoretical differences manifest as functional choices**
Different evaluation papers use distinct implicit models while claiming to compare the same "capability," leading to a lack of comparability. This paper addresses this by placing all candidate theories into a single framework, where their differences converge into the choice of the generative function $f(\theta, \text{item})$. For CTT, $\phi_i = \theta_i + \epsilon_i$ is extended to $\phi_i = \theta_i + s(x_i) + \epsilon_i$, introducing **Assumption 4.1 (mean-zero perturbations)**: $\mathbb{E}_{x_i \sim \mathcal{P}_i}[s(x_i)] = 0$. Table 1 provides unified expressions for CTT/IRT/CDM/BNSM after incorporating $s(x_i)$. Once the form of $f$ is fixed, the definition of ability is locked, allowing for a determinable answer to whether two papers are measuring the same thing.

**2. Two-stage sampling diagnostic: Proving that single-phrasing benchmarks cannot identify $\theta_i$**
Prompt sensitivity complaints have remained anecdotal; this paper elevates them to a formal identifiability proposition. The key is splitting benchmark item generation into two stages: Stage 1 draws an item $i$ from the task space $\mathbb{P}$, and Stage 2 draws a specific phrasing $x_i$ from the phrasing distribution $\mathcal{P}_i$. While curators strictly control Stage 1, Stage 2 often involves a single $x_i$ handwritten by a single team. This introduces structural dependencies and violates Assumption 4.1, meaning $\theta_i$ is unidentifiable under $\phi_i = \theta_i + s(x_i) + \epsilon_i$. Proposition B.3 suggests a remedy: approximating the true $\mathcal{P}_i$ with multiple perturbations $\{x_{ij}\}_{j=1}^{m_i}$ to reduce bias, where bias $|\delta_i|$ shrinks monotonically with the distance between the approximate distribution $\tilde{\mathcal{P}}_i$ and the true $\mathcal{P}_i$.

**3. Evaluation Card: Mandating the disclosure of modeling decisions**
Since no capability theory is strictly superior, the only basis for comparison is the transparency of assumptions. Following the success of Datasheets for Datasets and Model Cards, this paper provides a template for evaluation papers. Table 2 breaks the Evaluation Card into four sections: (a) Meaning of Capability (e.g., CTT reports mean / IRT reports latent ability); (b) Task Structure (e.g., whether latent skills are assumed, whether aggregation uses DINA or DINO); (c) Sources of Systematic Variation (which confounders are treated as noise vs. explicitly modeled); (d) Data Considerations (e.g., item parameters for IRT, skill-to-item mapping for CDM). The goal is to replace "finding the right model" with "clarifying assumptions."

### Loss & Training
While the four theories do not involve training, each corresponds to an inference algorithm with item-level bootstrapping for uncertainty estimation: CTT uses clustered bootstrap (items as clusters); IRT uses Fisher scoring / Newton-Raphson for $\hat{\theta}$ MLE; CDM uses MAP with logistic likelihood and Gaussian priors (relaxing $\alpha \in \{0,1\}^K$ to $\alpha \in \mathbb{R}^K$); BNSM uses Bayesian networks for posterior inference.

## Key Experimental Results

### Main Results

Experimental setup: 7 open-source instruction-tuned LLMs (families: Llama-3.2, Qwen-2.5, Gemma) $\times$ 2 benchmarks (BBH, LMEntry) $\times$ 4 subtasks each (AWFC/FA/ML/RW, CJ/MR/FF/S), using the perturbation version from mizrahi2024state as a proxy for the phrasing distribution.

| Dimension | CTT (Mean) | IRT (Latent Ability) | Key Differences |
|-----------|------------|----------------------|-----------------|
| Model Rank Consistency | baseline | Generally consistent with CTT | Consistent |
| Inter-model Separation | Small | Significantly amplified | IRT assigned Qwen-3.5B the highest score on AWFC (due to better performance on difficult items) |
| Sample Complexity | Full set | Adaptive testing | IRT achieved the same inference with fewer samples |
| Ability Discrimination | Indistinguishable | Distinguishable | Qwen-3.5B had identical accuracy on AWFC and S, but IRT calculated lower ability for S because AWFC items are harder |

### Ablation Study

| Configuration | Key Findings | Explanation |
|---------------|--------------|-------------|
| CTT vs IRT | Ranks largely consistent but separation differs significantly | "High accuracy $\neq$ high ability"; difficult items carry more information weight |
| CDM vs BNSM (Movie Recommendation) | BNSM rated Social Reasoning significantly higher | CDM can only explain poor MR performance via low $Soc$; BNSM introduces World Knowledge ($W$) as an additional skill, attributing MR errors to low $W$ |
| Original Phrasing vs. Perturbation Mean | Difference $D_i = \phi_i^{\text{orig}} - \bar{\phi}_i$ is significantly non-zero | Single-phrasing benchmarks act as biased estimators |

### Key Findings
- **Theory choice directly changes conclusions**: For the same raw data, applying CTT vs. IRT can conclude "models have the same ability" vs. "models differ by 2 standard deviations." This is a product of modeling decisions, not measurement error.
- **CDM/BNSM skill structure assumptions are highly influential**: In the MR task, introducing a single latent skill can double the Social Reasoning score, highlighting the importance of the "Task Structure" section in the Evaluation Card.
- **No "correct" theory exists**: The authors deliberately refrain from recommending one theory. The core takeaway is that which theory was chosen and whether its assumptions hold for AI must be disclosed.
- **Perturbations are for identification, not just truth-seeking**: The authors re-interpret prompt perturbation work; their primary contribution is expanding the phrasing space to improve the identifiability of $\theta_i$, rather than merely measuring "true ability."

## Highlights & Insights

- **Elevating anecdotal complaints to statistical identifiability**: While prompt sensitivity is a common complaint, this work uses two-stage sampling to prove that single-phrasing benchmarks leave $\theta$ unidentifiable under $\phi = \theta + s(x) + \epsilon$, turning an engineering nuisance into a refutable mathematical problem.
- **Identifying that IRT-based evaluations are "secretly changing definitions"**: Methods like tinyBenchmarks and adaptive testing, often viewed as "sample-saving optimizations," actually shift the definition of capability. This is a sharp observation and perhaps the most impactful insight of the paper.
- **Redefining AI "deliberation time"**: Section 3.3 treats reasoning token counts or FLOPs as the AI version of "response time" in log-normal models, providing an extensible interface for evaluating the speed-accuracy tradeoff in agentic or o1-style models.
- **The Evaluation Card Paradigm**: By following the path of Datasheets and Model Cards, this provides a practical checklist for writing, reviewing, and reproducing AI evaluations.

## Limitations & Future Work

- **Author Admissions**: The list of capability theories is not exhaustive (omitting Mokken scaling, BKT, PFA, SDT, DDM, etc.); prompt perturbation is only one confounder, while temperature, context, and decoding strategies have not yet been empirically validated.
- **Methodological Limitations**: DINA models require an expert-provided Q-matrix (skill-to-item mapping), and BNSM needs a prior skill dependency graph. Defining these priors for general benchmarks remains an open and subjective problem.
- **Experimental Scale**: The study only covers 7 open LLMs and 8 subtasks, excluding frontier closed-source models. The perturbation quality of the reused dataset was not independently verified.
- **Lack of Link to Downstream Validity**: While the authors discuss alternative views (interpretability $\neq$ downstream validity), the paper lacks empirical evidence on whether explicit capability estimates better predict real-world deployment performance.
- **Future Improvements**: Systematizing the confounder taxonomy (temperature, system prompts, etc.) and providing "recommended Evaluation Card entries" for common benchmarks as community infrastructure.

## Related Work & Insights

- **vs Datasheets for Datasets / Model Cards (gebru2021datasheets)**: Shared philosophy of mandating transparency in modeling decisions, but targets the **evaluation process** itself rather than models or data.
- **vs tinyBenchmarks (polo2024tiny) / adaptive testing (zhuang2023static)**: This paper provides a theoretical framework for these IRT-based methods, interpreting them as a shift in capability definitions rather than just sample reduction.
- **vs Robustness literature (mizrahi2024state; sclar2023quantifying)**: Repositions prompt perturbation as a means to "improve identifiability of $\theta_i$" rather than just "measuring robustness," providing a unified motivation for fragmented perturbation techniques.
- **vs Mitchell (mitchell2024debates) / Hardt (hardt2025emerging)**: This work serves as a technical implementation of the growing call for a "science of benchmarks."

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing psychometrics to LLM evaluation isn't entirely new, but formalizing it into a comparable family of theories with empirical demos is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐ The 7×8 scale is relatively small and lacks closed models; the value is primarily in proof-of-concept.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation, well-balanced between formulas and intuition, and the "Alternative Views" section demonstrates strong self-awareness.
- Value: ⭐⭐⭐⭐ The Evaluation Card has strong potential for adoption in peer review and could significantly influence the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICLR 2026\] Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice](../../ICLR2026/image_generation/overshoot_and_shrinkage_in_classifier-free_guidance_from_theory_to_practice.md)

</div>

<!-- RELATED:END -->
