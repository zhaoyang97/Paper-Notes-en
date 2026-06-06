---
title: >-
  [Paper Note] Position: AI Evaluations Should be Grounded on a Theory of Capability
description: >-
  [ICML 2026][Image Generation][AI Evaluation] The authors argue that "benchmark score = capability" is an **implicit inference** rather than a direct measurement. They call for explicitly modeling AI evaluation as a stati…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AI Evaluation"
  - "Theory of Capability"
  - "Psychometrics"
  - "IRT"
  - "Perturbation Robustness"
date: 2026-05-08
content_hash: 5843b1f9c8bf7b7a
---

# Position: AI Evaluations Should be Grounded on a Theory of Capability

**Conference**: ICML 2026  
**arXiv**: [2509.19590](https://arxiv.org/abs/2509.19590)  
**Code**: https://github.com/nathanaj99/ai_stat_test (Yes)  
**Area**: LLM Evaluation / Psychometrics / Position Paper  
**Keywords**: AI Evaluation, Theory of Capability, Psychometrics, IRT, Perturbation Robustness

## TL;DR
The authors argue that "benchmark score = capability" is an **implicit inference** rather than a direct measurement. They call for explicitly modeling AI evaluation as a statistical inference task and suggest using four psychometric theories (CTT/IRT/CDM/BNSM) as templates, providing an "Evaluation Card" for evaluators to justify their assumptions.

## Background & Motivation

**Background**: Current LLM evaluation almost exclusively adopts the paradigm of "running on benchmarks and reporting average accuracy"—MMLU, BBH, and HELM all default to *score = capability*, and leaderboards (HuggingFace, Vellum) use aggregated scores directly for ranking.

**Limitations of Prior Work**: It has become a consensus that evaluation results are fragile and irreproducible—scores for the same model can vary by over ten points under different prompt phrasings, temperatures, or system messages. Furthermore, the meaning of "capability" across different benchmarks is mutually incompatible. Even worse, new IRT-based methods (adaptive testing, tinyBenchmarks, etc.) are quietly rewriting the definition of "capability" without explicit declaration.

**Key Challenge**: There is an **undeclared statistical model** between scores and capability. In classical ML (e.g., disease detection), capability definitions are clear (recall/precision) and metric selection is well-founded; however, generative models are general-purpose. A single item on a benchmark simultaneously tests factual memory, linguistic ability, and reasoning—what *latent ability* average accuracy actually estimates is an unstated modeling choice.

**Goal**: (1) Reveal that current evaluations implicitly commit to some theory of capability; (2) Provide a menu of optional theories (CTT, IRT, CDM, BNSM, RT) and point out how they need to be adapted for AI systems; (3) Prove through a concrete experiment (input perturbation sensitivity) that different theories yield **systematically different** conclusions; (4) Propose an Evaluation Card to formalize evaluators' modeling decisions.

**Key Insight**: Psychometrics has accumulated 60 years of methodology for "how to infer latent ability from limited observations"—the work of Lord and Rasch was designed for human intelligence testing. AI evaluation is essentially the same type of problem and should inherit this mindset.

**Core Idea**: Reposition AI evaluation as an **inference task**: first write the generative model of capability $\phi_i = f(\theta, \text{item}_i) + \text{noise}$, then discuss which assumptions of this model are untenable in AI scenarios and how to correct them.

## Method

As a position paper, this work does not propose a new algorithm; the "Method" consists of a **formal comparison of four theories of capability + systematic modifications for AI + a proof-of-concept experiment**.

### Overall Architecture
The paper proceeds in three layers:
- **Layer 1 (Section 2)**: Points out that current evaluations default to CTT (Classical Test Theory), i.e., $\phi_i = \theta_i + \epsilon_i$, $\theta = \mathbb{E}_i[\theta_i]$, which is equivalent to "all items have equal information and errors are i.i.d." Emerging IRT-based methods (e.g., tinyBenchmarks), though not explicitly stated, actually change the definition of capability: $\theta$ is a latent variable, and two models with the same accuracy can have different abilities under IRT because their errors fall on items with different discrimination.
- **Layer 2 (Section 3)**: Integrates five theories (CTT/IRT/CDM/BNSM/RT) into a unified formula $\phi_i = \theta_i + s(x_i) + r(h) + g(c) + \epsilon_i$, where $s(x_i)$ is phrasing perturbation, $r(h)$ is the influence of hyperparameters (temperature, top-$p$), and $g(c)$ is context influence. It points out that **core assumptions of human psychometrics—conditional independence and mean-zero noise—are systematically violated by AI systems** (Potemkin understanding, prompt sensitivity, temperature dependence).
- **Layer 3 (Section 4)**: Uses prompt perturbation as a concrete example to formulate the four theories as executable inference algorithms (CTT with clustered bootstrap, IRT with Fisher scoring, CDM with penalized MAP, BNSM with Bayesian network posterior), comparing the "capability estimates" provided by the four theories across 7 open-source LLMs and 8 benchmark sub-tasks.

### Key Designs

1. **Unified "Capability + Perturbation" Formula Family**:
    - **Function**: Fits all candidate theories into a single framework, making their differences explicit as choices of the function form $f(\theta, \text{item})$ for easier comparison.
    - **Mechanism**: Taking CTT as an example, it extends $\phi_i = \theta_i + \epsilon_i$ into $\phi_i = \theta_i + s(x_i) + \epsilon_i$ and proposes **Assumption 4.1 (mean-zero perturbations)**: $\mathbb{E}_{x_i \sim \mathcal{P}_i}[s(x_i)] = 0$. Table 1 provides the unified expressions for CTT/IRT/CDM/BNSM after incorporating $s(x_i)$.
    - **Design Motivation**: To solve the problem where different evaluation papers use different models but pretend to compare the same thing. Once the form of $f$ is defined, the capability definition is locked, making the results comparable.

2. **Two-Stage Sampling Diagnosis for Benchmark Independence Violations**:
    - **Function**: Explains from a probabilistic modeling perspective why a *single-phrasing benchmark* cannot identify $\theta_i$.
    - **Mechanism**: Splits benchmark item generation into two stages—Stage 1 samples item $i$ from task space $\mathbb{P}$; Stage 2 samples specific phrasing $x_i$ from phrasing distribution $\mathcal{P}_i$. Curators control Stage 1 (independent sampling), but Stage 2 usually involves only one hand-written $x_i$, often produced by the same team → structural dependencies exist between phrasings, violating Assumption 4.1 → $\theta_i$ is unidentifiable under $\phi_i = \theta_i + s(x_i) + \epsilon_i$. Proposition B.3 further proves that approximating $\mathcal{P}_i$ with multiple perturbations $\{x_{ij}\}_{j=1}^{m_i}$ reduces bias, with bias $|\delta_i|$ shrinking monotonically as $\tilde{\mathcal{P}}_i$ approaches the true $\mathcal{P}_i$.
    - **Design Motivation**: To upgrade anecdotal observations like "prompt perturbations change rankings" to an identifiability problem—it is not that the model is fragile, but that the benchmark itself is statistically underspecified.

3. **Evaluation Card: Making Modeling Decisions Explicitly Public**:
    - **Function**: Provides a mandatory template for evaluation papers, listing four categories of **decisions that must be explicitly declared**.
    - **Mechanism**: Table 2 breaks the Evaluation Card into four columns: (a) Meaning of Capability (CTT reports mean / IRT reports latent / CDM reports skill mastery); (b) Task Structure (latents assumed, DINA or DINO aggregation); (c) Sources of Systematic Variation (which confounders are noise, which are explicitly modeled); (d) Data Considerations (item parameters for IRT, skill-to-item prior maps for CDM).
    - **Design Motivation**: Following the same logic as Datasheets for Datasets and Model Cards—substituting "finding the most correct model" with "clarifying assumptions," as the paper acknowledges there is **no strictly better theory of capability**, only more transparent assumptions.

### Loss & Training
N/A for training, but each theory corresponds to an inference algorithm: CTT uses clustered bootstrap (items as clusters, perturbations as within-cluster observations), IRT uses Fisher scoring/Newton-Raphson for $\hat{\theta}$ MLE, CDM uses MAP with logistic likelihood + Gaussian prior (relaxing $\alpha \in \{0,1\}^K$ to $\alpha \in \mathbb{R}^K$), and BNSM uses Bayesian network posterior inference, all paired with item-level bootstrap for uncertainty.

## Key Experimental Results

### Main Results

Experimental setup: 7 open-source instruction-tuned LLMs (Llama-3.2, Qwen-2.5, Gemma families) × 2 benchmarks (BBH, LMEntry) × 4 sub-tasks each (AWFC/FA/ML/RW, CJ/MR/FF/S), using the perturbed versions released by mizrahi2024state as phrasing distribution proxies.

| Comparison Dimension | CTT (Mean) | IRT (Latent Ability) | Key Difference |
|----------------------|------------|-----------------------|----------------|
| Ranking Consistency  | baseline   | Generally consistent with CTT | Consistent |
| Inter-model Separation| Small      | Significantly amplified | IRT pulls Qwen-3.5B to the top on AWFC (due to performance on hard items) |
| Sample Complexity    | Full       | Adaptive testing      | IRT achieves same inference with fewer samples (bolded in results) |
| Same accuracy, different ability | Indistinguishable | Distinguishable | Qwen-3.5B has same accuracy on AWFC and S, but IRT yields lower ability for S because AWFC items are harder |

### Ablation Study

| Configuration | Key Findings | Explanation |
|---------------|--------------|-------------|
| CTT vs IRT | Rankings broadly consistent but separation differs significantly | "High accuracy ≠ High ability"; hard items carry more weighted information |
| CDM vs BNSM (Movie Recommendation) | BNSM rates Social Reasoning significantly higher | CDM can only explain poor MR performance via low $Soc$; BNSM introduces World Knowledge ($W$) as extra latent skill, attributing MR/CJ errors to low $W$ rather than low $Soc$ |
| Original phrasing vs Perturbation mean | Difference $D_i = \phi_i^{\text{orig}} - \bar{\phi}_i$ is significantly non-zero across tasks | Single-phrasing benchmarks are biased estimators |

### Key Findings
- **Theory choice directly changes conclusions**: For the same raw answer data, applying CTT vs IRT can lead to conclusions ranging from "the two models have the same capability" to "they differ by 2 standard deviations." This is a product of modeling decisions, not measurement error.
- **Skill structure assumptions in CDM/BNSM are extremely powerful**: The introduction or omission of a single latent skill in the MR task can double the Social Reasoning score—highlighting the importance of the second column in the Evaluation Card (Task Structure).
- **No "correct" theory exists**: The authors intentionally do not recommend any single theory—the real takeaway is that "which one was chosen, why, and whether its assumptions hold in AI scenarios" must be stated in the evaluation paper.
- **Perturbations are for identification, not just finding ground truth**: The authors reinterpret work on prompt perturbation—their essential contribution is expanding phrasing space coverage to improve the identifiability of $\theta_i$, rather than "measuring true capability."

## Highlights & Insights

- **Upgrading anecdotal community complaints to statistical identifiability problems**: Everyone complains about prompt sensitivity, but this paper cleanly proves using two-stage sampling (task space + phrasing space) that "single-phrasing benchmarks under $\phi = \theta + s(x) + \epsilon$ leave $\theta$ unidentifiable," turning an engineering problem into a falsifiable mathematical one.
- **Pointing out that IRT-based evaluations are "stealthily changing definitions"**: Works like tinyBenchmarks or adaptive testing, often treated as "sample-saving engineering optimizations," actually change the definition of capability. This is a sharp observation and the most impactful point of the paper.
- **Redefining "deliberation time" for AI vs Humans**: Section 3.3 treats reasoning token count / FLOPs as the AI version of response time, applying log-normal models. This provides an extensible interface for evaluating speed-accuracy tradeoffs in agentic / o1-style systems, applicable to any inference model that "calculates more or less."
- **The Evaluation Card paradigm**: Directly replicates the successful path of Datasheets/Model Cards. It can be used as a checklist when writing, reviewing, or reproducing AI evaluations—much more practical than simply calling for "better evaluations."

## Limitations & Future Work

- **Self-acknowledged limitations**: The list of capability theories is non-exhaustive (missing Mokken scaling, BKT, PFA, SDT, DDM, etc.); prompt perturbation is only one confounder—temperature, context, and decoding strategies have not yet been empirically tested.
- **Methodological limitations**: The DINA model requires experts to provide a Q-matrix (skill-to-item mapping), and BNSM requires prior skill dependency graphs—defining these priors for general benchmarks remains an open problem, essentially shifting "subjectivity" from accuracy to the priors.
- **Experimental scale**: Only 7 open-source LLMs × 8 sub-tasks were used, excluding frontier closed-source models; the perturbation dataset reused mizrahi2024state, and perturbation quality was not independently verified.
- **Lack of connection to downstream validity**: The authors acknowledge in Alternative View 2 that "interpretability ≠ downstream validity," but the paper lacks empirical evidence on whether "abilities estimated via explicit theories better predict deployment performance." This is an obvious next step.

## Related Work & Insights

- **vs Datasheets for Datasets / Model Cards (gebru2021datasheets)**: Consistent logic (publicly declaring modeling decisions), but targets a different layer—those specify data and models, while this specifies the **evaluation process** itself.
- **vs tinyBenchmarks (polo2024tiny) / adaptive testing (zhuang2023static)**: These IRT-based methods are reinterpreted here—their true contribution isn't "reducing samples" but "changing the definition of capability," and this paper provides them with a theoretical framework.
- **vs Robustness literature (mizrahi2024state; sclar2023quantifying; zheng2023large)**: Repositions prompt perturbation as "improving the identifiability of $\theta_i$" rather than "measuring robustness," providing a unified motivation for otherwise scattered perturbation techniques.
- **vs Mitchell (mitchell2024debates) / Hardt (hardt2025emerging) "benchmark science" calls**: This paper is a concrete methodological implementation of this trend, treatable as a technical white paper for the "science of benchmarks" movement.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing psychometrics to LLM evaluation isn't the first, but formalizing it into a comparable family of theories with empirical demos is a relatively new combination.
- Experimental Thoroughness: ⭐⭐⭐ The 7×8 scale is small and excludes closed-source models; the value lies more in proof-of-concept than benchmarking.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentative chain is clear, math and intuition are well-interspersed, and the Alternative Views section shows proper debate awareness for a position paper.
- Value: ⭐⭐⭐⭐ The Evaluation Card has the potential to be immediately adopted as a reviewing standard and has a directional impact on the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)

</div>

<!-- RELATED:END -->
