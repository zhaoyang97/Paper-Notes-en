---
title: >-
  [Paper Note] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost
description: >-
  [ICML 2026][Image Generation][Paper Note] This position paper argues that "introducing AI into an organization does not automatically equate to productivity gains." It identifies five human and environmental moderators ignored by traditional economic models (personnel composition, individual baseline capability, learning curves, equitable use incentives, and g
tags:
  - ICML 2026
  - Image Generation
date: 2026-05-08
content_hash: 5f7b81583490ab80
---
# Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost

**Conference**: ICML 2026  
**arXiv**: [2605.24688](https://arxiv.org/abs/2605.24688)  
**Code**: None  
**Area**: AI Governance / Position Paper / Economics of AI Productivity  
**Keywords**: AI Adoption, Productivity Paradox, Gries-Naudé Model, Organizational Factors, Learning Curve

## TL;DR
This position paper argues that "introducing AI into an organization does not automatically equate to productivity gains." It identifies five human and environmental moderators ignored by traditional economic models (personnel composition, individual baseline capability, learning curves, equitable use incentives, and goal flexibility). By incorporating organizational effectiveness $\Omega$, capability adjustment $\phi(z,\kappa_i)$, learning curves $\lambda_i(\tau)$, and effective automation thresholds $\tilde N_{IT}$ into the Gries-Naudé (2022) partial equilibrium model, the authors derive a revised production function that explains the massive output gap between organizations with identical AI investments.

## Background & Motivation

**Background**: Following the explosion of generative AI—from GPT-3 and Codex to ChatGPT and agentic frameworks—enterprises and educational institutions are rushing to deploy AI. In economics, researchers such as Graetz-Michaels, Acemoglu, and Gries-Naudé have begun treating AI directly as a "productivity factor" in aggregate output models, assuming that stronger AI capabilities naturally lead to higher Total Factor Productivity (TFP).

**Limitations of Prior Work**: Empirical evidence does not support this linear extrapolation. Brynjolfsson et al. (2025) found that AI boosted novice performance by 36% while having almost zero effect on experienced workers in customer service. Dell'Acqua et al. (2023) observed in management consulting that when tasks exceeded AI's capability boundaries, users performed 19 percentage points worse than non-users. Cross-country firm-level data from Calvino-Fontanelli show that AI adoption is highly concentrated in already dominant large firms. Solow's classic observation—"You can see the computer age everywhere but in the productivity statistics"—is recurring as the "AI Productivity Paradox."

**Key Challenge**: Existing economic models treat parameters like $\gamma_{IT}(z)$, $b_{IT}$, and $A_{IT}/A_L$ as purely exogenous technical variables. However, the variables that actually determine whether AI capability can be monetized are **endogenous human and environmental variables**: organizational structure, individual capability, learning dynamics, usage incentives, and goal flexibility. These have been largely overlooked.

**Goal**: (1) Explicitly identify these five types of moderators and their mechanisms; (2) Modify the Gries-Naudé partial equilibrium model to explicitly incorporate them into the production function; (3) Provide a "to-do list" for researchers, practitioners, and policymakers, emphasizing the distinction between industry and education.

**Key Insight**: Rather than denying the growth of AI capability, the authors shift the focus from "how strong the AI model is" to "how the organization utilizes AI." AI capability is a necessary but insufficient condition; the five organizational/individual factors serve as the "transmission ratio" that translates capability into output.

**Core Idea**: In summary—**True AI Productivity = Technical Capability × Organizational Effectiveness $\Omega$ × Capability-Task Fit $\phi$ × Learning Progress $\lambda$ × Goal Flexibility $F$**. Ignoring the latter four components makes any estimation of AI investment ROI misleading.

## Method

As a position paper, this is not an algorithmic work; its "method" refers to the revised framework for the Gries-Naudé (2022) partial equilibrium model. To understand the revision, one must first understand the base model: tasks are arranged on a continuous interval $[N-1, N]$, where each task $z$ can be produced by either standard labor or IT services (including AI). These are aggregated into total human-service output $H$ via CES. The model defines an automation threshold $N_{IT}$ that splits the interval into "AI-accessible" and "manually reliant" segments. The flaw in the original model is that variables like $\gamma_{IT}$, $b_{IT}$, and $N_{IT}$ are treated as exogenous technical constants. This paper decomposes these three variables by layering modulators across organizational, individual, and temporal dimensions to endogenize heterogeneity.

### Overall Architecture
The authors qualitatively identify five neglected moderators (personnel composition, individual baseline capability, learning curves, equitable use incentives, and goal flexibility) and compress them into four mathematical modulators: organizational effectiveness $\Omega$, capability-adjusted productivity $\tilde\gamma_L/\tilde\gamma_{IT}$, the learning curve $\lambda_i(\tau)$, and the endogenous task boundary $\tilde N_{IT}$. These components form a three-dimensional revised human-service production function $\tilde h_i(z,\tau)$ (individual-task-time), which is then aggregated to the sub-organizational level $\tilde H(\tau)$ via CES. The framework takes observable organizational and individual attributes as inputs and outputs a production function capable of explaining why identical AI investments yield vastly different outputs.

### Key Designs

**1. Organizational Effectiveness $\Omega = \omega_C \cdot \omega_I$: Separating "Expert Availability" from "Expert Accessibility."** The original model uses a single parameter $b_{IT}$ for expert availability, assuming hiring experts automatically translates to assistance. The authors introduce two discount factors: $\omega_C \in [0,1]$ for organizational alignment (flat AI task groups approach 1, while deep hierarchies with policy-execution gaps approach 0) and $\omega_I \in [0,1]$ for incentive alignment (asymmetric competition erodes the motivation for peer-to-peer equitable AI use). Their product $\Omega$ discounts $b_{IT}$, yielding effective accessibility $\tilde b_{IT}(z) = \Omega \cdot b_{IT}(z)$. This explains Calvino-Fontanelli's observation: stronger firms benefit more not just because they have more experts, but because their structures allow experts to reach frontline executors.

**2. Capability-Task Interaction $\phi(z,\kappa_i)$ and the Reliable Frontier $N_R$: Opposite effects for the same tool on different tasks.** A simple $\gamma_{IT}/\gamma_L$ ratio cannot describe the non-monotonic relationship where "novices surge while experts slightly decline." The authors introduce individual baseline capability $\kappa_i \in [0,1]$ and further partition the automatable interval $[N-1, N_{IT}]$ into the "AI Reliable Zone" $[N-1, N_R]$ and the "AI Unreliable Zone" $(N_R, N_{IT}]$, formalizing Dell'Acqua's "jagged frontier." Inside the reliable zone, $\partial \phi_{\text{in}}/\partial \kappa_i \leq 0$ (weaker baselines gain more, as in Brynjolfsson 2025). In the unreliable zone, $\partial \phi_{\text{out}}/\partial \kappa_i > 0$, as only high-capability individuals can identify and correct AI errors. Raw labor productivity is also modulated as $\tilde\gamma_L(z,\kappa_i) = \gamma_L(z) \cdot g(\kappa_i)$, where $g$ increases with $\kappa_i$. This piecewise function endogenizes non-monotonic heterogeneous gains.

**3. Learning Curve $\lambda_i(\tau) = 1 - e^{-\rho_i \tau}$ and Flexible Threshold $\tilde N_{IT}$: Incorporating temporal evolution and organizational rigidity.** Using time since adoption $\tau$ as a parameter, the authors define learning progress $\lambda_i(\tau) \in [0,1)$ for individual $i$. The learning rate $\rho_i$ is heterogeneous (fast learners pull ahead, creating Matthew effect risks). Multiplying capability and learning terms yields effective AI task productivity $\tilde\gamma_{IT}(z,\kappa_i,\tau) = \gamma_{IT}(z) \cdot \phi(z,\kappa_i) \cdot \lambda_i(\tau)$. Finally, organizational goal flexibility $F \in [0,1]$ shrinks the technical threshold $N_{IT}$ into an effective threshold $\tilde N_{IT} = (1-F)(N-1) + F \cdot N_{IT}$. When $F=1$, it restores the original model; when $F<1$, rigid assessments mean the organization only delegates a fraction of technically automatable tasks to AI. This links the technically given $N_{IT}$ to leadership's willingness to rearrange KPIs, explaining massive variance in actual AI penetration under the same tech stack.

Combining the organizational, individual, and temporal components yields the core revised equation (Eq. 9):

$$\tilde h_i(z,\tau) = \begin{cases} \tilde\gamma_L(z,\kappa_i) l_i(z) A_L + \tilde\gamma_{IT}(z,\kappa_i,\tau) \cdot \Omega \cdot b_{IT}(z) A_{IT} D, & z \in [N-1, \tilde N_{IT}] \\ \tilde\gamma_L(z,\kappa_i) l_i(z) A_L, & z \in (\tilde N_{IT}, N] \end{cases}$$

In the automatable segment, AI and human labor contribute in parallel, with the AI term discounted by $\Omega$, while segments beyond the effective threshold rely solely on labor. Summing over individuals and applying CES aggregation $\tilde H(\tau) = \big( \int_{N-1}^N (\sum_i \tilde h_i(z,\tau))^{(\sigma-1)/\sigma} dz \big)^{\sigma/(\sigma-1)}$ yields the total human-service output at the sub-organizational level.

**Mechanism**: This position paper does not rely on algorithms but follows a four-step argumentative strategy: first, citing empirical counterexamples (Brynjolfsson, Dell'Acqua, Calvino-Fontanelli, Acemoglu) to show linear extrapolation is untenable; second, performing a minimal-intrusion revision on the Gries-Naudé mathematical skeleton; third, applying the framework to industry and education cases; and finally, responding to three types of counter-arguments (technological determinism, measurement issues, and wage cost arguments) in Section 4 by re-interpreting their points within this framework.

## Key Experimental Results

As a position paper, this contains no experiments. This section provides two tables mapping empirical evidence to framework factors.

### Empirical Evidence → Framework Factor Mapping

| Empirical Phenomenon | Source | Corresponding Factor |
|---|---|---|
| Novices boost 36%, experts almost zero | Brynjolfsson et al. 2025 | $\phi_{\text{in}}(\kappa_i)$ decreasing with $\kappa_i$ |
| Accuracy drops 19pp outside AI boundary | Dell'Acqua et al. 2023 | $\phi_{\text{out}}(\kappa_i)$ increasing with $\kappa_i$; $N_R < N_{IT}$ |
| AI adoption concentrated in large/strong firms | Calvino-Fontanelli 2023 | High $\Omega = \omega_C \omega_I$ |
| TFP growth far below expectations | Acemoglu 2025 | Aggregate $F \cdot \Omega \cdot \lambda$ is $\ll 1$ |
| ChatGPT raises grades but not self-efficacy | Deng et al. 2025 | Misalignment of $F$ and $\omega_I$ in education |

### Gries-Naudé Original Determinants → Ours (Ref. Table 1)

| Original Determinant | Our Revision |
|---|---|
| Automation Threshold $N_{IT}$ | Effective Threshold $\tilde N_{IT}$, determined by Goal Flexibility $F$ |
| Task Productivity Ratio $\gamma_{IT}(z)/\gamma_L(z)$ | Identity- and time-dependent $\tilde\gamma_{IT}/\tilde\gamma_L = \gamma_{IT}\phi\lambda / (\gamma_L g)$ |
| Expert Availability $b_{IT}$ | Effective Accessibility $\tilde b_{IT} = \Omega \cdot b_{IT}$ |
| Relative Capability Ratio $A_{IT}/A_L$ | Modulated as $\propto \phi(z,\kappa_i) \lambda_i(\tau) / g(\kappa_i)$ |

### Key Findings
- **Productivity Ratio Monotonicity Constraint**: The authors require $\tilde\gamma_{IT}/\tilde\gamma_L$ to be monotonic relative to $z$ given fixed $\kappa_i, \tau$ to define $N_R$ and $\tilde N_{IT}$ without ambiguity—an implicit prerequisite for model self-consistency.
- **Aggregation Granularity**: Aggregating at the team/department level rather than the firm level is justified by the heavy-tailed distribution of firm-wide performance (the "10x engineer" effect), whereas sub-unit distributions are more manageable. This preserves individual heterogeneity via $\kappa_i, \phi, \rho_i$.
- **Fundamental Difference between Education and Industry**: In industry, AI gains are high when $F$ (KPI flexibility) is high, but the beneficiaries shift from employees to management. In education, because the beneficiary is the student themselves, the "productivity for whom" problem vanishes, but misalignment between $F$ and $\omega_I$ can directly harm learning objectives.

## Highlights & Insights
- **Minimal Intrusion by Turning "Parameters" into "Variables"**: The authors layered four moderators ($\Omega, \phi, \lambda, F$) onto the original $\gamma, b_{IT}, N_{IT}$. This "patching rather than rewriting" approach allows new readers to adapt with zero friction while retaining the qualitative conclusions of the base model.
- **Elegant Response to "Technological Determinism"**: Instead of denying technical progress, the authors argue that "stronger AI $\to$ higher $N_R \to$ lower importance of $\phi_{\text{out}}$, yet constraints shift to $F$ and $\omega_I$." It reinterprets opposing views rather than simply refuting them.
- **"Productivity for whom" - A Political Economy Dimension**: Most AI economic papers ignore this. By contrasting "Industry (management benefits)" with "Education (aligned benefits)," the paper forces any "AI productivity" study to define whose productivity is being measured.
- **Transferable Design**: The $\Omega, \phi, \lambda, F$ modulators are applicable to any "new technology adoption" economic model, including cloud computing, remote work, or low-code platforms.

## Limitations & Future Work
- **Acknowledged Limitations**: The specific functional forms for $\phi, g, \rho$ are kept as "general forms" to avoid masking cross-domain differences. However, this means the framework **lacks falsifiable predictions** and functions more as a narrative tool than a fittable model.
- **Abstraction of Baseline Capability $\kappa_i$**: It is difficult to measure uniformly in practice. The authors do not provide an operational definition for "baseline capability" across different tasks (coding skill? critical thinking? domain knowledge?), leaving it to "organizational behavior researchers" in §6.1.
- **Lack of Endogenized Direct AI Costs**: Section 4.3 admits licensing fees and compute costs are "another variable" but excludes them from the framework, meaning full decision-making on AI investment still requires external analysis.
- **Conflict between Individual Summation and CES Assumptions**: Summing over $i$ before CES aggregation implies individuals are perfectly substitutable across tasks, whereas the Gries-Naudé CES describes task substitution elasticity $\sigma$. The mixing of individual and task substitution requires more rigorous theoretical justification.
- **Improvement Directions**: (i) Assume parametric families (e.g., sigmoid or power laws) for $\phi_{\text{in}}, \phi_{\text{out}}, g, \rho(\kappa)$ to run counterfactual simulations; (ii) Use Brynjolfsson et al. (2025) data for model calibration to compare the fit of the original vs. revised Gries-Naudé model.

## Related Work & Insights
- **vs. Gries & Naudé (2022)**: Directly extends their partial equilibrium CES model by endogenizing $\gamma_{IT}, b_{IT}, N_{IT}$ as functions of organization, individual, and time.
- **vs. Acemoglu (2025)**: While Acemoglu calculates low TFP contributions by distinguishing "easy/hard to learn" tasks, this paper provides a micro-mechanism: it’s not just task difficulty, but organizational failure to delegate tasks even when they are learnable ($F<1$, $\Omega<1$).
- **vs. Dell'Acqua et al. (2023)**: Formalizes the "jagged frontier" empirical observation into production function primitives using $\phi(z, \kappa_i) + N_R$.
- **vs. Brynjolfsson et al. (2025)**: Extends the "novice gains most" finding via the inequality $\partial \phi_{\text{in}}/\partial \kappa_i \leq 0$, while noting the inverse $\partial \phi_{\text{out}}/\partial \kappa_i > 0$ in the unreliable zone.
- **Insight**: This framework provides an excellent checklist for studying any technology's economic impact: Consider $\Omega$ (deployment), $\phi$ (matching), $\lambda$ (learning), and $F$ (KPI adjustment).

## Rating
- Novelty: ⭐⭐⭐⭐ Mapping vague "organizational factors" into four modulators for a CES model is a clear framework contribution, though individual factors have been discussed in OB/IS literature.
- Experimental Thoroughness: ⭐⭐ As a position paper, it lacks experiments. Coverage of empirical literature is excellent, but the lack of calibration or counterfactuals makes the framework less falsifiable.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure (Background $\to$ Five Factors $\to$ Revised Model $\to$ Counter-arguments $\to$ Cases $\to$ Call to Action). Precise math with minimal intrusion.
- Value: ⭐⭐⭐⭐ A rare bridge between the ML community and productivity economics. High value as a checklist for organizations and policymakers estimating AI ROI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[CVPR 2026\] PositionIC: Unified Position and Identity Consistency for Image Customization](../../CVPR2026/image_generation/positionic_unified_position_and_identity_consistency_for_image_customization.md)
- [\[AAAI 2026\] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models](../../AAAI2026/image_generation/hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)

</div>

<!-- RELATED:END -->
