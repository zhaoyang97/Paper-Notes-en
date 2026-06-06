---
title: >-
  [Paper Note] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost
description: >-
  [ICML 2026][Image Generation][AI Adoption] This position paper argues that "introducing AI into an organization does not automatically equate to productivity gains." It identifies five human and environmental moderators…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AI Adoption"
  - "Productivity Paradox"
  - "Gries-Naudé Model"
  - "Organizational Factors"
  - "Learning Curves"
date: 2026-05-08
content_hash: 0098e6cfd9d5433b
---

# Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost

**Conference**: ICML 2026  
**arXiv**: [2605.24688](https://arxiv.org/abs/2605.24688)  
**Code**: None  
**Area**: AI Governance / Position Paper / AI Productivity Economics  
**Keywords**: AI Adoption, Productivity Paradox, Gries-Naudé Model, Organizational Factors, Learning Curves

## TL;DR
This position paper argues that "introducing AI into an organization does not automatically equate to productivity gains." It identifies five human and environmental moderators ignored by traditional economic models (personnel composition, individual baseline capability, learning curves, equitable use incentives, and goal flexibility). By incorporating organizational effectiveness $\Omega$, capability adjustment $\phi(z,\kappa_i)$, learning curve $\lambda_i(\tau)$, and effective automation threshold $\tilde N_{IT}$ into the Gries-Naudé (2022) partial equilibrium model, the authors derive a revised production function that explains why organizations with identical AI investments see vast disparities in output.

## Background & Motivation

**Background**: Following the explosion of Generative AI—ranging from GPT-3, Codex, and ChatGPT to agentic frameworks—enterprises and educational institutions are racing to deploy AI. In economics, scholars like Graetz-Michaels, Acemoglu, and Gries-Naudé have begun inserting AI directly as a "productivity factor" into aggregate output models, defaulting to the assumption that stronger AI capability leads to higher Total Factor Productivity (TFP).

**Limitations of Prior Work**: Empirical evidence does not support this linear extrapolation. Brynjolfsson et al. (2025) found that while AI improved novice performance by 36% in customer service, it had near-zero impact on experienced staff. Dell’Acqua et al. (2023) observed in management consulting that when tasks fall outside the "AI capability frontier," AI users' accuracy was 19 percentage points lower than non-users. Calvino-Fontanelli’s cross-country firm-level data shows that AI adoption is highly concentrated in already dominant large firms. Solow’s classic observation, "You can see the computer age everywhere but in the productivity statistics," is recurring as the "AI Productivity Paradox."

**Key Challenge**: Existing economic models treat parameters like $\gamma_{IT}(z)$, $b_{IT}$, and $A_{IT}/A_L$ as purely technical exogenous variables. However, the variables that actually determine whether AI capability can be monetized—organizational structure, individual capability, learning dynamics, usage incentives, and goal flexibility—are **endogenous human and environmental variables** that have been largely overlooked.

**Goal**: (1) Formally identify these five moderators and their mechanisms; (2) Modify the Gries-Naudé partial equilibrium model to explicitly include them in the production function; (3) Provide "actionable checklists" for researchers, practitioners, and policymakers, emphasizing the distinction between industrial and educational domains.

**Key Insight**: The authors do not deny that AI capabilities are increasing. Instead, they shift the perspective from "how powerful the AI model is" to "how the organization utilizes AI." AI capability is a necessary but insufficient condition; the five organizational/individual factors serve as the "transmission ratio" that translates capability into output.

**Core Idea**: In short—**True AI Productivity = Technical Capability × Organizational Effectiveness $\Omega$ × Capability-Task Fit $\phi$ × Learning Progress $\lambda$ × Goal Flexibility $F$**. Ignoring the latter four factors makes any ROI estimation for AI investment misleading.

## Method

As a position paper, its "method" refers to the revised framework of the Gries-Naudé model rather than an algorithmic procedure. The original model describes how each task $z$ on the interval $[N-1, N]$ is produced by standard labor or IT services (including AI), aggregated by CES into total human-service output $H$, with an automation threshold $N_{IT}$ distinguishing tasks where AI can intervene.

### Overall Architecture
The authors qualitatively identify five moderators and provide five corresponding revisions: organizational effectiveness $\Omega$, capability-adjusted productivity $\tilde\gamma_L, \tilde\gamma_{IT}$, learning curve $\lambda_i(\tau)$, and endogenous task boundary $\tilde N_{IT}$. These are synthesized into a revised human-service production function $\tilde h_i(z,\tau)$ across "individual-task-time" dimensions, which is then CES-aggregated to the sub-organizational level $\tilde H(\tau)$.

### Key Designs

1.  **Organizational Effectiveness Correction $\Omega = \omega_C \cdot \omega_I$**:
    *   **Function**: Distinguishes between "absolute number of AI experts $b_{IT}$" and "effective expert accessibility $\tilde b_{IT}$" to characterize internal friction.
    *   **Mechanism**: Defines $\omega_C \in [0,1]$ as organizational structural alignment (flat AI-integrated teams approach 1, deep hierarchies approach 0) and $\omega_I \in [0,1]$ as incentive alignment. If only a few receive "AI transformation" rewards, motivation collapses due to asymmetric competition. The discount factor is $\tilde b_{IT}(z) = \Omega \cdot b_{IT}(z)$.
    *   **Design Motivation**: Explains the Calvino-Fontanelli observation—"strong firms gain more from AI" is not just about having more experts, but about structural designs that allow experts to reach task executors.

2.  **Capability-Task Interaction Function $\phi(z,\kappa_i)$ and Reliability Frontier $N_R$**:
    *   **Function**: Captures Dell’Acqua’s "jagged frontier"—where the same AI tool has opposite effects on the same person across different tasks.
    *   **Mechanism**: Introduces individual baseline capability $\kappa_i \in [0,1]$. Tasks $[N-1, N_{IT}]$ are divided into a "reliable zone" $[N-1, N_R]$ and an "unreliable zone" $(N_R, N_{IT}]$. In the reliable zone, $\partial \phi_{\text{in}}/\partial \kappa_i \leq 0$ (novices gain more); in the unreliable zone, $\partial \phi_{\text{out}}/\partial \kappa_i > 0$ (only high-ability users can identify and correct AI errors).
    *   **Design Motivation**: A single $\gamma_{IT}/\gamma_L$ ratio cannot describe non-monotonic heterogeneity (novices surge, experts decline). A piecewise function endogenizes this relationship.

3.  **Learning Curve $\lambda_i(\tau) = 1 - e^{-\rho_i \tau}$ and Flexible Threshold $\tilde N_{IT}$**:
    *   **Function**: Integrates "AI gain evolution over time" and "organizational goal rigidity" into the framework.
    *   **Mechanism**: Uses time since adoption $\tau$ as a comparative static parameter. Individual learning rate $\rho_i$ is heterogeneous (risk of Matthew effect). Effective AI task productivity becomes $\tilde\gamma_{IT}(z,\kappa_i,\tau) = \gamma_{IT}(z) \cdot \phi(z,\kappa_i) \cdot \lambda_i(\tau)$. Organizational flexibility $F \in [0,1]$ shrinks the technical threshold $N_{IT}$ to an effective threshold $\tilde N_{IT} = (1-F)(N-1) + F \cdot N_{IT}$.
    *   **Design Motivation**: Whereas the original $N_{IT}$ was purely technical, $F$ links it to "leadership's willingness to rearrange KPIs based on AI capabilities," explaining why different organizations with the same tech stack show vastly different AI penetration.

Combining these components yields the core revised equation (Eq. 9):

$$\tilde h_i(z,\tau) = \begin{cases} \tilde\gamma_L(z,\kappa_i) l_i(z) A_L + \tilde\gamma_{IT}(z,\kappa_i,\tau) \cdot \Omega \cdot b_{IT}(z) A_{IT} D, & z \in [N-1, \tilde N_{IT}] \\ \tilde\gamma_L(z,\kappa_i) l_i(z) A_L, & z \in (\tilde N_{IT}, N] \end{cases}$$

The total sub-organizational output is obtained via CES aggregation: $\tilde H(\tau) = \big( \int_{N-1}^N (\sum_i \tilde h_i(z,\tau))^{(\sigma-1)/\sigma} dz \big)^{\sigma/(\sigma-1)}$.

### Mechanism
The argumentation strategy involves: (i) Citing empirical evidence (Brynjolfsson, Dell’Acqua, Calvino-Fontanelli, Acemoglu) as "counter-examples to the status quo"; (ii) Performing a minimal invasive revision on the Gries-Naudé mathematical skeleton; (iii) Applying the framework to industry and education cases; (iv) Directly addressing three types of counter-arguments (technological determinism, measurement issues, wage costs) in Section 4.

## Key Experimental Results

As a position paper, this section provides an "Empirical Evidence × Framework Correspondence" mapping.

### Mapping Empirical Evidence to Framework Factors

| Empirical Phenomenon | Source | Corresponding Term in Framework |
| :--- | :--- | :--- |
| Novices +36%, Experts near zero | Brynjolfsson et al. 2025 | $\phi_{\text{in}}(\kappa_i)$ decreases with $\kappa_i$ |
| Accuracy -19pp outside capability frontier | Dell'Acqua et al. 2023 | $\phi_{\text{out}}(\kappa_i)$ increases with $\kappa_i$; $N_R < N_{IT}$ |
| AI adoption concentrated in large/strong firms | Calvino-Fontanelli 2023 | High $\Omega = \omega_C \omega_I$ |
| TFP growth far below expectations | Acemoglu 2025 | Overall $F \cdot \Omega \cdot \lambda$ is far less than 1 |
| ChatGPT raises grades but not self-efficacy | Deng et al. 2025 | Mismatch between $F$ and $\omega_I$ in education |

### Original Gries-Naudé Determinants vs. Ours (Revised)

| Original Determinant | Our Revision |
| :--- | :--- |
| Automation threshold $N_{IT}$ | Effective threshold $\tilde N_{IT}$, determined by flexibility $F$ |
| Productivity ratio $\gamma_{IT}(z)/\gamma_L(z)$ | Individual-time dependent $\tilde\gamma_{IT}/\tilde\gamma_L = \gamma_{IT}\phi\lambda / (\gamma_L g)$ |
| Expert availability $b_{IT}$ | Effective availability $\tilde b_{IT} = \Omega \cdot b_{IT}$ |
| Relative capability ratio $A_{IT}/A_L$ | Modulated by $\propto \phi(z,\kappa_i) \lambda_i(\tau) / g(\kappa_i)$ |

### Key Findings
*   **Productivity Ratio Monotonicity Constraint**: The authors require $\tilde\gamma_{IT}/\tilde\gamma_L$ to be monotonic over $z$ for fixed $\kappa_i, \tau$ to ensure $N_R$ and $\tilde N_{IT}$ are uniquely defined—a hidden prerequisite for model consistency.
*   **Aggregation Granularity**: Aggregating at the team/department level rather than the whole firm allows for manageable distributions (avoiding "10x engineer" heavy tails) while preserving individual heterogeneity.
*   **Industry vs. Education**: In industry, high $F$ (KPI flexibility) yields large AI gains, but beneficiaries drift from employees to management. In education, the student is the beneficiary, so the "productivity for whom" tension disappears, but a mismatch between $F$ and $\omega_I$ can directly harm learning goals.

## Highlights & Insights
*   **Minimal Invasive Revision**: By only overlaying four modulation terms ($\Omega, \phi, \lambda, F$) onto the original $\gamma, b_{IT}, N_{IT}$, the authors maintain compatibility with prior qualitative conclusions. This "patch, don't rewrite" approach is highly effective for acceptance in economic communities.
*   **Countering "Technological Determinism"**: Instead of denying AI progress, the authors argue that as AI grows stronger ($N_R$ moves up), the bottleneck merely shifts to $F$ and $\omega_I$. This incorporates the opponent's argument into a larger framework.
*   **Political Economy Dimension**: Many AI papers ignore "productivity for whom." This paper forces a distinction between industrial efficiency and educational outcomes, highlighting the drift of benefits.
*   **Transferable Design**: The $\Omega, \phi, \lambda, F$ modulators are applicable to any "new technology adoption" economic model, including cloud computing or remote work.

## Limitations & Future Work
*   **Limitations**: Functional forms for $\phi, g, \rho$ are kept as "general forms" to accommodate cross-domain differences, but this means the framework lacks falsifiable predictions and behaves more like a narrative tool.
*   **Abstract Baseline Capability $\kappa_i$**: Defining "baseline capability" (coding skill? critical thinking?) is difficult to operationalize and is left for organizational behavior researchers.
*   **Lack of Direct AI Cost Endogenization**: While license fees and compute costs are acknowledged, they are excluded from the core framework, meaning "profitability" decisions require external valuation.
*   **Aggregation Conflicts**: Summing over $i$ before CES aggregation assumes perfect substitutability between individuals on tasks, which may conflict with the CES substitution elasticity $\sigma$ between tasks.
*   **Future Work**: (i) Propose parametric families (e.g., sigmoid or power laws) for $\phi_{\text{in}}, \phi_{\text{out}}, g, \rho(\kappa)$ for counterfactual simulations; (ii) Perform model calibration using the Brynjolfsson 2025 dataset to compare the original vs. revised Gries-Naudé fit.

## Related Work & Insights
*   **vs. Gries & Naudé (2022)**: This paper extends their partial equilibrium model by endogenizing technical exogenous variables into functions of organization, individual, and time.
*   **vs. Acemoglu (2025)**: Acemoglu attributes low TFP gain to "easy vs. hard tasks." This paper provides the micro-mechanism: it's not task difficulty, but organizational failure to delegate automatable tasks ($F < 1$).
*   **vs. Dell’Acqua et al. (2023)**: Formalizes the empirical "jagged frontier" as $\phi(z, \kappa_i)$ and $N_R$, elevating empirical terminology to theoretical primitives.
*   **Insight**: Any study claiming "Technology X brings Growth Y" should first answer the four questions: $\Omega$ (Can the organization land it?), $\phi$ (Can users match it?), $\lambda$ (How long to learn?), and $F$ (Do KPIs allow it?).

## Rating
*   Novelty: ⭐⭐⭐⭐ Systematically integrating organizational factors into a CES model is a clear framework-level contribution, even if individual factors are known in OB/IS literature.
*   Experimental Thoroughness: ⭐⭐ As a position paper, it lacks internal experiments and relies on external calibration. Falsifiability is currently weak.
*   Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure and precise use of minimal math to communicate deep economic concepts.
*   Value: ⭐⭐⭐⭐ A rare bridge between the ML community and productivity economics; provides a necessary checklist for stakeholders estimating AI ROI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[AAAI 2026\] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models](../../AAAI2026/image_generation/hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)

</div>

<!-- RELATED:END -->
