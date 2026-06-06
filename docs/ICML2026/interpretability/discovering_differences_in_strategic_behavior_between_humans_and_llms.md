---
title: >-
  [Paper Note] Discovering Differences in Strategic Behavior Between Humans and LLMs
description: >-
  [ICML 2026][Interpretability][Behavioral Game Theory] This paper utilizes AlphaEvolve (an LLM-based program synthesis framework) to "evolve" interpretable Python behavioral models directly from behavioral data. By compar…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Behavioral Game Theory"
  - "AlphaEvolve"
  - "Program Synthesis"
  - "Iterated Rock-Paper-Scissors"
  - "Opponent Modeling"
date: 2026-05-08
content_hash: 12e3be250ebe2ff8
---

# Discovering Differences in Strategic Behavior Between Humans and LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.10324](https://arxiv.org/abs/2602.10324)  
**Code**: Not yet released  
**Area**: Interpretability / Behavioral Game Theory / LLM Evaluation  
**Keywords**: Behavioral Game Theory, AlphaEvolve, Program Synthesis, Iterated Rock-Paper-Scissors, Opponent Modeling

## TL;DR
This paper utilizes AlphaEvolve (an LLM-based program synthesis framework) to "evolve" interpretable Python behavioral models directly from behavioral data. By comparing humans with frontier LLMs in Iterated Rock-Paper-Scissors (IRPS), the study finds that Gemini 2.5 Pro/Flash and GPT 5.1 significantly outperform humans in both win rates and "opponent modeling" dimensions, whereas GPT OSS 120B exhibits deteriorating performance over time.

## Background & Motivation

**Background**: LLMs are increasingly deployed in social interaction scenarios (negotiations, customer service, virtual companions) and are utilized as "cheap humans" for behavioral simulation in social science and market research. Distinguishing LLM behavior from human behavior is a traditional topic in Behavioral Game Theory (BGT)—mainstream methods either report aggregate statistics (win rates, first-hand distributions) or manually design mathematical models with parameters (e.g., EWA, Sophisticated EWA, Cognitive Hierarchy) to fit human behavior.

**Limitations of Prior Work**: Aggregate statistics describe trends but cannot explain underlying mechanisms. Manual BGT models are designed for humans, and their priors (e.g., "Rock preference for the first hand," "win-stay-lose-shift") may not effectively characterize the non-human behaviors of LLMs. On the other hand, black-box neural network models fit data well, but the interpretability cost is pushed to the model decoding stage, failing to provide direct "mechanistic hypotheses."

**Key Challenge**: A trade-off exists between interpretability and fitting capability—traditional BGT formulas are readable but have low capacity and are biased toward humans; neural networks have high capacity but are unreadable. To "fit LLMs while directly discovering mechanistic differences," it is necessary to break through the "human-predefined mathematical templates."

**Goal**: (1) Openly search a **program space** rather than a predefined family of formulas, allowing data to select the most concise structure that characterizes the agent (human/LLM); (2) Compare the strategic behaviors of humans and multiple frontier LLMs under a unified interpretable representation, pinpointing "which mechanisms cause the differences" at a structural level.

**Key Insight**: The "behavioral model" of BGT is rewritten as a Python function with a fixed signature: `agent(params, choice, opp_choice, reward, state) → (logits, state)`. AlphaEvolve (LLM-driven program evolution) performs an outer-loop search over the program space, combined with SGD in the inner-loop for parameter fitting, constructing a bi-level optimization process.

**Core Idea**: The process of "finding the behavioral model that best fits the agent" is transformed from "manually writing formulas" to "LLM-based program evolution." Multi-objective Pareto optimization (likelihood + program simplicity) is used to filter for the "simplest but best" programs as mechanism hypotheses, followed by a comparison between humans and LLMs at the program text level.

## Method

### Overall Architecture
The end-to-end workflow consists of four steps: (1) **Unified Behavioral Model Interface** — Any BGT model is implemented with the same Python function signature (Inputs: previous action, opponent action, reward, internal state; Outputs: next action distribution, new state); (2) **Data Collection** — The IRPS dataset of 411 humans / 129,087 choices from Brockbank & Vul (2024) is reused, and alignment data is collected for 4 LLMs (20 games × 15 bots × 300 rounds = 90,000 choices per LLM); (3) **AlphaEvolve Bi-level Optimization** — The outer layer uses an LLM (Gemini 2.5 Flash) to evolve program structures, while the inner layer uses SGD to fit parameters $\theta$, targeting the multi-objective fitness `(likelihood, -Halstead_effort)`; (4) **SBB Selection + Cross-Agent Generalization** — The "simplest but best" programs are selected along the Pareto frontier as mechanistic hypotheses, and agent similarities are quantified via a cross-generalization matrix.

The IRPS setting is fixed at 300 rounds against 15 bots (including non-adaptive transition-based and adaptive following bots), with rewards of +3 for a win, 0 for a draw, and -1 for a loss. Data for each agent (Human, Gemini 2.5 Pro/Flash, GPT 5.1, GPT OSS 120B) is processed through the same AlphaEvolve pipeline to ensure fair comparison.

### Key Designs

1. **AlphaEvolve for Evolving Interpretable Behavioral Programs**:

    - **Function**: Directly searches the program space for the Python function that "best predicts the agent's next move," providing both mechanism readability and fitting performance.
    - **Mechanism**: Starting from a template program equivalent to the Nash equilibrium (outputting uniform logits), the LLM is provided with parent programs, historical samples, and corresponding fitness scores in each generation to "propose modifications for higher scores." Parameters $\theta$ for each candidate are fitted via SGD maximizing likelihood $\arg\max_\theta \sum_t \log \hat{p}_{\theta}(a_t \mid h_t)$. The normalized likelihood from two-fold cross-validation serves as the performance axis, forming a bi-level outer-program / inner-parameter optimization. Compared to FunSearch + BIC, this work employs the stronger AlphaEvolve and replaces BIC with Halstead effort as the complexity metric (as parameter and data sizes are uniform, BIC degenerates).
    - **Design Motivation**: BGT research has long been constrained by "researchers only testing conceptualized formulas." Since frontier LLMs have absorbed vast behavioral science knowledge, they serve as excellent "hypothesis generators." Using programs instead of formulas or neural networks maintains Turing-complete expressivity while preserving readable structures (conditional branches, Q-tables, opponent frequency tables), eliminating the need for secondary explanation.

2. **Multi-objective Pareto + SBB Selection Rule**:

    - **Function**: Extracts representative programs from candidate generations that are both predictive and concise enough to serve as mechanism hypotheses.
    - **Mechanism**: Fitness tracks both cross-validated likelihood $\ell(\phi)$ and negative Halstead effort $s(\phi)$. Non-dominated solutions form the Pareto Frontier $\mathrm{PF}(\hat{\Phi})$. The Simplest-But-Best (SBB) is defined as: $\mathrm{SBB}(\epsilon) \in \arg\max_{\phi \in \mathrm{PF}(\hat{\Phi})} \{ s(\phi) \mid \ell(\phi) > \max_{\phi'} \ell(\phi') - \epsilon \}$, with $\epsilon = 0.005$. This "select the simplest while likelihood is nearly maintained" approach embeds Occam's razor directly into the selection rule.
    - **Design Motivation**: The programs that truly reveal mechanisms are often those near the "elbow" of the Pareto frontier rather than the best-fit programs, which are frequently bloated. This rule transforms qualitative mechanism interpretation into a reproducible algorithmic step.

3. **Cross-Agent Alignment & Cross-Generalization Matrix**:

    - **Function**: Collects aligned human and LLM data in identical IRPS environments and quantifies mechanism similarity via cross-fitting.
    - **Mechanism**: The 4 LLMs utilize the same 15 bots, reward matrices, and 300-round lengths as the human dataset; prompts are adapted from human subject instructions to avoid environment-driven variance. A $5 \times 5$ cross-generalization matrix $M_{ij}$ is constructed, where the $i$-th row and $j$-th column represent the likelihood achieved by fitting agent $j$'s SBB program to agent $i$'s data. Diagonal dominance confirms that SBB programs capture agent-specific characteristics.
    - **Design Motivation**: Aligned datasets eliminate the interference of "behavioral differences due to envionment," converging differences on the "strategic mechanisms of the agents." The cross-matrix statistically demonstrates that models designed for humans systematically fail to predict LLM behavior, validating that LLMs are not cheap human substitutes.

### Loss & Training
Inner-parameter optimization uses SGD implemented in JAX to maximize the negative NLL. Outer-program evolution is conducted by AlphaEvolve via island-based evolution on multi-objective fitness, run independently 3 times per dataset. Baselines include Nash equilibrium, Contextual Sophisticated EWA (CS-EWA, which maintains independent attraction vectors based on joint history of length L=2), and a GRU-based RNN with full hyperparameter search.

## Key Experimental Results

### Main Results
**Win Rate Comparison** (Average win rate against 15 bots over 300 rounds):

| Agent | Avg Win Rate (Non-adaptive Bots) | Convergence Speed | Gap with Oracle |
|--------|------|------|------|
| Random Baseline | ~0% (Zero-sum) | N/A | Large |
| Human | Medium | Slow | Medium |
| Gemini 2.5 Flash / Pro | Significantly higher than human | Fast (Approaches oracle in tens of rounds) | Small |
| GPT 5.1 | Significantly higher than human | Fast | Small |
| GPT OSS 120B | Near human levels initially, **decreases over time** | Reverse | Increasing |

**Behavioral Model Quality** (Normalized likelihood gain over Nash; two-fold cross-validation):

| Model | Human | Gemini 2.5 Pro | Gemini 2.5 Flash | GPT 5.1 | GPT OSS 120B |
|------|------|----------------|------------------|---------|--------------|
| Nash | 0 (Base 1/3) | 0 | 0 | 0 | 0 |
| CS-EWA | Significant gain | Significant gain | Significant gain | Significant gain | Significant gain |
| GRU-RNN | Comparable to AlphaEvolve | < AlphaEvolve | < AlphaEvolve | < AlphaEvolve | Comparable to AlphaEvolve |
| **AlphaEvolve** | **Comparable/Slightly better** | **Better than RNN** | **Better than RNN** | **Better than RNN** | **Comparable to RNN** |

All comparisons of AlphaEvolve vs. CS-EWA yielded $p < 0.001$ (Wilcoxon signed-rank + Bonferroni correction, $Z \le -7.148$).

### Ablation Study
Comparison of SBB program "mechanism configurations":

| Agent | Q-table Dimension | Opponent Model Dimension | Choice Stickiness |
|--------|----------------------|--------------|----------|
| Human | 3×3×3 ($Q(a_t, a^o_{t-1}, a_{t-1})$) | 1D (Frequency only) | Yes |
| Gemini 2.5 Flash | 3×3×3 | **3×3** (Conditional on previous) | Yes |
| Gemini 2.5 Pro | 3×3×3 | **3×3** + Counterfactual updates | Yes |
| GPT 5.1 | 3×3×3 | **3×3×3** (Highest dimension) | No |
| GPT OSS 120B | **1D** ($Q(a_t)$, weakest) | 1D | Yes |

### Key Findings
- **Dimension is Capability**: The strategic advantage of frontier LLMs is localized to higher-dimensional opponent modeling matrices. Gemini 2.5 Pro further introduces counterfactual reward updates (updating values for unchosen actions), a mechanism absent in human SBB programs.
- **GPT OSS 120B as Negative Control**: It is the only agent where the Q-table degenerated to 1D and performance worsened over time, consistent with the phenomenon that "weaker LLMs fail strategic synthesis in long contexts." This suggests opponent modeling is an emergent capability of model scale/power.
- **Humans are Poor Predictors of LLMs and Vice-Versa**: The cross-generalization matrix shows that while Gemini 2.5 Pro/Flash and GPT 5.1 predict each other well, using LLM-derived programs to predict humans (or vice-versa) results in significant performance drops ($p < 0.001$), falsifying the "LLMs as cheap human substitutes" assumption.
- **All Agents are Level-1 Players**: No SBB program exhibited recursive "the opponent is modeling me" structures. Consequently, against level-1 adaptive bots, all agents struggled, with win rates falling near random levels.

## Highlights & Insights
- **Advancing BGT to "Data-Driven Program Synthesis"**: While traditional BGT depends on manual formulas, this study proves LLM + evolution can automatically produce readable Python behavioral models that match or exceed RNNs in fitting quality while retaining mechanistic interpretability.
- **Algorithmic Occam's Razor via SBB(ε)**: This rule formalizes the subjective selection of "mechanism representatives," making structural comparisons across different agents reproducible.
- **Localizing Structural Differences to Scalar Dimensions**: The human-LLM strategic gap is reduced to a concrete structural variable: "the tensor dimension of the opponent model," which is far more measurable than vague "intelligence" conclusions and provides a lever for alignment.
- **New Supplement for CoT Monitoring**: This method does not rely on reasoning traces (which may be unfaithful to behavior), serving as a valuable complement to CoT-based monitoring in AI safety.

## Limitations & Future Work
- **Single Game Validation**: IRPS is simple; transferability to multi-player or asymmetric information games (e.g., Diplomacy, negotiation) remains unverified.
- **Average Human Behavior**: Human data is aggregated across 411 subjects, potentially masking individual differences (e.g., human experts might possess higher-dimensional models).
- **Mechanism Hypothesis $\neq$ Mechanism Itself**: An SBB program is an acting hypothesis; it does not prove the LLM internally maintains specified tables. Verification using mechanistic interpretability (probes, logit lens) is necessary.
- **LLM-driven Evolution Bias**: Using Gemini as a generator to study Gemini may introduce inductive biases, necessitating tests with different generator LLMs.

## Related Work & Insights
- **vs. Fan et al. (2024)**: They found LLMs were inferior to humans in IRPS; this study reverses that with newer LLMs and longer horizons (300 rounds), highlighting a generational jump in capability.
- **vs. Castro et al. (2025)**: This work advances "program synthesis for cognitive modeling" by using AlphaEvolve and multi-objective (Likelihood, Halstead) fitness.
- **vs. Traditional BGT**: AlphaEvolve significantly outperforms the CS-EWA baseline and handles non-human structures.
- **vs. RNN Benchmarks**: AlphaEvolve provides readable functions and overcomes the interpretability-performance trade-off, outperforming RNNs on LLM-generated data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](../../AAAI2026/interpretability/elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICML 2026\] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)

</div>

<!-- RELATED:END -->
