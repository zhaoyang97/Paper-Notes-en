---
title: >-
  [Paper Note] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation
description: >-
  [ICML 2026][LLM Agent][Hawkes Process] HawkesLLM grafts a multivariate Hawkes point process onto the LLM agent text simulation loop: Hawkes is responsible for scheduling "when and by which node the generation occurs" and…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Hawkes Process"
  - "Semantic Uncertainty"
  - "Agentic Text Simulation"
  - "Cascade Generation"
  - "Memory Selection"
date: 2026-05-08
content_hash: ec8ffb277e46cc4f
---

# HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation

**Conference**: ICML 2026  
**arXiv**: [2605.23043](https://arxiv.org/abs/2605.23043)  
**Code**: To be confirmed  
**Area**: LLM Agent / Text Simulation / Uncertainty  
**Keywords**: Hawkes Process, Semantic Uncertainty, Agentic Text Simulation, Cascade Generation, Memory Selection

## TL;DR
HawkesLLM grafts a multivariate Hawkes point process onto the LLM agent text simulation loop: Hawkes is responsible for scheduling "when and by which node the generation occurs" and "which historical node outputs are used as compressed memory," while the LLM is only tasked with writing the selected memory into the next event. It achieves late-stage semantic alignment that increases over time under a compact prompt budget on the GDELT Artemis II news cascade.

## Background & Motivation
**Background**: Current research on LLM uncertainty primarily focuses on single-turn generation, such as semantic entropy, black-box confidence estimation, and internal uncertainty awareness or tool-calling decisions of agents. These methods treat uncertainty as an "answer to the current question."

**Limitations of Prior Work**: In "read-while-write" agentic text simulation (e.g., news cascades, social media narratives, multi-step agent interactions), every previously generated text becomes part of the subsequent prompt. Early semantic ambiguity propagates along the trajectory. Single-step uncertainty measures fail to capture this path dependency, and long-context LLMs have been proven not to utilize all context uniformly; indiscriminately stacking history does not solve the problem.

**Key Challenge**: To stabilize subsequent generation, structured signals are needed to determine "which history should enter the prompt"; however, relying purely on the LLM's own selection is neither explainable nor controllable. While graph cascade models provide node structures, they lack text.

**Goal**: Decouple "temporal influence modeling" from "text generation"—the former determines when and who speaks and whom to look back at, while the latter handles only the language layer. Additionally, measure semantic uncertainty at the trajectory level rather than just at the end.

**Key Insight**: The authors observe that a multivariate Hawkes point process simultaneously provides two things—node incident intensity (deciding which node is active next) and the cumulative node-to-node excitation matrix (deciding whom to look back at). Fitting it to the event stream serves as a "visible memory scheduling signal."

**Core Idea**: Use a multivariate Hawkes process to drive node selection and prompt memory weighting, allowing the LLM to generate the next event based on "compressed memory selected according to temporal influence scores." This transforms the propagation of semantic uncertainty into a monitorable trajectory problem.

## Method

### Overall Architecture
The model models text propagation on a fixed directed graph $\mathcal{G}_0=(\mathcal{N},\mathcal{E})$, where each node is a "text generation agent." Each event $e_m=(\tau_m, n_m, x_m)$ consists of a timestamp, node index, and generated text. Starting from a seed event $e_0$, the following loops for $L$ steps:

1. Use the fitted Hawkes process on the current history $\mathcal{H}_t$ to perform Ogata thinning, sampling the next timestamp $\tau_t$ and node $n_t$;
2. Use the learned Hawkes excitation weights to score each candidate precursor node, applying threshold filtering + Top-$k$ to obtain the compressed memory $\mathcal{M}_t$;
3. Concatenate $(\tau_t, n_t)$, node style instructions $a_{n_t}$, the selected precursor texts, and their normalized weights into a prompt $p_t$;
4. The LLM samples $x_t \sim g_{\text{LLM}}(\cdot \mid p_t)$ and writes it into the history.

In this workflow, the LLM only generates text, while nodes, time, and memory are controlled by Hawkes; this completely separates "scheduling" from "verbalization" architecturally.

### Key Designs

1. **Multivariate Hawkes Temporal Influence Layer**:

    - **Function**: Estimates the incident intensity $\lambda_i(s)$ for each node and provides the cumulative influence of node $j$ on node $i$ via the "excitation matrix" $G_{j,i}=\alpha_{j,i}/\beta$.
    - **Mechanism**: The conditional intensity is written as $\lambda_i(s) = \mu_i + \sum_{(j,i)\in\mathcal{E}} \sum_{\tau_m<s, n_m=j} \phi_{j,i}(s-\tau_m)$, using an exponential kernel $\phi_{j,i}(u)=\alpha_{j,i} e^{-\beta u}$. A set of $\beta$ candidates is fixed, and for each $\beta$, $(\boldsymbol{\mu},\boldsymbol{\alpha})$ is maximized under log-likelihood with a contraction penalty $\eta\Omega(\boldsymbol{\alpha})$. The best stable fit is chosen based on likelihood and stability (spectral radius $\rho(\mathbf{G})$).
    - **Design Motivation**: Compared to neural TPPs, the parameterized Hawkes provides a readable $\alpha_{j,i}$ matrix and a single decay $\beta$, which can be directly exposed to the prompt as an explainable memory scheduling signal; this is the prerequisite for the controllability of the subsequent steps.

2. **Hawkes Memory Strategy (Node-level Top-$k$ Selection)**:

    - **Function**: For a sampled $(\tau_t, n_t)$, calculate which historical nodes should enter the prompt and their respective weights to form $\mathcal{M}_t$.
    - **Mechanism**: For each candidate precursor node $j$, let $r_t(j)$ be the index of its most recent active event, the node-level decay state be $h_{j,t}=\sum_{m<t,n_m=j} e^{-\hat{\beta}(\tau_t-\tau_m)}$, and the cumulative Hawkes contribution to the current node be $q_{j,t}=\hat{\alpha}_{j,n_t} h_{j,t}$. Negligible nodes are filtered using a raw threshold $\epsilon_{\text{raw}}$ and a normalized threshold $\epsilon_{\text{norm}}$, followed by a Top-$k$ selection for set $\mathcal{I}_t$. The final weights are $w_{j,t}=q_{j,t}/\sum_{\ell\in\mathcal{I}_t} q_{\ell,t}$. For each retained node, only one "latest" representative text is included, with weights written into the prompt as text annotations.
    - **Design Motivation**: The node-level excitation from Hawkes naturally provides continuous scores for "whom to look back at and how important they are"; node-level (rather than event-level) aggregation ensures prompt compactness, preventing LLMs from losing focus in long contexts; Top-$k$ and thresholds are engineering compressions, but the semantics originate from Hawkes itself, allowing the scheduling and language layers to be audited independently.

3. **Local/Global Drift Diagnosis + Local Semantic Alignment**:

    - **Function**: Decomposes "trajectory-level semantic uncertainty" into global drift relative to the seed and local drift relative to the weighted precursor center, measuring whether the process remains within the topic neighborhood using semantic alignment $S_t$ with a "locally retained reference set."
    - **Mechanism**: Use an embedding function $\mathbf{z}(\cdot)$ to compute $S_t = \cos(\mathbf{z}(x_t), \frac{1}{|\mathcal{R}_t|}\sum_{r\in\mathcal{R}_t} \mathbf{z}(r))$, where $\mathcal{R}_t$ is the set of real held-out texts for the same node within $\pm 12$ hours (relaxed to $\pm 24$ if necessary); simultaneously define $D_t^{\text{global}}=1-\cos(\mathbf{z}(x_t),\mathbf{z}(x_0))$ and $D_t^{\text{local}}=1-\cos(\mathbf{z}(x_t),\bar{\mathbf{z}}_t)$, where $\bar{\mathbf{z}}_t=\sum w_{j,t}\mathbf{z}(x_{r_t(j)})$ is the precursor center weighted by memory scores.
    - **Design Motivation**: Precise continuation is irrecoverable (multiple media outlets may write different headlines at the same moment), but semantic stability within a local neighborhood is comparable; decomposing uncertainty into "distance from seed" and "distance from memory just fed into the prompt" distinguishes between "long-range drift" and "local decoupling" failure modes.

### Loss & Training
The LLM side involves no fine-tuning and is simply called via Qwen2.5 / Ollama (temperature 0.35, top-p 0.9, max 75 new tokens). Training occurs only at the Hawkes layer: for a given $\beta$, maximize the log-likelihood $\ell_\beta(\boldsymbol{\mu},\boldsymbol{\alpha};\mathcal{D})=\sum_m \log\lambda_{n_m}(\tau_m) - \sum_i \int_0^T \lambda_i(s)ds$ with a contraction penalty $\eta\Omega(\boldsymbol{\alpha})$, selecting $\beta$ based on likelihood and stability. In held-out evaluation, Hawkes is re-fitted on the train segment (198 events).

## Key Experimental Results

### Main Results
The data is derived from the Artemis II reporting window on GDELT from 2026-04-01 to 11, resulting in 248 deduplicated English events spanning approximately 263 hours; nodes are defined as 5 curated media categories (local_tv / mass_market / specialist_science_tech / business_finance / general_news). The data is split temporally into 198 train / 50 test. Each generated event is matched with the real held-out headline set within $\pm 12$ h for the same node; all 62 generated events were successfully matched.

| Method | k | Mean $S_t$ | Trend | Late-stage $S_t$ |
|------|---|-----------|------|-----------------|
| **HawkesLLM** | 3 | **0.636** | **Increasing** | **0.682** |
| Chronological last-$k$ | 3 | 0.581 | Decreasing | 0.541 |
| Random-$k$ | 3 | 0.621 | Decreasing | 0.594 |

Ours is the only method where "semantic alignment increases over time," with the late-stage performance being approximately 14 percentage points higher than the nearest baseline.

### Ablation Study ($k$ Sensitivity)

| Method | $k$ | Mean $S_t$ | Trend | Late-stage $S_t$ |
|------|-----|-----------|------|-----------------|
| HawkesLLM | 3 | 0.635 | Increasing | **0.703** |
| HawkesLLM | 5 | 0.634 | Increasing | 0.703 |
| HawkesLLM | 7 | 0.634 | Increasing | 0.703 |
| Chronological | 3 | 0.578 | Decreasing | 0.497 |
| Chronological | 5 | 0.556 | Decreasing | 0.454 |
| Chronological | 7 | 0.694 | Flat | 0.636 |
| Random | 3 | 0.633 | Decreasing | 0.557 |
| Random | 5 | 0.597 | Decreasing | 0.537 |
| Random | 7 | 0.642 | Decreasing | 0.627 |

Drift diagnosis (average over multiple runs): Global drift $0.450\pm 0.019$, local drift $0.185\pm 0.072$. All runs satisfy global > local.

### Key Findings
- **The true sweet spot for Hawkes is in "compact budget + late-stage performance"**: When increasing $k$, HawkesLLM remains almost unchanged (since most steps only have fewer than 3 meaningful weighted neighbors), while chronological can raise the mean at $k=7$ but still loses to HawkesLLM in the late stage. This indicates that Hawkes provides structural gains through "whom to look back at" rather than simply "stuffing more content."
- **Global drift consistently exceeds local drift**: The trajectory slowly drifts away from the seed while remaining close to the memory just fed into the prompt at each step. This picture of "local stability + global accumulation" validates that path-dependent uncertainty must be measured hierarchically.
- **The only increasing alignment curve**: Baselines decrease over time while HawkesLLM increases, suggesting that structured memory scheduling delays semantic de-anchoring—a rare phenomenon in text-level chain-of-events systems.

## Highlights & Insights
- **Handing "when/who/lookback" entirely to a fittable and readable probabilistic model**: Rather than letting the LLM select history itself, Hawkes provides an explicit $\alpha_{j,i}$ matrix and decay $\beta$. The memory strategy thus becomes "reading + sorting + truncation," making all scheduling decisions auditable.
- **Node-level rather than event-level aggregation**: By including only the latest text for each retained node and compressing "temporal decay" into a "at most one per node" format, the system cleverly bypasses the attention dilution issues of long-context LLMs.
- **Drift decomposition can migrate to any agent loop**: As long as there are "seed text" and "weighted precursor center" anchors in the loop, global drift and local decoupling can be tracked simultaneously. This is a general tool for diagnosing multi-step agents, RAG, or self-dialogue.
- **Decoupling philosophy**: Replacing the scheduling layer with a neural TPP or graph diffusion model would not break other modules; similarly, changing the language layer to a stronger LLM does not require modifying Hawkes. This layering is more scalable in engineering than end-to-end "all-in-one" agents.

## Limitations & Future Work
- **Node classification uses a manually curated small vocabulary**: The 5 media categories were created specifically for the Artemis II task; particularly, the specialist_science_tech category is sparse in the test segment. Re-design is required for other topics or finer-grained nodes.
- **Evaluation embeddings share the same source as the generator**: $\mathbf{z}(\cdot)$ and $g_{\text{LLM}}$ are both Qwen2.5/Ollama, which may overestimate self-consistency. The authors suggest supplementing with independent embedding backends, human evaluation, and fact-checking.
- **Covers only headline-level text and a single news window**: This is a case study rather than a benchmark, with short text, few events, and only 3 repeated runs. Expansion to body-level, cross-topic, and cross-lingual scenarios is needed.
- **Time kernel is fixed as exponential**: A single global decay $\beta$ is consistent across all node pairs, which may mask heterogeneous influence durations; this could be replaced with multi-kernel or neural kernels for conditional decay.
- **Top-$k$ truncation + thresholds are engineering parameters**: Robustness scanning of hyperparameters was not performed; thresholds that are too strict might discard useful memory, while those that are too loose might break compactness. Calibration methods are required.

## Related Work & Insights
- **vs. Agent Uncertainty (Han et al. 2024 / Zhao et al. 2025)**: They treat uncertainty as a control signal for "whether tool-calling should occur in a single step"; Ours shifts the focus to the generation trajectory itself, concerning "how early ambiguity amplifies along the path," which is a perspective shift from the decision layer to the state layer.
- **vs. Semantic Entropy / Black-box Confidence (Farquhar et al. 2024, Lin et al. 2023)**: They compare multiple candidate answers for the same question; Ours measures alignment with local reference neighborhoods at each step of the sequence, focusing on trajectory drift rather than single-turn confidence.
- **vs. Neural Temporal Point Processes (Mei & Eisner 2017; Zuo et al. 2020)**: Neural TPPs are more expressive but have unreadable weights; Ours insists on parameterized Hawkes so that the "excitation matrix" can directly enter the prompt as an explainable scheduling signal, reflecting a trade-off of "auditability > expressiveness."
- **vs. Classic RAG (Lewis et al. 2020)**: RAG uses semantic similarity to retrieve external documents, with state maintained by the retriever; the "memory" in Ours comes entirely from its own generated history and is scheduled by temporal cascade dynamics rather than semantic similarity, effectively replacing the "retriever" with a "temporal influence estimator."
- **vs. LLM Social Simulation (Park et al. 2023; Sun et al. 2024)**: Those works focus on the emergence of multi-agent behavior with mostly heuristic scheduling; Ours provides a fittable probabilistic scheduling layer that can be embedded back into social simulations to replace ad-hoc rules.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly embedding multivariate Hawkes as a "memory scheduler" into the LLM agent loop and proposing accompanying local/global drift diagnosis is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐ Diagnostic evaluation only on the GDELT Artemis II single window + 3 runs + headline-level text; the authors acknowledge it is a case study. It is highly illustrative but does not constitute a full benchmark.
- Writing Quality: ⭐⭐⭐⭐ Uniform formula notation, complete algorithm blocks, and clear explanations of "what/why/evaluation mapping" with many reproducible details.
- Value: ⭐⭐⭐⭐ Methodological contribution to agents, multi-step generation, and news cascade simulation by decoupling scheduling from generation, exposing explainable influence matrices, and providing operational measures for path-dependent uncertainty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation](mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)
- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](../../AAAI2026/llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)
- [\[ACL 2026\] Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities](../../ACL2026/llm_agent/uncertainty_quantification_in_llm_agents_foundations_emerging_challenges_and_opp.md)
- [\[ACL 2026\] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](../../ACL2026/llm_agent/hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)

</div>

<!-- RELATED:END -->
