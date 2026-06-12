---
title: >-
  [Paper Note] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
description: >-
  [ICML 2026][LLM Reasoning][Latent Chain-of-Thought] The authors treat latent CoT as an intervenable Structural Causal Model (SCM), performing step-wise `do`-intervention + early-stop decoding + teacher-forced readout for…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Latent Chain-of-Thought"
  - "Causal Intervention"
  - "do-intervention"
  - "Structural Causal Model"
  - "Coconut"
  - "CODI"
date: 2026-05-08
content_hash: ecb81e3f38280c4c
---

# Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure

**Conference**: ICML 2026  
**arXiv**: [2602.08783](https://arxiv.org/abs/2602.08783)  
**Code**: https://github.com/J1mL1/causal-latent-cot  
**Area**: LLM Reasoning / Interpretability  
**Keywords**: Latent Chain-of-Thought, Causal Intervention, do-intervention, Structural Causal Model, Coconut, CODI

## TL;DR
The authors treat latent CoT as an intervenable Structural Causal Model (SCM), performing step-wise `do`-intervention + early-stop decoding + teacher-forced readout for each continuous "thought step." They systematically quantify the step-level necessity, propagation structure, and trajectory superposition of Coconut/CODI in mathematical and commonsense reasoning. The study discovers that latent steps are not homogenized "deepening" but represent a structured interface characterized by high heterogeneity, non-local routing, and output-layer commitment preceding representation-layer commitment.

## Background & Motivation
**Background**: While explicit CoT achieves significant results in reasoning tasks such as GSM8K and CommonsenseQA, it suffers from high decoding costs, verbose outputs, and potential post-hoc rationalization rather than reflecting true model computation. To alleviate these issues, methods like Coconut, CODI, and Sim-CoT replace "token-by-token thinking" with $T$ steps of latent thinking (latent CoT) in a continuous representation space, using the last-layer hidden state as the next step's input before finally decoding the answer.

**Limitations of Prior Work**: Intermediate computations in latent CoT are no longer discrete, readable, or editable tokens. Traditional interpretability methods for explicit CoT, such as "deleting paragraphs / modifying rationales / shuffling," become entirely ineffective. Existing evaluations mostly rely on probes to observe correlations, but high probe activation may reflect that the "model is actually using this step" or simply that "information happens to be linearly separable."

**Key Challenge**: Correlation-based methods cannot answer whether a specific step has a causal effect on the final answer, nor can they characterize how information flows between $T$ steps or distinguish whether an early output bias toward "Yes" implies representational collapse. Essentially, a causal framework at the intervention level is missing.

**Goal**: To establish a step-resolved causal evaluation for latent CoT to answer three questions: (RQ1) Which latent steps are causally necessary for correctness, and at which step does the answer become decodable; (RQ2) How do influences propagate between steps and whether they approximate a chain-like structure as in explicit CoT; (RQ3) Whether intermediate trajectories simultaneously retain multiple candidate answer patterns and the gap between output-layer commitment and representation-layer commitment.

**Key Insight**: Treat each latent state $h_t$ as a variable in an SCM. By performing $do(h_t \leftarrow \tilde h_t)$ to directly overwrite it and recomputing downstream transitions, combined with teacher-forced readout to quantify effects, classical causal mediation analysis can be naturally extended to continuous thinking trajectories.

**Core Idea**: Use a unified "intervention + readout" protocol to recast latent CoT from a "black-box depth" into a "manipulable causal system," thereby placing the three questions of step-level necessity, propagation structure, and superposition-commitment into a single reproducible experimental framework.

## Method

### Overall Architecture
The evaluation protocol, as shown in Figure 2, models a minimal SCM:

$$\text{(Hidden)}\; H_t = f_t(H_{<t}, x, \epsilon_t; \theta), \quad t=1,\dots,T; \qquad \text{(Output)}\; Y = g(H_{1:T}, x, \epsilon_y; \theta).$$

For a given prompt $x$, standard propagation provides a baseline trajectory $h_{1:T}$ and answer $y$. Based on this, three types of counterfactuals are constructed:

1.  **Early-stop decoding**: Truncate latent computation at step $k$ and decode directly from $h_k$ to detect the earliest point the answer is readable.
2.  **Single-step intervention**: Perform $do(h_t \leftarrow \tilde h_t)$ (the paper uses zero intervention $\tilde h_t = \mathbf 0$), then re-forward while keeping downstream $f_{t'>t}$ unchanged to obtain $\tilde y^{(t)}$.
3.  **Intervention + Early-stop readout**: Intervene at $t$ and use teacher forcing to read out at $s > t$, obtaining a pair of distributions $p_{\text{base}}^{(s)}$ and $p_{\text{do}(t)}^{(s)}$. The KL divergence between them measures the influence intensity from $t \to s$.

The experiments focus on Coconut and CODI latent reasoning paradigms across GPT-2, Llama3-1B, and Qwen3-4B-Instruct. Datasets include GSM8K and CommonsenseQA, with StrategyQA used for binary Yes/No classification in RQ3.

### Key Designs

1.  **Step-wise `do`-intervention + Flip rate (RQ1: Necessity and early-stop decodability)**:
    - **Function**: Measures the causal necessity of each latent step for the final decision and locates where the answer is "locked."
    - **Mechanism**: For each sample, a baseline trajectory and a counterfactual trajectory (where only $h_t$ is zeroed) are run, with downstream transitions and readout remaining unchanged. The proportion of samples with flipped predictions $\mathrm{Flip}(t)$ is used as an indicator of decision dependency for step $t$. For early-stop decoding, the earliest correct step $k_i = \min\{k: \hat y_i^{(\le k)} = y_i^*\}$ and the cumulative solve rate $S(k) = \frac{1}{N}\sum_i \mathbb{1}\{k_i \le k\}$ are defined.
    - **Design Motivation**: Unlike correlation metrics like probes, single-step intervention directly measures if the prediction changes without that step, providing clean counterfactual necessity. Zero intervention is chosen for its stability across different backbones. Distinguishing "necessity" (intervene) from "sufficiency" (early stop) prevents misinterpreting early readability as the subsequent steps being useless.

2.  **Influence matrix $W_{t,s}$ + Principal Influence Graph (RQ2: Propagation and Routing)**:
    - **Function**: Characterizes how perturbations at step $t$ reach step $s$ as a directed weighted graph, comparing it with explicit CoT to see if latentization changes the internal "reasoning circuitry."
    - **Mechanism**: Under the intervention + teacher-forced readout protocol, sample-level position-averaged KL is defined as $\mathrm{KL}^{(i)}_{t\to s} = \frac{1}{|y_i^*|} \sum_u \mathrm{KL}(p_{\text{base}}^{(s)}(\cdot\mid y^*_{i,<u}) \| p_{\text{do}(t)}^{(s)}(\cdot \mid y^*_{i,<u}))$. Taking the expectation yields $W_{t,s} = \mathbb E_i[\mathrm{KL}^{(i)}_{t\to s}]$. The principal influence graph is obtained by sparsification using a threshold $\alpha = 0.1 \cdot \max(W)$ and keeping the top-1 outgoing edge per node. Structural metrics like locality, span, early-out, and late-in are calculated on normalized $W$.
    - **Design Motivation**: Unlike point-wise flip rates, the influence matrix distinguishes whether a step is intrinsically critical or acts as a relay for long-distance routing. Using teacher-forced readout instead of sampling suppresses noise, ensuring $W_{t,s}$ reflects propagation effects rather than temperature fluctuations.

3.  **Superposition score + Dual-readout comparison (RQ3: Representational vs. Output Commitment)**:
    - **Function**: Determines if intermediate steps simultaneously retain Yes/No candidate answers and the relative time lag between "output layer bias" and "representational collapse."
    - **Mechanism**: Perform $K$ random rollouts for the same prompt on StrategyQA, keeping only "dual-mode" prompts where both Yes and No appear. At each latent step $t$, estimate probabilities $p_Y(t), p_N(t)$ using: (i) teacher-forced template scoring, and (ii) a light probe trained on frozen latents. The superposition score $\mathrm{SS}(t) = \min(p_Y(t), p_N(t))$ measures the degree of superposition.
    - **Design Motivation**: Relying solely on teacher-forced readout creates an illusion that the model commits to an answer very early. Conversely, probes might overestimate "hesitation" in intermediate steps. Comparing both readouts separates distributional early bias (output commitment) from whether the alternative answer is still decodable in the representation (representational commitment).

### Loss & Training
The paper does not train new models but uses official CODI weights and a reproduction of Coconut across three backbones; all analyses are inference-only. Probes are small linear classifiers trained on frozen latents for RQ3.

## Key Experimental Results

### Main Results: Step-level Necessity and Early-stop Decodability (RQ1)

| Setting | Phenomenon | Value/Direction |
| :--- | :--- | :--- |
| Single-step `do(0)` $\mathrm{Flip}(t)$ | Significant inter-step variation | Mid-stage peaks observed for most GSM8K backbones |
| Dataset Contrast | Arithmetic vs. Commonsense | GSM8K $\mathrm{Flip}\sim 0.1$–$0.2$; CommonsenseQA $<0.1$ |
| Paradigm Contrast | Coconut vs. CODI | Coconut shows higher flip rates, especially on GSM8K |
| Backbone Strength | Stronger backbones suppress flips | Qwen3-4B is significantly lower than GPT-2, but shape persists |
| Early stop $S(k)$ | Dataset disparity | CommonsenseQA saturates early; GSM8K rises until $k=6$ |

### Ablation Study: Propagation Structure (RQ2, GSM8K)

| Configuration | Locality | Span | Late-in | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| CoT-SFT (Explicit) | $\ge 0.6$ | Low | Low | Chain-like, adjacent propagation |
| Coconut (Latent) | Significantly lower | Large | High | Dominated by early-to-late long-range jumps |
| CODI (Latent) | Relatively lower | Large | Relatively high | Deviates from chain but less extreme than Coconut |

Principal influence graphs (Figure 5 vs. Figure 6) visually confirm that CoT-SFT has almost exclusively adjacent edges, while Coconut/CODI are filled with long-distance edges skipping intermediate steps.

### Key Findings
*   **Causal leverage is highly heterogeneous**: The single-point flip rate varies greatly across different steps, indicating "high-leverage" and "low-leverage" steps, contradicting the intuition that latent steps are homogenized depth.
*   **Latent CoT does not inherit the chain topology of explicit CoT**: Even though Coconut/CODI are distilled/compressed from explicit CoT, the influence structure becomes skip-dominant, suggesting that latentization changes internal routing rather than just surface format.
*   **Output commitment precedes representational commitment**: Teacher-forced readout shows the model biases toward an answer early, but probes show the representation still supports the alternative answer until the final step.

## Highlights & Insights
*   The work advances latent CoT interpretability from correlation-based probes to the intervention-causal level. This framework is naturally transferable to other hidden-state reasoning paradigms.
*   The "phenomenon–mechanism–nature" logic is clear: heterogeneous leverage (phenomenon) $\to$ non-local routing (mechanism) $\to$ output vs. representational commitment (nature).
*   It provides engineering insights: latent budgets are not homogenized depth, suggesting that supervising or regularizing steps based on their "functional roles" (rather than a uniform loss) may be the correct direction for next-generation latent reasoning.

## Limitations & Future Work
*   **Methodological constraints**: All interventions use zeroing; more semantic interventions (e.g., swapping $h_t$ with another sample) were only briefly tested. The influence graph relies heavily on teacher-forced readout.
*   **Discovery constraints**: RQ3 was primarily conducted on StrategyQA's binary scenarios. T=6 steps were fixed by the pre-trained models, and it remains unknown if structural phase transitions occur with different budgets or significantly larger models.
*   **Future work**: Integrating "high-leverage step identification and routing protection" into latent CoT regularization, and designing commitment-aware stopping strategies.

## Related Work & Insights
*   **vs. Coconut / CODI / Sim-CoT**: This work does not propose a new training algorithm but serves as the first step-resolved causal evaluation protocol for these subjects. 
*   **vs. Explicit CoT Faithfulness**: While prior works used rationale deletion/permutation, this work extends these ideas to hidden layers to provide comparable faithfulness metrics for latent CoT.
*   **vs. Causal Mediation Analysis**: While sharing the `do`-intervention framework, the goal shifts to "step-level structure of reasoning trajectories" rather than locating factual knowledge within layers.

## Rating
*   Novelty: ⭐⭐⭐⭐ First systematic application of SCM and do-intervention to latent CoT.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple paradigms, backbones, and datasets across three RQs.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from phenomena to nature with actionable conclusions.
*   Value: ⭐⭐⭐⭐ Provides specific insights for latent budget allocation and stopping strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ICML 2026\] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](on_robustness_and_chain-of-thought_consistency_of_rl-finetuned_vlms.md)

</div>

<!-- RELATED:END -->
