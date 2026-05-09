---
title: >-
  [Paper Note] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
description: >-
  [ICLR2026][LLM Reasoning][Latent Chain-of-Thought] This paper models latent CoT as a structural causal model (SCM) and analyzes the Coconut and CODI paradigms via step-wise do-interventions, revealing that latent reasoning steps exhibit heterogeneous causal leverage, non-local jump-based propagation structures, and a persistent gap between early output commitment and late representational commitment.
tags:
  - ICLR2026
  - LLM Reasoning
  - Latent Chain-of-Thought
  - Causal Analysis
  - do-Intervention
  - Structural Causal Model
  - Interpretability
date: 2026-05-08
content_hash: 68096b6fbe61da33
---

# Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure

**Conference**: ICLR2026
**arXiv**: [2602.08783](https://arxiv.org/abs/2602.08783)
**Code**: [GitHub](https://github.com/J1mL1/causal-latent-cot)
**Area**: LLM Reasoning
**Keywords**: Latent Chain-of-Thought, Causal Analysis, do-Intervention, Structural Causal Model, Interpretability

## TL;DR
This paper models latent CoT as a structural causal model (SCM) and analyzes the Coconut and CODI paradigms via step-wise do-interventions, revealing that latent reasoning steps exhibit heterogeneous causal leverage, non-local jump-based propagation structures, and a persistent gap between early output commitment and late representational commitment.

## Background & Motivation
**Inherent Limitations of Explicit CoT**: While Chain-of-Thought improves reasoning accuracy, it introduces substantial decoding overhead and verbose outputs, and may produce post-hoc rationalization rather than faithfully reflecting the model's internal computation.

**The Rise and Challenges of Latent CoT**: Methods such as Coconut and CODI shift reasoning into continuous representation spaces to reduce decoding costs; however, intermediate computation is no longer exposed as discrete, editable steps, rendering traditional step-editing and ablation methods inapplicable.

**Limitations of Existing Analysis**: Current understanding of latent CoT relies primarily on correlation-based probes and lacks systematic causal analysis, leaving key questions such as "which steps are causally necessary" unanswered.

**Unknown Nature of Step Budgets**: Does a fixed latent step budget (e.g., $T=6$) contribute additional computational depth uniformly, or do individual steps serve distinct functional roles? How is information routed across steps?

**Unclear Relationship Between Output and Representational Commitment**: At what point does the output layer "lock in" an answer? Is this synchronized with the state of internal representations? Do competing hypotheses persist across intermediate steps?

**Absence of a Unified Evaluation Framework**: A standardized intervention–readout protocol applicable across different latent reasoning paradigms is needed to enable comparable causal analysis.

## Method

### Core Framework: Latent CoT as a Causal System
The latent state sequence of latent CoT is modeled as causal variables within an SCM. Given input $x$, the model produces a latent trajectory $H_{1:T}$ and output $Y$:
$$H_t = f_t(H_{<t}, x, \epsilon_t; \theta), \quad t=1,\ldots,T$$
$$Y = g(H_{1:T}, x, \epsilon_y; \theta)$$
where $f_t$ is the transition mechanism and $g$ is the decoding mechanism. A do-intervention $\mathrm{do}(h_t \leftarrow \tilde{h}_t)$ severs the causal link between step $t$ and its upstream context, enabling observation of downstream effects. The counterfactual trajectory is propagated as:
$$\tilde{h}_{t'} := f_{t'}(\tilde{h}_{<t'}, x, \tilde{\epsilon}_{t'}; \theta), \quad t' > t$$

### RQ1: Step Necessity and Sufficiency
**Zero Intervention**: The target step's hidden state is set to zero via $\mathrm{do}(h_t \leftarrow \mathbf{0})$, and the flip rate—the fraction of samples whose final prediction changes after intervention—is computed:
$$\mathrm{Flip}(t) = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}[\tilde{y}_i^{(t)} \neq y_i]$$
This metric quantifies the causal necessity of each step for the final decision. Zero intervention is chosen for its determinism, parameter-free nature, and fairness across architectures.

**Early-Stop Decoding**: Decoding is performed directly after truncation at step $k$, defining the earliest decodable step $k_i$ and the cumulative solve rate $S(k)$:
$$k_i = \min(\{k : \hat{y}_i^{(\leq k)} = y_i^*\} \cup \{\infty\}), \quad S(k) = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{k_i \leq k\}$$

### RQ2: Information Flow and Inter-Step Influence Structure
Single-step interventions are combined with downstream early readouts, and the directed propagation strength from step $t$ to step $s$ is quantified via the KL divergence of teacher-forced output distributions:
$$\mathrm{KL}_{t \to s}^{(i)} = \frac{1}{|y_i^*|}\sum_{u=1}^{|y_i^*|}\mathrm{KL}(p_{\text{base}}^{(s)}(\cdot \mid y_{i,<u}^*) \| p_{\mathrm{do}(t)}^{(s)}(\cdot \mid y_{i,<u}^*))$$
An influence matrix $W_{t,s} = \mathbb{E}_i[\mathrm{KL}_{t \to s}^{(i)}]$ is constructed by aggregation. For visualization, only top-1 outgoing edges with weight $> 0.1 \cdot \max(W)$ are retained to form a dominant influence graph. Four normalized structural metrics are also computed:
- **Locality**: Concentration of influence mass near the diagonal
- **Span**: Expected jump distance
- **Early-out**: Proportion of influence originating from early steps
- **Late-in**: Proportion of influence converging to late steps

### RQ3: Superposition and Commitment
On StrategyQA (Yes/No binary labels), two-mode prompts are obtained via random sampling. For each prompt, $K$ rollouts are sampled and partitioned into $\mathcal{C}_Y$ and $\mathcal{C}_N$, with support for both answers measured at each step using two readout methods:
- **Teacher-forced readout**: Token-level log-probabilities computed over fixed answer templates
- **Probe readout**: A fixed probe maps $h_t$ to next-token probabilities

The superposition score is defined as $\mathrm{SS}(t) = \min(p_Y(t), p_N(t))$; a high score indicates that both answers remain competitive at intermediate steps.

### Key Design Choices
- **Robustness validation of intervention operators**: Six intervention types (zero/mean/mean_step/gaussian_h/gaussian_mu/gaussian_mu_step) are compared; qualitative results are consistent, and zero intervention is selected for its determinism.
- **Comparison of two reasoning paradigms**: Coconut (recurrent latent tokens) and CODI (self-distillation-compressed CoT) differ architecturally but share the same intervention interface.
- **Three-tier progressive analysis**: Phenomenon (RQ1: which steps matter) → Mechanism (RQ2: how information propagates) → Essence (RQ3: how competing hypotheses evolve).

## Key Experimental Results

### Table 1: RQ1 Step Necessity — Key Findings on Flip Rate

| Setting | Task | Flip Rate Range | Pattern |
|------|------|------|------|
| Coconut (GPT-2) | GSM8K | 0.10–0.20+ | Mid-step peak, high variance |
| CODI (GPT-2) | GSM8K | 0.05–0.15 | Lower than Coconut on same backbone |
| Coconut (Llama3-1B) | GSM8K | Higher | Backbone strengthens but does not eliminate structure |
| CODI (Llama3-1B) | GSM8K | Moderate | More stable relative to Coconut |
| Coconut (Qwen3-4B) | GSM8K | Lower | Strong backbone substantially suppresses flip |
| CODI (Qwen3-4B) | GSM8K | Lowest | Strong backbone + CODI is most stable |
| All paradigms | CommonsenseQA | Generally <0.1 | Commonsense tasks are more robust to intervention |

### Table 2: RQ2 Information Flow Structural Metrics (GSM8K)

| Model Type | Locality (↑=local) | Span (↑=long-range) | Early-out | Late-in |
|------|------|------|------|------|
| CoT-SFT (GPT-2) | ≥0.6 | Low | Moderate | Moderate |
| CoT-SFT (Llama3-1B) | ≥0.6 | Low | Moderate | Moderate |
| Coconut (all backbones) | Significantly lower than CoT | High | High | High |
| CODI (all backbones) | Lower than CoT but higher than Coconut | Moderate–High | Moderate | High |

### Table 3: RQ3 Superposition Score Comparison (StrategyQA)

| Readout Method | Coconut SS Trend | CODI SS Trend |
|------|------|------|
| Teacher-forced | Consistently low and nearly flat — early output commitment | Consistently low and nearly flat — early output commitment |
| Probe | Higher at intermediate steps, significant drop at final step | Higher than Coconut throughout, drop at final step |

## Key Findings

1. **Heterogeneous Distribution of Causal Leverage**: Flip rates vary substantially across step indices, exhibiting non-uniform or mid-step-peaked patterns. Different steps serve distinct functional roles, and the removal of certain "high-leverage" steps causes disproportionate disruption to downstream computation.
2. **Task-Dependent Decision Fragility**: Flip rates on GSM8K (mathematics) are far higher than on CommonsenseQA, indicating that arithmetic reasoning depends more heavily on intermediate latent state computation, while commonsense reasoning is more robust to step interventions.
3. **Non-Local Jump-Based Propagation**: Latent CoT influence graphs contain extensive skip connections, with information frequently bypassing intermediate steps to propagate directly from early to late steps—in stark contrast to the near-chain (local) propagation of explicit CoT. Coconut favors direct early→final connections, while CODI exhibits more distributed routing.
4. **Desynchronization of Output and Representational Commitment**: Teacher-forced readout reveals that the output layer locks in an answer early (low SS), whereas probe readout shows that intermediate representations continue to maintain competing hypotheses (high SS) until the final step, at which point they collapse. This demonstrates that "decodable" does not imply "committed."
5. **Orthogonal Effects of Paradigm and Backbone**: Stronger backbones reduce the absolute flip rate without altering the step-dependency structure; Coconut is more fragile than CODI under matched backbones, indicating that the paradigm itself shapes the causal structure.
6. **Task Differences in Early-Stop Decoding**: $S(k)$ on CommonsenseQA saturates rapidly within the first 2–3 steps, whereas $S(k)$ on GSM8K continues to grow through step 6, confirming that mathematical tasks genuinely require more latent computation steps.

## Highlights & Insights
- **First Causal Analysis of Latent CoT**: A unified intervention–readout protocol is established, distinguishing between availability and stability of latent representations.
- **Progressively Layered Analytical Framework**: The three RQs advance from phenomenon (step importance) to mechanism (propagation structure) to essence (mode competition and commitment), forming a logically rigorous structure.
- **Core Design Insight Revealed**: The latent step budget is not a uniform source of "additional depth," but rather a staged functional interface with non-local routing—improving latent reasoning should focus on shaping routing and commitment mechanisms rather than simply increasing the number of steps.
- **The Output vs. Representational Commitment Finding** has far-reaching implications for reasoning system design: a model may appear to have "made a decision" at the output level while its internal representations remain in a state of "deliberation."

## Limitations & Future Work
- Only two latent CoT paradigms (Coconut and CODI) are studied; methods such as Token Assorted and SoftCoT are not covered.
- Although the robustness of zero intervention (zeroing out) has been validated, it may introduce off-manifold distributional shift.
- The fixed latent step budget $T=6$ is used throughout; changes in causal structure across different budget lengths are not explored.
- RQ3 is conducted only on StrategyQA (binary labels); mode analysis on open-ended tasks (e.g., GSM8K) is infeasible due to the large output space.
- No concrete training or decoding improvements are proposed; the analysis motivates directions but does not validate them empirically.
- The sparsification threshold $\alpha=0.1$ for the influence graph and the early/late boundary $m=2/5$ are chosen somewhat subjectively.

## Related Work & Insights

| Dimension | Ours | Wu et al. (2025) "Single-Thread Reasoning" |
|------|------|------|
| Core Claim | Representations retain competing hypotheses at intermediate steps (high probe SS) | Continuous reasoning is inherently greedy/single-threaded |
| Analysis Granularity | Step-level causal intervention + readout | Behavioral/output-level analysis |
| Key Distinction | Reveals "output commitment ≠ representational commitment"; the two views are complementary rather than contradictory | Does not distinguish between output-level and representation-level commitment |

| Dimension | Ours | Classical Mechanistic Interpretability (Elhage et al.) |
|------|------|------|
| Unit of Analysis | Latent reasoning "steps" (macro-level) | Neurons / attention heads / features (micro-level) |
| Intervention Method | Step-level do-intervention + teacher-forced readout | Activation patching / ablation |
| Complementarity | Step-level analysis → reveals functional routing | Micro-level → localizes specific computational mechanisms |

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First causal analysis of latent CoT; the three progressively layered RQs form a unified and extensible framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple paradigms (Coconut/CODI) × multiple backbones (GPT-2/Llama/Qwen) × multiple tasks (GSM8K/CommonsenseQA/StrategyQA).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear structure; the "phenomenon → mechanism → essence" progression is maintained consistently throughout.
- **Value**: ⭐⭐⭐⭐ — Offers important guidance for latent reasoning system design (routing/commitment rather than stacking steps), though no concrete improvements are proposed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](../../ACL2026/llm_reasoning/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](../../AAAI2026/llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)

</div>

<!-- RELATED:END -->
