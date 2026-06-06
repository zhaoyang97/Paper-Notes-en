---
title: >-
  [Paper Note] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain
description: >-
  [ICML 2026 (Position Paper)][LLM Reasoning][Self-evolving LLM] The authors argue that the current collapse of "LLM Self-Play" within a few rounds is fundamentally caused by self-synthetic data failing to provide learnabl…
tags:
  - "ICML 2026 (Position Paper)"
  - "LLM Reasoning"
  - "Self-evolving LLM"
  - "Triadic roles (Proposer/Solver/Verifier)"
  - "Learnable information"
  - "epiplexity"
  - "Self-synthetic data pipeline"
date: 2026-05-08
content_hash: 159fdea6cab0fc4d
---

# Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2603.02218](https://arxiv.org/abs/2603.02218)  
**Code**: None  
**Area**: LLM Reasoning / Self-Evolution / Self-Play / Information Theory  
**Keywords**: Self-evolving LLM, Triadic roles (Proposer/Solver/Verifier), Learnable information, epiplexity, Self-synthetic data pipeline

## TL;DR
The authors argue that the current collapse of "LLM Self-Play" within a few rounds is fundamentally caused by self-synthetic data failing to provide learnable information gain. They formalize "learnable information" using bounded MDL/epiplexity and propose three system-level designs—Asymmetric Co-evolution, Capacity Budget Growth, and Proactive Information Seeking—to ensure monotonically increasing learnable information in the triadic (Proposer-Solver-Verifier) self-evolution loop.

## Background & Motivation

**Background**: LLM self-evolution systems typically employ the same model to play three roles: Proposer (problem generation), Solver (problem solving), and Verifier (scoring), forming a closed-loop training via multi-reward reinforcement learning (RL) without external labels. Representative works include Absolute Zero, R-Zero, Dr. Zero, SPIN, Self-Rewarding, URPO, and Cooper.

**Limitations of Prior Work**: These systems generally exhibit a pattern of "rapid early gains followed by collapse after a few rounds"—Proposers degenerate into generating trivial problems ($f(x)=x$), Solver performance plateaus and then declines, and ground truth must be injected periodically to avoid "self-hallucination." Even with sophisticated reward designs (e.g., maintaining a 50% pass rate), multi-reward RL remains unstable.

**Key Challenge**: Existing methods equate self-evolution with "self-play RL" and focus solely on monotonic reward increases. However, rewards can be hacked, achieved through rote memorization of pre-training knowledge, or inflated by repeatedly sampling isomorphic problems—**task-level metrics improve, but the "learnable structure" in newly synthesized data does not increase.** Once learnable information saturates, the model stops true learning.

**Goal**: (1) Provide a metric to distinguish "illusory progress" from "genuine evolution"; (2) Identify system-level conditions that guarantee monotonic growth of learnable information across iterations; (3) Unify existing self-play, triadic-loop, and curriculum approaches into a single analytical framework and identify their failure modes.

**Key Insight**: The authors leverage the concept of epiplexity from Finzi et al. (2026). Under a bounded observer (fixed parameter budget $C$ and inference budget $T$), MDL is decomposed into "learnable structure $S_{C,T}(X)$" and "residual entropy $H_{C,T}(X)$." Since the same data may be noise to a weak observer but structured to a strong one, "learnable information" is a **relative quantity** that must be co-designed with the observer's budget.

**Core Idea**: Self-evolution is not an RL game but a **self-synthetic data pipeline**. The loop only avoids collapse if $S_{C,T}(D^{(t)})$ increases monotonically across iterations $t$. This requires the synchronization of three components: the generation end (asymmetry), the receiver end (capacity), and the raw material end (external information).

## Method

As a position paper, this work does not propose specific training algorithms but clarifies the necessary conditions for "genuine self-evolution" through three layers: information-theoretic metrics, three design principles, and diagnostic experiments.

### Overall Architecture

The self-evolution loop is abstracted as a pipeline with a "single information source + multi-directional synthesis" (Figure 1). Pre-trained weights of a single LLM serve as the sole source, producing data flows $X_d$ along three synthesis directions (synthesis question / solution / feedback), which are then fed back for self-training. "Genuine evolution" is determined by whether the sequence $\{S_{C^{(t)},T^{(t)}}(D^{(t)})\}_t$ increases monotonically.

The measurement tool is a bounded MDL optimizer: solve $P^{\star}=\arg\min_{P}\{|P|+\mathbb{E}[\log 1/P(X)]\}$ within an observer family $\mathcal{P}_{C,T}$, defining $S_{C,T}(X):=|P^{\star}|$ (epiplexity, learnable structure) and $H_{C,T}(X):=\mathbb{E}[\log 1/P^{\star}(X)]$ (bounded entropy, residual noise). This metric naturally identifies a "Goldilocks Zone"—data must be neither too simple (low $S$, low $H$) nor too difficult (low $S$, high $H$), but must fall in the middle ground: "complex enough to be non-trivial yet structured enough to be learnable."

### Key Designs

1.  **Asymmetric Co-evolution**:
    *   **Function**: Leverages the computational asymmetry where "verifying/proposing is easier than solving." It uses weak Proposers/Verifiers to train a stronger Solver via RL (weak-to-strong), then synchronizes the stronger Solver back to the environment to refresh the Proposer/Verifier (strong-to-weak).
    *   **Mechanism**: Although the three roles share the same weight source, $S_{C,T}(X_d)$ produced along different directions $d(P,S,V)$ varies for a bounded observer. Taking one-way permutations as a limiting case, one can prove $H_{\text{poly}}(X|Y)-H_{\text{poly}}(Y|X)\ge c\log n$, showing an $\Omega(\log n)$ bit difficulty gap between forward (proposing) and inverse (solving) directions. Training converts this residual uncertainty into reusable structures. Implementation requires: (i) organizing directions based on asymmetric gaps (from small to large to inverse gaps: e.g., grammar correction → math proof → medical diagnosis); (ii) using back-translation (Magicoder, MathGenie, InverseCoder) for the Proposer to re-extract problems from strong Solver data; (iii) attempting verifier-free RL for the Verifier to share beliefs with the Solver.
    *   **Design Motivation**: Existing RL only completes the weak-to-strong half. If Proposers/Verifiers do not keep pace as the Solver improves, the task stream becomes "low structure" relative to the observer, causing the loop to collapse into triviality.

2.  **Capacity Budget Growth**:
    *   **Function**: Allows parameter budget $C^{(t)}$ and inference budget $T^{(t)}$ to expand with iterations, ensuring the observer family $\mathcal{P}_{C,T}$ can keep pace with new learnable structures in self-synthetic data.
    *   **Mechanism**: For a fixed $(C,T)$, $S_{C,T}(X)$ is upper-bounded. Since $\mathcal{P}_{C_1,T_1}\subseteq\mathcal{P}_{C_2,T_2}$ implies $\mathrm{MDL}_{C_2,T_2}(X)\le\mathrm{MDL}_{C_1,T_1}(X)$, capacity expansion shifts the "learnable/unlearnable" boundary. This involves asymmetric role scaling (weak Proposer/Verifier training a large Solver) or cross-iteration scaling (Net2Net, Stacking, MoE subset growth) on the parameter axis, and adaptive reasoning tokens or Mixture-of-Recursions on the inference axis.
    *   **Design Motivation**: Empirical evidence shows that fixed $C^{(t)}$ leads to early loss saturation, forcing the Proposer to degenerate into directions the current model class can easily solve. Fixed $T^{(t)}$ misinterprets "truncated inference" as "lack of knowledge." Both mismatches reduce subsequent learnable information.

3.  **Proactive Information Seeking**:
    *   **Function**: Enables the Proposer+Verifier to actively select external context $d^{(t)}$ in each round and learn new synthesis directions around it, injecting external information as conditioning context $(Y^{(t)}\mid d^{(t)})$.
    *   **Mechanism**: Define conditional bounded MDL $\mathrm{MDL}(Y\mid d):=\min_{P}\{|P|+\mathbb{E}[\log 1/P(Y\mid d)]\}$, treating $S_{C,T}(Y\mid d)$ as conditional learnable information. Strategies include: (i) Proposer generating queries from Solver failures/Verifier disagreements to retrieve $d$ and synthesize tasks requiring explicit use of $d$ (citation support, multi-document synthesis, contradiction detection); (ii) converting $d$ into multiple difficulty levels scheduled via curriculum; (iii) evolving retrievers/rerankers/memory using self-synthetic signals (Verifier relevance).
    *   **Design Motivation**: Zero-data systems are capped by pre-trained weights. Fixed external corpora lead to fine-tuning on that corpus. Fixed retrieval mechanisms (static RAG) exceed Solver budget early on and become routine later. These regimes consume information "reactively" and fail to expand the source of learnable information.

### Loss & Training

On the measurement side, **Prequential Coding** is used to estimate epiplexity (Algorithm 1): split the dataset into training/validation, accumulate online loss $\mathcal{L}_{\text{online}}=\sum_i -\log P_{\theta_i}(Z_i)$ during the first pass over $\mathcal{D}_{\text{train}}$, and at the end of each epoch calculate $S=(\mathcal{L}_{\text{online}}-\mathcal{L}_{\text{train}})/\ln 2$ as model cost and $(\mathcal{L}_{\text{val}}/\ln 2)/N_{\text{val}}$ as data cost. The $S^{\star}$ corresponding to the epoch with minimum MDL is used as the estimation for learnable information. This quantity is equivalent to the "cumulative online regret the model pays to learn the data." The three principles are not bound to specific losses; rather, the authors list practical methods (back-translation, verifier-free RL, parameter stacking, adaptive depth, retrieval co-evolution) in the Practice section for future integration.

## Key Experimental Results

Experiments are **diagnostic**, aiming to validate two points using epiplexity: (1) significant differences in learnable information across combinations of directions and Proposer/Solver capacities; (2) current self-play loops do **not** exhibit monotonic increases in learnable information after multiple iterations. Tasks follow Absolute Zero (Zhao et al., 2025a) coding problems: abduction (infer input from code/output), deduction (infer output from code/input), and induction (infer code from input/output).

### Main Results (Experiment 1: Single-round epiplexity distribution)

| Axis of Variation | Values | Observed epiplexity Trend | Conclusion |
| :--- | :--- | :--- | :--- |
| Proposer Capacity | Qwen2.5 7B → Qwen2.5 14B → Qwen3 4B | Monotonic increase | Stronger Proposers generate more learnable information. |
| Solver Capacity | Small to Large | **Increase then Decrease** | Consistent with "emergence" (Finzi et al., 2026): small models forced to learn structure, beyond a threshold they shift to memorization. |
| Synthesis Direction | abduction / deduction / induction | induction ≫ abduction ≈ deduction | Learnable information varies significantly by direction; increasing Proposer capacity alone may not suffice. |

### Ablation Study (Experiment 2: epiplexity trajectory in multi-round self-play)

| Configuration | epiplexity Behavior across Iterations | Behavioral Observation |
| :--- | :--- | :--- |
| Multi-reward RL self-play (no explicit loop mechanism) | **Violent oscillation**, non-monotonic | Solver ability declines, Proposer problem patterns collapse. |
| (Implicit Control) With Three Principles | Authors argue for monotonic growth | Pending community verification. |

### Key Findings
*   **Strong Proposer $\neq$ Good Data**: When Solver capacity exceeds a threshold, the gain from a "stronger Proposer providing more information" is negated by "Solver degenerating into memorization"—providing empirical evidence that Capacity Growth must scale Proposer/Solver/Verifier **simultaneously**.
*   **Direction Over Quantity**: Induction possesses significantly higher learnable information than abduction/deduction, proving that "adding tokens or problems" is inferior to "changing synthesis directions"—the core gap dimension of Asymmetric Co-evolution.
*   **Multi-reward RL is Insufficient**: Fixed $(C,T)$ multi-reward self-play causes violent epiplexity oscillations rather than monotonic growth, explaining from an information-theoretic perspective why reward shaping alone fails.

## Highlights & Insights
*   Translates "whether self-evolution is truly evolving" into a computable quantity $S_{C,T}(D^{(t)})$, distinguishing reward increases from information gain—a critical step in turning "model collapse" into a monitorable metric.
*   Uses the $\Omega(\log n)$ gap of one-way permutations as a limiting case for "asymmetry," upgrading the intuition that "verifying is easier than solving" into a citable lower bound applicable to any "forward easy, inverse hard" task (e.g., creative writing constraints).
*   The "Goldilocks Zone (High $S$, Moderate $H$)" serves as a signal for curriculum scheduling: by calculating the $(S, H)$ position each round, one can decrease difficulty if $H$ is too high or change directions if $S$ is too low, offering better interpretability than pass rates.
*   The narrative of the three principles (Asymmetry as the generator, Capacity as the receiver, Information Seeking as the open inlet) is highly transferable for diagnosing any self-play or agentic system.

## Limitations & Future Work
*   Epiplexity is a very recent concept (Finzi et al., 2026) not yet widely validated, and prequential coding carries significant computational overhead for large models.
*   The three designs are currently easier to close-loop in easy-to-verify domains (code, math); measuring and training the inverse gap in hard-to-verify domains (open QA, medical) remains an open problem.
*   Experiments are small-scale diagnostic tests; the lack of a "monotonically increasing" counter-proof after applying all three designs is a gap typical of position papers.
*   Learnable information is a macro metric and may not correlate perfectly with downstream accuracy—it is possible to learn structures intrinsic to data but irrelevant to the task.
*   The bottleneck for Proactive Information Seeking is "knowing what you don't know," which is itself an open research question (Yin et al., 2023).

## Related Work & Insights
*   **vs Self-Training (STaR / ReST / Rejection-Sampling)**: These rely on fixed verifiers; this paper points out they saturate after initial distributions are exhausted due to "lack of Information Seeking."
*   **vs Solver-Verifier Co-evolution (Self-Rewarding / SPIN / URPO / Cooper)**: They lack strong-to-weak synchronization to ensure Verifiers keep up with Solvers, and task distributions do not expand—"lack of Asymmetric Co-evolution's reverse loop."
*   **vs Proposer-Solver Self-Play (Absolute Zero / R-Zero / Dr. Zero / Self-Questioning)**: Their collapse is attributed to "Proposers drifting toward trivial/unsolvable" and the omission of Verifier synchronization—"Asymmetry not forming a ladder."
*   **vs Triadic Loops (SPELL / SPICE / Socratic-Zero / GenEnv)**: Closest to this framework but still report plateaus; this paper provides the missing unified criterion: epiplexity.
*   **vs Curriculum / Co-evolution / "Scaling is All You Need"**: The authors argue these are necessary but insufficient without the explicit constraint of learnable information.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Attributing self-play collapse to the lack of "learnable information" and formalizing it with epiplexity is a first for self-evolving LLM design criteria.
*   **Experimental Thoroughness**: ⭐⭐⭐ Includes only two small-scale diagnostic experiments; the lack of positive verification for the three designs is a notable but expected gap for a position paper.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear structure (Framework → Metric → Principles → Failure Modes → Mapping). The consistent template (Design/Perspective/Gaps/Practice) makes it highly actionable for engineering.
*   **Value**: ⭐⭐⭐⭐⭐ Provides a unified diagnostic vocabulary and design principles for the self-evolving LLM and agentic RL community. This paper will likely be a frequent reference for its three principles and epiplexity monitoring paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](../../ACL2026/llm_reasoning/self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](../../ACL2026/llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)

</div>

<!-- RELATED:END -->
