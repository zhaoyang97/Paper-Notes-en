---
title: >-
  [Paper Note] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning
description: >-
  [AAAI 2026][LLM Reasoning][Large Language Models] This paper proposes an entropy-guided adaptive LLM reasoning framework that combines dynamic in-context retrieval with adaptive chain-of-thought (CoT) reasoning. On the T…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Large Language Models"
  - "Game Theory"
  - "Entropy-Guided Reasoning"
  - "Chain-of-Thought"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 37a11c7b7d24ab5d
---

# LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning

**Conference**: AAAI 2026
**arXiv**: [2601.10775](https://arxiv.org/abs/2601.10775)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Large Language Models, Game Theory, Entropy-Guided Reasoning, Chain-of-Thought, Retrieval-Augmented Generation

## TL;DR
This paper proposes an entropy-guided adaptive LLM reasoning framework that combines dynamic in-context retrieval with adaptive chain-of-thought (CoT) reasoning. On the Tic-Tac-Toe benchmark, the framework improves the average game outcome of LLMs from $-11.6\%$ to $+9.5\%$ while maintaining a low number of LLM queries.

## Background & Motivation

### State of the Field
Large language models (LLMs) excel at single-step reasoning, language understanding, and few-shot generalization. However, they face significant challenges in structured sequential decision-making problems, where each decision affects all subsequent states and outcomes, requiring both local consistency and long-term planning.

### Limitations of Prior Work

**Unstable CoT Reasoning**: Chain-of-thought prompting improves complex reasoning, but performance varies considerably across tasks and domains, and evaluation metrics are often insufficiently fine-grained.

**Wasted Computational Resources**: Multi-path reasoning methods such as Tree-of-Thoughts improve performance but incur substantial computation even when the model is already confident.

**Lack of Theoretically Optimal Reference**: Most reasoning tasks lack objectively optimal solutions, making it difficult to quantitatively evaluate CoT reasoning quality.

**Static Context Retrieval**: Conventional RAG retrieves a fixed number of examples without dynamically adjusting to model uncertainty.

### Root Cause
The core tension lies in balancing computational efficiency and reasoning quality—committing additional computational resources to deep reasoning only when the model is uncertain.

### Starting Point
The paper exploits the token-level entropy of LLM generation as a proxy for uncertainty, dynamically controlling two dimensions: (1) the number of retrieved examples—increasing retrieval under high entropy; and (2) the number of CoT reasoning paths—expanding multi-path exploration under high entropy. Tic-Tac-Toe is selected as a controlled testbed because the minimax algorithm provides known optimal solutions for all states, enabling precise quantification of reasoning quality.

## Method

### Overall Architecture
The game loop proceeds as follows: current board state → autoencoder encodes the state into a vector → cosine similarity retrieves nearest-neighbor game states → a structured prompt is constructed (board + player + retrieved examples) → the LLM generates reasoning and a move → output is parsed → the board is updated → the process repeats until the game ends.

### Key Designs

1. **Board Representation and Contrastive Learning Encoder**:

    - The board is represented as $B \in \{0,1,2\}^{3\times 3}$, flattened into a 9-dimensional vector.
    - An autoencoder maps the input to a low-dimensional latent space: $z = f_\theta(x) \in \mathbb{R}^d$.
    - Training objective: reconstruction loss + contrastive learning loss.
    - The contrastive loss pulls together board states sharing the same optimal action and pushes apart those with different optimal actions: $\mathcal{L}_{con}$ uses margin $\tau$ to control the minimum separation distance.
    - Final loss: $\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_{con}$
    - The vector database stores approximately $20\%$ of all possible board states along with their corresponding minimax-optimal actions.
    - **Design Motivation**: Ensures that retrieved historical states are strategically similar, not merely visually similar.

2. **Entropy-Guided In-Context Retrieval**:

    - The number of retrieved examples $k$ is dynamically adjusted based on model prediction entropy: $k = \min(k_{max}, \lceil k_0 + \alpha \cdot H_q \rceil)$
    - Low entropy (high confidence) → fewer retrievals, preserving context space for reasoning.
    - High entropy (high uncertainty) → more retrievals, providing additional reference information.
    - Total context length is constrained by a token budget $L_{max}$.
    - **Design Motivation**: Avoids wasting the context window on straightforward decisions.

3. **Adaptive Chain-of-Thought Reasoning**:

    - Four reasoning modes of increasing complexity are defined:
        - **Direct output**: directly outputs an action without simulation.
        - **Multi-CoT**: generates $n$ independent reasoning paths and selects the action by majority vote.
        - **Tree-based CoT**: tree expansion, generating $n$ candidate moves per step along with all opponent responses.
        - **Entropy-guided CoT**: expands multiple paths only under high entropy.
    - Token-level entropy: $H_{t,k}^{token} = -\sum_{i=1}^{|V|} p_{t,k}^{(i)} \log p_{t,k}^{(i)}$
    - Step-level entropy: $H_t^{step} = \frac{1}{L_t} \sum_{k=1}^{L_t} H_{t,k}^{token}$
    - Ordered threshold mechanism: $H_t^{step} \in [H_j, H_{j+1}) \Rightarrow n_t = n_j$
    - High entropy → more branches; low entropy → fewer branches.
    - Only the top-$k$ branches are retained at each step to control computational cost.
    - **Design Motivation**: Concentrates computational resources where the model genuinely needs them.

4. **Opponent Modeling**:

    - The opponent ranks all legal moves using the minimax table.
    - A skill level parameter $\alpha \in [0,1]$ is introduced ($\alpha=0.95$ in experiments).
    - Selection probability peaks at the action corresponding to the skill level and decays linearly toward both ends.
    - **Design Motivation**: A sub-optimal opponent better reflects real-world scenarios.

### Loss & Training
- LLaMA-7B is used as the baseline model; preliminary experiments use Gemma 3 270M.
- Each configuration is evaluated over 100 games.
- LLM hyperparameters: temperature$=0.1$, top-$p$/top-$k$ sampling disabled, beam search$=2$.
- A maximum of 10 tokens is generated per step; random seed is fixed at 42.

## Key Experimental Results

### Main Results

| Reasoning Mode | No Context $S(\%)$ | Queries | Fixed Context $S(\%)$ | Queries | Entropy-Guided Context $S(\%)$ | Queries |
|---|---|---|---|---|---|---|
| No CoT | -11.6 | 3 | -5.2 | 4 | -2.8 | 4 |
| Single CoT | -8.2 | 13 | -2.6 | 13 | -0.1 | 15 |
| Multi CoT | -7.5 | 24 | -1.2 | 26 | +4.8 | 28 |
| Tree-based CoT | -2.7 | 165 | +4.5 | 178 | **+9.8** | 188 |
| Entropy-guided CoT | -4.1 | 48 | +3.8 | 56 | **+9.5** | **48** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| No context + No CoT | $S=-11.6\%$ | Worst baseline |
| Fixed context + No CoT | $S=-5.2\%$ | Context retrieval is effective |
| Entropy-guided context + No CoT | $S=-2.8\%$ | Dynamic retrieval further improves |
| Entropy-guided CoT + Entropy-guided context | $S=+9.5\%$ (48 queries) | Best efficiency |
| Tree-based CoT + Entropy-guided context | $S=+9.8\%$ (188 queries) | Best performance, but $4\times$ the cost |

### Key Findings

1. **Complementarity of Context Retrieval and CoT**: The two mechanisms improve performance along orthogonal dimensions—knowledge injection and reasoning depth—and their combination yields the best results.
2. **Negative Correlation between Entropy and Optimality**: Spearman $\rho=-0.471$ ($p<10^{-3}$), Kendall $\tau=-0.346$ ($p<10^{-3}$), confirming that high-entropy tokens correspond to sub-optimal actions.
3. **Computational Efficiency**: Entropy-guided CoT achieves performance comparable to Tree-based CoT ($+9.5\%$ vs. $+9.8\%$) using only approximately one-quarter of the queries (48 vs. 188).
4. **Two Sources of High Entropy**: Genuine uncertainty (true positives) and multiple equally optimal actions (false positives). The latter predominantly occurs at the opening move and is mitigated by not applying entropy branching on the first step.
5. **Zero-Shot Nature**: The LLM undergoes no task-specific fine-tuning; all reasoning stems from in-context conditioning.

## Highlights & Insights
1. Using entropy as a proxy for uncertainty to dynamically allocate computational resources is an elegant and generalizable idea.
2. The choice of Tic-Tac-Toe as a testbed is well-motivated—its known optimal solutions enable precise evaluation of per-step decision quality.
3. The contrastive learning design of the retrieval encoder is clever, organizing the latent space by optimal strategy rather than board appearance.
4. The experimental design is systematic and comprehensive: 5 reasoning strategies $\times$ 3 context configurations $= 15$ configurations with clear cross-comparison.
5. Statistical testing is rigorous (Spearman and Kendall correlation analyses).

## Limitations & Future Work
1. Validation is limited to Tic-Tac-Toe, which has an extremely small state space; scalability to more complex games (Connect Four, Go) remains unknown.
2. LLaMA-7B has limited reasoning capacity; stronger models may handle such reasoning intrinsically.
3. Token-level entropy assumes that linguistic uncertainty equals decision uncertainty, which may not hold in ambiguous scenarios.
4. The retrieval database is derived from a complete game tree and may not be applicable to domains with imperfect information.
5. The logical quality of reasoning chains is not analyzed; only the accuracy of final decisions is evaluated.
6. Extending the framework to environments with uncertain dynamics and partial observability remains an open direction.

## Related Work & Insights
- **Tree-of-Thoughts (Yao 2023)**: Branch-based search framework → this paper adds adaptive control on top of this paradigm.
- **RAG (Lewis 2021)**: Retrieval-augmented generation → this paper dynamizes the retrieval quantity.
- **AlphaZero**: Policy network + search → this paper replaces these components with LLM + retrieval + CoT.
- **GridPuzzle / EIC**: Reasoning chain analysis → this paper leverages game-theoretic optimal solutions for more precise evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of entropy-guided adaptive reasoning is novel)
- Experimental Thoroughness: ⭐⭐⭐ (Tic-Tac-Toe is too simple; validation in more complex environments is lacking)
- Writing Quality: ⭐⭐⭐⭐ (Formalization is clear; experimental design is systematic)
- Value: ⭐⭐⭐ (Core ideas are strong but the scope of application is limited; scalability remains to be demonstrated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](../../ICLR2026/llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[AAAI 2026\] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension](relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)
- [\[ICLR 2026\] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](../../ICLR2026/llm_reasoning/thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)

</div>

<!-- RELATED:END -->
