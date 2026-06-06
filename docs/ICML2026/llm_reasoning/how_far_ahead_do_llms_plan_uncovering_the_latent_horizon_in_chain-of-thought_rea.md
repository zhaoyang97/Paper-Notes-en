---
title: >-
  [Paper Note] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Chain-of-Thought] This paper employs a low-rank adapter probe called Tele-Lens to systematically measure the predictive power of LLM latent states for "future reasoning" across 12 cross-domain…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Latent Probing"
  - "Planning Horizon"
  - "Uncertainty Estimation"
  - "CoT Bypass"
date: 2026-05-08
content_hash: fce1c5dade662cfb
---

# How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.02103](https://arxiv.org/abs/2602.02103)  
**Code**: https://github.com/lxucs/tele-lens (Available)  
**Area**: LLM Reasoning / CoT Interpretability  
**Keywords**: Chain-of-Thought, Latent Probing, Planning Horizon, Uncertainty Estimation, CoT Bypass

## TL;DR
This paper employs a low-rank adapter probe called Tele-Lens to systematically measure the predictive power of LLM latent states for "future reasoning" across 12 cross-domain tasks. It reveals that the internal planning of LLMs is **myopic**—the answer is precisely locked only at the end of the CoT. Based on this, the "Wooden Barrel Principle" is proposed, using the uncertainty of sparse pivot positions to represent the entire CoT. This significantly improves uncertainty calibration and achieves a 16% CoT bypass rate.

## Background & Motivation

**Background**: CoT has become the standard paradigm for eliciting multi-step reasoning in LLMs, and models like DeepSeek-R1 have further amplified "long-chain thinking" capabilities through RL.

**Limitations of Prior Work**: Research regarding whether CoT is truly essential has yielded conflicting evidence. One school (Pal et al. 2023, Azaria & Mitchell 2023) found that **early latent states already encode subsequent reasoning and the final answer**, suggesting CoT might just be a playback of pre-calculated trajectories. Another school, based on Transformer expressivity theory (Merrill & Sabharwal 2023, Abbe et al. 2024), proved that explicit intermediate steps are necessary for compositional reasoning and length generalization, making "knowing the answer in advance" structurally impossible for such tasks.

**Key Challenge**: Existing evidence mostly stems from single domains or single probe dimensions, leading to conflicting conclusions that have not been aligned on a common scale. Do LLMs possess a "global blueprint" before starting CoT, or are they merely "locally greedy" step-by-step generators? This question concerns both interpretability and the design premises for adaptive thinking and early-exit mechanisms in future models like GPT-5 and Claude.

**Goal**: The problem is decomposed into two sub-questions—Q1: To what extent do latent states encode a global reasoning blueprint versus supporting only local incremental transitions? Q2: How does the planning horizon conversely affect the estimation of CoT uncertainty and necessity?

**Key Insight**: Authors argue that prior contradictions exist because researchers **focused on a single dimension (mostly "early prediction of final answers") within a single task category**. Systematic probing across multiple dimensions (Final Answer / Subsequent Tokens / Total Length) $\times$ multiple task types (Explicit Compositional / Implicit Compositional / Knowledge Semantic) can reconcile these conflicts.

**Core Idea**: A "Tele-Lens" diagnostic probe is trained—a low-rank adapter projecting each layer's latent state followed by an LM head to predict three types of "teleological" signals: future tokens, final answer, and reasoning length. From this, the "Wooden Barrel Uncertainty" principle is derived and applied to two downstream problems: CoT calibration and necessity estimation.

## Method

### Overall Architecture
The approach consists of two parts: **Probing (Section 2)** for diagnosis to answer Q1, and **Application (Section 3)** to answer Q2. The diagnostic input comprises full CoT trajectories $T=\{t_1,\dots,t_n\}$ and latent states $H_i^k\in\mathbb{R}^d$ across 12 tasks, producing accuracy/correlation curves for three types of probes across positions and layers to identify the "myopic horizon." The application side leverages the intuition that "sparse pivots are the true signals" for two tasks: Uncertainty AUROC (averaging uncertainty at top-k pivot positions) and CoT Bypass (skipping CoT if the normalized entropy of the first 5 tokens is below a threshold). Two LLM backbones are evaluated: Qwen3-32B (off-the-shelf with thinking mode) and an In-Domain LLM (based on Qwen2.5-7B-Instruct, trained via GRPO to minimize confounding variables).

### Key Designs

1.  **Tele-Lens Probes: Low-rank adapters + Offset Embeddings + Three-dimensional "Teleological" Targets**:
    - **Function**: Predicts three types of future signals—the next $m$ tokens, the final answer, and the remaining reasoning length—from any position $i$ and layer $k$ latent state $H_i^k$.
    - **Mechanism**: Following the Logit Lens tradition of connecting intermediate layers to the LM head, a bottleneck adapter is added to prevent overfitting: $\widetilde{H}_i^k = \operatorname{GeLU}\big((H_i^k + \operatorname{Emb}^k(\delta))A^k\big)B^k$, where $A^k\in\mathbb{R}^{d\times r}$ and $B^k\in\mathbb{R}^{r\times d}$ are low-rank matrices with $r=256$. $\operatorname{Emb}^k(\delta)$ is an **offset embedding**—given $\delta=1,2,\dots,m$, it specifies whether to predict the "next token" or the "8th token later," unifying multi-step prediction within the same adapter. The output is $\mathcal{P}_i^k=\operatorname{Softmax}(\widetilde{H}_i^k L)$, with a frozen LM head $L$. The offset embedding is removed for final answer prediction, and a regression head is used for reasoning length.
    - **Design Motivation**: Previous probes were either target-specific or limited to the final layer. Tele-Lens provides a low-cost, unified framework covering three semantically different teleological dimensions across layers, allowing for a 3D panorama (layer, position, dimension) to align previously contradictory findings.

2.  **12-Task Classification Protocol: Aligning "Compositional vs. Knowledge" on a Comparative Graph**:
    - **Function**: Systematically samples diverse task types to prevent bias toward any single category.
    - **Mechanism**: Tasks are categorized into **Explicit Compositional** (Parity / Cycle / Subsum, requiring strict multi-step algorithms), **Implicit Compositional** (GSM8K / MATH / AIME / MuSR / Zebra, with hidden multi-step reasoning), and **Knowledge Semantic** (CSQA / MMLU / QuALITY / GPQA, relying on world knowledge and pattern matching). All are converted to multiple-choice questions with fixed answer spaces (distractors generated by GPT-4o), simplifying the final-answer probe to a 20-label classification.
    - **Design Motivation**: Contradictions in prior work stemmed from different task distributions—CSQA leads to the conclusion that "answers are encoded early," while Parity suggests CoT is mandatory. Placing all three categories under the same probe reveals the unified explanation: "myopic horizons + coarse answer perception for simple tasks."

3.  **Wooden Barrel Uncertainty Principle: Replacing Global Averages with Sparse Pivots**:
    - **Function**: Converts the long-standing problem of "perplexity/entropy being insensitive to long CoT" into a simple strategy of averaging select key positions.
    - **Mechanism**: Based on the diagnostic finding that most CoT tokens are high-confidence "syntactic fillers," while reasoning correctness depends on a few "logical bridge points." This is formalized as selecting top-$k$ pivot positions from a CoT based on extrema (lowest entropy for Tele-Lens final-answer probes, or highest for general perplexity/entropy/Self-Certainty). The uncertainty of the path is defined by the average of these $k$ positions. The Self-Certainty formula is $\operatorname{SC}(X)=-\frac{1}{N|\mathcal{V}|}\sum_{i}\sum_{w\in\mathcal{V}}\log(|\mathcal{V}|\cdot P(w|x_{<i}))$. For CoT Bypass, normalized entropy $\bar{\mathrm{H}}(\mathbf{p})=-\sum_{i=1}^{C}p_i\log p_i / \log C$ of the first 5 tokens is used; if any position falls below a 0.1 threshold, the thinking mode is bypassed.
    - **Design Motivation**: This completes the loop with the "myopic horizon" diagnosis. Since global planning is weak and key signals are concentrated, global averages are diluted by "filler tokens." The Wooden Barrel Principle translates the "shortest plank" concept into a plug-and-play calibration trick without model retraining.

### Loss & Training
Each adapter is trained for ~5K steps with early stopping on the dev set ($r=256$). A separate adapter is trained for each probe dimension and layer. The In-Domain LLM uses GRPO (Shao et al. 2024) on Qwen2.5-7B-Instruct for task-aware RL, producing cleaner CoTs (~1K characters vs. 10K+ for Qwen3) as a performance upper bound.

## Key Experimental Results

### Main Results: Tele-Lens Probes Reveal the Myopic Horizon

**Evolution of final-answer probe by CoT position** (In-Domain LLM, Parity task, random guess 0.5):

| CoT Position (dist. to end) | -4 | -3 | -2 | -1 | 0 (End) |
|---|---|---|---|---|---|
| In-Domain LLM | 0.49 | 0.51 | 0.51 | 0.97 | 0.99 |
| Off-the-Shelf Qwen3 | 0.50 | 0.52 | 0.51 | 0.94 | 0.97 |

The conclusion is clear: answer probability stays near the random baseline until the penultimate step, then spikes to $\ge 94\%$ at the final step—a classic "myopic tail mutation." The Cycle task shows the same pattern; probability jumps only after a complete path or loop is observed.

**Early Coarse Signals vs. True CoT vs. w/o CoT** (Off-the-Shelf Qwen3-32B): In nearly all tasks, "early planning" accuracy from initial-position final-answer probes is lower than w/o CoT (direct answering) and much lower than w/ CoT. This indicates that early "answer perception" is merely coarse pattern-matching intuition, not pre-arranged reasoning.

### Uncertainty Estimation AUROC (Closer to 1 is better)

Using Tele-Lens signals as pivots on the In-Domain LLM:

| Method | GSM8K | Zebra | MMLU | GPQA | Average |
|---|---|---|---|---|---|
| Perplexity (Global Avg) | 0.70 | 0.58 | 0.53 | 0.50 | 0.57 |
| Entropy (Global Avg) | 0.72 | 0.60 | 0.52 | 0.50 | 0.58 |
| Self-Certainty | 0.76 | 0.67 | 0.53 | 0.51 | 0.60 |
| **Tele-Lens Top-5** | **0.87** | **0.77** | **0.73** | **0.56** | **0.69** |
| Tele-Lens Top-10 | 0.81 | 0.75 | 0.72 | 0.56 | 0.68 |
| Tele-Lens Top-50 | 0.78 | 0.69 | 0.56 | 0.47 | 0.64 |

On Off-the-Shelf Qwen3-32B using top-100 selection for general metrics (Avg CoT > 7K tokens): Perplexity 0.72 $\rightarrow$ 0.78, Entropy 0.72 $\rightarrow$ 0.75, Self-Certainty 0.65 $\rightarrow$ 0.70. Improvements reach up to 6% absolute.

### CoT Bypass (Qwen3-32B, Threshold 0.1)

| Task | Parity | CSQA | MMLU | GPQA | Avg Bypass | Perf. Change |
|---|---|---|---|---|---|---|
| Bypass Ratio | 0% | 16.2% | 12.4% | 1.2% | 2.8% | **-0.03** |
| Threshold 0.2 | 0% | 28.8% | 20.2% | 3.2% | 6.2% | -0.37 |

Crucially, the bypass mechanism **correctly retains CoT for tasks like Parity (0% bypass)** while achieving double-digit bypass rates on CSQA/MMLU with negligible accuracy loss.

### Key Findings
- **5 pivots are enough**: Top-5 is the optimal $k$ for Tele-Lens; Top-50 is 5% lower. More pivots dilute the signal, confirming the "Wooden Barrel" intuition.
- **Middle layers are strongest**: The best probes are at layer 21 (out of 28) for the In-Domain LLM and layer 48 (out of 64) for Qwen3, consistent with findings by Reif et al. (2019) regarding semantic richness in middle layers.
- **Shortcuts warning**: "Reasoning length predictability" for Parity and Subsum was confounded by input length. The Cycle task (where length depends on the path, not the input) exposes this; "apparent global planning" often stems from surface heuristics.
- **Pivot distribution differences**: Tele-Lens pivots cluster at the CoT end, while general entropy pivots are scattered; their complementarity suggests potential for signal fusion.

## Highlights & Insights
- **Reconciling contradictions**: By using a dual narrative of "myopic horizons + coarse gist for simple tasks," the paper elegantly harmonizes the "early encoding" and "intermediate step necessity" schools.
- **Elegant diagnosis-to-application loop**: The Wooden Barrel Principle is derived directly from diagnostic plots showing that final-answer entropy peaks at sparse locations. This logic—identifying signal sparsity to justify top-k selection—is a reusable framework for any latent signal analysis.
- **Portability of CoT bypass**: Using probe entropy of the first 5 tokens as a switch for long-chain thinking achieves a 16% bypass rate at almost zero cost. This suggests adaptive thinking routers (like those in GPT-5 or Claude Code) can be simple probes rather than complex classifier models.
- **Transferable trick**: The top-$k$ pivot strategy can be combined with any token-level confidence metric (perplexity, entropy, or SC), requiring only a change in the aggregation function during inference.

## Limitations & Future Work
- **Necessity estimation relies on fixed answer spaces**: The CoT bypass calculates entropy over a fixed 20-token label set, which is not directly applicable to open-ended generation (coding, long-form writing).
- **In-Domain LLM size (7B)**: The "upper bound" model might not represent the true ceiling of larger 30B+ thinking models, which might possess stronger global planning.
- **Tele-Lens requires training**: Unlike pure logit lens, adapters must be trained for each layer and dimension, incurring deployment costs. Shift toward OOD task transferability has not been systematically evaluated.
- **Future Directions**: Weighting or learned fusion of Tele-Lens pivots with general entropy pivots; utilizing the "Wooden Barrel Principle" for reward shaping during RL, which might be more efficient than rewarding the entire sequence.

## Related Work & Insights
- **vs. Logit Lens / Tuned Lens (Belrose et al. 2023)**: Traditional lenses focus on "inter-layer interpretability" for the current token. Tele-Lens extends this to "prospective prediction" using offset embeddings.
- **vs. Early Answer Probing (Azaria & Mitchell 2023, Afzal et al. 2025)**: These works conclude answers are encoded early. This paper shows such findings are task-dependent and often reflect shallow pattern matching rather than global planning.
- **vs. CoT Early-exit (Yong et al. 2025)**: While early-exit makes decisions during CoT generation, this bypass mechanism decides "whether to think" before the thinking mode even starts.
- **vs. Reward Shaping (Li et al. 2026b)**: Similar to observations that most tokens are low-entropy fillers, the Wooden Barrel Principle successfully operationalizes this into a calibration algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐ Probe method is an extension, but the "myopic horizon" narrative and "Wooden Barrel" application are novel and cohesive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 tasks, two backbones, three probe dimensions, and two downstream applications with comprehensive appendix data.
- Writing Quality: ⭐⭐⭐⭐ Clear structure driven by dual sub-questions; the "Wooden Barrel" metaphor effectively unifies the work.
- Value: ⭐⭐⭐⭐ Directly serves adaptive thinking and CoT compression; the pivot trick is zero-cost and highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ACL 2026\] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](../../ACL2026/llm_reasoning/how_chain-of-thought_works_tracing_information_flow_from_decoding_projection_and.md)
- [\[ICML 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)

</div>

<!-- RELATED:END -->
