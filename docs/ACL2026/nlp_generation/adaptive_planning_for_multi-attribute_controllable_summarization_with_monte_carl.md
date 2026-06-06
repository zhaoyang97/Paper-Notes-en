---
title: >-
  [Paper Note] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search
description: >-
  [ACL 2026][Text Generation][Controllable Summarization] This paper proposes PACO, which reformulates "multi-attribute controllable summarization" as a planning problem to find an optimal "attribute control sequence." It…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Controllable Summarization"
  - "Multi-attribute Control"
  - "MCTS"
  - "Training-free"
  - "Sequential Planning"
date: 2026-05-08
content_hash: 4f75001e4eef5841
---

# Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search

**Conference**: ACL 2026  
**arXiv**: [2509.26435](https://arxiv.org/abs/2509.26435)  
**Code**: Not yet public  
**Area**: Text Generation / Controllable Summarization  
**Keywords**: Controllable Summarization, Multi-attribute Control, MCTS, Training-free, Sequential Planning

## TL;DR
This paper proposes PACO, which reformulates "multi-attribute controllable summarization" as a planning problem to find an optimal "attribute control sequence." It utilizes a customized Monte Carlo Tree Search—where nodes represent complete summaries and actions represent single-attribute adjustments—to identify the optimal adjustment path during the prompting stage. Without any attribute-specific training, Llama-3.2-1B with PACO achieves controllability comparable to a Llama-3.3-70B baseline, while 70B+PACO outperforms all existing methods.

## Background & Motivation

**Background**: Controllable summarization aims to generate summaries according to multiple user-specified attributes (length, extractiveness, specificity, topic, speaker, etc.). Existing mainstream solutions either use MoE (e.g., HydraSum, where each decoder learns one attribute) or hard prompt + soft prefix tuning (e.g., MACSum). Both require fine-tuning for each individual attribute, leading to poor flexibility and difficulty in generalizing to unseen combinations of preferences.

**Limitations of Prior Work**: (1) **Complex correlations between attributes**: For instance, increasing extractiveness may passively change the length, and adjusting specificity might affect topic alignment. Simultaneously forcing all attributes in a single decoding pass often leads to a "whack-a-mole" effect; (2) Autoregressive generation in LLMs is naturally ill-suited for satisfying multiple numerical constraints in a single forward pass; (3) Even for step-by-step adjustments, the number of possible adjustment sequences explodes with the number of attributes, lacking a systematic exploration mechanism.

**Key Challenge**: There is a structural conflict between the "one-shot satisfaction of all constraints" and the "token-by-token generation paradigm of language models." Furthermore, the search space for determining the "optimal sequence of attribute adjustments" is too large to be covered by manual heuristics.

**Goal**: (1) Replace attribute-specific fine-tuning with an inference-time method to achieve a training-free approach; (2) Transform multi-attribute satisfaction from a single-pass generation problem into a sequential decision-making problem; (3) Use a search algorithm to automatically find the optimal control sequence.

**Key Insight**: It is observed that attribute control in summarization is essentially "iterative modification"—humans adjust summaries by "tuning length first, then checking the topic, then refining named entity density." If each single-attribute adjustment is viewed as an action in an MDP and the current summary as a state, attribute control becomes a tree search problem over the summary space.

**Core Idea**: Use MCTS for planning at the summary level (rather than the token level)—where nodes = complete summaries, actions = adjusting a specific attribute, and rewards = alignment with target attributes. This avoids the search space explosion of token-level MCTS in long-text generation while systematically exploring the optimal adjustment path.

## Method

### Overall Architecture
PACO models multi-attribute controllable summarization as an MDP: state $s$ = current summary (including all historical adjustments), action $a \in \{ext, len, spc, top, spk\}$ = adjustment of a specific attribute, and root $s_0$ = the initial summary obtained by prompting all attributes at once. The LLM acts as the policy $\pi$, where each action is implemented by "prompting the LLM to re-generate a summary optimized only for that specific attribute." The termination condition is either satisfying all attributes or reaching the maximum depth. After running MCTS, the node with the highest degree (attribute alignment) is selected from the entire tree as the final output. This is crucial as it means PACO does not force the refinement of all attributes but finds the best compromise.

### Key Designs

1.  **Summary-Level MCTS Node Design**:
    -   **Function**: Increases the search granularity from the traditional token/sentence level in LLM-MCTS to the summary level, making search complexity manageable and ensuring each node is a complete, adoptable answer.
    -   **Mechanism**: Every time a node is expanded, all actions are enumerated (i.e., all attributes are legal actions, even if previously adjusted). The LLM is prompted to generate $s_{t+1}$ based on the complete history $s_0, s_1, \ldots, s_t$. The selection step uses a PUCT variant $a = \arg\max_a[Q(s,a) + U(s,a)]$, where $U(s,a) = c_{\text{puct}}\cdot\pi_\theta(s,a)\cdot\sqrt{\sum_b N(s,b)}/(1+N(s,a))$, and $c_{\text{puct}}$ uses a logarithmic form to dynamically balance exploration and exploitation. Repeatedly adjusting the same attribute is allowed because subsequent actions might disrupt previous effects, necessitating an opportunity for the model to "backtrack" and fix them.
    -   **Design Motivation**: Token-level MCTS faces an immense search space for long summaries and cannot evaluate attribute satisfaction holistically. Summary-level MCTS simplifies each expansion into a full generation with a search depth of only 5–10 steps; the trade-off is a higher cost per expansion but a more appropriate semantic granularity.

2.  **Attribute-Type Aware Reward Design**:
    -   **Function**: Differentiates reward calculations based on the quantifiable nature of attributes, allowing the search to support both "exact match" and "the higher the better" objectives.
    -   **Mechanism**: Attributes are categorized into two types: **deterministic** (extractiveness, length, specificity—requiring exact numerical matches), which use the inverse of the Mean Absolute Deviation (MAD) as the reward; and **non-deterministic** (topic, speaker—the more aligned, the better), which use BERTScore directly. Local reward $= \alpha / (avg_{\text{det}} + \varepsilon) + (1/\beta) \cdot avg_{\text{non-det}}$, where $\alpha, \beta$ control the relative importance of the two categories. Specific metrics: extractiveness = proportion of summary words in the source; length = word count; specificity = named entities / total words; topic = average BERTScore between topic keywords and summary words; speaker = BERTScore between the summary and the target speaker's utterances.
    -   **Design Motivation**: Attributes in controllable summarization are inherently heterogeneous—length is a "must be 50 words" constraint, while topic is "the closer, the better." Mixing them with a single metric results in numerical incommensurability and distorts the MCTS reward signal. Categorization allows for precise control of hard constraints and best-effort optimization for soft constraints.

3.  **DimensionAware Termination / Selection Strategy**:
    -   **Function**: Selects the node with the "highest degree in the entire tree" rather than just the leaf node after MCTS, supporting flexible termination when only a subset of attributes is satisfied.
    -   **Mechanism**: While standard MCTS selects the most-visited leaf, PACO selects the highest-degree node from the entire tree. Since the "optimal reachable degree" varies by instance, forcing the satisfaction of all attributes might lead the search to a maximum depth that degrades quality. Selecting the globally highest-degree node is equivalent to "adaptively abandoning attributes that cannot be simultaneously satisfied." Reward aggregation offers three pluggable strategies: Weighted Mean (default, includes recency-decay $\lambda$ to weight later actions higher), Geometric Mean (emphasizes balance across attributes), and Min Score (for safety-critical scenarios). Backpropagation follows the standard $W(s,a) \leftarrow W(s,a) + V(s_l)$, $Q = W/N$.
    -   **Design Motivation**: In reality, exact satisfaction of all attributes is often unfeasible due to conflicts. Forcing it leads to quality collapse. Allowing MCTS to discover "at which step the gain is maximized" is more robust than rigid constraints and aligns better with actual user preferences.

### Loss & Training
**Entirely training-free**, utilizing pure inference-time MCTS. The tree defaults to width=5 (number of actions) and depth=5. Each simulation uses a weighted mean to aggregate token-level rewards. Recency decay $\lambda=0.5$ is the "sweet spot" (ablations show performance drops at $\lambda=0$ or $\lambda=2.0$). Any LLM can serve as the backbone.

## Key Experimental Results

### Main Results
Three benchmarks: MACSumDial (meeting transcripts, 5 attributes including speaker), MACSumDoc (CNN/DailyMail, 4 attributes without speaker), and DialogSum (daily conversations). Three backbones: Llama-3.2-1B, Qwen2.5-7B, and Llama-3.3-70B. Metrics: MAD for deterministic attributes (lower is better), alignment for non-deterministic (higher is better), and ROUGE/BERTScore for quality.

| Backbone | Method | Ext MAD↓ | Len MAD↓ | Spc MAD↓ | Top↑ | Spk↑ | ROUGE↑ |
|----------|------|---------|---------|---------|------|------|--------|
| HP+SP (BART-large trained) | - | 6.66 | 34.66 | 7.08 | 0.807 | 0.804 | 0.315 |
| Llama-3.2-1B | base | 10.79 | 55.68 | 9.30 | 0.783 | 0.795 | 0.270 |
| Llama-3.2-1B | **PACO** | 9.30 | **17.96** | 7.22 | 0.792 | 0.794 | 0.288 |
| Qwen2.5-7B | base | 9.70 | 17.82 | 6.99 | 0.797 | 0.795 | 0.301 |
| Qwen2.5-7B | **PACO** | 8.72 | **11.79** | 5.43 | 0.799 | 0.794 | 0.302 |
| Llama-3.3-70B | base | 6.43 | 15.72 | 7.11 | 0.800 | 0.798 | 0.328 |
| Llama-3.3-70B | Implicit self-plan | 7.35 | 27.70 | 8.09 | 0.802 | 0.795 | 0.304 |
| Llama-3.3-70B | Explicit self-plan | 7.44 | 28.19 | 7.32 | 0.808 | 0.794 | 0.287 |
| Llama-3.3-70B | Joint-iterative | 5.19 | 11.19 | 5.18 | 0.797 | 0.797 | 0.319 |
| Llama-3.3-70B | Random sequential | 5.44 | 11.16 | 4.24 | 0.797 | 0.797 | 0.322 |
| Llama-3.3-70B | **PACO** | **4.91** | **7.63** | **3.81** | 0.795 | 0.798 | **0.328** |

### Ablation Study (DeepSeek 70B / MACSumDial)

| Configuration | Avg Det MAD↓ | Top↑ |
|------|------------|------|
| Full PACO (local reward + DA filter) | 5.45 | 0.795 |
| Only local reward (no heuristic) | 5.45 | 0.795 |
| Only heuristic (binary probability of "can model complete remaining attributes") | 5.53 | 0.796 |
| Local + heuristic combination | 5.67 | 0.795 |
| Joint-iterative (equal inference budget) | 7.19 | 0.797 |
| Random sequential (equal inference budget) | 6.95 | 0.797 |

### Key Findings
-   **Small Model + PACO ≈ Large Model baseline**: Llama-3.2-1B + PACO reduced the length control MAD from 55.68 to 17.96, matching the 70B baseline. This proves that "planning > model scale" holds for controllability.
-   **Budget-matched comparisons prove gains from planning, not just compute**: Using the same number of inference passes, Joint-iterative and Random sequential are outperformed by PACO by 1.3 MAD, demonstrating that structured tree search is more effective than simple resampling.
-   **Self-planning (letting the LLM plan itself) fails**: Both Implicit and Explicit self-planning perform worse than the baseline, proving that current LLM planning capabilities are insufficient to replace explicit search algorithms. External MCTS is necessary.
-   **Length is the most difficult attribute**: In MACSumDial, length consistently has the highest MAD due to its high coupling with other attributes. 70B + PACO tends to place length adjustments at the end as a "finishing touch."
-   **Heuristics are less effective than pure local rewards**: Intuitively, heuristics should provide lookahead signals, but LLMs cannot reliably predict whether a partial summary can satisfy all remaining attributes, introducing noise into the search.
-   **No quality degradation**: While significantly improving controllability, PACO's ROUGE/BERTScore remains on par with or slightly higher than the baseline (incremental adjustments are less damaging to quality than one-shot constraints).

## Highlights & Insights
-   **Correct selection of granularity**: Setting MCTS nodes at the "complete summary" level rather than "token / sentence" is key to the method's success. It bypasses the search explosion of token-level MCTS in long-text generation and ensures every node is a usable intermediate product. This granularity choice is transferable to other "long-form controllable generation" tasks.
-   **Attribute ontology is key to reward design**: Categorizing attributes into deterministic/non-deterministic types and using inverse MAD vs. alignment is simple yet effective. This approach of "designing reward structures based on physical meanings" is more robust than normalizing all attributes into a [0, 1] sum.
-   **"Highest degree node" selection is unconventional**: Standard MCTS selects the most-visited node based on the assumption that high-quality solutions are repeatedly confirmed. In controllable summarization, however, optimal solutions are often near the root, and deeper search can lead astray. PACO's highest-degree selection avoids this "over-search."
-   **Negative results are valuable**: The failure of self-planning and heuristic value functions provides a significant insight—LLM meta-reasoning (planning, lookahead evaluation) is much weaker than its execution capability. Outsourcing planning to deterministic algorithms remains a more reliable engineering choice.

## Limitations & Future Work
-   High computational overhead: Each instance requires 5–10 LLM generations, which is 5–10× slower than a single pass. Latency-sensitive applications may require batch scheduling or more aggressive pruning (e.g., progressive widening).
-   The current 5 attributes are near the reliable control limit of LLM prompting. Expanding to 10+ attributes may require hierarchical search or attribute clustering.
-   Lack of learning signal: All feedback is discarded at inference time. If successful trajectories were distilled back into the model via SFT, a 1B model could theoretically approach PACO performance zero-shot.
-   Evaluation metrics (extractiveness via word overlap, specificity via NER ratio) are relatively coarse and may have a gap with human perception of control.
-   Inconsistent control preferences across datasets (e.g., length is easiest in DialogSum but hardest elsewhere) suggest that current reward functions are sensitive to annotation styles and lack normalization.

## Related Work & Insights
-   **vs. HydraSum / HP+SP MACSum** (Goyal 2022, Zhang 2023): These models fine-tune dedicated modules for each attribute. PACO achieves better results using a 70B model without training.
-   **vs. Tree of Thoughts / Reasoning via Planning** (Yao 2023, Hao 2023): These also use MCTS but with nodes as tokens or thought fragments. PACO uses summary-level nodes for long-text generation and allows repeated actions since attribute adjustment is not monotonic.
-   **vs. Best-of-N sampling**: The Random sequential baseline is essentially BoN (multiple trials within the same budget). PACO uses PUCT signals to guide the search, proving that structured exploration > random resampling.

## Rating
-   Novelty: ⭐⭐⭐⭐ "Summary-level MCTS" and "attribute-type aware reward" are clean designs, reformulating controllable summarization as a sequential planning problem for the first time.
-   Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 3 backbones + budget-matched controls + self-planning controls + full ablations + cross-domain analysis; exceptionally comprehensive.
-   Writing Quality: ⭐⭐⭐⭐ Clear visualizations, thorough discussion of negative self-planning results, and concise mathematical/algorithmic expressions.
-   Value: ⭐⭐⭐⭐ Training-free, plug-and-play, and allows small models to reach large model performance. High reference value for practical controllable generation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)
- [\[ICML 2026\] Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces](../../ICML2026/nlp_generation/score-repellent_monte_carlo_toward_efficient_non-markovian_sampler_with_constant.md)
- [\[ACL 2026\] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)

</div>

<!-- RELATED:END -->
