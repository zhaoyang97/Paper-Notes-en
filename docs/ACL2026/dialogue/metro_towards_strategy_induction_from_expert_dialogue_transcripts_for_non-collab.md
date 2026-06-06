---
title: >-
  [Paper Note] Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues
description: >-
  [ACL 2026][Dialogue Systems][strategy induction] Metro automatically inducts expert dialogue transcripts into a "Strategy Forest"—a collection of trees rooted in K-Means clustered dialogue states. Nodes represent LLM-exp…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "strategy induction"
  - "Strategy Forest"
  - "negotiation"
  - "persuasion"
  - "planning logic"
date: 2026-05-08
content_hash: 10e197a1670970f1
---

# Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues

**Conference**: ACL 2026  
**arXiv**: [2604.11427](https://arxiv.org/abs/2604.11427)  
**Code**: https://github.com/Humphrey-0125/METRO (Available)  
**Area**: Dialogue Strategy / Non-collaborative Dialogue / Knowledge Induction / LLM agent  
**Keywords**: strategy induction, Strategy Forest, negotiation, persuasion, planning logic

## TL;DR
Metro automatically inducts expert dialogue transcripts into a "Strategy Forest"—a collection of trees rooted in K-Means clustered dialogue states. Nodes represent LLM-expanded micro-principle actions, and branches represent complete action trajectories pruned based on Wilson confidence lower bounds and MCTS-style value backpropagation scoring. During inference, the system retrieves a tree to extract parallel short-term (breadth) and long-term (depth) suggestions. Without any training, Metro achieves an average performance gain of approximately 10% over baselines such as PRINCIPLES, PPDPP, and GDP-Zero on two non-collaborative dialogue tasks, P4G and CB.

## Background & Motivation

**Background**: Non-collaborative dialogues (e.g., price negotiation, charitable persuasion, debt collection) require an agent to "win" even when the opponent has conflicting interests. The standard paradigm relies on (i) domain experts manually codifying strategy action sets (such as negotiation acts from He et al. 2018 or persuasion acts from Wang et al. 2019), followed by (ii) training a plug-in planner (PPDPP) or running MCTS (GDP-Zero) to select actions. This pipeline is not scalable, as experts must redefine action sets for every new domain.

**Limitations of Prior Work**: (i) Expert dependence—the quality of the action set determines the upper bound, yet human-written coverage is limited (e.g., human actions cover $\le 30\%$ of clusters in the CB dataset); (ii) Lack of planning logic—while PRINCIPLES (Kim 2025) can extract principles in the form of "When [situation], you should [action A] rather than [action B], because [reason]," it treats strategies as independent units and loses multi-turn planning logic regarding "which move to follow next"; (iii) Expensive training—PPDPP requires SFT followed by RL, and GDP-Zero's MCTS inference is computationally intensive.

**Key Challenge**: Strategy involves two types of knowledge: "what to do" (action set) and "when to do it" (planning logic). The former can be summarized by LLMs, but the latter requires preserving the temporal structure of multi-turn contexts. Traditional induction methods excel at extracting flat rules but struggle to weave trajectories into hierarchical structures.

**Goal**: Enable LLMs to directly induct both (a) an expanded action set and (b) multi-turn planning logic tied to dialogue states from raw transcripts, storing them in a retrieval-friendly structure without requiring training during inference.

**Key Insight**: The authors observe that multiple historical trajectories may lead to success or failure under the same dialogue state. By clustering these trajectories by state and performing prefix-merging, a "state-centric tree" naturally forms—where each root is a dialogue state and each branch is a multi-turn planning path. Pruning with the Wilson lower bound and value backpropagation filters out unreliable branches that appear successful but have small sample sizes.

**Core Idea**: Induct expert transcripts into a "Strategy Forest" where breadth (immediate children of the root) provides short-term tactical responses and depth (root-to-leaf paths) provides long-term strategic foresight. Both types of knowledge are injected into LLM decision-making via retrieval-augmented prompting.

## Method

### Overall Architecture
Metro consists of two phases: **offline induction** and **online inference**. In the induction phase: (i) Action Extraction & Expansion pulls turn-level actions from transcripts and expands them into "do/avoid micro-principles" using an LLM; (ii) Dialogue State Identification encodes each history prefix $d'_i$ using bge-large-en-v1.5 and applies K-Means clustering ($K=150$ for P4G, $K=80$ for CB); (iii) Strategy Forest Induction builds a state-centric tree for each cluster—the root is the cluster centroid, and all future trajectories passing through that state are inserted via prefix-merge. Outcome values are backpropagated to each node, and unreliable branches are pruned using the Wilson lower bound and Beam Search. In the inference phase: the system retrieves the root most similar to the current $d_{t-1}$, extracts breadth (micro-principles) and depth (highest-value branch), and asks the LLM to reinterpret these into context-aware short/long-term prompts to generate the final response.

### Key Designs

1.  **Action Expansion via self-reflection micro-principle**:
    *   **Function**: Upgrades raw actions $a_i$ in transcripts to self-reflective "do/avoid" micro-principles $\hat a_i$, allowing the action set to transcend the limitations of the transcript's coverage.
    *   **Mechanism**: For every agent turn, GPT-4o-mini acts as a critic to judge if the turn was better, worse, or neutral based on local context (history + current utterance + opponent response), using a majority vote from five independent evaluations. For "better" turns, the LLM summarizes a reusable principle in the form "When [situation], do [...]" without seeing $a_i$. For "worse/neutral" turns, it wraps the original action into "When [situation], avoid [$a_i$]". Each micro-principle is explicitly conditioned on the opponent's previous utterance to facilitate semantic retrieval.
    *   **Design Motivation**: Raw actions are case-specific ("Could you go down to \$50?"), which limits generalization. Micro-principles abstract strategies to a level (e.g., when to concede vs. persist) that is reusable across transcripts and transferable across tasks. The dual "do/avoid" path learns from both positive and negative signals, preventing distribution shifts caused by training only on successful samples.

2.  **State-centric Tree + Confidence-aware Value Estimation**:
    *   **Function**: Explicitly encodes the multi-turn planning logic of "when to use which action" into trees rooted in dialogue states.
    *   **Mechanism**: Each transcript $D$ is sliced into history prefixes $\{d'_1, \ldots, d'_{T-1}\}$, which are clustered via K-Means. All future trajectories $(a_{t+1}, \ldots)$ within a cluster are inserted into a tree via prefix-merge. Each node $u$ aggregates statistics: empirical success $\hat p(u)=s(u)/n(u)$ and outcome value $v(d,t) = r(d) - \lambda_{\text{len}}(t+1)/N_d$ (including a length penalty). Values are backpropagated using a depth-discount $\gamma^k$ (MCTS style). The total score for a node is $S(u)=w_{\text{sr}}\cdot p_{\text{lb}}(u) + w_{\text{val}}\cdot \bar V(u) + w_{\text{cnt}}\cdot \log(1+n(u))$, where $p_{\text{lb}}$ is the Wilson score lower bound. This penalizes branches that appear successful but were only observed once. Finally, Beam Search retains the Top-K branches.
    *   **Design Motivation**: Viewing transcripts as an unfolded MCTS history, merging prefixes naturally creates search trees. The Wilson lower bound is introduced to combat "success by chance" in small samples—a common pitfall in strategy induction. The length penalty prevents the agent from learning degenerate strategies that achieve success merely by stalling.

3.  **Breadth + Depth Dual-scale Inference Enhancement**:
    *   **Function**: Translates the Strategy Forest into two complementary prompt-time suggestions: tactical next steps (short-term) and strategic paths (long-term).
    *   **Mechanism**: At turn $t$, $d_{t-1}$ is embedded and compared against all roots using cosine similarity to retrieve the most similar tree $f$. **Breadth**: The system uses the opponent's most recent utterance as a query to retrieve the Top-5 micro-principles within the cluster and asks the LLM to reinterpret these into specific next steps. **Depth**: The system selects the single complete branch in $f$ with the highest average node value and reinterprets it into a high-level planning directive (e.g., "Build trust gradually before proposing a donation"). Both suggestions are concatenated into the prompt to guide generation.
    *   **Design Motivation**: Using only the Top-1 action can be short-sighted, while using only a full trajectory can be too rigid. The dual-scale approach provides the LLM with both "what to say now" and "the overall plan," corresponding to exploitation and plan rollout in MCTS. Unlike MCTS, Metro is entirely offline; per-turn inference involves only a table lookup and a single LLM generation, making it significantly cheaper than GDP-Zero.

### Loss & Training
Metro **requires no training**. Offline induction uses GPT-4o-mini for criticism and expansion, bge-large-en-v1.5 for encoding, and scikit-learn for K-Means. The online inference LLM backbone is GPT-3.5-turbo (consistent with baselines). Key hyperparameters: $\lambda_{\text{len}}=0.2$, $\gamma=0.9$, $w_{\text{sr}}=1.0, w_{\text{val}}=0.2, w_{\text{cnt}}=0.05$, Wilson $z=1.96$, Breadth Top-K=5, and Depth Top-1 full branch.

## Key Experimental Results

### Main Results
Evaluated on P4G (charity persuasion) and CB (CraigslistBargain price negotiation) with 200 LLM simulators and 5 human participants:

| Method | P4G SR↑ | P4G AT↓ | CB SR↑ | CB SL%↑ | P4G* SR↑ (Human) | CB* SR↑ (Human) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard | 0.620 | 4.56 | 0.185 | 0.154 | 0.333 | 0.283 |
| GDP-Zero (MCTS) | 0.660 | 5.35 | 0.495 | 0.125 | 0.600 | 0.450 |
| PPDPP (Trained planner) | 0.730 | 4.67 | 0.250 | 0.150 | 0.633 | 0.383 |
| PRINCIPLES (Induction) | 0.770 | 5.24 | 0.485 | 0.149 | 0.600 | 0.467 |
| **Metro** | **0.780** | 4.76 | **0.575** | **0.189** | **0.661** | **0.483** |

*   Metro achieves an average improvement of 10.24% over PRINCIPLES and 9.93% over the second-best method, without requiring training or test-time MCTS.

### Ablation Study
Breakdown of Planning Logic (Table 2):

| Configuration | P4G SR | CB SR | CB SL% | Note |
| :--- | :--- | :--- | :--- | :--- |
| Full (Top-5 Nodes + Top-1 Full Branch) | 0.780 | 0.575 | 0.189 | Default |
| Breadth Top-1 Node | 0.760 | 0.465 | 0.140 | Breadth narrowing drops 11 pp |
| Depth 1-hop Branch | 0.770 | 0.535 | 0.150 | Depth truncation drops 4 pp |
| Depth Top-3 Branches | 0.760 | 0.485 | 0.150 | Adding more branches decreases perf |
| w/o Exp. Action | (Fig 3) | ↓ | ↓ | Remove LLM-expanded principles |
| w/o Depth | ↓ on P4G | **↑ on CB** | — | High redundancy in CB makes depth noise |
| w/o Breadth | ↓ | ↓ | — | Short-term suggestions are more stable |

*   The marginal gain of breadth is generally higher than depth for SR. Depth's effectiveness depends heavily on the diversity of source transcripts.

### Key Findings
*   **Action diversity is central to Metro**: Cluster Coverage analysis shows Metro covers ~80% of clusters at K=100, while PRINCIPLES covers ~50% and manual actions cover less than 30%.
*   **Strong cross-task transferability**: Transferring from CB → P4G yields an SR of 0.755 (near baseline). When the action space is expanded to ALL (CB+P4G), Metro maintains performance (0.770), while PPDPP and GDP-Zero degrade or face computational explosions.
*   **Transcript Quality > Source**: Using LLM-generated transcripts instead of expert transcripts resulted in a higher SR on CB (0.500 vs. 0.440), suggesting the induction framework is sensitive to quality rather than "expert status."
*   **Generalization across personalities**: Across Big-Five traits and decision-making styles, Metro ranked first or second in 9/10 subgroups, with a standard deviation (~0.08) half that of the baselines.

## Highlights & Insights
*   **Abstracting strategy as a "state-action prefix tree" is a principled design**: It upgrades the flat memory of PRINCIPLES into state-conditioned hierarchical memory, using a dual-scale prompt (short + long) rarely seen in dialogue agent literature.
*   **The Wilson lower bound is robust**: It automatically prunes unreliable branches, a common issue in strategy induction where small sample sizes lead to false positives. This technique is easily transferable to any trajectory-mining scenario.
*   **Zero training + cross-task transfer**: Compared to PPDPP (requires RL) and GDP-Zero (expensive test-time MCTS), Metro is highly cost-effective and engineering-friendly.
*   **Honest reporting on CB results**: The authors explicitly note that "w/o Depth" improves performance on CB due to structural limitations in the source data (high repetition), and demonstrate in Appendix C.3 that depth becomes effective again when using LLM-generated transcripts.

## Limitations & Future Work
*   Source transcripts are from crowdsourced users, not real experts, making it difficult to verify performance on truly professional dialogues—a common data issue in this field.
*   The LLM-as-judge pipeline is long (critic → expansion → retrieval → generation), creating a risk of accumulated hallucinations. The paper only validates final downstream metrics without manual verification of intermediate steps.
*   Breadth/depth weights and the clustering parameter $K$ are manually tuned. Optimal $K$ varies significantly between tasks (P4G=150, CB=80), requiring a grid search for new tasks.
*   Taking only the Top-1 branch for depth might lose diversity; more sophisticated branch ensemble or adaptive aggregation methods represent obvious future directions.

## Related Work & Insights
*   **vs PRINCIPLES (Kim et al. 2025)**: Both extract strategies from transcripts, but PRINCIPLES stores rules as unit units. Metro's state-action trees preserve multi-turn logic, leading to a 10.24% average gain.
*   **vs PPDPP (Deng et al. 2024)**: PPDPP trains a RoBERTa planner; Metro is zero-training and relies on retrieval-augmented reinterpretation. Metro outperforms PPDPP by 5 pp on P4G and 32 pp on CB.
*   **vs GDP-Zero (Yu et al. 2023)**: GDP-Zero runs expensive test-time MCTS. Metro pre-computes the Strategy Forest offline, making inference significantly faster.
*   **vs MERMAID / Dialogue Flow Extraction**: Existing tools perform flat flow induction without value-aware pruning. Metro's outcome-driven Wilson evaluation is better suited for outcome-sensitive non-collaborative scenarios.

## Rating
*   Novelty: ⭐⭐⭐⭐ The Strategy Forest representation and Wilson pruning are novel in dialogue strategy, though MCTS-style backprop and retrieval prompting are established techniques.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across two tasks, 8 baselines, LLM/Human evaluation, cross-task transfer, 9 user personas, and detailed ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear framework and consistent notation; some formulas (e.g., scoring $S(u)$) are fully explained only in the Appendix.
*   Value: ⭐⭐⭐⭐⭐ Zero-training, cross-task capability, and a 10% SOTA improvement make this highly practical for industry applications and transferable to other multi-turn strategic tasks like medical consultation or debate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2026\] Simulated Students in Tutoring Dialogues: Substance or Illusion?](simulated_students_in_tutoring_dialogues_substance_or_illusion.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)

</div>

<!-- RELATED:END -->
