---
title: >-
  [Paper Note] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving
description: >-
  [ACL 2025][LLM (Other)][Automatic Theorem Proving] This paper challenges the conventional wisdom that "automatic theorem proving requires complex search methods (such as MCTS or value functions)" by proposing the BFS-Prover system. Through three key innovations (expert iteration with data filtering, DPO based on compiler feedback, and length normalization), it scales simple best-first search (BFS) into a high-performance theorem prover, achieving a state-of-the-art (SOTA) sco…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Automatic Theorem Proving"
  - "Best-First Search"
  - "Expert Iteration"
  - "DPO"
  - "Lean4"
date: 2026-05-08
content_hash: bc88eb2deb70fa0b
---

# BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving

**Conference**: ACL 2025  
**arXiv**: [2502.03438](https://arxiv.org/abs/2502.03438)  
**Code**: [https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B](https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B)  
**Area**: LLM/NLP  
**Keywords**: Automatic Theorem Proving, Best-First Search, Expert Iteration, DPO, Lean4

## TL;DR
This paper challenges the conventional wisdom that "automatic theorem proving requires complex search methods (such as MCTS or value functions)" by proposing the BFS-Prover system. Through three key innovations (expert iteration with data filtering, DPO based on compiler feedback, and length normalization), it scales simple best-first search (BFS) into a high-performance theorem prover, achieving a state-of-the-art (SOTA) score of 72.95% on the MiniF2F test set.

## Background & Motivation

**Background**: Using LLMs for Lean4 formal theorem proving is a core benchmark for evaluating AI's reasoning capabilities. Mainstream methods rely on Monte Carlo Tree Search (MCTS) combined with value functions (critic models) to navigate the vast proof search space, such as DeepSeek-Prover, InternLM-StepProver, and HunyuanProver.

**Limitations of Prior Work**: Although MCTS excels in board games, its application to theorem proving poses unique difficulties: (1) proof search lacks clear termination state judgment (unlike games with wins/losses), making intermediate states difficult to evaluate; (2) the branching factor is enormous and dynamically changing; (3) feedback is sparse and delayed. Furthermore, training and maintaining an additional critic model increases system complexity and inference overhead (requiring both policy and value inferences for each state expansion).

**Key Challenge**: There is a common belief in the community that BFS is unsuitable for large-scale theorem proving due to its lack of exploration mechanisms and bias towards deep paths (cumulative log probability penalizes long paths). However, has this assumption been rigorously validated?

**Goal**: To verify whether BFS, under an appropriate scaling strategy, can achieve performance comparable to complex search methods.

**Key Insight**: The authors argue that the two "flaws" of BFS can be mitigated through training strategies—constantly learning better tactics via expert iteration, and eliminating bias towards deep paths via length normalization.

**Core Idea**: To transform simple BFS into a powerful theorem prover using a well-designed scaling strategy (expert iteration + DPO + length normalization), challenging the traditional perspective that "complex search must be used."

## Method

### Overall Architecture
BFS-Prover consists of three core components: a policy LLM (based on Qwen2.5-Math-7B), an environment interface interacting with Lean4 via LeanDojo, and a length-normalized priority queue-driven BFS search engine. The system continuously improves the policy LLM through multiple rounds of expert iteration, collecting new proof data and retraining the model in each round.

### Key Designs

1. **Expert Iteration with Self-Filtering**:

    - **Function**: Gradually accumulate more challenging training data to continuously enhance the policy LLM.
    - **Mechanism**: Each iteration round consists of four steps: (a) Beam Search Filtering—using deterministic beam search to identify theorems that the current model can easily prove, **deliberately excluding** these simple data from the training set; (b) Data Collection—using temperature-sampled BFS to search for remaining unproven theorems and collecting successfully proved (state, tactic) pairs; (c) SFT Training—re-finetuning the base model on all accumulated data; (d) DPO Refinement—further optimizing the policy using compiler error signals.
    - **Design Motivation**: Filtering out simple problems is a key innovation. Without filtering, the training set would be overwhelmed by numerous simple proofs, preventing the model from learning complex reasoning patterns. As iterations progress and the model's capabilities improve, the threshold for "simple" theorems raised, continuously shifting the training data towards harder directions.

2. **Direct Preference Optimization from Compiler Feedback**:

    - **Function**: Utilize Lean compiler error feedback as negative signals to improve the sample efficiency of BFS.
    - **Mechanism**: During the tree search process, starting from the same proof state, the tactic on the correct proof path serves as the positive instance $a_w$, while the tactic that causes a compiler error serves as the negative instance $a_l$, forming a preference pair $(a_w, a_l)$. The model is trained using the DPO loss $-\mathbb{E}[\log\sigma(\beta(r_\theta(s,a_w) - r_\theta(s,a_l)))]$, where the implicit reward is $r_\theta(s,a) = \log p_\theta(a|s) - \log p_{\text{ref}}(a|s)$.
    - **Design Motivation**: SFT only learns from positive instances, while DPO utilizes both positive and negative instances. Compiler errors provide free and accurate negative signals, the utilization of which significantly improves policy quality without additional annotation. DPO sharpens the policy distribution, reducing invalid expansions in BFS.

3. **Length-Normalized BFS Scoring**:

    - **Function**: Eliminate the inherent bias of BFS against deep proof paths.
    - **Mechanism**: Traditional BFS uses cumulative log probability $\sum \log p(a_t|s_t)$ as node priority, where longer paths receive lower scores. This paper introduces length normalization: $\text{score}(s_L) = \frac{\sum_{t=0}^{L-1}\log p(a_t|s_t)}{L^\alpha}$, where $\alpha \in [0,1]$ controls the normalization strength. When $\alpha=0$, it is equivalent to no normalization, and when $\alpha=1$, it represents full normalization (average log probability per step).
    - **Design Motivation**: Many theorems require long proof chains (20+ tactic steps), which traditional BFS struggles to explore due to cumulative penalties. Length normalization allows paths of different depths to compete fairly on the same scale.

### Loss & Training
The SFT phase uses the standard next-token prediction loss for 3 epochs, with the learning rate decaying from $2\times10^{-5}$ to $10^{-6}$. The DPO phase runs for 1 epoch with a learning rate of $5\times10^{-6}$ and a KL regularization parameter $\beta=10$. The choice between SFT or DPO depends on the volume of data generated in that iteration round—SFT is used for large volumes, while DPO is used for small volumes (leveraging the sample efficiency advantages of negative instances).

## Key Experimental Results

### Main Results

| Method | Critic | Search Method | Tactic Budget | MiniF2F-test |
|------|--------|----------|------------|--------------|
| DeepSeek-Prover-V1.5 | None | MCTS | 32×16×400 | 63.5% |
| InternLM2.5-StepProver | Yes | BFS | 256×32×600 | 65.9% |
| HunyuanProver | Yes | BFS | 600×8×400 | 68.4% |
| **BFS-Prover** | **None** | **BFS** | 2048×2×600 | **70.83%** |
| **BFS-Prover (Cumulative)** | **None** | **BFS** | Cumulative | **72.95%** |

### Ablation Study

| Configuration | pass@64 | pass@2048 | Description |
|------|---------|-----------|------|
| SFT only | 64.58% | 70.38% | Baseline SFT model |
| SFT + DPO | 64.98% | 70.83% | DPO brings consistent improvement |
| w/o beam search filtering | ~62% | ~68% | Filtering strategy is crucial for data quality |
| α=0.0 (no normalization) | - | 67.8% | Deep proofs are difficult to discover |
| α=0.5 | - | 70.83% | Optimal configuration |

### Key Findings
- BFS outperforms all MCTS or BFS+critic methods without requiring a critic model, demonstrating the potential of simple methods paired with appropriate scaling strategies.
- SFT+DPO consistently improves by about 0.4-0.5 percentage points compared to pure SFT. The advantage of DPO lies in sample efficiency—the gap is small at the start (pass@64), but the advantage is stable as the scale grows.
- The distribution of proof lengths shifts significantly during expert iteration—averaging 10.2 steps in early stages and 16.7 steps in later stages, indicating that the model progressively learns deeper reasoning.
- BFS performance scales logarithmically with the number of search passes, showing diminishing returns.

## Highlights & Insights
- The approach of challenging complex methods with a "simple method + careful scaling" paradigm is highly inspiring. BFS-Prover proves that in ATP, data quality and training strategy might be more important than the search algorithm itself—disrupting the common perception of the community.
- The beam search filtering strategy is ingenious: by deliberately discarding simple data, it guarantees the increasing difficulty of the training data, similar to an "anti-curriculum" design in curriculum learning.
- Utilizing compiler feedback as negative instances for DPO is extremely natural and efficient. This idea of "leveraging naturally occurring error signals from the environment" can be transferred to tasks with clear environmental feedback, such as code generation and robotics.

## Limitations & Future Work
- Only a 7B parameter model is used; larger models may capture more complex mathematical reasoning patterns, but at the cost of higher inference overhead.
- The logarithmic scaling law implies diminishing returns from relying solely on increasing computation, highlighting the need to explore new breakthroughs.
- It relies on a specific formal language (Lean4) and compiler, and the cost of generalizing to other proof systems remains unknown.
- The quality and coverage of formal corpora (900,000 statements) might limit the system's performance in certain mathematical domains.

## Related Work & Insights
- **vs DeepSeek-Prover-V1.5**: DeepSeek uses whole-proof generation + MCTS, while BFS-Prover uses step-level generation + BFS, achieving better results within a simpler framework.
- **vs HunyuanProver**: HunyuanProver requires an additional critic model which increases inference cost, while BFS-Prover remains stronger even without a critic.
- **vs InternLM-StepProver**: Both use BFS, but InternLM-StepProver relies on value function assistance, whereas BFS-Prover demonstrates that a value function is not necessary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of beating complex methods with a simple approach is highly impactful, and the three innovative designs are tightly integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons with several SOTA systems, with thorough scaling analysis and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and strong motivation.
- Value: ⭐⭐⭐⭐⭐ SOTA results + challenging current paradigms, driving significant progress for the ATP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theorem Prover as a Judge for Synthetic Data Generation](theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)
- [\[ACL 2025\] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)

</div>

<!-- RELATED:END -->
