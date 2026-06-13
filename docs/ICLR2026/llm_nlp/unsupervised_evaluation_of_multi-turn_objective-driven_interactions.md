---
title: >-
  [Paper Note] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions
description: >-
  [ICLR2026][LLM/NLP][Unsupervised Evaluation] Three **unsupervised** metrics are proposed—LLM-guided clustering (goal identification), interaction completeness detection via fine-tuned completion models…
tags:
  - "ICLR2026"
  - "LLM/NLP"
  - "Unsupervised Evaluation"
  - "Multi-Turn Dialogue"
  - "Goal Completion"
  - "LLM Uncertainty"
  - "Response Tree"
  - "LLM-Guided Clustering"
date: 2026-05-08
content_hash: 1ff2058c06f45ed4
---

# Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions

**Conference**: ICLR2026  
**arXiv**: [2511.03047](https://arxiv.org/abs/2511.03047)  
**Code**: Not released (paper indicates release with final version)  
**Area**: LLM/NLP  
**Keywords**: Unsupervised Evaluation, Multi-Turn Dialogue, Goal Completion, LLM Uncertainty, Response Tree, LLM-Guided Clustering  
**Authors**: Emi Soroka, Tanmay Chopra, Krish Desai, Sanjay Lall (Stanford & Emissary Technologies)

## TL;DR

Three **unsupervised** metrics are proposed—LLM-guided clustering (goal identification), interaction completeness detection via fine-tuned completion models, and response trees (LLM uncertainty quantification)—for evaluating multi-turn objective-driven dialogues without labeled data or LLM-as-a-judge, achieving performance that matches or exceeds a 70B judge using only an 8B model.

## Background & Motivation

**Evaluation of enterprise LLM systems is challenging**: Task-oriented dialogue, AI agents, and customer service systems involving objective-driven interactions are increasingly prevalent, yet evaluation methods lag significantly behind—data are complex and unannotated, and manual labeling does not scale.

**LLM-as-a-judge is unreliable**: Well-documented issues include position bias, verbosity bias, familiarity bias, output inconsistency, and sensitivity to prompt phrasing.

**Distribution shift**: Objective-driven systems introduce reasoning, tool invocation, multi-agent interaction, and shared environment manipulation, all of which diverge from the base conversational distribution on which LLMs are pretrained, further complicating evaluation.

**Limitations of existing metrics**: ROUGE/BLEU require reference answers; perplexity is informationally limited; custom metrics can only monitor known error types.

**Core Problem**: Design evaluation metrics that require **zero annotations** and **zero reference answers**, and that can automatically discover user goals, detect interaction completeness, and quantify LLM uncertainty.

## Method

### Metric 1: LLM-Guided Clustering (User Goal Identification)

**Objective**: Automatically discover and label user goal categories from unannotated multi-turn dialogues.

**Three-stage algorithm** (Algorithm 1):

**Preprocessing**: For each dialogue $c_i$, an LLM is prompted to generate a free-text goal summary $s_i$, which is then embedded as $v_i \in \mathbb{R}^{1536}$ using text-embedding-3-small.

**Phase 1 — Initial Clustering + Labeling**:
- K-means is applied to $v_1, \dots, v_n$ to obtain $k_1$ initial clusters (with $k_1$ set as a generous overestimate)
- For each cluster, 10 positive and 10 negative samples are drawn, and an LLM is prompted to generate a cluster description $L_i$
- All descriptions are embedded to obtain $d_1, \dots, d_{k_1}$

**Phase 2 — Iterative Merging**:
- A pairwise cosine similarity matrix is computed: $D_{ij} = \frac{d_i^\top d_j}{\|d_i\|_2 \|d_j\|_2}$
- The pair with the maximum $D_{ij}$ is iteratively selected, and the LLM is prompted to decide whether to merge (with 10 positive and negative samples each time)
- Merged clusters receive regenerated descriptions; termination occurs when all current cluster pairs are rejected for merging

**Advantages**: Combines the stability of k-means with the semantic understanding of LLMs, producing interpretable clusters with natural language labels.

### Metric 2: Interaction Completeness Detection (Goal Completion)

**Core Idea**: A fine-tuned LLM learns the "completion distribution" and detects completeness by predicting whether a dialogue should terminate.

**Formal definition**: Given a distribution $D$ of complete dialogues, a new distribution $D'$ is constructed in which the final response of each complete dialogue is appended with an `end` token. This yields:

$$P_{D'}(\texttt{end} \mid c) = P(\text{llm}_{D'}(\text{concat}(p_1, r_1, \dots, p_n, r_n)) = \texttt{end})$$

For a complete dialogue $c$ and a truncated dialogue $c'$ (with $k < n$ turns), the following is expected to hold:

$$P_{D'}(\texttt{end} \mid c) > P_{D'}(\texttt{end} \mid c')$$

**Implementation**:
- **Base distribution** (e.g., LMSYS): LLaMA3.1-8B-Instruct with a short prompt is used directly
- **Domain-specific distribution** (e.g., insurance underwriting, code debugging): A LoRA adapter is trained to fine-tune LLaMA3.2-8B as a completion model
    - Input: $\text{concat}(p_1, r_1, \dots, p_n)$
    - Target: $r_n$ + `end` token
    - Training: AdamW 8-bit, lr = 0.0002, weight decay = 0.01, 3 epochs, 50% of data
- **Incomplete dialogues**: The model does not output `end` but instead generates subsequent turns $p_{n+1}, r_{n+1}, \dots$, which can additionally **summarize the remaining tasks the LLM has not yet completed**

### Metric 3: Response Trees (Response Uncertainty)

**Objective**: Quantify LLM response uncertainty for a given prompt without repeated high-temperature sampling.

**Response tree definition**: Given a prompt $p$ and a threshold probability $\alpha$, $\text{rtree}_{D,\alpha}(p)$ returns a tree of all branches whose traversal probability is $\ge \alpha$.

**Construction**:
1. Generate one response along with its top-$k$ log probabilities
2. If the log probability of the 2nd through $k$-th tokens exceeds $\alpha$, generate separate branches for each
3. Recurse until no log probability exceeds $\alpha$ or a computational threshold is reached

**Uncertainty quantification**:
- **Leaf node count**: More leaves → more possible responses → higher uncertainty → greater likelihood of error
- **Maximum log probability**: Higher values indicate greater model confidence in the optimal response
- Both metrics exhibit low correlation with dialogue length ($r$ ranging from $-0.25$ to $0.41$), suggesting that response trees capture more complex uncertainty than length-dependent signals

## Key Experimental Results

### Datasets

| Dataset | Size | Domain | Objective-Driven | Tool Use |
|--------|------|------|:--------:|:--------:|
| LMSYS-Chat-1M | 1000 | Unstructured dialogue | ✗ | ✗ |
| Code-Feedback | 1000 | Code generation & debugging | ✓ | ✗ |
| Insurance | 380 | Insurance underwriting | ✓ | ✓ |
| WebShop | 351 | Online shopping interaction | ✓ | ✓ |
| SQL+OS+KB | 1043 | SQL / terminal / knowledge base | ✓ | ✓ |

### Main Results

#### Completeness Detection Results

| Dataset (Evaluator) | Accuracy | Precision | Recall | F1 |
|------------------|----------|-----------|--------|-----|
| LMSYS (70B judge) | 0.43 | 0.77 | 0.25 | 0.38 |
| **LMSYS (8B completion)** | **0.74** | **0.79** | **0.85** | **0.82** |
| Code-Feedback (70B judge) | 0.53 | 0.53 | 0.46 | 0.49 |
| Code-Feedback (FT 8B) | 0.47 | 0.71 | 0.12 | 0.21 |
| Insurance (70B judge) | 0.95 | 1.0 | 0.91 | 0.95 |
| **Insurance (FT 8B)** | **0.91** | **0.94** | **0.87** | **0.91** |
| WebShop (70B judge) | 0.92 | 1.0 | 0.83 | 0.91 |
| **WebShop (FT 8B)** | **0.92** | **0.89** | **1.0** | **0.94** |
| SQL+OS+KB (70B judge) | 0.97 | 0.96 | 0.97 | 0.96 |
| **SQL+OS+KB (FT 8B)** | **0.98** | **0.99** | **0.98** | **0.99** |

### Key Findings

- The fine-tuned 8B model **matches or surpasses the 70B LLM judge** on most datasets
- The `end` token is a critical design choice (removing it on Insurance reduces F1 from 0.91 to 0.72)
- Code-Feedback presents the greatest difficulty due to loosely structured dialogues that can be continued at any point

#### Response Tree Statistics

| Metric | LMSYS | Code-Feedback | Insurance | WebShop | KB+OS+SQL |
|------|-------|---------------|-----------|---------|-----------|
| Max logprob vs. length | -0.11 | -0.19 | -0.25 | 0.16 | 0.41 |
| Max logprob vs. leaf count | -0.49 | -0.46 | -0.10 | -0.19 | -0.06 |

- KB+OS+SQL exhibits the highest uncertainty, as tool invocation, SQL, and terminal interactions diverge most from the base distribution
- LMSYS and Code-Feedback show the highest confidence, being closest to the pretraining distribution

#### Clustering Stability
- LMSYS, WebShop, and SQL+OS+KB produce **highly stable clusters** across runs
- Code-Feedback and Insurance show slightly lower stability (amenable to multi-dimensional labeling: language vs. task type)
- Compared to a GPT-4.1 LLM-only labeling baseline: the LLM-only approach degenerates to a single cluster ("Online Shopping and Purchase") on WebShop

## Highlights & Insights

1. **Three complementary metrics providing complete coverage**: goal identification (what) + completeness detection (whether) + uncertainty quantification (how confident), spanning the core dimensions of evaluation
2. **Zero annotations, zero references**: Genuinely unsupervised, requiring neither ground truth nor an LLM judge
3. **Strong performance from a small model**: An 8B fine-tuned model matches or exceeds a 70B judge, making online deployment and real-time monitoring feasible
4. **Response tree innovation**: More structured and informationally richer than semantic entropy, which requires multiple high-temperature samples
5. **Distribution adaptability**: LoRA fine-tuning adapts to domain-specific token distributions

## Limitations & Future Work

1. **Clustering depends on the initial setting of $k_1$**, bounding the maximum number of discoverable clusters
2. **Completeness detection performs poorly on loosely structured dialogues** (e.g., Code-Feedback, where the first turn may suffice and subsequent turns are follow-up questions)
3. **Multi-label classification is not supported** (a single dialogue may involve multiple goals)
4. **Response trees lack ground-truth validation** (no direct evidence that high uncertainty corresponds to errors)
5. **Limited fine-tuning data** (Insurance uses only 190 training samples, leading to larger performance variance)
6. **Validation is confined to synthetic/public datasets**, with no deployment testing in real enterprise systems

## Related Work & Insights

| Method | Requires Labels | Requires References | Requires LLM Judge | Model Scale | Multi-Turn |
|------|:--------:|:----------:|:--------------:|:--------:|:--------:|
| ROUGE/BLEU | ✗ | ✓ | ✗ | — | ✗ |
| BERTScore | ✗ | ✓ | ✗ | ~110M | ✗ |
| HelpSteer | ✓ | ✗ | ✗ | — | ✓ |
| G-EVAL | ✗ | ✗ | ✓ | >70B | ✓ |
| DeepEval | ✗ | ✗ | ✓ | >70B | ✓ |
| **Ours** | **✗** | **✗** | **✗** | **8B** | **✓** |

**Additional insights**:
- **Online intervention potential**: Completeness detection can terminate unproductive dialogues early to conserve tokens; uncertainty quantification can trigger human escalation
- **Response trees + sampling strategies**: If the LLM's sampling strategy is known, response trees can provide statistical guarantees over output probabilities
- **Complementarity with conformal prediction**: This work targets the unsupervised setting, while conformal methods offer supervised guarantees; the two approaches are combinable
- **LoRA fine-tuning as a distribution adapter**: A general paradigm—fine-tuning an 8B model with small-data LoRA to adapt to domain-specific token distributions

## Rating
- Novelty: ⭐⭐⭐⭐ — All three metrics offer individual contributions; LLM-guided clustering and response trees represent novel advances
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six datasets and multiple ablations, though response trees lack direct effectiveness validation
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and rigorous; appendix is comprehensive
- Value: ⭐⭐⭐⭐ — Fills a gap in unsupervised evaluation of multi-turn objective-driven dialogues with strong practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)
- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/llm_nlp/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](../../ACL2026/llm_nlp/when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](../../ICML2026/llm_nlp/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](../../NeurIPS2025/llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)

</div>

<!-- RELATED:END -->
