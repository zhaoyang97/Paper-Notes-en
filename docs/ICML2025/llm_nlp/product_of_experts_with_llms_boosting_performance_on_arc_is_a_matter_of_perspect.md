---
title: >-
  [Paper Note] Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective
description: >-
  [ICML 2025][LLM (Other)][Product-of-Experts] By employing the LLM simultaneously as a candidate generator and a scorer, this work leverages a DFS-based search algorithm to generate high-probability candidate solutions and subsequently utilizes Product of Experts (PoE) scoring under multi-perspective augmentation to select the optimal answer. This approach achieves an open-source SOTA accuracy of 71.6% on the ARC-AGI public evaluation set, surpassing the average human performa…
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Product-of-Experts"
  - "ARC-AGI"
  - "Depth-First Search"
  - "Data Augmentation"
  - "Test-Time Training"
date: 2026-05-08
content_hash: bc53262d34e1e112
---

# Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective

**Conference**: ICML 2025  
**arXiv**: [2505.07859](https://arxiv.org/abs/2505.07859)  
**Code**: [GitHub](https://github.com/da-fr/Product-of-Experts-ARC-Paper)  
**Area**: LLM/NLP  
**Keywords**: Product-of-Experts, ARC-AGI, Depth-First Search, Data Augmentation, Test-Time Training

## TL;DR

By employing the LLM simultaneously as a candidate generator and a scorer, this work leverages a DFS-based search algorithm to generate high-probability candidate solutions and subsequently utilizes Product of Experts (PoE) scoring under multi-perspective augmentation to select the optimal answer. This approach achieves an open-source SOTA accuracy of 71.6% on the ARC-AGI public evaluation set, surpassing the average human performance (60.2%) with a single-task inference cost of only around $0.02.

## Background & Motivation

The Abstraction and Reasoning Corpus (ARC-AGI), proposed by Chollet in 2019, is an abstract reasoning benchmark containing 900 reasoning tasks (400 training + 400 public evaluation + 100 private evaluation). Each task consists of several input-output grid pairs (ranging from 1×1 to 30×30, with 10 colors) and a test input. Models are required to infer transformation rules from the examples and apply them to the new input. While simple for humans (averaging 60.2%), this task is extremely challenging for AI.

Limitations of Prior Work:

**Scale is not everything**: Although LLMs perform exceptionally well on many tasks, scaling up the model size alone does not fundamentally solve the abstract reasoning challenges of ARC.

**Representation and tokenization hinder performance**: A body of research indicates that many failure modes of LLMs stem from data representation and tokenization methods rather than a lack of reasoning capability itself.

**Limitations of autoregressive architectures**: Autoregressive models predict the next token based only on previously generated tokens. This can cause high-confidence errors when global information is required (e.g., solving a Sudoku puzzle internally before predicting the first cell).

**High cost of closed-source methods**: OpenAI’s o3 reaches 82.8% on ARC-AGI, but inherits a computational cost of $17 per task and lacks reproducibility.

The Core Idea of this paper is: **Models often already possess the latent capability to solve ARC, and the key is to create conditions that allow these capabilities to be reliably expressed.** Reviewing the same problem from multiple perspectives via semantic-preserving augmentation of inputs can effectively boost reasoning robustness.

## Method

### Overall Architecture

The method consists of three stages, with data augmentation integrated throughout:

1. **Training Phase**: Initial fine-tuning on RE-ARC data followed by task-specific Test-Time Training (TTT).
2. **Generation Phase**: Systematically searching for high-probability candidate solutions across multiple augmented perspectives using DFS.
3. **Selection Phase**: Scoring candidate solutions using Product of Experts under multiple perspectives to select the final answer.

Formal definition: Given a task $p = ((x_i, y_i)_{i=1}^k, \hat{x})$, the goal is to find $s_p^* = \arg\max_{s \in S_p} P(s|p)$. A family of semantic-preserving augmentation transformations is defined as $\Phi = \{\phi_1, \ldots, \phi_m\}$ (satisfying $P(s|p) = P(\phi_j(s)|\phi_j(p))$), which includes rotations, reflections, color permutation, and example order shuffling.

### Key Designs

#### 1. Data Representation and Tokenization

- The vocabulary size is **significantly reduced from 120,000+ tokens to only 64 tokens**. It comprises letters A-Z/a-z (excluding I, O, i, o to avoid confusion), numbers 0-9 (encoding 10 colors), newline characters, input/output markers, and start/end/padding tokens.
- Each grid cell is represented by one token without number merging or compression (avoiding tokenization artifacts).
- A short block of extra tokens (an alphabetical sequence) is added at the beginning of the task to act as a "computation buffer." During fine-tuning, the model learns to utilize these tokens to improve subsequent predictions.
- This simplification drastically reduces the embedding layer size and eliminates the ambiguity caused by number merging in BPE tokenization.

#### 2. DFS Candidate Generation

Unlike multinomial sampling, a **Threshold-Based Depth-First Search (DFS)** is used to systematically explore the solution space:

$$\mathcal{C}_{p,T} := \{s \in S_p \mid \exists \phi_j \in \Phi: \hat{P}(\phi_j(s)|\phi_j(p)) > T \}$$

- DFS is performed over the token sequence of the solution. Any partial path is pruned immediately when its cumulative probability falls below the threshold $T$.
- DFS is run independently for all 16 augmentations ($D_8$ symmetry group × 2 sets of random color permutations and example reorderings).
- Identical solutions generated from different augmentations (modulo the augmentation transformation) are merged into a single candidate.
- **Caching intermediate computations** accelerates inference: For the second and subsequent augmentations, the current best candidate is first passed as an initial guess in a forward pass (which is much faster than token-by-token generation) before starting the backtracking search.
- Compared to multinomial sampling: DFS with $T=9\%$ achieves a comparable coverage rate (76.0% vs 77.3%) in 1/4 of the inference time (9:32h vs 39:47h) with only half the number of false positives.
- Compared to Beam Search: Under equivalent accuracy, VRAM requirements are halved (7.3GB vs 14GB), and speed is 4 times faster.

#### 3. Product of Experts Candidate Ranking

After generating the candidate pool, **product-based scoring under multi-perspective augmentation** is used to select the optimal solution:

$$\text{score}_{\text{agg}}(s) = \prod_{\phi_j \in \Phi} \hat{P}(\phi_j(s)|\phi_j(p))$$

Key ideas:

- This step **does not perform generative sampling**; instead, it directly computes the log-likelihood of each candidate under all augmented inputs.
- 16 **newly randomized** augmentations (independent of the generation phase) are used for scoring.
- The product form is highly sensitive to outliers: even if a candidate has a high probability in most perspectives, a low probability assigned by a single perspective will filter it out.
- The final selection is $s_p^* = \arg\max_{s \in \mathcal{C}_{p,T}} \text{score}_{\text{agg}}(s)$.

This is equivalent to geometric mean ensembling: $\bar{P}(s) = \frac{1}{Z} \prod_{j=1}^m [\hat{P}_j(s)]^{1/m}$

**Theoretical Guarantee (Theorem 4.1)**: The KL divergence of the PoE ensemble satisfies $\text{KL}(P \| \bar{P}) = \frac{1}{m}\sum_{j=1}^m \text{KL}(P \| \hat{P}_j) + \log Z$, where $\log Z \leq 0$. This implies that the error of PoE is bounded by the average error of each augmented predictor, and when augmented predictors disagree (which autoregressive architectures naturally do), PoE provides a tighter estimate.

### Loss & Training

#### Initial Fine-Tuning

- **Base Model**: Mistral-NeMo-Minitron-8B-Base (best performing); Llama-3.2-3B was also evaluated.
- **Training Data**: Exclusively uses RE-ARC data (massive training samples generated by procedural generators of the 400 training tasks) to avoid "concept leakage."
- **LoRA Configuration**: Rank 256, applied to all layers (including input/output embeddings), 4-bit quantization, gradient checkpointing.
- **Gradient Computation**: Gradients are calculated only on the output grids (from the second example onward) and the final solution, not on the input grids.
- **Data Augmentation**: All $D_8$ symmetry transformations + color permutations + example reordering.
- **Training Budget**: 1200 epochs for the NeMo model (98h, 1×H100), 368 epochs for the Llama model (15h, 1×H100).

#### Test-Time Training (TTT)

- For each task in the evaluation set, secondary fine-tuning is performed using its example pairs.
- LoRA rank 32, 64 steps, batch size 1.
- Starts from the initially fine-tuned model.
- Average time per task is 51 seconds (NeMo, RTX 4090) / 12 seconds (Llama).
- TTT alone more than doubles performance (NeMo: 18.3% → 44.5%).

## Key Experimental Results

### Main Results

| Method | Public Eval Accuracy | Open Source |
|------|------|------|
| o1-preview | 21% | ✗ |
| Ryan Greenblatt | 42% | ✗ |
| Jeremy Berman | 58.5% | ✗ |
| GPT o3 | 82.8% | ✗ |
| Human Average | 60.2% | - |
| TTT | 53.5% | ✓ |
| BARC | 56.75% | ✓ |
| TTT+BARC | 62.8% | ✓ |
| **Ours** | **71.6%** | ✓ |

### Ablation Study

| Model | Baseline | +TTT | +16xAug | +PoE | +DFS |
|------|------|------|------|------|------|
| Llama-3.2-3B | 14.9% | 40.9% | 52.9% | 59.5% | 61.4% |
| NeMo-Minitron-8B | 18.3% | 44.5% | 62.5% | 67.6% | 71.6% |

Sampling vs. Selection Strategy Comparison (NeMo-Minitron-8B, RTX 4090):

| Sampling Method | Coverage | Avg. Candidates | Sampling Time | VRAM | PoE Accuracy | Total Time |
|------|------|------|------|------|------|------|
| Greedy | 70.8% | 6.7 | 9:39h | 7.0 GB | 67.6% | 18:52h |
| Stochastic (4x) | 77.3% | 17.6 | 39:47h | 7.0 GB | 70.8% | 58:55h |
| Beam Search (4x) | 79.0% | 34.7 | 37:36h | 14.0 GB | 71.6% | 71:39h |
| **DFS T=9% (Ours)** | **76.0%** | **9.3** | **9:32h** | **7.3 GB** | **71.6%** | **20:50h** |
| DFS T=0.5% | 83.5% | 84.7 | 80:56h | 7.3 GB | 71.8% | 134:43h |

### Key Findings

1. **PoE significantly outperforms other aggregation strategies**: Under $T=9\%$ DFS, PoE is 5% higher than average probability scoring (71.6% vs 66.6%) and consistently wins across all sampling methods.
2. **DFS is the most efficient sampling method**: It achieves the same accuracy (71.6%) as Beam Search (4x) but in only 1/3.4 of the time (20:50h vs 71:39h) and with half the VRAM.
3. **The contribution of each component is additive**: TTT (+26.2%), Aug (+18.0%), PoE (+5.1%), DFS (+4.0%) progressively improve performance, demonstrating that the modules are complementary.
4. **Strong generalization ability**: It achieves 73.3% on ConceptARC (without hyperparameter tuning), 53% on Sudoku tasks (far surpassing SOTA LLMs' < 3%), and correct Sudoku solutions are selected 100% of the time when sampled.
5. **Cost is only a thousandth of o3**: $0.02/task vs $17/task.

## Highlights & Insights

- **"Perspective as an Expert"**: The core innovation is treating semantic-preserving augmentations as different "experts." By exploiting the inconsistencies naturally produced by autoregressive models under different input orderings, it establishes an effective ensemble system using a single model.
- **Dual Roles of the LLM**: The same model acts as both generator and scorer. The generation phase utilizes DFS to search high-probability paths (forward capability), and the scoring phase utilizes log-likelihood calculation (verification capability), with the two complementing each other.
- **Clever Application of DFS**: The next-token probabilities of the LLM are used as search heuristics for DFS. Threshold pruning avoids wasteful exploration, while its determinism guarantees that no solutions above the threshold are missed.
- **Counterintuitively Effective Minimalist Tokenization**: Reducing the vocabulary from over 120k tokens to 64 looks like a loss of information, but it systematically eliminates BPE merging ambiguities, minimizes embedding size overhead, and boosts fine-tuning efficiency.
- **"Computation Buffer" Tokens**: Adding an alphabet sequence before the input allows the model to learn to treat it as an implicit scratchpad—an interesting finding that could potentially be generalized.

## Limitations & Future Work

1. **Augmentations are limited to structured domains**: Present augmentations (rotations, reflections, color swap) are highly specific to grid-based reasoning. Generalizing to areas like natural language inference requires designing new types of semantic-preserving transformations.
2. **Reliance on hand-crafted augmentations**: The set of augmentations $\Phi$ requires domain knowledge to define, lacking a mechanism to automatically discover novel valid transformations.
3. **Threshold $T$ requires tuning**: The probability threshold for DFS is a hyperparameter and its optimal value varies across models and training configurations.
4. **Gap with o3 persists**: 71.6% vs 82.8%, although the authors bridged much of this gap at an extremely low cost.
5. **Promising Future Directions**: Text-domain augmentations (e.g., phrasing/style shifts), and evaluation on wider structured reasoning tasks such as logical deduction and program synthesis.

## Related Work & Insights

- **TTT (Akyürek et al., 2024)**: Test-time training was the first to substantially boost LLM performance on ARC; it is adopted as a key component in this work.
- **BARC (Li et al., 2025)**: Distinguishes between Induction (inferring code programs) and Transduction (generating solutions directly) reasoning paths. This work follows the Transduction path.
- **Product of Experts (Hinton, 1999, 2002)**: Classical ensemble theory, elegantly combined here with data augmentation for structured reasoning tasks.
- **RE-ARC (Hodel, 2024)**: Provides procedural generators for 400 training tasks, making large-scale training possible without introducing concept leakage.
- **Key Takeaway**: The framework of PoE coupled with semantic-preserving augmentations has general potential. As long as symmetry group transformations can be defined for a problem class, a single model can be converted into an effective ensemble.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Clever assembly of PoE, DFS, and augmentations; "perspective as expert" is an intriguing idea |
| Technical Depth | ⭐⭐⭐⭐ | Backed by a clean theoretical analysis, gracefully designed DFS search |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Detailed ablations, comparison against multiple baselines, cross-domain verification (ConceptARC + Sudoku) |
| Value | ⭐⭐⭐⭐⭐ | Fully open-source, highly cost-effective (runnable on an RTX 4090), and highly reproducible |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, maintaining a strong balance between theory and experiments |
| Overall | ⭐⭐⭐⭐½ | A masterpiece combining engineering and theory; sets a new open-source ARC SOTA with massive practical value |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](../../ECCV2024/llm_nlp/promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)
- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](../../ACL2025/llm_nlp/unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](../../ACL2025/llm_nlp/praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)
- [\[ICML 2026\] Multi-Agent Teams Hold Experts Back: Why Self-Organized LLM Teams Fail to Retain "Experts"](../../ICML2026/llm_nlp/multi-agent_teams_hold_experts_back.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)

</div>

<!-- RELATED:END -->
