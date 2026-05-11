---
title: >-
  [Paper Note] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners
description: >-
  [NeurIPS 2025][LLM Evaluation][self-improvement reasoning] This work identifies that random data sampling in STaR (Self-Taught Reasoner) leads to severely imbalanced observation training frequencies—easy problems are ove…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "self-improvement reasoning"
  - "STaR"
  - "adaptive sampling"
  - "curriculum learning"
  - "data efficiency"
date: 2026-05-08
content_hash: f6448f068c0c2434
---

# AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners

**Conference**: NeurIPS 2025
**arXiv**: [2505.16322](https://arxiv.org/abs/2505.16322)
**Code**: [GitHub](https://github.com/reiss-koh/AdaSTaR)
**Area**: LLM Evaluation
**Keywords**: self-improvement reasoning, STaR, adaptive sampling, curriculum learning, data efficiency

## TL;DR
This work identifies that random data sampling in STaR (Self-Taught Reasoner) leads to severely imbalanced observation training frequencies—easy problems are over-trained while hard problems are under-trained—and proposes AdaSTaR, which combines adaptive diversity sampling (prioritizing under-trained samples) with adaptive curriculum sampling (adjusting difficulty based on model strength) to achieve the highest accuracy on all 6 benchmarks while reducing training FLOPs by 58.6%.

## Background & Motivation

**Background**: STaR (Self-Taught Reasoner) / RFT (Rejection sampling Fine-Tuning) is the core training paradigm for LLM self-improvement in reasoning—models generate chain-of-thought (CoT), verify correct answers, and fine-tune accordingly. It is adopted by frontier models such as DeepSeek-R1 and Kimi k1.5.

**Limitations of Prior Work**: STaR uses random observation sampling, resulting in: (a) easy problems being trained repeatedly (10–13 times) while hard problems are trained rarely (1–2 times) → wasted computation; (b) 72% of under-trained and 91% of over-trained observations remain unchanged after 3 iterations → a persistent, self-non-correcting problem.

**Key Challenge**: Directly prioritizing hard problems increases false positives (correct answers but incorrect CoT) → a balance between training diversity and CoT quality is required.

**Key Insight**: Two adaptive principles—diversity (prioritize under-trained samples) and curriculum (sample more easy problems when the model is weak).

**Core Idea**: A hierarchical min-heap sorts observations by (last sampling time + difficulty), while training accuracy $\alpha$ serves as a curriculum regulator to automatically balance difficulty.

## Method

### Overall Architecture
An adaptive sampling module is inserted into the data sampling step of the STaR loop: maintain per-observation statistics $(\tilde{t}_i, w_i)$ → sort via a hierarchical min-heap → prioritize under-trained and difficult observations → curriculum regulation limits the proportion of hard samples → proceed with standard training.

### Key Designs

1. **Adaptive Diversity Sampling (AdaD)**:

    - Function: Ensures all observations receive balanced training opportunities.
    - Core data structure: Hierarchical min-heap HieMinHeap, with sorting key $(\tilde{t}_i, w_i)$:
        - First priority: $\tilde{t}_i$ (iteration of last sampling)—earlier sampled observations are prioritized → promotes diversity.
        - Second priority: $w_i$ (win rate statistic)—within the same iteration, harder observations (lower win rate) are prioritized → focuses on difficult problems.
    - Win rate statistic: $w_i = \frac{1}{K}\sum_{k=1}^K \mathbb{I}[y_i = \hat{y}_i]$—the proportion of correct answers across $K$ CoT samples at the last sampling step.
    - Key advantage: Computing $w_i$ incurs **zero additional overhead**, as the $K$ CoT samples are an inherent part of STaR.
    - Non-exhaustive sampling (Remark 1): A while-loop terminates upon collecting $\beta^t$ correct samples, avoiding unnecessary computation.

2. **Adaptive Curriculum Sampling (AdaC)**:

    - Function: Suppresses excessive hard samples when the model is weak, preventing a rise in false positives.
    - Mechanism: Uses training accuracy $\alpha \in [0,1]$ at the current iteration as a proxy for model strength.
    - Implementation: At each iteration, $m$ observations are sampled, but statistics are updated only for the top $\lfloor m \alpha^2 \rfloor$ observations.
    - Effect: When $\alpha$ is low, most observation statistics are not updated → old priorities are retained → those observations will be re-sampled → effectively increases the proportion of easy samples.
    - $f(\alpha) = \alpha^2$: Allows repeated easy problems when the model is weak, and rapidly relaxes the constraint as the model strengthens.
    - Zero computational overhead: $\alpha$ is a byproduct of the training step.

### Loss & Training
- Base models: Llama 3.2 3B, Qwen 2.5 3B, Gemma 7B
- Cumulative STaR (STaR-Acc) is adopted: training continues from the previous iteration's model.
- $K=2$ (standard CoT sampling count) for fair comparison.
- Evaluation: zero-shot greedy decoding.

## Key Experimental Results

### Main Results (Llama 3.2 3B)

| Method | ARC-C | CQA | CLadder | ANLI | GSM8K | SVAMP | Avg Acc. | Avg FLOPs |
|--------|-------|-----|---------|------|-------|-------|----------|-----------|
| STaR | baseline | baseline | baseline | baseline | - | baseline | baseline | baseline |
| STaR-Acc | better | better | better | better | - | better | better | higher |
| B-STaR* | good | good | good | good | - | good | good | much higher |
| **AdaSTaR** | **best** | **best** | **best** | **best** | **best** | **best** | **6/6 best** | **−58.6%** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| STaR-Acc (baseline) | baseline | random sampling |
| AdaD (diversity only) | +accuracy but ↑false positives | prioritizes hard problems but degrades CoT quality |
| **AdaSTaR (AdaD + AdaC)** | **best** | curriculum regulation eliminates false positive increase |
| $f(\alpha) = \alpha$ | near-best but slightly worse | $\alpha^2$ is more conservative and performs better |

### Key Findings
- **Best on all 6/6 benchmarks**: AdaSTaR achieves the highest accuracy on every evaluated dataset.
- **58.6% FLOPs reduction**: Computation is reduced by nearly 60% compared to the strongest accuracy baseline.
- **Quantified impact of training diversity**: AdaD alone increases false positives by 9%; adding AdaC recovers this.
- **Strong generalization**: Consistently effective across three model families—Llama, Qwen, and Gemma.
- **Simultaneous efficiency and effectiveness gains**: Performance is not traded for efficiency; both are improved concurrently.

## Highlights & Insights
- **Zero-overhead difficulty estimation**: Win rates are computed using the $K$ CoT samples inherent to STaR, requiring no additional forward passes—an elegant reuse of existing computation.
- **Training accuracy as a curriculum signal**: $\alpha$ is a free byproduct of the training process; using it to regulate sampling difficulty constitutes zero-cost adaptive curriculum learning.
- **Hierarchical min-heap data structure**: Encodes both diversity ($\tilde{t}_i$) and difficulty ($w_i$) in a hierarchical heap, achieving $O(\log N)$ sampling efficiency.
- **In-depth analysis of STaR training dynamics**: The work uncovers the persistent nature of training frequency imbalance (72%/91% remaining unchanged), an observation with significant value for understanding STaR systems.

## Limitations & Future Work
- **Outcome verification only**: Only final answer correctness is checked; process reward models (PRMs) are not utilized.
- **Manual selection of $\alpha^2$**: The curriculum function $f(\alpha) = \alpha^2$ is manually chosen and may not be optimal.
- **No comparison with RL-based methods**: AdaSTaR targets the SFT/STaR pipeline and does not directly compare with RL-based methods such as GRPO.
- **Future directions**: (1) Integrate PRMs for more precise false positive filtering; (2) learn $f(\alpha)$ rather than selecting it manually; (3) transfer the adaptive sampling concept to RL-based reasoning training.

## Related Work & Insights
- **vs. STaR**: AdaSTaR is a sampling-enhanced variant of STaR; the core contribution lies in the adaptive data sampling strategy.
- **vs. ReSTEM**: ReSTEM also addresses over/under-training via truncation thresholds; AdaSTaR's hierarchical heap + curriculum approach demonstrates superior performance.
- **vs. B-STaR**: B-STaR employs PRMs for finer-grained verification at the cost of substantial computation; AdaSTaR requires no additional reward model.
- **vs. Curriculum Learning**: Traditional curriculum learning requires predefined difficulty metrics; AdaSTaR naturally estimates difficulty via win rates.

## Rating
- Novelty: ⭐⭐⭐⭐ Integrates adaptive sampling and curriculum learning into STaR; zero-overhead difficulty estimation is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks × 3 model families, extensive baselines, dual metrics of FLOPs and accuracy, thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth motivation analysis (quantified and persistent training imbalance), precise method description (algorithmic pseudocode + complexity analysis).
- Value: ⭐⭐⭐⭐⭐ A practical improvement to the widely adopted STaR/RFT training paradigm; the 58.6% efficiency gain is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[CVPR 2026\] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](../../CVPR2026/llm_evaluation/ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)

</div>

<!-- RELATED:END -->
