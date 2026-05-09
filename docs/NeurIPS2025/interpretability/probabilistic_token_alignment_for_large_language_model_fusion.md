---
title: >-
  [Paper Note] Probabilistic Token Alignment for Large Language Model Fusion
description: >-
  [NeurIPS 2025][Optimal Transport] This work reformulates the token alignment problem in LLM fusion as an Optimal Transport (OT) problem, replacing traditional hard mappings with soft probabilistic alignment via dynamic token pairing and the Sinkhorn algorithm. On 78 tasks across 6 benchmarks, PTA-LLM achieves an average improvement of +1.72% over FuseLLM, while substantially mitigating performance degradation on challenging tasks (from −13.04% to −4.07%).
tags:
  - NeurIPS 2025
  - Optimal Transport
  - Sinkhorn Algorithm
  - Probabilistic Token Alignment
  - Logit Fusion
  - Cross-Architecture Model Fusion
  - Knowledge Fusion
date: 2026-05-08
content_hash: bcc92a5d557d80c3
---

# Probabilistic Token Alignment for Large Language Model Fusion

**Conference**: NeurIPS 2025
**arXiv**: [2509.17276](https://arxiv.org/abs/2509.17276)
**Code**: [runjia.tech/neurips_pta-llm](https://runjia.tech/neurips_pta-llm)
**Area**: Interpretability
**Keywords**: Optimal Transport, Sinkhorn Algorithm, Probabilistic Token Alignment, Logit Fusion, Cross-Architecture Model Fusion, Knowledge Fusion

## TL;DR
This work reformulates the token alignment problem in LLM fusion as an Optimal Transport (OT) problem, replacing traditional hard mappings with soft probabilistic alignment via dynamic token pairing and the Sinkhorn algorithm. On 78 tasks across 6 benchmarks, PTA-LLM achieves an average improvement of +1.72% over FuseLLM, while substantially mitigating performance degradation on challenging tasks (from −13.04% to −4.07%).

## Background & Motivation

**Background**: Training LLMs from scratch is prohibitively expensive, making model fusion an efficient alternative for building stronger baselines. Mainstream approaches fall into three categories: model ensembling, weight merging, and knowledge fusion.

**Limitations of Prior Work**: Knowledge fusion methods such as FuseLLM rely on manually defined hard token mappings (based on minimum edit distance), which suffer from two critical flaws: ❶ hard mappings oversimplify alignment and fail to capture diverse token correspondence patterns across different contexts, introducing bias and reducing learning capacity; ❷ the top-$k$ token sets of source and target models are aligned independently without considering probability values or the overall distribution, yielding only locally optimal solutions.

**Key Challenge**: Model ensembling incurs high inference overhead from running multiple models simultaneously; weight merging requires identical architectures and thus lacks generality; knowledge fusion is flexible but its coarse token alignment degrades performance on certain tasks.

**Key Insight**: Although different LLMs may have distinct vocabularies and token IDs, their logit probability distributions encode semantically similar knowledge—aligning at the distribution level rather than through string-level hard matching enables more coherent fusion.

**Core Idea**: Token alignment is reformulated as an optimal transport problem, solved via the Sinkhorn algorithm to obtain a global transport plan that enables soft probabilistic alignment of logit distributions.

## Method

### Overall Architecture
PTA-LLM follows the knowledge fusion paradigm: the probability distribution matrices of multiple source LLMs are aligned and fused into a target LLM (Llama-2 7B) via token alignment, with a weighted combination training objective $\mathcal{L} = \lambda \mathcal{L}_{\text{CLM}} + (1-\lambda) \mathcal{L}_{\text{Fusion}}$. The core innovation lies in a two-stage probabilistic token alignment pipeline: dynamic token pairing → OT-based probabilistic alignment.

### Key Designs

1. **Dynamic Token Pairing**:

    - **Function**: Resolves sequence length discrepancies caused by different tokenizers across source and target models, finding the optimal token pairing.
    - **Core Recurrence**: $f(k,j) = \min\{f(k-1,j)+c, f(k,j-1)+c, f(k-1,j-1)+c\}$, where $c(\mathcal{B}_k, \mathcal{A}_j)$ is a predefined distance.
    - **Key Contribution**: Relaxes the conventional one-to-one constraint, allowing a single source token to correspond to multiple target tokens (and vice versa), accommodating granularity differences across tokenization schemes.
    - **Computational Efficiency**: Dynamic programming avoids brute-force search over $L \times N$ candidates.

2. **Probabilistic Token Alignment via OT**:

    - **Function**: Performs logit-level soft alignment between paired tokens, resolving the token ID mismatch problem.
    - **Formulation**: For each token pair $(\mathcal{A}_j \in \mathbb{R}^{V_s}, \mathcal{B}_k \in \mathbb{R}^{V_t})$, the optimal transport plan is obtained by solving $\hat{\mathcal{T}} = \arg\min_{\mathcal{T} \geq 0} \sum_{x=1}^n \sum_{y=1}^m c_{xy} \mathcal{G}_{xy}$.
    - **Marginal Constraints**: $\sum_y \mathcal{G}_{xy} = \mathcal{A}_j[x]$, $\sum_x \mathcal{G}_{xy} = \mathcal{B}_k[y]$—ensuring conservation of probability mass.
    - **Cost Matrix**: $c_{xy}$ is defined as the minimum edit distance between decoded texts, jointly optimized at both surface and logit levels.
    - **Fused Logit Selection**: For each row of the transport matrix, the target index corresponding to the maximum value is selected; probabilities mapped to the same target index are accumulated.
    - **Window Size**: Default top-10 logits ($n=m=10$).

3. **Sinkhorn Algorithm**:

    - Initializes the transport matrix as $\mathcal{T} = \exp(-\lambda C)$.
    - Alternately scales rows and columns to match the marginal distributions of source and target token probabilities.
    - Convergence threshold of $10^{-5}$ yields the best results (stricter constraints produce more coherent fusion).

4. **Fusion Strategy**:

    - Cross-entropy loss is used to evaluate the prediction quality of each source LLM.
    - The source distribution matrix with the lowest cross-entropy is selected (MinCE strategy outperforms AvgCE).
    - Combination weight $\lambda = 0.8$—a lower value implies greater reliance on the fused matrix.

### Loss & Training
- Target model: Llama-2 7B
- Source models: OpenLLaMA 7B + MPT 7B
- Training data: MiniPile, batch size 256, max sequence length 2048
- Implementation: PyTorch + HuggingFace Transformers + FlashAttention

## Key Experimental Results

### Main Results (6 Benchmarks, 78 Tasks)

| Benchmark [Tasks] | OpenLLaMA | MPT | Llama-2 | FuseLLM | PTA-LLM | Gain |
|---|---|---|---|---|---|---|
| GSM [1] | 7.81 | 9.17 | 14.18 | 14.56 | **14.71** | +1.03% |
| BBH [27] | 33.87 | 33.38 | 39.70 | 41.01 | **41.08** | +0.17% |
| MultiPL-E [10] | 18.11 | 17.26 | 14.63 | 15.56 | **15.88** | +2.06% |
| MMLU [17] | 42.11 | 27.84 | 46.94 | 48.77 | **49.38** | +1.25% |
| ToxiGen [14] | 18.94 | 18.42 | 18.56 | 18.19 | **18.89** | +3.85% |
| TyDi QA [9] | 27.32 | 22.11 | 31.42 | 32.99 | **34.07** | +3.27% |
| **Average [78]** | 24.69 | 21.36 | 27.57 | 28.51 | **29.00** | **+1.72%** |

### Stability on Challenging Tasks (Tasks Where FuseLLM Degrades)

| Task | Llama-2 | FuseLLM | PTA-LLM | PTA vs Llama-2 |
|---|---|---|---|---|
| Causal Judgement [BBH] | 50.80 | 46.52 (−8.43%) | **50.80** | +0.00% |
| Geometric Shapes [BBH] | 34.40 | 22.80 (−33.72%) | **26.80** | −22.09% |
| Tracking Shuffled (7) [BBH] | 11.20 | 10.40 (−7.14%) | **14.00** | **+25.00%** |
| Chemistry [MMLU] | 35.97 | 34.98 (−2.75%) | **36.96** | +2.75% |
| Arabic [TyDi QA] | 8.47 | 5.65 (−33.29%) | **7.49** | −11.57% |
| **Average (7 Tasks)** | 30.22 | 26.28 (−13.04%) | **28.99** | **−4.07%** |

### Quantitative Interpretability Analysis

| Metric | FuseLLM | PTA-LLM | Description |
|---|---|---|---|
| Inner Distance | 257.83 | **239.44** | Fused tokens are more compact |
| Center Distance | 136.95 | **22.25** | Closer to the target token distribution |

### Ablation Study

| Parameter | Choice | BBH | ME | MMLU |
|---|---|---|---|---|
| OT Convergence Threshold | 1e-4 → **1e-5** | +1.33% | −0.38% | +0.80% |
| Alignment Window | 5 → **10** | +0.97% | +1.70% | Flat |
| Combination Weight | 0.9 → **0.8** | +1.71% | +1.04% | +0.92% |
| Fusion Function | AvgCE → **MinCE** | +1.38% | +1.23% | +1.00% |

### Key Findings
- **Consistent Improvement**: PTA-LLM surpasses FuseLLM on all 6 benchmarks, with an average gain of +1.72% across 78 tasks.
- **Substantial Mitigation on Hard Tasks**: On 7 tasks where FuseLLM degrades, the average regression is reduced from −13.04% to −4.07% (an 8.97% improvement in mitigation).
- **Notable Gains in Safety and Multilingual Settings**: ToxiGen +3.85%, TyDi QA +3.27%—FuseLLM even falls below the original Llama-2 on safety benchmarks, whereas PTA-LLM fully recovers this gap.
- **More Source Models Yield Better Performance**: Scaling from 1 to 2 to 3 source models leads to consistent performance improvements.

## Highlights & Insights
- **Elegant Problem Reformulation**: Recasting token alignment from "string matching" to "probability distribution transport" naturally introduces the OT framework—both mathematically principled and practically effective.
- **Soft Alignment vs. Hard Mapping Intuition**: Hard mapping resembles "relocation" (changing position without redistributing mass), whereas probabilistic alignment resembles "logistics routing" (distributing probability mass to optimal positions)—an intuition well supported by the paper's visualizations.
- **Quantitative Interpretability Validation**: Token distributions are visualized via Isomap+PCA dimensionality reduction, with inner and center distances quantified, providing a distributional explanation for why the alignment is effective.
- **MinCE Outperforms AvgCE**: Selecting the source model with the best current prediction (rather than averaging across models) proves more effective, suggesting that "selection" matters more than "mixture."

## Limitations & Future Work
- **Limited to 7B-Scale Models**: All experiments use Llama-2 7B / OpenLLaMA 7B / MPT 7B; the approach has not been validated at larger scales.
- **Modest Absolute Gains**: An average relative improvement of +1.72% may be insufficient for certain applications.
- **Marginal Gains on BBH**: The +0.17% improvement on BBH suggests that when source models are weak in a domain, more precise alignment may still propagate noisy knowledge.
- **Top-10 Window Limitation**: OT computation is restricted to the top-10 logits, potentially discarding information from long-tail tokens.
- **Future Directions**: (1) Scaling to larger models to validate generalizability; (2) exploring cost functions incorporating semantic embedding distances; (3) dynamically adjusting the OT window size; (4) combining with parameter-efficient methods such as LoRA to reduce training costs.

## Related Work & Insights
- **vs. FuseLLM**: PTA-LLM is a direct improvement over FuseLLM; the key difference lies in replacing hard-mapped token alignment with OT-based soft probabilistic alignment.
- **vs. Model Ensemble**: Ensembling requires maintaining multiple models at inference time (high memory and latency), whereas PTA-LLM retains only a single model after fusion.
- **vs. Weight Merging (e.g., SWA)**: Weight merging requires identical architectures, while PTA-LLM supports cross-architecture fusion.
- **vs. Traditional Token Alignment**: Conventional methods rely solely on character-level edit distance for hard mapping; PTA-LLM additionally leverages logit probability distributions to achieve globally optimal matching.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing OT into token alignment is a natural yet previously unexplored direction; the problem formulation is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 benchmarks, 78 tasks, challenging task analysis, ablation studies, and visualizations, though limited to 7B-scale models.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, mathematical derivations are complete, and visual analyses are convincing.
- **Value**: ⭐⭐⭐ A practical improvement to the knowledge fusion paradigm, though the absolute performance gains are limited and the applicable scenarios are somewhat narrow.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)

<!-- RELATED:END -->
