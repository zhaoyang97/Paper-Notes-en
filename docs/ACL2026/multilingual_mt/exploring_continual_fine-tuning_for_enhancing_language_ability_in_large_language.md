---
title: >-
  [Paper Note] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Continual Fine-tuning] This paper proposes a two-phase continual fine-tuning (CFT) framework—first fine-tuning on English instruction data…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Continual Fine-tuning"
  - "Multilingual Adaptation"
  - "Catastrophic Forgetting"
  - "Dataset Similarity"
  - "Representation Drift"
date: 2026-05-08
content_hash: 08f6cf2e3361b335
---

# Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2410.16006](https://arxiv.org/abs/2410.16006)  
**Code**: None  
**Area**: Multilingual / Continual Learning
**Keywords**: Continual Fine-tuning, Multilingual Adaptation, Catastrophic Forgetting, Dataset Similarity, Representation Drift

## TL;DR

This paper proposes a two-phase continual fine-tuning (CFT) framework—first fine-tuning on English instruction data, then on multilingual data—and finds that instruction similarity between the two phases is the key factor determining whether English capability degrades. Generative replay and heuristic layer freezing are shown to effectively mitigate representation drift and English forgetting caused by dissimilar datasets.

## Background & Motivation

**Background**: The multilingual user base of LLMs continues to grow, yet models perform substantially worse on low-resource languages. Training from scratch is prohibitively expensive, making fine-tuning the preferred approach. Joint fine-tuning on mixed multilingual datasets introduces an English bias, while fine-tuning exclusively on non-English data leads to catastrophic forgetting of English performance.

**Limitations of Prior Work**: (1) Existing methods such as InstructAlign require parallel data and data from previous tasks, incurring high computational overhead; (2) Direct fine-tuning on mixed datasets results in an imbalance between English and multilingual performance; (3) There is no systematic understanding of under what conditions multilingual fine-tuning degrades English capability; (4) Regularization methods such as EWC require storing both old and new parameters, reducing computational efficiency.

**Key Challenge**: A fundamental tension exists between improving multilingual ability (MA) and preserving English ability (EA)—ideally a single model should excel at both, avoiding the cost of maintaining multiple models.

**Goal**: Within a two-phase CFT framework, understand the mechanism underlying English degradation during multilingual adaptation and propose efficient mitigation strategies.

**Key Insight**: The paper focuses on the *instruction similarity* between phases—if both phases encode the same instructions in different languages, English capability can be preserved or even improved.

**Core Idea**: The root cause of English degradation is representation drift—dissimilar phase datasets cause large shifts in the model's hidden representation space, which can be controlled through data distribution replay and layer freezing.

## Method

### Overall Architecture

Two-phase CFT: Phase 1 fine-tunes on English instruction datasets (Alpaca/OpenOrca); Phase 2 fine-tunes on multilingual datasets (MultiAlpaca/mOpenOrca). Compared to single-phase mixed fine-tuning, two-phase CFT achieves consistently better average performance under the same number of training steps.

### Key Designs

1. **Dataset Embedding Similarity (DES)**:

    - Function: Quantifies instruction similarity between the two phase datasets.
    - Mechanism: Uses the language-agnostic sentence encoder LaBSE to encode instructions from each dataset and computes the normalized dot product of the two mean embeddings. A higher DES indicates greater instruction similarity. Experiments show DES = 0.924 for the Alpaca–MultiAlpaca pair (same-source) versus 0.746 for the Instruct–MultiAlpaca pair (cross-source).
    - Design Motivation: A language-agnostic metric is needed to predict whether Phase 2 will cause English degradation.

2. **Model Parameter Divergence (MPD)**:

    - Function: Quantifies the differential impact of each dataset on the model from a parameter-space perspective.
    - Mechanism: Fine-tunes from the same base model separately on each dataset and computes the L2 norm of the parameter difference between the two resulting models. A smaller MPD indicates greater dataset similarity—Alpaca–MultiAlpaca yields MPD = 0.29, while Instruct–MultiAlpaca yields 1.00.
    - Design Motivation: DES measures similarity from a data perspective; MPD measures it from a model perspective—the two are complementary and jointly validate the similarity hypothesis.

3. **Representation Drift Mitigation Strategies**:

    - Function: Controls hidden-layer representation shifts induced by Phase 2 fine-tuning.
    - Mechanism: (a) *Generative Replay (GR)*—uses the Phase 1 model to generate responses to the English counterpart instructions of the Phase 2 dataset, which are then mixed into Phase 2 training at a 5% or 10% ratio. The intuition is that generated data bridges the distribution gap between phases. (b) *English Replay (ER)*—uses actual English parallel data for replay. (c) *Layer Freezing (LF)*—selectively freezes layers based on the highest change during Phase 1 fine-tuning (LF_H2), random selection (LF_H1), or signal-to-noise ratio (Spectrum).
    - Design Motivation: Representation drift is the core mechanism of English forgetting—replay reduces drift by maintaining distributional continuity between phases, while layer freezing physically constrains the drift space.

### Loss & Training

Full-parameter fine-tuning in bf16 precision. Phase 1 and Phase 2 each use their respective full datasets. English ability is evaluated on IFEval, Alpaca Eval, MMLU, HellaSwag, and XLSUM_en; multilingual ability is evaluated on MLQA, XQuAD, XLSUM, and GMMLU, covering 11 languages (French, Arabic, German, Spanish, Indonesian, Japanese, Korean, Portuguese, Russian, Thai, Vietnamese).

## Key Experimental Results

### Main Results

| Model | Phase 1 | Phase 2 | EA Avg. | MA Avg. | Overall |
|-------|---------|---------|---------|---------|---------|
| Mistral-7B | Alpaca | MultiAlpaca | 0.371 ↑ | 0.338 ↑ | 0.355 |
| Mistral-7B | Instruct | MultiAlpaca | 0.332 ↓ | 0.302 ↑ | 0.317 |
| LLaMA-3-8B | Alpaca | MultiAlpaca | 0.265 ↑ | 0.427 ↑ | 0.346 |
| LLaMA-3-8B | Instruct | MultiAlpaca | 0.178 ↓ | 0.301 ↓ | 0.240 |
| Mistral-7B | Mixed | - | 0.371 | 0.278 | 0.325 |
| LLaMA-3-8B | Mixed | - | 0.335 | 0.289 | 0.312 |

### Ablation Study

| Strategy | Mistral EA | Mistral MA | LLaMA EA | LLaMA MA |
|----------|-----------|-----------|----------|----------|
| No mitigation (Instruct→MA) | 0.332 | 0.302 | 0.178 | 0.302 |
| GR_5 | 0.394 | 0.298 | 0.236 | 0.348 |
| GR_10 | 0.394 | 0.274 | 0.173 | 0.204 |
| ER_10 | 0.404 | 0.276 | 0.345 | 0.359 |
| LF_H2 | 0.294 | 0.263 | 0.306 | 0.320 |
| Spectrum | 0.363 | 0.237 | 0.329 | 0.261 |
| LoRA | 0.341 | 0.321 | 0.142 | 0.075 |

### Key Findings

- Two-phase CFT consistently outperforms mixed fine-tuning—Mistral-7B overall 0.355 vs. 0.325; LLaMA-3-8B 0.346 vs. 0.312.
- Similar dataset pairs (Alpaca→MultiAlpaca) not only preserve English ability but improve it, as both phases encode the same instructions.
- Dissimilar dataset pairs (Instruct→MultiAlpaca) cause severe English degradation—LLaMA-3-8B IFEval drops from 0.735 to 0.182.
- Representation drift visualization confirms that dissimilar datasets induce 3–4× greater covariance shift in upper layers compared to similar datasets.
- ER_10 achieves the best overall performance on Mistral; GR_5 achieves the strongest multilingual performance on LLaMA.
- LoRA yields extremely poor multilingual performance on LLaMA (0.075), indicating that parameter-efficient methods do not necessarily preserve multilingual capability.

## Highlights & Insights

- The finding that "instruction similarity determines the degree of forgetting" is highly practical—when selecting Phase 2 data, datasets encoding the same instructions as Phase 1 in different languages should be preferred over arbitrary multilingual data.
- DES and MPD provide complementary validation of the similarity hypothesis from data and model perspectives respectively, strengthening the credibility of the conclusions.
- Generative replay does not require access to the original Phase 1 data (satisfying real-world constraints), and as little as 5% replay data suffices to effectively mitigate drift.
- Covariance matrix drift analysis intuitively reveals the layer-wise distribution of English forgetting—concentrated in upper layers for Mistral and distributed across all layers for LLaMA.

## Limitations & Future Work

- Experiments are conducted only on Mistral-7B and LLaMA-3-8B; generalizability to larger models or different architectures is unknown.
- DES and MPD as similarity proxies may not capture all instruction-level differences.
- The best performance of ER_10 depends on the availability of parallel data, which may not always be accessible in practice.
- Extension to multi-phase (>2) continual fine-tuning is not explored.

## Related Work & Insights

- **vs. InstructAlign**: The latter requires cross-lingual alignment, episodic replay, and parallel data, incurring high costs; the proposed GR requires only the Phase 1 model to generate English responses.
- **vs. Shaham et al. (2024)**: The latter introduces multilinguality in the first phase; this work introduces it in the second phase and systematically analyzes the conditions for forgetting.
- **vs. EWC and other regularization methods**: These require storing both old and new parameters at high computational cost; layer freezing achieves a similar effect in a more lightweight manner.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-phase CFT framework and similarity metrics are novel, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple dataset pairs, multiple models, detailed ablations, and mitigation strategies, though model scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and effective visualizations, though the notation system is slightly complex.
- Value: ⭐⭐⭐⭐ Provides practical guidance for multilingual continual fine-tuning—selecting similar datasets and applying lightweight replay can substantially mitigate forgetting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)

</div>

<!-- RELATED:END -->
