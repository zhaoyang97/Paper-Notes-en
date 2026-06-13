---
title: >-
  [Paper Note] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Continual Fine-tuning] This paper proposes a two-phase continual fine-tuning (CFT) framework—fine-tuning first on English instruction data and then on multilingual data. It…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Continual Fine-tuning"
  - "Multilingual Adaptation"
  - "Catastrophic Forgetting"
  - "Dataset Similarity"
  - "Representation Drift"
date: 2026-05-08
content_hash: 51a171bc78aa538a
---

# Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2410.16006](https://arxiv.org/abs/2410.16006)  
**Code**: None  
**Area**: Multilingual / Continual Learning  
**Keywords**: Continual Fine-tuning, Multilingual Adaptation, Catastrophic Forgetting, Dataset Similarity, Representation Drift

## TL;DR

This paper proposes a two-phase continual fine-tuning (CFT) framework—fine-tuning first on English instruction data and then on multilingual data. It identifies that the instruction similarity between datasets across phases is the key factor determining whether English proficiency degrades. Furthermore, it effectively mitigates representation drift and English forgetting caused by dissimilar datasets through generative replay and heuristic layer freezing.

## Background & Motivation

**Background**: The multilingual user base of LLMs is growing steadily, yet models perform significantly worse in low-resource languages. Since training from scratch is extremely costly, fine-tuning is the preferred solution. Fine-tuning on mixed multilingual datasets often leads to English bias, while fine-tuning solely on non-English data results in English performance degradation due to catastrophic forgetting.

**Limitations of Prior Work**: (1) Existing methods like InstructAlign require parallel data and old task data, incurring high computational overhead; (2) direct fine-tuning on mixed datasets leads to an imbalance between English and multilingual performance; (3) there is a lack of systematic understanding regarding "under what conditions multilingual fine-tuning harms English proficiency"; (4) regularization methods such as EWC require saving both new and old parameters, leading to low computational efficiency.

**Key Challenge**: A tension exists between improving Multilingual Adaptation (MA) and maintaining English Adaptation (EA)—ideally, a single model should excel in both to avoid the cost of maintaining multiple models.

**Goal**: To understand the mechanism of English degradation during multilingual adaptation within a two-phase CFT framework and propose efficient mitigation strategies.

**Key Insight**: The study focuses on "instruction similarity" between datasets across phases—if both phases encode the same instructions (merely in different languages), English proficiency can be maintained or even improved.

**Core Idea**: The fundamental cause of English degradation is representation drift—dissimilar phase datasets cause significant shifts in the model's hidden representation space. This drift can be controlled through data distribution replay and layer freezing.

## Method

### Overall Architecture

Two-phase CFT: Phase 1 involves fine-tuning on an English instruction dataset (Alpaca/OpenOrca), followed by Phase 2 fine-tuning on a multilingual dataset (MultiAlpaca/mOpenOrca). Compared to single-phase mixed fine-tuning, two-phase CFT achieves better average performance under the same number of training steps.

### Key Designs

1.  **Dataset Embedding Similarity (DES)**:
    *   **Function**: Quantifies the instruction similarity between two phase datasets.
    *   **Mechanism**: Uses the language-agnostic sentence encoder LaBSE to encode instructions in the datasets and calculates the normalized dot product of the average embeddings. Higher DES values indicate higher instruction similarity. Experiments show DES = 0.924 for Alpaca-MultiAlpaca (homologous pair), while it is only 0.746 for Instruct-MultiAlpaca (heterologous pair).
    *   **Design Motivation**: A language-independent metric is needed to predict whether Phase 2 will cause English degradation.

2.  **Model Parameter Difference (MPD)**:
    *   **Function**: Quantifies the difference in the impact of datasets on the model from the parameter space perspective.
    *   **Mechanism**: Fine-tunes from the same base model on two different datasets separately, then calculates the $L_2$ norm difference of the parameters. Smaller MPD indicates higher dataset similarity—MPD = 0.29 for Alpaca-MultiAlpaca and 1.00 for Instruct-MultiAlpaca.
    *   **Design Motivation**: DES measures similarity from a data perspective, while MPD measures it from a model perspective—the two complement each other to validate the similarity hypothesis.

3.  **Representation Drift Mitigation Strategies**:
    *   **Function**: Controls the hidden representation shift caused by Phase 2 fine-tuning.
    *   **Mechanism**: (a) Generative Replay (GR)—using the Phase 1 model to generate responses for English-translated instructions of the Phase 2 dataset, mixing these as 5%/10% replay data into Phase 2 training. The intuition is that generated data bridges the distributions of the two phases; (b) English Replay (ER)—using actual parallel English data; (c) Layer Freezing (LF)—selectively freezing parts of the model based on layers with the largest changes in Phase 1 (LF_H2), random layers (LF_H1), or Signal-to-Noise Ratio (Spectrum).
    *   **Design Motivation**: Representation drift is the core mechanism of English degradation—replay reduces drift by maintaining data distribution continuity, while layer freezing limits the drift space through physical constraints.

### Loss & Training

Full-parameter fine-tuning with bf16 precision. Each phase uses the full respective dataset for training. Evaluation of English proficiency uses IFEval, Alpaca Eval, MMLU, HellaSwag, and XLSUM_en; multilingual proficiency is evaluated using MLQA, XQuAD, XLSUM, and GMMLU. Multilingual coverage includes 11 languages (FR, AR, DE, ES, ID, JA, KO, PT, RU, TH, VI).

## Key Experimental Results

### Main Results

| Model | Phase 1 | Phase 2 | Avg. EA | Avg. MA | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mistral-7B | Alpaca | MultiAlpaca | 0.371 ↑ | 0.338 ↑ | 0.355 |
| Mistral-7B | Instruct | MultiAlpaca | 0.332 ↓ | 0.302 ↑ | 0.317 |
| LLaMA-3-8B | Alpaca | MultiAlpaca | 0.265 ↑ | 0.427 ↑ | 0.346 |
| LLaMA-3-8B | Instruct | MultiAlpaca | 0.178 ↓ | 0.301 ↓ | 0.240 |
| Mistral-7B | Mixed | - | 0.371 | 0.278 | 0.325 |
| LLaMA-3-8B | Mixed | - | 0.335 | 0.289 | 0.312 |

### Ablation Study

| Strategy | Mistral EA | Mistral MA | LLaMA EA | LLaMA MA |
| :--- | :--- | :--- | :--- | :--- |
| No Mitigation (Instruct→MA) | 0.332 | 0.302 | 0.178 | 0.302 |
| GR_5 | 0.394 | 0.298 | 0.236 | 0.348 |
| GR_10 | 0.394 | 0.274 | 0.173 | 0.204 |
| ER_10 | 0.404 | 0.276 | 0.345 | 0.359 |
| LF_H2 | 0.294 | 0.263 | 0.306 | 0.320 |
| Spectrum | 0.363 | 0.237 | 0.329 | 0.261 |
| LoRA | 0.341 | 0.321 | 0.142 | 0.075 |

### Key Findings

*   Two-phase CFT consistently outperforms mixed fine-tuning—Mistral-7B overall 0.355 vs 0.325, LLaMA-3-8B 0.346 vs 0.312.
*   Similar datasets (Alpaca→MultiAlpaca) do not harm English proficiency but rather improve it, as both phases encode identical instructions.
*   Dissimilar datasets (Instruct→MultiAlpaca) cause severe English degradation—IFEval for LLaMA-3-8B plummeted from 0.735 to 0.182.
*   Visualization of representation drift confirms that dissimilar datasets produce covariance shifts at higher layers that are 3-4 times larger than those of similar datasets.
*   ER_10 achieves the best overall performance on Mistral, while GR_5 is strongest for LLaMA multilingual tasks.
*   LoRA exhibits extremely poor multilingual performance on LLaMA (0.075), indicating that parameter-efficient methods may not effectively maintain multilingual capability.

## Highlights & Insights

*   The discovery that "instruction similarity determines the extent of forgetting" is highly practical—when selecting Phase 2 datasets, preference should be given to versions encoding the same instructions as Phase 1, rather than using arbitrary multilingual data.
*   The DES and MPD metrics complement each other from data and model perspectives to validate the similarity hypothesis, enhancing the reliability of the conclusions.
*   Generative replay does not require original Phase 1 data (satisfying real-world constraints); merely 5% replay data can effectively mitigate drift.
*   Covariance matrix drift analysis intuitively reveals the hierarchical distribution of English degradation—concentrated in high layers for Mistral and across all layers for LLaMA.

## Limitations & Future Work

*   Only validated on Mistral-7B and LLaMA-3-8B; generalization to larger models or different architectures remains unknown.
*   DES and MPD as similarity proxies might not capture all instruction-level differences.
*   The optimal performance of ER_10 depends on the availability of parallel data, which may not always be accessible in practice.
*   Multi-phase (>2) continual fine-tuning extension has not been explored.

## Related Work & Insights

*   **vs InstructAlign**: The latter requires cross-lingual alignment, exemplar replay, and parallel data, which are costly; the GR in this paper only requires the Phase 1 model to generate English responses.
*   **vs Shaham et al. (2024)**: The latter introduces multilinguality in the first phase, whereas this paper introduces it in the second phase and systematically analyzes forgetting conditions.
*   **vs EWC and other regularization**: These require saving both new and old parameters, leading to low computational efficiency; layer freezing achieves similar effects in a more lightweight manner.

## Rating

*   Novelty: ⭐⭐⭐⭐ The two-phase CFT framework and similarity metrics are innovative, though individual components have precedents.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Includes various dataset pairs, multiple models, detailed ablations, and mitigation strategies, though model scales are limited.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure and effective visualizations, though the notation system is slightly complex.
*   Value: ⭐⭐⭐⭐ Provides practical guidance for multilingual continual fine-tuning—forgetting can be significantly mitigated by selecting similar datasets and using lightweight replay.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)

</div>

<!-- RELATED:END -->
