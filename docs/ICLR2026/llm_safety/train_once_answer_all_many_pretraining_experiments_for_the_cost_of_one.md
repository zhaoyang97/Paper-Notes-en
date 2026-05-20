---
title: >-
  [Paper Note] Train Once, Answer All: Many Pretraining Experiments for the Cost of One
description: >-
  [ICLR 2026][LLM Safety][Pretraining Experiments] This paper proposes a methodological framework for running multiple independent experiments simultaneously within a single LLM pretraining run. Training a 2.7B-parameter m…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Pretraining Experiments"
  - "LLM"
  - "Experiment Independence"
  - "Continual Pretraining"
  - "Data Contamination"
date: 2026-05-08
content_hash: cbb37af3f0f4c691
---

# Train Once, Answer All: Many Pretraining Experiments for the Cost of One

**Conference**: ICLR 2026
**arXiv**: [2509.23383](https://arxiv.org/abs/2509.23383)  
**Code**: [Python Package](https://arxiv.org/abs/2509.23383) (OLMo-2 experiment package provided in the paper)  
**Area**: AI Safety
**Keywords**: Pretraining Experiments, LLM, Experiment Independence, Continual Pretraining, Data Contamination

## TL;DR

This paper proposes a methodological framework for running multiple independent experiments simultaneously within a single LLM pretraining run. Training a 2.7B-parameter model on 210B tokens, the framework concurrently executes 10 experiments, successfully replicates the results of 5 prior works, and conducts 3 novel experiments. It further introduces Continual Pretraining Dependence Testing (CPDT) to verify inter-experiment independence.

## Background & Motivation

**Controlled pretraining experiments are the gold standard for studying LLM behavior**: Training models from scratch to systematically isolate the effects of specific interventions—data changes, architectural modifications, learning objective alterations—offers conceptual simplicity and scientific rigor superior to alternative approaches.

**Computational cost is the primary bottleneck for pretraining experiments**: The current standard practice of one independent training run per experiment means that, for any given aspect of model behavior, the expected insights from a single project rarely justify the cost of training a model from scratch.

**The multi-task nature of pretraining provides a theoretical foundation**: A model simultaneously learns many tasks during pretraining, suggesting that independent interventions targeting different tasks can in principle be applied concurrently. This also mirrors practical model development, where practitioners routinely combine multiple interventions in a single training run.

**Core Problem: Do inter-experiment interactions exist?**: If one experiment influences the outcome of another, the validity of joint training is compromised. A principled method for detecting and quantifying such dependencies is therefore necessary.

## Method

### Overall Architecture

Building on the OLMo-2 model family, 10 experiments are conducted simultaneously within a single pretraining run. Each experiment is implemented by modifying a small fraction of the training data (a total of 3.7B tokens, representing 1.8% of the full training corpus). After training, each experiment is evaluated using its own designated metrics.

### Key Designs

**1. Synchronous Multi-Experiment Training**

- **Function**: Simultaneously executes 10 distinct experiments during the training of OLMo-2-1B-Exp.
- **Mechanism**: Each experiment independently modifies a non-overlapping subset of the training data and is evaluated by its own isolated metrics.
- **Design Motivation**: Exploits the multi-task nature of pretraining—interventions targeting different tasks are theoretically parallelizable without mutual interference.

The 10 experiments span three broad categories:
- **Learning & Generalization**: Knowledge Acquisition (KA, 26M tokens), Mathematical Reasoning (MR, 180M tokens)
- **Memorization & Privacy**: Benchmark Contamination (BC, 106M), Memorization Patterns (MemP, 246M), Verbatim Memorization (MemV, 1.1B), Gaussian Watermarking (GW, 210M)
- **Forgetting & Unlearning**: Pretraining Poisoning (PP, 235M), Forgetting Curves (FC, 19M), MUSE-News (152M), IID Replacement (1.5B)

**2. Dynamic Control Algorithm for Knowledge Acquisition**

- **Function**: Dynamically adjusts the frequency of factual knowledge in training data via a control algorithm, ensuring the model acquires specific knowledge by the end of training.
- **Mechanism**: Every 1,000 steps, the current probability of a knowledge probe is evaluated, and the frequency of the corresponding knowledge in subsequent data is adjusted based on the gap to the target.
- **Design Motivation**: Addresses the key research question: how many times must a model encounter a fact during pretraining in order to acquire it?

**3. Continual Pretraining Dependence Testing (CPDT)**

- **Function**: Detects inter-experiment dependencies via continual pretraining experiments prior to the full pretraining run.
- **Mechanism**: An intermediate checkpoint is taken; short continual pretraining runs are performed using the data from each experiment individually. An $n \times n$ dependency matrix is constructed to assess whether the data of experiment $j$ influences the outcome of experiment $i$.
- **Design Motivation**: Provides a practical method to verify in advance whether a set of experiments can be validly executed within a single training run.

Inter-experiment independence is formally defined as: experiments $E_1, \dots, E_n$ are independent if and only if $Y_i^{\{i\}} \stackrel{d}{=} Y_i^{\{i\} \cup T}$ holds for all $i$ and $T \subseteq [n] \setminus \{i\}$.

### Loss & Training

- Based on the OLMo-2-1B architecture; trained for 100,000 steps over 210B tokens.
- The learning rate schedule for the first 90,000 steps mirrors OLMo-2-1B; the final 10,000 steps apply linear decay to zero.
- Experiment data is distributed uniformly throughout training, replacing corresponding segments of the original pretraining data.
- Four model scales are trained concurrently: 179M, 546M, 1.5B, and 2.7B parameters.

## Key Experimental Results

### Main Results

Novel experiment results (OLMo-2-1B-Exp):

| Experiment | Key Metric | Result |
|---|---|---|
| Knowledge Acquisition | Final knowledge probe value / zero-shot accuracy | 0.05 (target: 0.08) / 25% |
| Mathematical Reasoning | Generalization beyond training difficulty | Generates 11-step optimal solutions |
| Gaussian Watermarking | TPR@1%FPR | Consistently above random baseline; later watermarks more detectable |
| Benchmark Contamination (4× repetition) | Degree of overfitting | ~1 percentage point |
| Benchmark Contamination (144× repetition) | Degree of overfitting | ~19 percentage points |

Model scale effects:

| Model | Parameters | Contamination Effect | Poisoning Success | Math Reasoning |
|---|---|---|---|---|
| OLMo-2-179M-Exp | 179M | Small | Effective | Not emergent |
| OLMo-2-546M-Exp | 546M | Moderate | Effective | Beginning to emerge |
| OLMo-2-1B-Exp | 1.5B | Large | Effective | Evident |
| OLMo-2-2.7B-Exp | 2.7B | Largest | Effective | Strongest |

### Ablation Study

CPDT dependency test results:

| Type | Comparison | Conclusion |
|---|---|---|
| Between language modeling benchmarks | ARC-Easy → ARC-Challenge | +6.2pp; significant dependency |
| Between experiments (off-diagonal) | All experiment pairs | All non-significant; no dependency |
| Joint vs. individual training | Diagonal vs. bottom row | Consistent results; independence confirmed |

Training dynamics impact:

| Metric | OLMo-2-1B | OLMo-2-1B-Exp |
|---|---|---|
| 10K holdout benchmark accuracy | 55.51% | 55.15% |
| Train / validation loss | Nearly identical | Nearly identical |

### Key Findings

1. **All 5 prior works successfully replicated**: Results for benchmark contamination, memorization patterns, verbatim memorization, pretraining poisoning, and forgetting curves are consistent with those from independent training runs.
2. **Experiments exert minimal impact on overall training dynamics**: Training loss, validation loss, and output-layer weight norms are nearly indistinguishable.
3. **Larger models exhibit stronger intervention effects**: Benchmark contamination, knowledge acquisition, and poisoning success rates all increase with model scale.
4. **Mathematical reasoning emerges only at ≥546M parameters**: Consistent with findings from prior independent training studies.
5. **Gaussian watermarking exhibits a "recency bias"**: Data encountered later in training is more readily detected, revealing a temporal characteristic of LLM learning dynamics.

## Highlights & Insights

1. **A major methodological contribution**: The paradigm shift from "one experiment per training run" to "many experiments per training run" has the potential to reduce the cost of LLM experimental research by an order of magnitude.
2. **CPDT is an elegant validation tool**: Using continual pretraining to detect dependencies before a full pretraining run is far less costly than full training, while providing sufficient statistical guarantees.
3. **Dynamic control for knowledge acquisition**: The work innovatively introduces control-theoretic concepts into pretraining, enabling on-demand adjustment of knowledge frequency and opening new possibilities for targeted knowledge injection.
4. **Practical implications for the community**: The training runs of open-source models (e.g., OLMo) could in principle accommodate multiple research experiments simultaneously; current open-source training practices exhibit insufficient "research density."

## Limitations & Future Work

1. **Limited experiment scale**: Each of the 10 experiments modifies at most 1.8% of training data; independence may not hold for interventions with larger data modification ratios (e.g., full synthetic data training).
2. **Data interventions only**: With the exception of Gaussian watermarking, all experiments modify training data; architectural changes, hyperparameter variations, and similar interventions cannot be parallelized under this framework.
3. **CPDT is an approximation**: Dependency detection via continual pretraining is an approximation of full pretraining, and carries a risk of false negatives.
4. **Model scale ceiling**: Validation extends only to 2.7B parameters; applicability to models at the 70B+ scale remains unverified.
5. **Higher-order interactions**: CPDT primarily detects pairwise dependencies; higher-order interactions among three or more experiments may be overlooked.

## Related Work & Insights

- **Bordt et al. (2025)**: The original benchmark contamination study; this work successfully replicates its core findings on forgetting.
- **Panda et al. (2025)**: The original memorization patterns / privacy leakage study; replicates the vulnerability of rare-token canaries.
- **Zhang et al. (2025b)**: The original pretraining poisoning study; replicates the effectiveness of backdoor attacks on small-scale models.
- **AdEMAMix (Pagliardini et al., 2025)**: The reference work for forgetting curve experiments; validates rapid forgetting of individual batches.
- Outlook: This methodology can be extended to pretraining experiments for vision and multimodal models, and has the potential to become a standard tool in AI research.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A pioneering methodological contribution; the "train once, answer all" concept is refreshingly original, and CPDT constitutes an entirely new testing instrument.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 experiments, 4 model scales, 5 successful replications, and 3 novel experiments; the experimental design is exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured with rigorous mathematical definitions and intuitive figures; however, experimental details are relegated to appendices due to space constraints.
- **Value**: ⭐⭐⭐⭐⭐ — Has the potential to shift the paradigm of LLM pretraining experimentation and provides substantive assistance to resource-constrained researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/llm_safety/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[ICLR 2026\] Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](faithful_bi-directional_model_steering_via_distribution_matching_and_distributed.md)

</div>

<!-- RELATED:END -->
