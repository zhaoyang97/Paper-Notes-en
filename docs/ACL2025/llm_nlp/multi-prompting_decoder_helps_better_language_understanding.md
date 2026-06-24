---
title: >-
  [Paper Note] Multi-Prompting Decoder Helps Better Language Understanding
description: >-
  [ACL2025][LLM (Other)][Model-as-a-Service] The Multi-Prompting Decoder (MPD) framework is proposed, which queries pre-trained language models (PLMs) with multiple prompts to obtain multiple sets of hidden states and class scores. Combined with optimal transport matching and calibrated decoding strategies, it significantly outperforms existing methods on few-shot classification tasks in MaaS (Model-as-a-Service) scenarios.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Model-as-a-Service"
  - "prompt tuning"
  - "optimal transport"
  - "few-shot learning"
  - "output-side adaptation"
date: 2026-05-08
content_hash: e441d1a01edcb30f
---

# Multi-Prompting Decoder Helps Better Language Understanding

**Conference**: ACL2025  
**arXiv**: [2406.06279](https://arxiv.org/abs/2406.06279)  
**Code**: To be confirmed  
**Area**: LLM/NLP  
**Keywords**: Model-as-a-Service, prompt tuning, optimal transport, few-shot learning, output-side adaptation

## TL;DR

The Multi-Prompting Decoder (MPD) framework is proposed, which queries pre-trained language models (PLMs) with multiple prompts to obtain multiple sets of hidden states and class scores. Combined with optimal transport matching and calibrated decoding strategies, it significantly outperforms existing methods on few-shot classification tasks in MaaS (Model-as-a-Service) scenarios.

## Background & Motivation

**Rise of MaaS Deployment Paradigm**: Large-scale pre-trained language models are increasingly provided as API services (e.g., GPT-3.5 Turbo, text-embedding series), where users can only obtain outputs (hidden states, class scores, text) but cannot access model parameters or gradients.

**Inefficiency of Input-side Adaptation**: Under MaaS settings, gradient-free optimization of continuous/discrete prompts (e.g., BBT, RLPrompt) requires querying PLMs thousands to tens of thousands of times, suffering from huge search spaces, optimization difficulties, and massive time costs.

**Single-Prompt Bottleneck in Output-side Adaptation**: Existing output-side methods (e.g., DecT) query the PLM using only a single prompt, making performance highly dependent on prompt quality. Experiments show that accuracy across different prompts can fluctuate by over 8% on datasets like SST2.

**Scarcity of Few-shot Data**: In few-shot settings with only 1-16 training samples per class, representation information obtained from a single prompt is extremely limited, further amplifying the risk of poor prompt selection.

**Predictive Bias of PLMs**: Models tend to predict tokens common in the pre-training distribution, leading to systematic bias in class scores. Directly using these scores for classification yields suboptimal results.

**Unexploited Potential of Multi-Prompting**: Multi-prompting can simultaneously mitigate dependence on a single prompt, alleviate data scarcity (by obtaining multiple representations from a single sample), and extract PLM knowledge from diverse perspectives. However, there is a lack of effective decoding mechanisms to exploit this multi-source information.

## Method

### Overall Architecture

MPD incorporates two decoding strategies: (1) multi-prompt hidden state decoding based on optimal transport; and (2) calibrated multi-prompt class score decoding. The final prediction is outputted via joint decoding.

### Key Designs

**Multi-Prompt Querying**: Each sample is wrapped with $P$ different templates to query the PLM, obtaining $P$ sets of hidden states (represented by the last-layer $[MASK]$ token representation). These are projected through a linear layer to form the text representation matrix $V_i \in \mathbb{R}^{P \times d}$.

**Optimal Transport-based Classification**: For each class $k$, $Q$ learnable prototypes $R_{k,n}$ are maintained. Optimal transport (Sinkhorn algorithm) is employed to solve the optimal matching plan $T^{i,k}$ between text representations and class prototypes. The OT score is computed as the weighted sum of the matching plan and cosine similarity. This design allows the representation of each prompt to align with the best-matching prototype, avoiding the drawbacks of rough averaging or independent classifiers.

**Calibrated Class Score Decoding**: (1) Label word set expansion (expanding 10 synonyms based on the cosine similarity of word vectors in the MLM prediction layer); (2) calibrating bias using the class score of empty input; and (3) averaging the calibrated scores across multiple prompts.

**Joint Decoding**: The final prediction is a weighted sum of the OT score and the calibrated class score, with $\beta$ as the balancing hyperparameter.

### Loss & Training

The standard cross-entropy loss is used to optimize the OT score. The learnable parameters consist only of the linear layer and class prototypes, rendering the model extremely lightweight (approximately 132K parameters).

## Key Experimental Results

### Main Results (Table 1 - 9 NLU datasets, RoBERTa-Large)

| Setting | Method | SST2 | AG | DBPedia | Yahoo | RTE | SNLI | Avg |
|------|------|------|-----|---------|-------|-----|------|-----|
| 1-shot | DecT | 90.8 | 79.9 | 78.8 | 55.2 | 56.0 | 47.7 | 70.0 |
| 1-shot | **MPD** | **92.3** | **83.2** | **84.4** | 53.6 | **57.6** | 46.6 | **71.5** |
| 4-shot | DecT | 87.6 | 81.9 | 89.1 | 59.9 | 56.7 | 53.2 | 71.8 |
| 4-shot | **MPD** | **92.6** | **85.9** | **92.8** | **62.2** | **59.2** | **57.1** | **75.2** |
| 16-shot | DecT | 91.0 | 86.4 | 94.6 | 64.2 | 59.7 | 60.5 | 75.5 |
| 16-shot | **MPD** | **91.9** | **87.9** | **96.7** | **68.3** | **61.7** | **62.4** | **77.8** |

MPD consistently achieves state-of-the-art performance across almost all datasets under all shot settings, with an average improvement of 1.5% in 1-shot and 2.3% in 16-shot.

### Efficiency Comparison (Table 2 - 16-shot)

| Method | Trainable Params | Queries | SST2 Acc | Training Time (s) |
|------|-----------|---------|----------|------------|
| BBT | 0.5K | 8,000 | 89.6 | 1619 |
| RLPrompt | 3100K | 12,000 | 87.0 | 82286 |
| DecT | 130K | 1 | 91.0 | 1.4 |
| **MPD** | 132K | 3 | **91.9** | 3.5 |

MPD requires only 3 queries ($P=3$ prompts) and 3.5 seconds of training, which is 462 times faster than BBT and 23,510 times faster than RLPrompt.

### Ablation Study

- **Number of Prompts**: $P=3$ is the optimal choice; $P=1$ degenerates to a single prompt, and diminishing marginal returns are observed when $P$ is too large.
- **Number of Prototypes**: $Q=3$ is optimal, as excessive prototypes lead to overfitting in few-shot settings.
- **OT vs. Averaging**: OT matching outperforms simple averaging of multi-prompt representations (approx. 1-2% improvement).
- **Contribution of Calibrated Decoding**: Removing the calibration score leads to a performance drop of 1-3%, validating the complementarity of prior knowledge.
- **Label Word Expansion**: Expanding to 10 label words improves performance by 2-4% compared to using only 1 label word (specifically on sentiment/topic classification tasks).

## Highlights & Insights

1. **Simple yet Effective**: The core idea is intuitively simple—querying with multiple prompts to obtain more stable representations—but it is fully exploited via the OT matching mechanism.
2. **Extremely Efficient**: Compared to input-side methods that require thousands of queries, MPD requires only $P$ queries and seconds of training, making it highly suitable for practical MaaS scenarios.
3. **Ingenious Use of OT Matching**: Instead of simply fusing multi-prompt information, it finds the best-matching prototype for each prompt, preserving prompt specificity.
4. **Prompt Robustness**: As shown in Figure 1 of the paper, while a single prompt's performance fluctuates by over 8%, the standard deviation of MPD is significantly reduced (e.g., from 0.5 to 0.1 on SST2 16-shot).

## Limitations & Future Work

1. Experiments are only conducted on RoBERTa-Large; its effectiveness on larger models or decoder-only architectures (e.g., GPT series) remains unvalidated.
2. Templates still require manual design; although results are insensitive to templates, it is not fully automated.
3. Label word expansion has limited effectiveness on NLI tasks (as the label words themselves have more abstract semantics).
4. Applicability when PLMs only provide text outputs (rather than hidden states/logits) was not explored.
5. The $\beta$ hyperparameter requires individual tuning on MNLI, leaving room for improvement in automation.

## Related Work & Insights

- **Relationship with DecT**: MPD is a natural extension of DecT—evolving from a single prompt to multiple prompts, and from hypersphere prototypes to OT matching.
- **Difference from PromptBoosting**: PromptBoosting integrates multiple weak learners via boosting, which is "model ensemble-based"; MPD is "representation fusion-based," which is more efficient.
- **OT Application in NLP**: Introducing OT into MaaS adaptation is novel and could inspire other tasks requiring multi-view representation matching.
- **Inspirations**: The multi-prompting concept can be extended to LLM black-box inference scenarios, such as querying ChatGPT API with multiple prompts and aggregating them, potentially improving stability.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of multi-prompting and OT decoding is highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (9 datasets, 3 shot configurations, and thorough ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear elaboration on methodology, and intuitive charts)
- Value: ⭐⭐⭐⭐ (Clear guiding significance for practical MaaS applications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ProgCo: Program Helps Self-Correction of Large Language Models](progco_program_helps_self-correction_of_large_language_models.md)
- [\[ACL 2025\] P3: Prompts Promote Prompting](p3_prompts_promote_prompting.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)

</div>

<!-- RELATED:END -->
