---
title: >-
  [Paper Note] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models
description: >-
  [AAAI 2026][Interpretability][Reward Model Evaluation] This paper proposes MRMBench, a benchmark that evaluates whether reward models (RMs) effectively capture multi-dimensional preferences via probing tasks across 6 dim…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Reward Model Evaluation"
  - "Preference Probing"
  - "MRMBench"
  - "Multi-Dimensional Preferences"
  - "Inference-Time Probing"
date: 2026-05-08
content_hash: 48d4f13c359daecc
---

# Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models

**Conference**: AAAI 2026
**arXiv**: [2511.12464](https://arxiv.org/abs/2511.12464)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Reward Model Evaluation, Preference Probing, MRMBench, Multi-Dimensional Preferences, Inference-Time Probing

## TL;DR

This paper proposes MRMBench, a benchmark that evaluates whether reward models (RMs) effectively capture multi-dimensional preferences via probing tasks across 6 dimensions (harmlessness, helpfulness, correctness, coherence, complexity, and verbosity). Probe performance is shown to strongly correlate with PPO alignment quality (Pearson $r > 0.8$), and an inference-time probing method is proposed that improves AlpacaEval win rate from 57.3% to 62.5%.

## Background & Motivation

**Background**: Reward models (RMs) are a core component of RLHF alignment, typically trained with Bradley-Terry loss on human preference data. Common RM evaluation methods compute pairwise ranking accuracy on fixed test sets (e.g., RewardBench), or assess end-to-end performance of the aligned LLM directly.

**Limitations of Prior Work**: (1) Pairwise ranking reduces evaluation to a binary decision (which response is better), failing to reveal an RM's capture ability across **individual preference dimensions**—e.g., whether a high score stems from detecting correctness or harmlessness. (2) End-to-end evaluation (training PPO + evaluating the LLM) is computationally prohibitive. (3) With the rise of multi-objective reward models, simple pairwise ranking is increasingly insufficient to evaluate dimensional balance.

**Key Challenge**: The RM's preference representation is a high-dimensional vector $\mathbf{h}_{[x,y]}$, yet the model ultimately outputs a single scalar reward $r_\phi(x,y) = \mathbf{h}_{[x,y]} \mathbf{W}_r$. Existing methods evaluate only the ranking correctness of this scalar, leaving the internal preference structure of the black box unexplained.

**Goal**: (1) Determine whether RMs effectively encode preferences along each preference dimension; (2) understand which dimensions an RM relies on when making reward predictions; (3) leverage this understanding to improve alignment quality.

**Key Insight**: Drawing inspiration from probing tasks in NLP for evaluating language model representations—if a simple linear classifier can successfully predict a given dimension (e.g., "harmful vs. harmless") from an RM's preference representation, this indicates that the RM internally encodes that dimension.

**Core Idea**: Diagnose reward model preference representations using probing classifiers, enabling for the first time fine-grained evaluation and mechanistic analysis of RMs' multi-dimensional preference capture capabilities.

## Method

### Overall Architecture

The framework consists of three components: (1) constructing MRMBench—a probing task benchmark across 6 dimensions; (2) evaluating RMs—freezing RM parameters and training lightweight linear classifiers, using their accuracy on each probing task to quantify preference capture; (3) inference-time probing—computing distances from preference representations to per-dimension cluster centroids via K-means, analyzing dimensional dependencies during RM prediction, and improving PPO training via confidence-based filtering.

### Key Designs

1. **MRMBench Benchmark Construction**

    - Function: Construct a probing task dataset covering 6 preference dimensions, with Easy (binary classification) and Hard (ternary classification) variants.
    - Mechanism: Binary/ternary classification tasks are constructed by merging original fine-grained labels from PKU-SafeRLHF (harmlessness) and HelpSteer (remaining 5 dimensions). For example, for harmlessness: the Easy variant merges original labels {1,2,3} into "harmful" and 0 into "harmless"; the Hard variant merges {2,3} into "harmful", retains 1 as "mildly harmful", and 0 as "harmless". Each task selects approximately 10K–15K training samples and 1K test samples with balanced class distributions.
    - Design Motivation: The Easy variant probes basic capture ability, while the Hard variant probes fine-grained discrimination. Label merging addresses class imbalance in the original data (e.g., label 0 accounts for only 8% in helpfulness).

2. **Probing Evaluation Method**

    - Function: Freeze the RM and extract preference representations $\mathbf{h}_{[x,y]} \in \mathbb{R}^d$ from the EOS token of the last Transformer layer; train a linear classifier $\mathbf{W}_c \in \mathbb{R}^{d \times k}$ for classification.
    - Mechanism: The classifier is trained for 1 epoch using standard cross-entropy loss $-\log(\text{softmax}(\mathbf{h}_{[x,y]} \mathbf{W}_c))$, with batch size 128 and learning rate selected from {5e-5, 2e-5, 1e-5}. Test set accuracy serves as the preference capture score for each dimension.
    - Design Motivation: The success of a linear classifier implies that preference information is linearly separable within the representation—constituting genuine "capture" rather than memorization. Freezing RM parameters ensures that the evaluation reflects the representations learned by the RM itself, not the fitting capacity of the classifier.

3. **Inference-Time Probing**

    - Function: Analyze which preference dimensions the RM relies on during inference, and construct a confidence metric to improve PPO training.
    - Mechanism: For each dimension, validation samples are grouped by label and K-means clustering is applied to obtain centroids $\mathcal{C}_{\text{dim}} = \{\mathbf{c}_1, \ldots, \mathbf{c}_k\}$. At inference time, the Euclidean distance from a new sample $\mathbf{h}_{[x',y']}$ to each dimension's centroids is computed: $d(x', y', \mathbf{c}_i) = \|\mathbf{h}_{[x',y']} - \mathbf{c}_i\|_2$. Smaller distance indicates greater reliance on that dimension. Dynamic RLHF: if the minimum distance $d_{\min}$ across all dimension centroids exceeds a threshold $d_\tau$ (indicating RM uncertainty), the sample is discarded and excluded from PPO updates.
    - Design Motivation: When an RM's prediction does not rely on any known preference dimension, the reward signal is unreliable; using it for PPO updates introduces noise and may cause reward hacking. Selectively discarding low-confidence samples improves alignment quality.

### Loss & Training

- Probing classifier: cross-entropy loss, 1 epoch, batch size 128. Learning rate selected from {5e-5, 2e-5, 1e-5} based on validation performance.
- PPO alignment: policy learning rate 1e-5, value model 5e-6, batch size 64, mini-batch PPO for 4 epochs, cold-start trick (value model updated only for the first 30 steps), reward queue (1K historical rewards) for normalization.
- All experiments conducted on 8×A800 GPUs. RM training uses Bradley-Terry loss, learning rate 1e-5, batch size 256, for 1 epoch.
- SFT stage uses 100K preferred completions from the Unified-Feedback dataset, learning rate 1e-5.
- PPO training saves checkpoints every 200 steps; the best checkpoint on the validation set is selected to mitigate reward over-optimization.

## Key Experimental Results

### Main Results (MRMBench-Easy Accuracy %)

| Model | Params | Harmlessness | Helpfulness | Correctness | Coherence | Complexity | Verbosity | Avg. |
|-------|--------|--------------|-------------|-------------|-----------|------------|-----------|------|
| GPM-LLaMA-3.1-8B | 8B | 90.9 | 71.1 | 72.6 | 69.9 | 91.1 | 82.2 | **79.6** |
| QRM-LLaMA-3.1-8B-v2 | 8B | 86.5 | 69.8 | 70.3 | 69.6 | 91.1 | 79.9 | 77.9 |
| Eurus-RM-7B | 7B | 82.2 | 70.0 | 72.1 | 72.7 | 90.9 | 82.2 | 78.4 |
| LLaMA-3.1-8B-Instruct (Baseline) | 8B | 80.4 | 66.3 | 69.4 | 67.0 | 89.1 | 79.1 | 75.2 |
| UltraRM-13B | 13B | 54.5 | 74.5 | 72.6 | 90.9 | 82.2 | 71.7 | 74.4 |

### Ablation Study (Inference-Time Probing for PPO Alignment)

| Method | AlpacaEval Win Rate | Notes |
|--------|---------------------|-------|
| Vanilla PPO | 57.3% | Standard PPO |
| Random Discard | 54.3% | Randomly discard equivalent samples |
| Inference-Time Probing ($d_\tau=140$) | **62.5%** | Selective discard based on dimensional distance |

### Key Findings

- **RMs do encode multi-dimensional preferences**: RMs trained on preference data achieve substantially higher average probe accuracy than baselines without preference training (e.g., GPM-LLaMA-3.1-8B 79.6% vs. LLaMA-3.1-8B-Instruct 75.2%).
- **No single RM excels across all dimensions simultaneously**: UltraRM-13B reaches 90.9% on coherence but only 54.5% on harmlessness, demonstrating that a single scalar reward struggles to balance multi-dimensional preferences and underscoring the necessity of multi-objective optimization.
- **MRMBench strongly correlates with alignment quality**: Pearson correlation coefficients exceed 0.8 across all dimensions ($p < 0.05$), confirming that probe accuracy is a reliable proxy for RM quality.
- **Fine-grained preference capture is substantially harder**: MRMBench-Hard accuracy drops markedly (e.g., GPM-LLaMA-3.1-8B from 79.6% to 67.0%), though harmlessness and coherence degrade less, indicating these two dimensions are better modeled by existing RMs.
- **Inference-time probing effectively improves alignment**: +5.2 win rate points (62.5% vs. 57.3%); random discarding actually degrades performance (54.3%), confirming that selective discarding—rather than simply reducing sample volume—drives the improvement.

## Highlights & Insights

- **Paradigm shift from "evaluating rankings" to "evaluating representations"**: Rather than asking only "did the RM rank correctly?", this work asks "what preference dimensions does the RM internally understand?" This diagnostic perspective offers far greater guidance for RM development than simple pairwise accuracy.
- **Practical utility of inference-time probing**: No modification to the RM architecture or retraining is required; a confidence metric constructed via K-means clustering alone suffices to improve PPO—an extremely lightweight approach that can be plug-and-play integrated into any RLHF pipeline.
- **Discovery of multi-dimensional preference imbalance**: The work reveals a structural weakness in existing RMs—they tend to model safety well but struggle with correctness—providing clear guidance for RM training data composition.
- **Lightweight evaluation replacing end-to-end validation**: Compared to the costly pipeline of training PPO and evaluating the LLM, probing evaluation requires only 1 epoch of linear classifier training, reducing computational cost by orders of magnitude while maintaining strong correlation with end-to-end performance.

## Limitations & Future Work

- Only 6 preference dimensions are covered; finer-grained dimensions (e.g., culturally specific harmlessness, creativity, factuality) are absent. The appendix provides extension guidelines and case studies on fairness and ethics dimensions.
- The threshold $d_\tau$ for inference-time probing must be set manually, without an adaptive strategy. Different scenarios may require different thresholds.
- The probing classifier uses only a linear layer, which may underestimate preference information encoded non-linearly within the RM. MLP probes may reveal additional structure.
- Alignment experiments are conducted with PPO only; effectiveness on DPO and other RLHF variants remains unverified.
- The evolution of RM preference representations across training stages has not been analyzed, which could yield further insights into RM training dynamics.
- Data sources are limited to PKU-SafeRLHF and HelpSteer; annotation quality and coverage may affect the representativeness of probing tasks.
- The inference-time probing method relies on Euclidean distance, which may be affected by the curse of dimensionality in high-dimensional spaces; cosine or Mahalanobis distance could be explored as alternatives.
- Using MRMBench to guide active sampling and data composition optimization for RM training remains unexplored.

## Related Work & Insights

- **vs. RewardBench (Lambert et al. 2024)**: RewardBench evaluates overall RM accuracy via pairwise ranking and cannot decompose performance into individual preference dimensions. MRMBench achieves dimension-level evaluation through probing tasks, providing substantially richer information.
- **vs. RMB (Zhou et al. 2024) / RM-Bench (Liu et al. 2024)**: Both are RM evaluation efforts but remain within the pairwise ranking paradigm. MRMBench is the first RM evaluation method to adopt a representation probing paradigm.
- **vs. Interpretable RMs (Wang et al. 2024)**: Interpretable RMs achieve interpretability via Chain-of-Thought or MoE architectures, but require retraining under new architectures. Inference-time probing requires no modification to the RM and can be applied directly to any existing RM.

## Rating

- Novelty: ⭐⭐⭐⭐ Probing multi-dimensional preferences in RMs is a novel perspective; inference-time probing offers strong practical utility
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 15+ RMs, includes end-to-end PPO validation and comprehensive Pearson correlation analysis
- Writing Quality: ⭐⭐⭐⭐ Three research questions provide a clear structure; benchmark design motivation is well articulated
- Value: ⭐⭐⭐⭐ Offers direct practical guidance for RM selection, training data composition, and alignment improvement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectifying Shortcut Behaviors in Preference-based Reward Learning](../../NeurIPS2025/interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](../../ICLR2026/interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)

</div>

<!-- RELATED:END -->
