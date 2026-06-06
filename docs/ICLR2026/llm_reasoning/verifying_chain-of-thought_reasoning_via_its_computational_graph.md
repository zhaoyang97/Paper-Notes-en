---
title: >-
  [Paper Note] Verifying Chain-of-Thought Reasoning via Its Computational Graph
description: >-
  [ICLR 2026][LLM Reasoning][Chain-of-Thought] This paper proposes CRV (Circuit-based Reasoning Verification), which constructs interpretable attribution graphs by replacing LLM MLPs with transcoders…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Attribution Graph"
  - "Transcoder"
  - "Reasoning Verification"
  - "Causal Intervention"
date: 2026-05-08
content_hash: 5fbf0bbf66c6b5da
---

# Verifying Chain-of-Thought Reasoning via Its Computational Graph

**Conference**: ICLR 2026
**arXiv**: [2510.09312](https://arxiv.org/abs/2510.09312)  
**Code**: [Available](https://github.com/facebookresearch/CRV)  
**Area**: LLM Reasoning / Mechanistic Interpretability
**Keywords**: Chain-of-Thought, Attribution Graph, Transcoder, Reasoning Verification, Causal Intervention

## TL;DR

This paper proposes CRV (Circuit-based Reasoning Verification), which constructs interpretable attribution graphs by replacing LLM MLPs with transcoders, extracts structural "fingerprints" of reasoning errors from these graphs, and enables white-box CoT reasoning verification with the capacity to correct erroneous reasoning via causal intervention.

## Background & Motivation

Existing CoT verification methods fall into two categories: **black-box methods** (analyzing output text or logit distributions) and **gray-box methods** (using probes over hidden activations or hidden-state trajectories). These approaches can detect correlates of errors but cannot reveal **why** reasoning fails—i.e., they cannot penetrate to the level of the model's computational process to understand the root cause of failure.

The authors' core hypothesis is that models internally implement specific "latent algorithmic circuits" to perform reasoning tasks, and reasoning failures are fundamentally defects in circuit execution. By constructing attribution graphs (analogous to execution traces in software debugging), identifiable error signals can be detected from the structural properties of the computational graph.

## Method

### Overall Architecture

CRV is a four-stage pipeline:

1. **Replace MLPs with Transcoders**: A transcoder (sparse overcomplete representation) is trained for each layer's MLP, with a TopK activation function to enforce sparsity, enabling internal computation to proceed over an interpretable feature basis.

2. **Construct Step-Level Attribution Graphs**: For each reasoning step $s_i$, a greedy path-finding algorithm traces high-attribution connections backward from the final logit, yielding a sparse directed graph $G_i = (\mathcal{V}, \mathcal{E})$ whose nodes include input tokens, transcoder features, and output logits.

3. **Extract Structural Feature Vectors**: A fixed-dimensional structural "fingerprint" $\mathbf{x}_i = \phi(G_i)$ is extracted from each attribution graph.

4. **Train a Diagnostic Classifier**: A gradient boosting classifier (GBC) is trained to predict the correctness of each reasoning step $\hat{y}_i = f_\theta(\mathbf{x}_i)$.

### Key Designs

1. **Transcoder-Based Interpretability Transformation**

    **Function**: Replaces each layer's MLP in the target model with a trained transcoder, routing the forward pass through a sparse, interpretable bottleneck layer.

    **Mechanism**: The transcoder is trained to satisfy $f(x) \approx \text{MLP}(x)$—approximating the input–output function of the MLP via a sparse overcomplete basis rather than performing autoencoder-style self-reconstruction. The output feature dimension satisfies $D \gg d$, with most entries being zero; each nonzero entry corresponds to an interpretable concept.

    **Design Motivation**: Standard SAEs merely reconstruct their own inputs, whereas transcoders serve as functional replacements for MLPs, performing equivalent computation in an interpretable manner and providing a semantic foundation for subsequent attribution graph analysis.

2. **Three-Level Structural Fingerprint Extraction**

    **Function**: Extracts features at three levels of granularity from the pruned attribution graph (retaining nodes/edges accounting for the top 80% of influence).

    **Mechanism**:
    - **Global graph statistics**: Number of active feature nodes, logit probability, and entropy—measuring computational complexity and uncertainty.
    - **Node influence statistics**: Mean, maximum, and standard deviation of activation values and influence scores, as well as per-layer histograms of active features—distinguishing between "few high-activation features driving computation" and "many weak features diffusing influence."
    - **Topological and path features**: Graph density, degree centrality, betweenness centrality, and connectivity—characterizing information flow structure.

    **Design Motivation**: Features at different levels are complementary; their combination is necessary for optimal detection performance. Ablation experiments confirm that node statistics are the most critical component (removal raises FPR@95 by 12 percentage points).

3. **Causal Intervention Verification**

    **Function**: Uses the error-relevant features identified by CRV to guide targeted model correction—suppressing or amplifying specific transcoder features to rectify reasoning errors.

    **Mechanism**: When CRV detects an erroneous reasoning step, it traces back to high-importance transcoder features (e.g., features encoding the "multiplication" concept) and clamps their activation values to zero via a forward hook, thereby redirecting the model's computational path.

    **Design Motivation**: This closed-loop verification establishes that the structural fingerprints identified by CRV bear a **causal** relationship to reasoning errors rather than a merely correlational one, opening a new direction for interpretable model debugging.

### Loss & Training

- Transcoders are trained with an L2 reconstruction loss and a TopK activation function.
- The diagnostic classifier is a gradient boosting classifier (GBC) trained directly on the tabularized extracted features.
- Dataset construction: synthetic tasks (Boolean/arithmetic) are automatically labeled via parsers; GSM8K uses Llama 3.3 70B Instruct as a semi-automatic annotator with subsequent human review.

## Key Experimental Results

### Main Results

| Method | Paradigm | Boolean AUROC↑ | Arithmetic AUROC↑ | GSM8K AUROC↑ |
|--------|----------|----------------|-------------------|--------------|
| MaxProb | Black-box | 58.81 | 61.87 | 54.91 |
| Energy | Black-box | 51.08 | 76.45 | 62.55 |
| CoE-C | Gray-box | 51.03 | 69.39 | 53.57 |
| MLP Probe | Gray-box | 53.63 | 54.41 | 56.02 |
| **CRV (Ours)** | **White-box** | **75.87** | **92.47** | **70.17** |

CRV consistently outperforms both black-box and gray-box baselines across all datasets. On arithmetic tasks it achieves an AUROC of 92.47, reducing FPR@95 to 37.09% (versus 63.33% for the strongest baseline).

### Ablation Study

| Feature Set | Arithmetic AUROC↑ | Arithmetic FPR@95↓ |
|-------------|-------------------|--------------------|
| CRV (all three types) | 92.47 | 37.09 |
| w/o global statistics | 89.62 | 44.54 |
| w/o node statistics | 88.31 | 49.07 |
| w/o topological statistics | 90.89 | 39.19 |

Node influence statistics constitute the most critical feature category.

### Key Findings

- **Error fingerprints are domain-specific**: Errors in different reasoning tasks (Boolean logic vs. arithmetic vs. natural-language mathematics) manifest as distinct structural patterns in the computational graph. A classifier trained solely on arithmetic transfers to GSM8K with only 57.04 AUROC.
- **Joint training recovers performance**: A classifier trained on the combined data from all three tasks achieves 70.62 AUROC on GSM8K, marginally surpassing the task-specific model (70.17).
- **Causal intervention succeeds**: On an arithmetic task, suppressing a single transcoder feature encoding the "multiplication" concept (ID 91814) successfully corrected an erroneous operation order (multiply-then-add) to the correct one (add-then-multiply), changing the answer from 105 to 147.

## Highlights & Insights

- This is the first work to use attribution graphs as "reasoning execution traces" for automated verification, bridging the gap between detection and understanding.
- The work reveals the existence of "computational integrity regions"—correct reasoning occupies a structural space unreachable by erroneous reasoning.
- The closed-loop design spanning detection, diagnosis, and correction represents a complete pipeline that traditional probing methods cannot achieve.

## Limitations & Future Work

- High computational overhead: training per-layer transcoders, constructing attribution graphs, and training classifiers makes CRV unsuitable as a plug-and-play verifier.
- Validation is limited to standard instruction-tuned models; advanced reasoning models with search/backtracking (e.g., o1) have not been tested.
- Cross-domain generalization is limited, requiring annotated data collection and classifier retraining for new tasks.
- Experiments use only Llama 3.1 8B Instruct; performance on larger models remains unknown.

## Related Work & Insights

- Complementary to PRMs (process reward models): PRMs are black-box step-level discriminators, whereas CRV provides white-box interpretable diagnostics.
- Builds on transcoder attribution graph techniques (Ameisen et al., 2025), advancing from qualitative visualization to quantitative automated verification.
- Potential direction: combining CRV's diagnostic capability with the scalability of PRMs to construct hybrid verification systems.

## Rating

⭐⭐⭐⭐ High methodological novelty; white-box attribution graph verification represents a genuinely new perspective, and causal intervention validates causality rather than mere correlation. Practical applicability is, however, constrained by computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](../../ICML2026/llm_reasoning/clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)

</div>

<!-- RELATED:END -->
