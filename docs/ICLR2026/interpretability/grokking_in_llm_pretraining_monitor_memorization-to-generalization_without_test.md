---
title: >-
  [Paper Note] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test
description: >-
  [ICLR 2026][grokking] This work is the first to validate the grokking phenomenon in near-single-epoch pretraining of a real-scale LLM (7B MoE)—different data groups exhibit asynchronous memorization and delayed generalization. By analyzing the evolution of MoE routing pathways (from instance-specific to structured/shared), two zero-cost metrics are proposed to monitor generalization progress without requiring instruction tuning or benchmark evaluation.
tags:
  - ICLR 2026
  - grokking
  - memorization
  - generalization
  - MoE pathway
  - pretraining dynamics
date: 2026-05-08
content_hash: 89b82bc1abe4c665
---

# Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test

**Conference**: ICLR 2026
**arXiv**: [2506.21551](https://arxiv.org/abs/2506.21551)
**Code**: None
**Area**: Interpretability
**Keywords**: grokking, memorization, generalization, MoE pathway, pretraining dynamics

## TL;DR
This work is the first to validate the grokking phenomenon in near-single-epoch pretraining of a real-scale LLM (7B MoE)—different data groups exhibit asynchronous memorization and delayed generalization. By analyzing the evolution of MoE routing pathways (from instance-specific to structured/shared), two zero-cost metrics are proposed to monitor generalization progress without requiring instruction tuning or benchmark evaluation.

## Background & Motivation

**Background**: Grokking (delayed generalization) is a counterintuitive phenomenon observed when training Transformers—long after training loss converges, generalization performance begins to rise sharply. Existing grokking research is limited to small models trained on algorithmic data for thousands of epochs.

**Limitations of Prior Work**: (a) LLM pretraining is near-single-epoch (~1 epoch), without repeated data replay, making the loss convergence mechanism fundamentally different from multi-epoch training; (b) LLMs are trained on heterogeneous cross-domain data, where memorization speed and the memorization-generalization relationship may differ across data types; (c) Monitoring LLM generalization is extremely costly—requiring instruction tuning followed by benchmark evaluation.

**Key Challenge**: What changes continue to occur inside the model after pretraining loss converges? Why does generalization improve while loss remains unchanged? Are there metrics that can track generalization without relying on external evaluation?

**Goal**: (a) Verify whether grokking exists in real-scale LLM pretraining; (b) Reveal the internal mechanism underlying the memorization-to-generalization transition; (c) Provide zero-cost generalization monitoring metrics.

**Key Insight**: The MoE architecture naturally organizes computation as expert-selection sequences (pathways), enabling tracking of how each sample's pathway evolves—from random/instance-specific (memorization) to structured/cross-sample shared (generalization).

**Core Idea**: Grokking exists in LLM pretraining in a local, asynchronous form; the evolution of MoE pathways from instance-specific to cross-sample shared constitutes an observable signal of the memorization-to-generalization transition.

## Method

### Overall Architecture
Based on a sequence of open-source pretraining checkpoints of OLMoE-7B, the work tracks the memorization time point of training data and the generalization time point on downstream benchmarks to validate local grokking. It then analyzes the dynamic evolution of MoE routing pathways, develops two metrics to quantify pathway complexity, and demonstrates their strong correlation with generalization performance.

### Key Designs

1. **Validation of Local Asynchronous Grokking**:

    - Function: Training data is grouped by memorization time point $t_i^*$, benchmark samples are grouped by the step at which predictions become correct, and groups are paired via Hungarian matching.
    - Core Finding: Different data groups are memorized at different steps, and generalization typically emerges with a lag after memorization. Mathematics and code tasks require memorizing more samples before generalization begins, while commonsense QA generalizes more rapidly.
    - Design Motivation: Demonstrates that grokking in LLMs is not globally synchronous but local and data-heterogeneous.

2. **Pathway Edit Distance (Inter-Sample Similarity)**:

    - Function: Measures the similarity of expert-selection sequences across different training samples at each MoE layer.
    - Mechanism: A pathway string $s_i = \text{concat}(e_1^{(i)}, ..., e_L^{(i)})$ is constructed per sample; pairwise Levenshtein edit distance $D_{path}(s_i, s_j)$ is computed.
    - Key Finding: Early pathways are nearly identical (low edit distance) → diverge during memorization (high edit distance) → **edit distance decreases after memorization**—semantically related samples begin converging to similar pathways, signaling the emergence of shared knowledge.

3. **Pathway Consistency (Layer-wise Smoothness)**:

    - Function: Measures the consistency of expert selection between adjacent layers for a single sample.
    - Mechanism: Weighted cosine similarity of selected expert embeddings between adjacent layers is computed.
    - Key Finding: Pathway consistency increases after memorization—expert selection becomes smoother and more structured across layers.

4. **Theoretical Support**:

    - A connection between pathway complexity and generalization bounds is established on a single-layer MoE.
    - More structured pathways → tighter generalization bounds.

### Loss & Training
- Analysis is based on 10 equally spaced pretraining checkpoints of OLMoE-7B.
- Generalization evaluation: LoRA instruction tuning is applied at each checkpoint, followed by standard benchmark evaluation.
- Metric computation is performed on training data at zero additional cost.

## Key Experimental Results

### Main Results

**Grokking Phenomenon Validation** (4 domains × multiple data groups):

| Domain | Post-Memorization Generalization Delay | Data Difficulty Effect |
|--------|----------------------------------------|------------------------|
| Mathematics | Long delay (requires memorizing many samples) | Later memorization → longer delay |
| Code | Long delay | Same as above |
| Commonsense QA | Short delay | Generalizes relatively quickly |
| Domain QA | Moderate delay | Moderate |

### Ablation Study

| Metric | Correlation with Generalization | Notes |
|--------|---------------------------------|-------|
| Pathway edit distance | Strong negative correlation | Decreasing edit distance → improved generalization |
| Pathway consistency | Strong positive correlation | Increasing consistency → improved generalization |
| Training loss | No significant correlation | Cannot predict generalization after convergence |

### Key Findings
- **Grokking genuinely exists in LLM pretraining**, but manifests locally and asynchronously—different data groups exhibit distinct memorization and generalization time points.
- **Training loss cannot predict generalization**: generalization continues to improve after loss convergence, with the magnitude varying by domain and difficulty.
- **Pathway transition from individualized to structured**: after memorization is complete, the model continues to "memorize more intelligently"—discovering cross-sample transferable knowledge structures.
- **Depth-dependent reorganization**: shallow-layer pathways share structure earliest (universal representations), while deeper layers retain greater flexibility (task specialization).
- **Both metrics correlate strongly with generalization** and can serve as zero-cost generalization monitoring tools.

## Highlights & Insights
- **"Smarter Memorization"**: Loss convergence does not imply learning has ceased—the model continues to discover more efficient encoding schemes (shared pathways), explaining why continued training improves generalization.
- **MoE as an Interpretability Tool**: The discrete nature of expert routing naturally provides a window for analyzing computational allocation, which is not feasible in dense models.
- **Practical Value of Zero-Cost Generalization Monitoring**: For LLM practitioners, the ability to determine when to stop pretraining without performing instruction tuning and benchmark evaluation is highly valuable.
- **Local Grokking Implies Data Curriculum Design**: Varying memorization-to-generalization delays across data types suggest that this knowledge could inform data mixing strategies.

## Limitations & Future Work
- Analysis is limited to OLMoE-7B; grokking behavior in larger models and dense architectures remains unvalidated.
- Pathway metrics depend on the MoE architecture and cannot be directly extended to dense Transformers.
- The choice of instruction tuning method (LoRA vs. full fine-tuning) may affect generalization measurement.
- Causal relationships are not fully established—it remains unclear whether pathway sharing causes generalization or is merely a consequence of it.

## Related Work & Insights
- **vs. Power et al. (original grokking)**: Small models + algorithmic data + thousands of epochs. This work is the first to validate grokking in 7B MoE pretraining at near-single-epoch scale, revealing its local and asynchronous nature.
- **vs. Nanda et al. (grokking mechanism)**: Mechanism explained via weight analysis. This work analyzes MoE pathways instead, which is more amenable to large-scale models.
- **vs. Merrill et al. (subnetwork sparsity)**: Associates grokking with sparsity in ReLU networks. The structured pathways found here echo this finding but are situated within the MoE framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of grokking in real-scale LLM pretraining, revealing local and asynchronous patterns.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 domains × multiple data groups + layer-wise analysis + theoretical support, though only a single model is studied.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is rigorously derived; the progressive revelation of findings is highly engaging.
- Value: ⭐⭐⭐⭐⭐ Fundamental contribution to understanding LLM training dynamics, with a practical zero-cost generalization monitoring tool.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings](../../CVPR2026/interpretability/inside-out_measuring_generalization_in_vision_transformers_through_inner_working.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](../../NeurIPS2025/interpretability/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)

<!-- RELATED:END -->
