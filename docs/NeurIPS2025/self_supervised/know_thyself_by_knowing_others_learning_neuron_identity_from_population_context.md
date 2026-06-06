---
title: >-
  [Paper Note] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context
description: >-
  [NeurIPS 2025][Self-Supervised Learning][neuron identity] This paper proposes NuCLR, a self-supervised framework that learns neuron-level representations enriched with population context via contrastive learning—pulling…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "neuron identity"
  - "contrastive learning"
  - "spatiotemporal Transformer"
  - "cell type decoding"
date: 2026-05-08
content_hash: f10f757eda27b715
---

# Know Thyself by Knowing Others: Learning Neuron Identity from Population Context

**Conference**: NeurIPS 2025
**arXiv**: [2512.01199](https://arxiv.org/abs/2512.01199)  
**Code**: [GitHub](https://github.com/nerdslab/nuclr)  
**Area**: Self-Supervised Learning
**Keywords**: neuron identity, self-supervised learning, contrastive learning, spatiotemporal Transformer, cell type decoding

## TL;DR
This paper proposes NuCLR, a self-supervised framework that learns neuron-level representations enriched with population context via contrastive learning—pulling together different temporal windows of the same neuron and pushing apart different neurons within a population. NuCLR achieves new state-of-the-art performance on cell type and brain region decoding, and is the first to demonstrate cross-animal zero-shot generalization and data scaling laws in this domain.

## Background & Motivation

### Limitations of Prior Work

1. **Background**: Understanding neuron identity (cell type, brain region, connectivity) is fundamental to neuroscience, yet traditional approaches (molecular markers, morphology) are costly and of limited coverage.
2. **Key Challenge**: Inferring identity from neural activity is challenging—existing methods rely solely on single-neuron features (waveforms, ISI) and ignore population context, and require retraining for each new animal.
3. **Core Idea**: Neuron identity is temporally stable, and population activity encodes rich contextual information. Contrastive learning can be applied to pull representations of the same neuron at different times closer together while pushing apart representations of different neurons.

## Method

### Overall Architecture
Self-supervised pretraining (contrastive learning) → frozen encoder → linear probe evaluation for cell type / brain region.

### Key Design 1: Spatiotemporal Transformer Architecture
- **Input**: Spike trains from a population of $N$ neurons, binned and divided into patches.
- **Temporal Transformer layers** ($L_T$ layers): Each neuron's temporal sequence is processed independently, with relative temporal positions encoded via RoPE.
- **Spatiotemporal Transformer layers** ($L_{ST}$ layers): Spatial attention (cross-neuron interaction at each time point) and temporal attention are applied in an interleaved fashion.
- **Permutation equivariance**: No fixed neuron ordering is assumed, naturally accommodating variable-size populations.
- Each neuron's representation vector is obtained by averaging over the temporal dimension.

### Key Design 2: Contrastive Learning Objective (NuCLR)
- Two temporal windows are sampled from the same recording session as two views.
- Random dropout of up to 50% of neurons is applied for robustness.
- **InfoNCE loss**: The same neuron across the two views forms a positive pair; different neurons within the same population form negative pairs.
- **Key distinction**: Negative pairs are not constructed across different animals or sessions, avoiding an excess of trivially easy negatives.

### Evaluation Protocols
- **Transductive**: The test population participates in pretraining; a classifier is trained on a subset of labeled neurons.
- **Transductive zero-shot**: The test population participates in pretraining but no labels are used.
- **Inductive zero-shot**: The test population is entirely unseen; no retraining is performed.

## Key Experimental Results

### Cell Type Decoding (Macro F1)

### Main Results

| Dataset | Setting | NuCLR | NeuPRINT | NEMO | LOLCAT |
|--------|------|-------|----------|------|--------|
| Allen VC (3 classes) | Ind. zero-shot | **0.720** | N/A | 0.419 | 0.412 |
| Bugeon E vs I | Transductive | **0.811** | 0.666 | N/A | 0.721 |
| Bugeon 5-class | Transductive | **0.610** | 0.495 | N/A | 0.290 |

### Brain Region Decoding (Macro F1)

### Ablation Study

| Dataset | Setting | NuCLR | NEMO |
|--------|------|-------|------|
| IBL 10 regions | Ind. zero-shot | **0.530** | 0.379 |
| Steinmetz 4 regions | Transductive | **0.959** | 0.699 |

### Scaling Analysis
- Doubling the number of pretraining animals yields greater gains in cell type decoding than doubling the number of supervised labels.
- Using only 12.5% of labels to train the classifier surpasses baselines trained with 100% of labels.

## Highlights & Insights
1. **Population context is critical**: Methods relying solely on single-neuron features (NEMO/LOLCAT) fall substantially behind.
2. **Zero-shot generalization**: The same pretrained model transfers directly to entirely new animals without retraining.
3. **First scaling analysis**: More unlabeled data (i.e., more animals) consistently improves performance.
4. **Cross-modality applicability**: The same framework applies to both electrophysiology and calcium imaging data.

## Limitations & Future Work
1. Decoding inhibitory neuron subtypes (5-class) achieves only F1 = 0.39 under the inductive setting, leaving room for improvement.
2. Spatial attention scales as $O(N^2)$ in population size, which may limit applicability to very large-scale recordings.
3. Validation is currently restricted to mouse data; generalization to primate or human brain data remains to be verified.
4. Each probe insertion is treated as an independent population, leaving cross-probe information unexploited.

## Related Work & Insights
- **vs. NEMO**: NEMO uses a CLIP-style approach with autocorrelograms and waveform templates, lacking population context.
- **vs. NeuPRINT**: Uses population-level statistical summaries but cannot generalize inductively.
- **vs. LOLCAT**: A supervised method that requires labels and relies on ISI distributions.
- **vs. POYO+**: A multi-session decoding method whose neuron embeddings substantially underperform NuCLR on cell type decoding.

## Related Work & Insights
- Contrastive learning in neuroscience as a paradigm: temporal invariance as a self-supervised signal.
- The permutation-equivariant spatiotemporal Transformer design is transferable to other set-structured data.
- The observed scaling laws suggest that the value of large-scale neural datasets (e.g., Allen Brain Atlas) will continue to grow.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Self-supervised learning with population context for neuron identity represents a new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, three evaluation protocols, scaling analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem-driven narrative, rigorous structure, and clear figures.
- **Value**: ⭐⭐⭐⭐ An important tool for computational neuroscience.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[NeurIPS 2025\] Long-Tailed Recognition via Information-Preservable Two-Stage Learning](long-tailed_recognition_via_information-preservable_two-stage_learning.md)

</div>

<!-- RELATED:END -->
