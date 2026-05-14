---
title: >-
  [Paper Note] LLM DNA: Tracing Model Evolution via Functional Representations
description: >-
  [ICLR 2026][Model Compression][LLM DNA] Drawing an analogy from biological DNA, this work formally defines LLM DNA as a low-dimensional bi-Lipschitz representation of a model's functional behavior…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "LLM DNA"
  - "model phylogenetic tree"
  - "functional representation"
  - "phylogenetic analysis"
  - "model provenance tracing"
date: 2026-05-08
content_hash: 4f7f058ae4489a01
---

# LLM DNA: Tracing Model Evolution via Functional Representations

**Conference**: ICLR 2026
**arXiv**: [2509.24496](https://arxiv.org/abs/2509.24496)
**Code**: [GitHub](https://github.com/Xtra-Computing/LLM-DNA)
**Area**: Model Compression
**Keywords**: LLM DNA, model phylogenetic tree, functional representation, phylogenetic analysis, model provenance tracing

## TL;DR
Drawing an analogy from biological DNA, this work formally defines LLM DNA as a low-dimensional bi-Lipschitz representation of a model's functional behavior, proves that it satisfies the properties of heritability and genetic determinism, and designs a training-free RepTrace pipeline to extract DNA from 305 LLMs and construct their evolutionary tree.

## Background & Motivation
Millions of LLMs exist on Hugging Face, derived from one another through fine-tuning, distillation, and adaptation, yet their evolutionary relationships are rarely documented. Tracing model evolution is critical for security auditing (e.g., tracking backdoor propagation), model governance (e.g., license compliance verification), and multi-agent system design.

Limitations of prior work:

**Task-specific representations** (HybridLLM, RouteLLM): trained for specific downstream tasks and lack generality.

**Fixed-model-set representations** (EmbedLLM): adding new models requires retraining; not an intrinsic property.

**Token/parameter-level comparison** (Nikolic et al.): relies on shared tokenizers or architectures and cannot generalize across heterogeneous models.

The core problem is: can one define an intrinsic, universal LLM "DNA" such that functionally similar models have similar DNA, and the DNA remains stable under small perturbations such as fine-tuning?

**Core Idea**: Define LLM DNA as a bi-Lipschitz mapping from the functional space to a low-dimensional space, prove its existence via the Johnson–Lindenstrauss lemma, and implement extraction via random linear projection.

## Method

### Overall Architecture
The RepTrace pipeline: sample an input set → each LLM generates text responses → a sentence embedding model encodes responses into semantic vectors → all response vectors are concatenated → random Gaussian projection into the low-dimensional DNA space.

### Key Designs
1. **Mathematical Definition of LLM DNA**:

    - Function: maps each LLM to a low-dimensional vector (DNA).
    - Mechanism: defines the DNA mapping to satisfy a bi-Lipschitz condition $c_1 \cdot d_H(f_1, f_2) \leq d_\tau(\tau_{f_1}, \tau_{f_2}) \leq c_2 \cdot d_H(f_1, f_2)$. The lower bound guarantees **genetic determinism** (similar DNA → similar function); the upper bound guarantees **heritability** (small modification → similar DNA).
    - Design Motivation: analogizes two core properties of biological DNA and provides rigorous mathematical guarantees.

2. **Existence Proof and Construction**:

    - Function: proves that a DNA satisfying the definition must exist and provides a constructive method.
    - Mechanism: first represents LLM functionality as a vector in a high-dimensional Hilbert space (Lemma A.4), then invokes the JL lemma to guarantee the existence of a low-dimensional bi-Lipschitz embedding. The DNA dimensionality is $L = O\left(\left[\frac{c_2+c_1}{c_2-c_1}\right]^2 \log K\right)$, where $K$ is the number of models.
    - Design Motivation: random projections from the JL lemma are optimal linear dimensionality reduction methods (Larsen & Nelson, 2014) and are computationally efficient.

3. **RepTrace Practical Pipeline**:

    - Semantics-aware representation: a sentence embedding model (e.g., Qwen3-Embedding-8B) encodes text responses into vectors, overcoming the limitations of surface-level text matching.
    - Random functional distance: $t$ representative prompts are sampled to approximate the true functional distance empirically, satisfying the concentration inequality $P(|\frac{1}{t}\hat{d}_f^2 - d_H^2| \geq \epsilon) \leq 2\exp(-\frac{2t\epsilon^2}{C_{\max}^2})$.
    - Implementation details: 100 samples from each of 6 datasets are used as inputs; responses are embedded and concatenated; a random Gaussian matrix $A \sim \mathcal{N}(0, 1/\sqrt{L})$ performs the projection.

### Loss & Training
RepTrace requires no training. The only prerequisites are the sampled input set and a pre-computed random projection matrix, both of which are one-time operations.

## Key Experimental Results

### Main Results (Relation Detection, 305 LLMs)

| Method | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|----|-----|
| Random | 50.0 | 50.0 | 50.0 | 50.0 | 0.500 |
| Greedy | ~65 | - | - | - | - |
| PhyloLM | ~80 | - | - | ~80 | ~0.85 |
| **DNA (Qwen-8B)** | **~95** | - | - | **~95** | **0.992** |
| DNA (BGE-0.3B) | ~95 | - | - | ~95 | 0.99+ |
| DNA (MPNet-0.1B) | ~95 | - | - | ~95 | 0.99+ |

### Ablation Study

| Configuration | AUC | Notes |
|---------------|-----|-------|
| 6-dataset mix (default) | 0.992 | Diverse inputs |
| Single dataset | Slightly lower | Insufficient coverage |
| Qwen3-Embedding-8B | 0.992 | Default embedding model |
| BGE-large-0.3B | 0.99+ | Small model equally effective |
| MPNet-0.1B | 0.99+ | Tiny model also applicable |
| Synthetic random inputs | Still effective | Strong robustness |

### Key Findings
- DNA achieves an AUC of 0.992 for relation detection across 305 LLMs, far surpassing PhyloLM.
- t-SNE visualization clearly reveals model family clustering (Qwen, Llama, etc.) and fine-tuning lineage.
- Multiple undocumented model relationships are discovered (e.g., Vicuna derived from Llama-base; Orca-2 derived from Llama-chat).
- DNA is robust to the choice of embedding model, input data distribution, and chat template variations.
- The constructed phylogenetic tree reflects the architectural transition from encoder-decoder to decoder-only models.

## Highlights & Insights
- The formal definition of LLM DNA (bi-Lipschitz + heritability + genetic determinism) provides a rigorous theoretical foundation for model analysis.
- The training-free, parameter-agnostic design makes it applicable to closed-source models (API access only is sufficient).
- DNA is independent of any fixed model set—the DNA of a new model can be computed independently without affecting existing models.
- Constructing phylogenetic trees introduces biological tooling into the domain of AI model governance.

## Limitations & Future Work
- A tightness trade-off exists between DNA dimensionality $L$ and the bi-Lipschitz constants—high fidelity requires high-dimensional DNA.
- The choice of sampled input set may introduce bias in detecting certain specific relationships.
- The current work focuses on text generation models; multimodal models are not yet covered.
- Analysis of "false positives" suggests recall exceeds precision, likely indicating the existence of undocumented true relationships.

## Related Work & Insights
- **vs. EmbedLLM**: DNA is an intrinsic property and does not depend on a fixed model set.
- **vs. PhyloLM**: based on semantics rather than token distributions, achieving better generalization across tokenizers.
- **vs. watermarking methods**: DNA is extracted post hoc and does not require modifying the training process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rigorously formalizes the biological DNA concept in the LLM domain with elegant theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale validation across 305 models with extensive ablation and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and clear experimental presentation.
- Value: ⭐⭐⭐⭐⭐ Far-reaching implications for model governance, security auditing, and ecosystem analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evolution and compression in LLMs: On the emergence of human-aligned categorization](evolution_and_compression_in_llms_on_the_emergence_of_human-aligned_categorizati.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](../../ACL2026/model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ICLR 2026\] A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [\[NeurIPS 2025\] Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs](../../NeurIPS2025/model_compression/benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)
- [\[ICLR 2026\] Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)

</div>

<!-- RELATED:END -->
