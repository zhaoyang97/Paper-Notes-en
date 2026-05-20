---
title: >-
  [Paper Note] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings
description: >-
  [AAAI 2026][Code Intelligence][Self-distillation] This paper proposes ModularStarEncoder (MoSE), a 1B-parameter multi-exit encoder that significantly enhances early-layer representations via a novel self-distillation mec…
tags:
  - "AAAI 2026"
  - "Code Intelligence"
  - "Self-distillation"
  - "Multi-exit network"
  - "Code retrieval"
  - "Early exit"
  - "Modular deployment"
date: 2026-05-08
content_hash: f6d11574c67e97be
---

# MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings

**Conference**: AAAI 2026
**arXiv**: [2503.03008](https://arxiv.org/abs/2503.03008)  
**Code**: [HuggingFace](https://huggingface.co/modularStarEncoder)  
**Area**: Code Intelligence
**Keywords**: Self-distillation, Multi-exit network, Code retrieval, Early exit, Modular deployment

## TL;DR
This paper proposes ModularStarEncoder (MoSE), a 1B-parameter multi-exit encoder that significantly enhances early-layer representations via a novel self-distillation mechanism in which higher layers guide the training of lower layers. MoSE surpasses all open-source models on code understanding tasks such as CodeSearchNet while supporting flexible compute–accuracy tradeoff deployment.

## Background & Motivation

### State of the Field
Large language models have achieved remarkable progress in NLP, yet their substantial computational demands pose severe deployment challenges. The community has proposed several mitigation strategies: quantization to reduce numerical precision, knowledge distillation to train smaller student models, and pruning to remove low-impact parameters. Model families such as LLaMA, Qwen, and Mistral represent a broader shift toward more efficient architectures.

### Limitations of Prior Work

**High cost of conventional distillation**: Separate training of teacher and student models is required, with costs scaling linearly with the number of target models.

**Fixed inference cost**: Standard models must traverse all layers before producing output, precluding adaptive computation based on task difficulty.

**Underutilized value of intermediate layers**: Studies such as SimCLR and Valeriani demonstrate that intermediate rather than final layers often hold the most semantically rich representations.

**Limitations of NSP loss**: Traditional Next Sentence Prediction loss provides negligible benefit after fine-tuning and inefficiently exploits the long-context window for code inputs.

### Root Cause
How can multiple sub-models of varying computational cost be trained simultaneously within a single model, while ensuring that early layers also produce high-quality representations?

### Starting Point
Multiple exit points are introduced within a single Transformer. Multi-layer losses propagate training signals from higher layers to lower layers (self-distillation) without requiring an external teacher model. In addition, an In-Context Classification (ICC) loss replaces NSP to improve context window utilization.

## Method

### Overall Architecture
A 1B-parameter bidirectional encoder based on the StarCoder-2 architecture with 36 layers. Exit heads are inserted at layers 4, 9, 18, 27, and 36. Each exit jointly computes MLM and ICC losses, which are combined via a weighted sum (self-distillation). At inference time, users may select any exit point, yielding a minimum footprint of 160M parameters.

### Key Designs

1. **Self-Distillation Multi-Layer Loss**:

    - Losses are computed independently at selected layers $\iota = \{4, 9, 18, 27, 36\}$
    - All layers share a classification head, augmented with layer-index positional encodings to differentiate them
    - Weighting coefficient $\alpha = i/|I|$, where $I = \{1, ..., 36\}$, assigning greater weight to deeper layers
    - Total loss: $\mathcal{L} = \sum_{i \in \iota} \mathcal{L}_i \cdot \alpha$
    - Effect: Training signals from higher layers propagate naturally to shared lower-layer parameters, encouraging lower layers to learn better representations
    - Design Motivation: A single training run yields multiple models of varying performance, eliminating the redundant cost of repeated distillation

2. **In-Context Classification (ICC) Loss**:

    - Replaces the conventional Next Sentence Prediction objective
    - Code snippets are randomly concatenated (separated by [SEP]), with a 50% probability of originating from different repositories
    - Classification target: whether the concatenated input derives from the same repository
    - Advantages:
        - Increases input density: average input length grows from 630 tokens to 1,300 tokens
        - Repositories are naturally modular and contain multilingual files, facilitating cross-lingual understanding
    - Combined pre-training objective: $\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{ICC}$
    - Design Motivation: NSP provides negligible benefit after fine-tuning and wastes context window capacity

3. **Architectural Modifications**:

    - Based on StarCoder-2: 36 hidden layers, 1B parameters
    - GQA (16 attention heads, 4 KV heads) + RoPE ($\theta=10^6$)
    - Hidden dimension 1024, intermediate dimension 12288
    - Key modifications:
        - Causal mask removed → bidirectional attention
        - Sliding window attention → full attention (to avoid receptive field limitations and ensure modularity)
        - FlashAttention V2 integrated
    - Context length: 2048 tokens

4. **SynthCoNL Dataset**:

    - Seeded from the CodeSearchNet dataset
    - Code translation performed using Qwen2.5-Coder-7B-Instruct
    - Generates 1,071,367 (natural language, code A, code B) triplets
    - Code B spans Go, Ruby, JS, Python, C++, PHP, C, and Java
    - Near-deduplication: LSH + Jaccard similarity threshold 0.7 + 256 permutations + character-level 5-gram
    - Design Motivation: Extends text–code benchmarks and adds cross-lingual code–code retrieval capability

### Loss & Training

**Pre-training**:
- Batch size 4M tokens, maximum context 2048 tokens
- 245,000 steps, processing approximately 1T tokens (TheStackV2 dataset)
- AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, $\varepsilon=1\text{e-}6$, weight decay $=0.1$)
- Multi-step learning rate schedule: 4,000 warmup steps, with staged decay at steps 120K / 185K / 220K / 230K / 240K
- 512 × NVIDIA Ampere (64 GB) GPUs, approximately 450,000 GPU hours

**Fine-tuning (Retrieval)**:
- CLIP-style loss combined with the multi-layer approach
- Five distinct projection heads corresponding to the five exit points
- Batch size 2048, with text–code and code–code pairs uniformly distributed
- Data augmentation: high-frequency tokens in code randomly replaced with 30% probability
- Learning rate $1\text{e-}5$, temperature parameter $10.0$

**Fine-tuning (Classification — BigCloneBench)**:
- Input format: [SEP] snippet-1 [SEP] snippet-2 [CLS]
- Final [CLS] token used for classification
- Learning rate $1\text{e-}5$, 2,000 warmup steps, batch size 64, 14,000 steps

## Key Experimental Results

### Main Results

| Model | Ruby | JS | Go | Python | Java | PHP | avg MRR | avg NDCG | POJ104 mAP |
|-------|------|----|----|--------|------|-----|---------|----------|------------|
| **MoSE** | 74.1 | 74.0 | 82.5 | **92.5** | 78.7 | 84.5 | **81.0** | **84.2** | **75.9** |
| CodeT5+ | 78.0 | 71.3 | 92.7 | 75.8 | 76.2 | 70.1 | 77.4 | - | 24.5 |
| UniXcoder | 74.0 | 68.4 | 91.5 | 72.0 | 72.6 | 67.6 | 74.4 | - | 41.0 |
| ModernBERT-large | - | - | - | - | - | - | - | 59.5 | 27.3 |
| OpenAI Embedding | 84.7 | 85.3 | 95.9 | 99.8 | 90.1 | 95.6 | 91.9 | 93.3 | 82.9 |

| Model | BigCloneBench F1 | Notes |
|-------|-----------------|-------|
| MoSE (L4) | 93.0 | Shallowest exit |
| MoSE (L9) | 93.4 | - |
| MoSE (L18) | **94.2** | Best layer |
| MoSE (L27) | 94.1 | - |
| MoSE (L36) | 94.1 | - |
| CodeT5+ (770M) | 95.1 | Only +0.9 higher |
| UniXcoder | 95.2 | - |

### Ablation Study

| Configuration | avg Recall@1 Gain | Notes |
|--------------|------------------|-------|
| Single-exit baseline L4 | Baseline | Trained only at layer 4 |
| Single-exit baseline L9 | Baseline | Trained only at layer 9 |
| Self-Distilled L9 | +4.36% | Multi-layer loss yields the largest gain |
| Self-Distilled L18–36 | Stable high performance | Performance gap narrows from layer 18 onward |
| ICC vs. NSP (CoIR CSN-CCR) | 10.1 vs. 6.2 | ICC substantially better |
| ICC vs. NSP (CoIR CT-DL) | 32.0 vs. 31.0 | ICC slightly better or comparable |

### Key Findings

1. **Open-source SOTA**: MoSE achieves MRR = 81.0 on CodeSearchNet, surpassing CodeT5+ by 3.6% and substantially narrowing the gap with closed-source OpenAI embeddings.
2. **90% compute reduction with only 6.4% performance loss**: Moving from layer 36 to layer 4 reduces FLOPs by approximately 90% while text–code retrieval MRR drops by only 6.4%.
3. **Task-dependent optimal exit layer**: The optimal layer for text–code retrieval is layer 18, while code–code retrieval peaks at shallower layers, corroborating the uneven distribution of semantic information across layers.
4. **Self-distillation vs. single-exit**: Multi-layer loss consistently outperforms single-exit baselines at all layers, with the largest gain at layer 9 (+4.36% Recall@1).
5. **ICC > NSP**: ICC significantly outperforms NSP on cross-context retrieval tasks while also providing denser input representations.
6. **Cross-architecture generalization**: The approach performs well across AlexNet, VGG11, and ResNet18.
7. **Permutation test**: Different layers produce significantly distinct similarity scores ($p < 0.001$), confirming that the model learns qualitatively different representations at each layer.

## Highlights & Insights
1. The self-distillation design is elegant: higher layers naturally guide lower layers without a separate teacher model, yielding multiple model variants from a single training run.
2. The ICC loss is cleverly designed: it simultaneously improves context utilization (630 → 1,300 tokens) and leverages repository-level structure to enhance cross-lingual understanding.
3. The paper reveals a systematic relationship between task type and optimal exit layer: cross-modal tasks require deeper layers, while same-modal tasks can be resolved at shallower layers.
4. The SynthCoNL dataset construction pipeline is generalizable: code translation models can be used to generate cross-lingual training pairs at low cost.
5. The modular design affords users significant flexibility, enabling on-demand selection between 160M and 1B parameters.

## Limitations & Future Work
1. Computational resource constraints limit thorough hyperparameter tuning and pre-training ablation studies.
2. The impact of fine-tuning on synthetic code translation data remains unclear.
3. A substantial gap from OpenAI Embedding persists (81.0 vs. 91.9 MRR), with the closed-source model's scale and training data remaining opaque.
4. The choice of exit points (4, 9, 18, 27, 36) is empirically determined; more principled exit configurations warrant exploration.
5. Combinatorial use of multiple exit points and a more systematic study of the relationship between task type and depth could be investigated.
6. Evaluation is limited to code understanding tasks; extension to natural language tasks remains as future work.

## Related Work & Insights
- **BranchyNet**: Early exit architecture → realized at scale in large Transformers in this work
- **Matryoshka Representation Learning**: Embedding dimension pruning → this work prunes layer depth instead
- **SimCLR / Valeriani**: Final layers are not necessarily optimal → the core assumption of this paper
- **CodeT5+**: Multi-task code pre-training → replaced here by self-distillation + ICC
- **Knowledge Distillation (Sanh 2019)**: Teacher–student paradigm → this work eliminates dependence on a teacher model

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of self-distillation and ICC is a first for code encoders)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (CodeSearchNet + POJ104 + BigCloneBench + CoIR with thorough ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, intuitive figures, rigorous argumentation)
- Value: ⭐⭐⭐⭐⭐ (High practical deployment value; significant open-source model and dataset contributions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](../../ACL2026/code_intelligence/deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](../../ICLR2026/code_intelligence/reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)

</div>

<!-- RELATED:END -->
