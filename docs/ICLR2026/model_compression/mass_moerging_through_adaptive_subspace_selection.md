---
title: >-
  [Paper Note] MASS: MoErging through Adaptive Subspace Selection
description: >-
  [ICLR 2026][Model Compression][model merging] MASS stores the low-rank singular subspaces updated by each task into a shared model. During inference, it utilizes a data-free and training-free "projection residual" router to automatically select the task subspace and classification head that best match the input without knowing the task identity, pushing model merging accuracy to approximately 98% of individually fine-tuned models.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "model merging"
  - "MoErging"
  - "task vectors"
  - "SVD"
  - "training-free routing"
date: 2026-05-08
content_hash: ff0300f0054cc219
---

# MASS: MoErging through Adaptive Subspace Selection

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CRBt6DNaBE](https://openreview.net/forum?id=CRBt6DNaBE)  
**Code**: [https://github.com/crisostomi/mass](https://github.com/crisostomi/mass)  
**Area**: Model Merging / Multi-task Model Merging  
**Keywords**: model merging, MoErging, task vectors, SVD, training-free routing  

## TL;DR
MASS stores the low-rank singular subspaces updated by each task into a shared model. During inference, it utilizes a data-free and training-free "projection residual" router to automatically select the task subspace and classification head that best match the input without knowing the task identity, pushing model merging accuracy to approximately 98% of individually fine-tuned models.

## Background & Motivation
- **Background**: Model merging aims to merge multiple models, which share the same pretrained backbone and are fine-tuned on different tasks, into a single set of parameters without training. From Task Arithmetic (directly adding task vectors "fine-tuned weights − pretrained weights") to Task Singular Vectors (TSV, discovering that the task update matrix $\Delta_i$ is low-rank and retains most accuracy by keeping only top-k singular vectors per layer), merging quality has steadily improved.
- **Limitations of Prior Work**: ① Pure merging methods (Weight Averaging / TA / TSV-M / Iso-C) consistently fail to reach the accuracy upper bound of "individual fine-tuning for each task"; ② Existing MoErging methods (SMILE, WeMoE, TwinMerging), despite adding routers, generally **default to assuming that task identity and the correct classification head are known during inference**—a bit of an unrealistic oracle assumption; ③ Training a real router often requires original task data, which is frequently unavailable in merging scenarios (e.g., models downloaded from HuggingFace).
- **Key Challenge**: Either assume the task is known (unrealistic) or fall back to compression (e.g., TSV-C achieves 99.5% normalized accuracy with 2× storage, nearly "solving" the known-task setting). The truly difficult and meaningful setting is: **unknown tasks, requiring automatic selection of both encoder subspaces and classification heads**.
- **Goal**: Create a single generalist model capable of handling all fine-tuned tasks without external supervision or prior knowledge of the task identity.
- **Core Idea**: **[Data-free, training-free weight-space routing]** Since TSV-M already encodes the top singular directions of each task into the orthogonal subspaces of a shared model, "determining which task the input belongs to" reduces to "measuring which subspace best reconstructs the input activation"—a single orthogonal projection residual calculation is sufficient, requiring neither labels nor retraining.

## Method

### Overall Architecture
MASS consists of two phases: **Fixed Merging** (one-time preprocessing) uses TSV-M to aggregate the low-rank singular subspaces of each task into a task-discriminative shared encoder $\theta_{MT}$; **Adaptive Merging** (during inference) executes two forward passes for each input—the first pass uses $\theta_{MT}$ to extract intermediate layer activations and routes to select relevant task subspaces, and the second pass temporarily merges the selected subspaces into the model and takes the maximum logit across the selected classification heads. The overall approach requires only ~2× storage and ~2× forward passes compared to a single pretrained model, and is independent of the number of tasks.

```mermaid
flowchart LR
    A[Input x] --> B[First Forward Pass<br/>θMT extract mid-activation zℓ]
    B --> C[Projection Routing<br/>Calculate residual rᵢ for each task]
    C --> D[softmax(-r)+threshold η+TopK<br/>Select task subset Ω]
    D --> E[Adaptive Merging Δada<br/>= Σ UΣVᵀ over Ω]
    E --> F[θMASS = θpre + αΔada]
    F --> G[Second Forward Pass<br/>Take max logit across heads in Ω]
    G --> H[Predicted Class c*]
```

### Key Designs

**1. Projection Residual Routing: Using reconstruction error as task likelihood.** Given the intermediate activation $z_\ell$ at layer $\ell$, MASS does not train any routing network but directly performs an orthogonal projection of $z_\ell$ onto the subspace $\mathrm{span}(V_i^{(\ell)})$ spanned by the right singular vectors of each task, calculating the Euclidean residual $r_i = \|z_\ell - V_i^{(\ell)}(V_i^{(\ell)})^\top z_\ell\|_2$. A smaller residual indicates that the subspace better explains the input. Task coefficients are obtained by passing $-r$ through a softmax, and tasks with coefficients below a threshold $\eta$ are filtered, keeping only the top-k if exceeded. This mechanism is entirely data-independent and training-free, directly applicable to merging scenarios. The authors further prove (Prop. 3.1): if the residuals follow isotropic Gaussian distributions $\varepsilon_i\sim\mathcal{N}(0,\sigma^2 I)$ and task priors are uniform, then **selecting the task with the smallest residual is equivalent to MAP estimation**, consistent with the maximum likelihood solution of Probabilistic PCA—in the absence of training data to fit complex distributions, this "least biased" isotropic prior is precisely suited for MASS.

**2. Redundancy Direction Removal: Preventing similar tasks from overshadowing each other.** Projection routing has a pitfall: if two tasks (e.g., MNIST and EMNIST) have highly similar training data and significantly overlapping right singular directions, the union of their subspaces will appear "wider and stronger" in certain feature regions, causing the router to misclassify samples belonging to a third similar task (KMNIST) as MNIST/EMNIST—because $\|z_\ell-\mathrm{Proj}_{V_{MN}\cup V_{EMN}}(z_\ell)\|_2 < \|z_\ell-\mathrm{Proj}_{V_{KM}}(z_\ell)\|_2$. MASS performs de-redundancy during the fixed merging stage: it selects one task matrix as a seed and examines remaining tasks one by one, flattening $\Delta_i$ into $\delta_i=\mathrm{vec}(\Delta_i)$, merging it only if its cosine similarity with all accepted tasks $\max_m \mathrm{sim}(\delta_i,\delta_{a_m}) \le \varepsilon$ (e.g., $\varepsilon=0.3$). This prevents highly similar subspaces from overshadowing rare tasks, ensuring "no task overwhelms others."

**3. Joint Encoder Subspace and Classification Head Selection: Completely discarding the oracle.** After the router selects the task subset $\Omega$, MASS merges these subspaces into $\theta_{MASS}=\theta_{pre}+\alpha\Delta_{ada}$ using TSV-M and runs the second forward pass to obtain the shared representation $z_{L-1}$. Unlike conventional merging that "assumes the oracle gives the correct classification head," MASS computes $z_i=h_i(z_{L-1})$ for the classification head $h_i$ of each task in $\Omega$ and takes the maximum logit across all classes of all heads: $(i^\star,c^\star)=\arg\max_{(i,c)\in\Omega\times\{1,\dots,C_i\}} z_i[c]$. This allows the most "confident" head to win, simultaneously determining the encoder subspace, classification head, and label space for each input—the key to handling the "unknown task" setting.

**4. Selection of routing layers.** The layer at which residuals are calculated is critical. Experiments found that for ViT-B-32 and ViT-B-16, layer 9 (with MLP layers slightly better than self-attention) is generally optimal, but the **optimal layer strongly depends on the task**: STL10 routes more accurately at earlier layers (3/4/5), while SUN397 performs better at later layers (9/10/11), with single-layer accuracy variance across tasks reaching up to 40%. This suggests that adaptive layer selection is a direction worth exploring; MASS currently uses a fixed best layer (layer 9) as a compromise.

## Key Experimental Results

### Main Results: Vision Task Merging (CLIP, normalized accuracy, percentage relative to individual fine-tuning upper bound in parentheses)

| Method | ViT-B-32 (8/14/20) | ViT-B-16 (8/14/20) | ViT-L-14 (8/14/20) |
|---|---|---|---|
| Finetuned Upper Bound | 90.3(100)/89.0(100)/89.5(100) | 92.4/91.3/91.9 | 94.2/93.4/94.0 |
| Task Arithmetic | 68.8(75.7)/64.6/64.0 | 73.0/70.6/69.0 | 84.4/80.4/76.9 |
| TSV-M (MASS Base) | 83.2(91.8)/78.6/75.6 | 85.5/81.4/78.8 | 91.2/88.8/87.5 |
| Iso-CTS | 82.0/80.6/77.0 | 88.7/84.1/80.7 | 92.8/91.1/89.2 |
| SMILE-2 | 84.4/76.4/74.1 | 89.0/82.7/80.4 | 92.0/87.1/85.5 |
| **MASS** | **87.0(96.5)/82.9(93.2)/81.1(90.9)** | **90.6/87.8/81.1** | **92.9/90.9/90.8** |

MASS achieved SOTA in 8 out of 9 benchmarks, with a maximum gain of about 6% over the best baseline; compared to its underlying TSV-M, routing brings an improvement of approximately 5%.

### Cross-modal validation: Flan-T5-Base on GLUE 8 tasks (normalized average)

| Method | Avg. |
|---|---|
| Task Arithmetic | 91.3 |
| SMILE-2 | 99.0 |
| **MASS** | **99.4** |

MASS achieved the highest absolute accuracy in 5 out of 9 GLUE subtasks, trailing slightly only in NLI tasks (MNLI/QNLI)—which the authors attribute to NLI requiring broader semantic reasoning, more diffuse task vectors, and higher rank. This proves the method is **modality-agnostic**.

### Ablation Study
**Router Comparison (ViT-L-14, normalized accuracy)**

| Router | 8 / 14 / 20 tasks | Notes |
|---|---|---|
| nn (Nearest Neighbor) | 94.0 / 92.1 / 92.0 | Requires storing embeddings for each task's val set |
| mlp (Trained MLP) | 98.9 / 99.5 / 98.3 | Requires labeled validation sets, violates merging premise |
| proj-PRE (Project from pretrained backbone) | 99.1 / 97.7 / 91.9 | Fails when tasks increase |
| **proj-TSV-M (Ours)** | 98.6 / 97.3 / **96.6** | Data-free/training-free, best scalability |

**Key Findings**:
- Projection routing from the TSV-M model (proj-TSV-M) significantly outperforms projection from the pretrained backbone (proj-PRE) when the number of tasks is large (20), with a gap of ~10% on ViT-B-32—confirming the core insight: TSV-M has already encoded task directions into orthogonal subspaces, and routing simply needs to "retrieve" the corresponding subspace.
- **Value**: MASS maintains a constant 2× parameter storage (independent of task count), while other MoErging baselines range between ~2.5× and 14×.
- **Batch Inference**: If a batch of samples belongs to the same domain, routing only needs to be performed once per batch, reaching ≥97% normalized accuracy in 8 out of 9 settings, nearly matching individually fine-tuned models.
- **Task Scaling (2→33 tasks)**: MASS maintains high accuracy throughout, and accuracy is not a monotonic function of the number of tasks—scalability depends more on task set composition than quantity, contrasting sharply with the steep degradation of TA.

## Highlights & Insights
- **Reformulating "routing" as "subspace projection residual"**: Because TSV-M encodes task directions into orthogonal subspaces, the router requires no learning and no data; it is simply a linear algebra operation, elegant and zero-cost.
- **MAP Perspective**: Residual minimization = MAP estimation under isotropic Gaussian noise, providing a clean probabilistic explanation for the training-free router.
- **Addressing the Oracle Assumption**: The first to perform MoErging in the more realistic "unknown task + unknown head" setting while still outperforming fixed-merging baselines that assume known tasks.
- **Constant 2× Storage**: Does not bloat with the number of tasks, a tangible engineering advantage over MoE routing methods that often require over ten times the storage.

## Limitations & Future Work
- **Fixed Routing Layer**: The optimal layer strongly depends on the task (single-layer variance across tasks reaches 40%); currently, a global best layer (layer 9) is used as a compromise, and adaptive layer selection remains unsolved.
- **Two Forward Passes**: Inference requires two forward passes, incurring higher costs than pure compression methods (e.g., TSV-C with zero extra inference overhead).
- **Manual Threshold $\varepsilon$ for De-redundancy**: Similar task removal relies on an artificial threshold (e.g., 0.3), lacking an adaptive mechanism.
- **Weakness in NLI-type Tasks**: The advantage of projection routing diminishes for task vectors with high rank and diffuse semantics, suggesting a need for stronger subspace characterization for "non-local feature shift" tasks.
- Still limited to merging isomorphic models sharing the same pretrained backbone.

## Related Work & Insights
- **Task Arithmetic / Ties / Consensus TA**: This family of training-free merging established the foundation; MASS inherits the "$\theta_{pre}+\alpha\sum\tau_i$" backbone but adds input-conditioned gating.
- **TSV-M / Iso-C (Gargiulo et al. 2025)**: Discovering the low-rank and subspace orthogonal structure of task updates is the direct premise for the fixed merging stage and routing feasibility in MASS.
- **MoErging: SMILE / WeMoE / TwinMerging**: Incorporating routing into merging, but all default to knowing the classification head; MASS's core differentiator is discarding this oracle.
- **Probabilistic PCA (Tipping & Bishop 1999)**: Provides a theoretical mirror of maximum likelihood for residual minimization routing.
- **Insight**: Once multiple experts are embedded into the same set of orthogonal subspaces, "expert selection" can completely reduce to "subspace retrieval via projection"—this idea could migrate to broader modular learning scenarios like dynamic LoRA library combinations or cross-modal expert routing.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reformulating routing as subspace projection residual with MAP explanation and addressing the "unknown task/head" setting is fresh and self-consistent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid coverage across 3 ViT types × {8,14,20} tasks + GLUE cross-modal + 2→33 task scaling + multi-dimensional ablation of routers/layers/batching.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation progresses logically; the MNIST/EMNIST/KMNIST redundancy example is intuitive, and theory aligns clearly with diagrams.
- **Value**: ⭐⭐⭐⭐ Approaches the fine-tuning upper bound (~98%) at a constant 2× storage without needing data or training, holding significant practical value for model merging deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[CVPR 2026\] Bridging Domains through Subspace-Aware Model Merging](../../CVPR2026/model_compression/bridging_domains_through_subspace-aware_model_merging.md)
- [\[ICLR 2026\] Achieving low-bit Muon through subspace preservation and grid quantization](achieving_low-bit_muon_through_subspace_preservation_and_grid_quantization.md)
- [\[ICLR 2026\] GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching](gptailor_large_language_model_pruning_through_layer_cutting_and_stitching.md)
- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](../../ICML2026/model_compression/mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)

</div>

<!-- RELATED:END -->
