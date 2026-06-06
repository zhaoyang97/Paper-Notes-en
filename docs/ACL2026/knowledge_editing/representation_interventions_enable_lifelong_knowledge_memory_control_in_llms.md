---
title: >-
  [Paper Note] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs
description: >-
  [ACL2026][Knowledge Editing][Representation Intervention] This paper proposes RILKE, which transforms lifelong knowledge editing from "modifying model weights" to "applying low-rank interventions in the hidden representa…
tags:
  - "ACL2026"
  - "Knowledge Editing"
  - "Representation Intervention"
  - "Lifelong Memory Control"
  - "Router"
  - "Low-rank Subspace"
date: 2026-05-08
content_hash: 5d69bfb50d711dea
---

# Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs

**Conference**: ACL2026  
**arXiv**: [2511.20892](https://arxiv.org/abs/2511.20892)  
**Code**: Undisclosed  
**Area**: knowledge_editing  
**Keywords**: Knowledge Editing, Representation Intervention, Lifelong Memory Control, Router, Low-rank Subspace

## TL;DR
This paper proposes RILKE, which transforms lifelong knowledge editing from "modifying model weights" to "applying low-rank interventions in the hidden representation space." Through robust training, query-adaptive routing, and shared subspace modules, RILKE maintains near-perfect editing success rates and strong generalization after 1,000 unstructured knowledge edits while significantly reducing storage overhead.

## Background & Motivation
**Background**: Once the parametric knowledge of LLMs is fixed after training, it is difficult to update with the real world. Common solutions include continual pre-training, Retrieval-Augmented Generation (RAG), and model editing. Continual training is costly and prone to forgetting; RAG does not change parameters but is affected by retrieval quality and parameter-memory conflicts; model editing attempts to directly change the internal knowledge of the model, which is suitable for low-cost factual error correction.

**Limitations of Prior Work**: Many editing methods still revolve around structured triplets, such as "what is a certain attribute of a certain subject." Real-world knowledge updates, however, are often unstructured, long-text, and context-dependent free-form answers. Furthermore, deployed models continuously receive new edits. Effectiveness in a single edit does not guarantee stability after multiple accumulations: weight editing suffers from edit collapse, and external memory modules may interfere with each other due to capacity limits and inaccurate routing.

**Key Challenge**: Lifelong knowledge control requires satisfying three objectives simultaneously: each edit must be precise without polluting unrelated questions; paraphrased queries must trigger the same edit; and the storage and training costs should not explode linearly as the number of edits grows. Existing methods usually only satisfy one or two of these.

**Goal**: The authors aim to write complex unstructured knowledge into LLMs in an accumulative, routable, and compressible way, keeping model weights frozen while activating only the relevant knowledge interventions during inference.

**Key Insight**: Starting from the geometry of hidden representations, it is observed that semantically similar questions are closer in the representation space of intermediate layers, and ReFT intervention subspaces trained independently for similar knowledge are also more aligned. This suggests that knowledge editing does not necessarily require weight changes; if local low-dimensional directions can be found in the representation space, model output can be controlled like "pluggable memory."

**Core Idea**: Manage lifelong knowledge using "hidden representation indexing + low-rank representation intervention modules + similarity routing." For each query, the system first finds the corresponding memory in the frozen model's representation space and then applies local interventions only to the relevant representations.

## Method
The core of RILKE is not training a new knowledge base model, but mounting lightweight intervention modules onto specific intermediate layers of a frozen LLM. Each edit is represented by a query $x$ and a target answer $y$. During training, the model learns to shift the hidden states of the original query into a region that generates the target answer. During inference, a router selects the most similar edited knowledge based on the hidden representation of the current query and activates the corresponding intervention if the similarity exceeds a threshold.

### Overall Architecture
The input is a stream of knowledge edit samples, and the output is a set of routable representation intervention modules. The process consists of three steps: First, use the frozen LLM to extract the hidden states of each edit query at a specified layer to serve as knowledge indices. Second, train a low-rank ReFT-style module for each piece of knowledge or semantic cluster, enabling the model to generate the target answer without modifying original weights. Third, during inference, extract the hidden representation at the same layer for new queries and perform cosine similarity matching against indices to route to the nearest module; if the maximum similarity is below a gating threshold $\tau_{sim}$, no intervention is applied.

### Key Designs
1. **Consistency-Robust Representation Intervention Training**:
    - **Function**: Ensures an edit is effective not only for the original query but also for semantically equivalent paraphrases.
    - **Mechanism**: Following the low-rank intervention form of ReFT, a transformation $\Phi(h)=h+R^\top(Ah+b-Rh)$ parameterized by $R, A, b$ is learned on the hidden state $h$. The authors further assume that hidden states of paraphrases fall within an $\epsilon$-ball around the original query. Therefore, perturbations are added to hidden representations during training, and a KL term is used to constrain consistency between the original and perturbed distributions. The final objective is language modeling cross-entropy plus a robust regularizer $\lambda_{robu} KL(p(h)\|p(h+\epsilon))$.
    - **Design Motivation**: Fitting only the target answer on the original query leads to overfitting on surface forms; maintaining output consistency within a representation neighborhood is equivalent to expanding an edit from a single point to a local semantic region.

2. **Query-Adaptive Routing and Gating**:
    - **Function**: Selects the correct module among massive lifelong edits and avoids erroneous rewriting of unrelated questions.
    - **Mechanism**: Layer representations $h_x^l$ of all edit queries are saved as keys after training. At inference, the representation $h_{\hat{x}}^l$ of a new query is matched via cosine similarity to all keys to route to the nearest module. If the maximum similarity is below $\tau_{sim}$, no intervention occurs. In experiments, the threshold for irrelevant knowledge is set to 0.9 to reduce spurious activations.
    - **Design Motivation**: Freezing the backbone means the key space is stable and won't drift with subsequent edits; meanwhile, semantically similar questions naturally cluster, allowing the router to send paraphrases to the same memory.

3. **Shared Subspace Cluster-Level Intervention**:
    - **Function**: Reduces the linear storage growth caused by having one adapter per piece of knowledge.
    - **Mechanism**: The authors use layer representations for hierarchical agglomerative clustering, requiring intra-cluster similarity above a threshold and cluster size below a limit. A shared intervention module is then trained for each semantic cluster. During inference, the system still finds the nearest knowledge item but maps it to the shared module of its cluster.
    - **Design Motivation**: The paper validates that ReFT subspaces of semantically similar knowledge are more aligned, allowing similar knowledge to share a single low-dimensional subspace; this compresses "one memory per knowledge item" into "one memory per knowledge cluster."

### Loss & Training
RILKE freezes LLaMA-3.1-8B-Instruct or Qwen2.5-7B-Instruct and only trains the representation intervention modules. Single-knowledge training uses teacher-forcing autoregressive cross-entropy with KL consistency regularization. The shared subspace version first clusters by hidden states and then trains one module per cluster batch. Inference uses a deterministic generation setting. Editing tasks are evaluated using Rouge-L, BertScore, MMLU retention, and reliability/generalization/locality metrics on ZsRE.

## Key Experimental Results

### Main Results
UnKE is the primary unstructured knowledge editing benchmark. Below are the key results for 1,000 sequential edits on LLaMA-3.1-8B-Instruct. Ours (RILKE) achieves near-perfect scores on original queries and significantly outperforms long-term editing baselines like WISE and GRACE on paraphrases, while MMLU retention remains close to the unedited model.

| Method | 1,000 edits Original BertS↑ | 1,000 edits Paraphrase BertS↑ | MMLU↑ | ZsRE Avg↑ | Major Observations |
|------|---------------------------|-------------------------------|-------|----------|----------|
| MEMIT | 0.033 | 0.034 | 0.188 | 0.00 | Collapses significantly after cumulative edits |
| GRACE | 0.810 | 0.521 | 0.594 | 0.49 | Accurate on original, lacks generalization |
| WISE | 0.681 | 0.673 | 0.584 | 0.73 | Stable but limited editing precision |
| **RILKE** | 1.000 | 0.963 | 0.622 | 0.88 | High success rate, high generalization, low utility loss |

Storage costs also demonstrate the efficiency of representation intervention. Individual RILKE modules are already more memory-efficient than WISE, and shared subspaces further compress overhead to approximately one-third.

| Method | UnKE Storage Cost | Relative to WISE | Description |
|------|---------------|-----------|------|
| WISE | 224.0 MiB | 100% | Stores external memory/sub-modules |
| RILKE (Individual) | 96.1 MiB | 42.9% | One low-rank module per knowledge |
| RILKE (Shared) | 29.4 MiB | 13.1% | One shared module per semantic cluster |

### Ablation Study
Robust training mainly improves paraphrase generalization without harming original query success. Shared subspaces provide significant compression with limited generalization loss.

| Configuration | T=100 Original BertS↑ | T=100 Paraphrase BertS↑ | T=1,000 Original BertS↑ | T=1,000 Paraphrase BertS↑ |
|------|--------------------|--------------------------|----------------------|----------------------------|
| w/o $\mathcal{L}_{robu}$ | 1.000 | 0.959 | 0.999 | 0.909 |
| w/ $\mathcal{L}_{robu}$ | 1.000 | 0.984 | 1.000 | 0.963 |

| Configuration | Original BertS↑ | Paraphrase BertS↑ | MMLU↑ | Description |
|------|----------------|--------------------|-------|------|
| RILKE (Individual) | 1.000 | 0.963 | 0.622 | Best precision and generalization |
| RILKE (Shared) | 0.999 | 0.901 | 0.621 | Slight drop in generalization, massive storage saving |
| Batched RILKE | 1.000 | 0.834 | - | Intra-cluster joint training is stronger |
| Sequential RILKE | 0.742 | 0.723 | - | Strict online absorption still beats AnyEdit/UnKE |

### Key Findings
- Semantic locality in representation space is the true lever: the closer distance between paraphrases and original queries provides a foundation for both routing and robust training.
- Shared subspaces are not just a compression trick but are built on the empirical property that "similar knowledge has similar intervention directions"; randomly batching dissimilar knowledge significantly biases the edit vectors.
- RILKE's strength lies in long-tail, unstructured, and cumulative edits, making it particularly suitable for scenarios where models continuously receive new knowledge post-deployment.

## Highlights & Insights
- The most ingenious part is decoupling knowledge editing into a "stable key space" and "pluggable value modules." By freezing the backbone, intermediate representations become a searchable index, avoiding parameter drift caused by repeated weight editing.
- Robust KL regularization acts as expanding a single-point edit into a semantic neighborhood edit. It improves paraphrase performance without explicitly collecting large datasets, suggesting that hidden space perturbation is a cheap form of data augmentation.
- Shared subspaces offer a scalable direction for lifelong editing. Future work could combine online clustering or periodic retraining to write new knowledge individually and merge them into cluster-level modules in the background.
- This paper serves as a reminder that knowledge editing does not always have to aim for "permanently writing facts into parameters." In many applications, reversible, routable, and auditable representation interventions may better fit engineering requirements.

## Limitations & Future Work
- The authors explicitly leave systemic risk analysis for future work, including malicious edits, bias amplification, and robustness under biased edit policies.
- The routing threshold is a critical hyperparameter. Too low a threshold activates irrelevant knowledge; too high misses paraphrases. Large-scale open-domain scenarios might require calibration, confidence estimation, or multi-stage retrieval.
- RILKE requires accessing and storing target layer hidden representations and training specific intervention modules for the target model; modules cannot be directly transferred if the model or layer changes.
- Shared subspaces sacrifice some paraphrase generalization, indicating that fine-grained conflicts may still exist between similar knowledge. Future work could consider intra-cluster Mixture-of-Adapters, dynamic rank, or conflict detection.
- While the editing effect is strong, discussions on safety boundaries, reversibility, audit logs, and permission control are limited, which are crucial for real-world knowledge management systems.

## Related Work & Insights
- **vs ReFT**: ReFT provides the basic form of low-rank representation intervention; RILKE extends it to knowledge editing and adds paraphrase robustness, routing, and lifelong accumulation management.
- **vs MEMIT / locate-then-edit**: Methods like MEMIT directly modify weights, suitable for single or small-batch factual edits; RILKE freezes weights to avoid edit collapse after many edits.
- **vs GRACE / WISE**: These external memory methods also preserve original parameters but usually learn control within a single sub-module or external memory; RILKE uses hidden representation keys for fine-grained routing and reduces storage via low-rank modules.
- **vs RAG**: RAG inserts retrieval evidence at the text level, prone to retrieval failure and parameter-knowledge conflicts; RILKE directly changes the generation trajectory at the representation layer, acting more like conditional control of the model's internal state.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reconstructs lifelong knowledge editing through representation geometry, routing, and shared subspaces with a complete and distinct logic.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers UnKE, EditEverything, ZsRE, MMLU, and various ablations, though real-world open-domain safety requires more systematic evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation and methodology with sufficient tables; relationships between geometric properties and engineering hyperparameters could be further detailed.
- Value: ⭐⭐⭐⭐⭐ Directly inspiring for lifelong knowledge updates, enterprise knowledge customization, and controllable model memory, especially for scenarios requiring pluggable edits.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](../../NeurIPS2025/knowledge_editing/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ACL 2026\] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](../../ICLR2026/knowledge_editing/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [\[ACL 2026\] Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse](spectral_characterization_and_mitigation_of_sequential_knowledge_editing_collaps.md)

</div>

<!-- RELATED:END -->
