---
title: >-
  [Paper Note] ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation
description: >-
  [ICML 2026][Graph Learning][Generalist graph anomaly detection] ProMoS treats a frozen self-supervised GNN as a "normality prior teacher" and distills it into a set of shared and sparsely activated lightweight student br…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Generalist graph anomaly detection"
  - "prototype distillation"
  - "Mixture-of-Students"
  - "zero-shot"
  - "self-supervised GNN"
date: 2026-05-08
content_hash: 4a513cbbbb708812
---

# ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation

**Conference**: ICML 2026  
**arXiv**: [2605.26857](https://arxiv.org/abs/2605.26857)  
**Code**: https://github.com/yimingxu24/ProMoS  
**Area**: Graph Learning / Anomaly Detection / Self-supervised  
**Keywords**: Generalist graph anomaly detection, prototype distillation, Mixture-of-Students, zero-shot, self-supervised GNN  

## TL;DR
ProMoS treats a frozen self-supervised GNN as a "normality prior teacher" and distills it into a set of shared and sparsely activated lightweight student branches. By aligning teachers and students to a cross-graph shared semantic space via learnable prototypes, it achieves the first fully label-free, zero-shot, cross-graph generalist graph anomaly detector.

## Background & Motivation
**Background**: The current mainstream of Graph Anomaly Detection (GAD) follows the "one model per graph" paradigm: either using scarce labels for supervised learning (DOMINANT / BWGNN / GHRN) or employing self-supervised proxy tasks (reconstruction, contrastive learning) to model normal structures. A few recent works (ARC, UNPrompt, AnomalyGFM) have explored generalist GAD, aiming to train once and apply across different graphs.

**Limitations of Prior Work**: Existing generalist methods still rely heavily on anomaly labels from training graphs or require a few support samples from the target domain during inference. Annotation costs are high, and open-world anomalies are constantly evolving. Furthermore, cross-graph migration inherently faces heterogeneity: financial graphs have numerical node features and hub-centric topologies, while social graphs feature text and community structures. Instance-level alignment easily overfits to the fine-grained patterns of specific training graphs.

**Key Challenge**: To be "generalist," a method must move beyond labels and fine-grained alignment. However, learning normality from scratch in a purely unsupervised manner often leads to a "narrow, non-representative normal manifold." An ill-designed unsupervised objective causes deviations in defining normality, inevitably degrading downstream anomaly scoring.

**Goal**: To construct the first generalist GAD framework that satisfies three criteria: "unsupervised + zero-shot + cross-graph." This involve two sub-problems: (i) how to learn comprehensive and representative normal patterns without labels; (ii) how to overcome the heterogeneity gap in node semantics and topology across graphs.

**Key Insight**: The authors observe that when self-supervised GNNs (such as GraphMAE) are pre-trained on large graphs, their representations already capture normality rules like "neighbor matching," as normal nodes constitute the vast majority. This provides a "normality prior teacher" for free, eliminating the need to relearn normality from scratch.

**Core Idea**: Knowledge distillation is used to inject the normality prior from a frozen self-supervised GNN teacher into lightweight student modules. A set of learnable semantic prototypes aligns teacher and student outputs at the "prototype level" rather than the "instance level," allowing normal patterns to exist as transferable abstract concepts. During inference, the deviation between teacher and student in the prototype space, combined with geometric deviation, serves as the anomaly score.

## Method

### Overall Architecture
The training target of ProMoS consists of a frozen self-supervised GNN teacher $f_T$ with a lightweight adapter $g_\phi$, and student branches $S^g, S^\ell$. Given an attributed graph $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$, neighborhood residual enhancement is calculated as $\tilde{\mathbf{x}}_i = \mathbf{x}_i + (\mathbf{x}_i - \frac{1}{|\mathcal{N}(v_i)|}\sum_{v_j\in\mathcal{N}(v_i)}\mathbf{x}_j)$. The workflow splits into two paths: (1) Teacher path: $\mathbf{Z}_T = g_\phi(f_T(\mathbf{X},\mathbf{A}))$, where only $g_\phi$ is trainable while teacher weights remain frozen; (2) Student path: $\tilde{\mathbf{x}}_i$ is fed into a "shared student $S^g$" and a "personalized student pool $S^\ell$" containing $N$ lightweight students, the latter activated sparsely via a top-$K$ router. Both teacher and student predictions are projected onto learnable prototypes to obtain soft distributions, which are aligned using KL divergence. Simultaneously, commitment and refinement objectives stabilize the teacher's prototype space and evolve the prototypes. During inference, the KL distillation bias and geometric quantization deviation are used as node anomaly scores without further training.

### Key Designs

1.  **Mixture-of-Students (MoS) Module**:
    - **Function**: Expresses both "graph-wide universal normality" and "local diverse normality" without significantly increasing parameters.
    - **Mechanism**: The shared branch $\mathbf{h}_i^g = f_g(\tilde{\mathbf{x}}_i \odot \mathbf{m}_i) + \tilde{\mathbf{x}}_i$ is always active, learning global patterns confirmed by the teacher. The personalized branch is a pool of $N$ lightweight MLPs $\{f_p\}$. A router outputs $\mathbf{r}_i = \mathrm{softmax}(\mathbf{W}_r \tilde{\mathbf{x}}_i)$, assigning non-zero weights $g_{i,p}$ to the top-$K$ students to obtain $\mathbf{h}_i^\ell = \sum_p g_{i,p} f_p(\tilde{\mathbf{x}}_i \odot \mathbf{m}_i) + \tilde{\mathbf{x}}_i$. Random masking $\mathbf{m}_i$ and identity skips provide robustness and encourage students to learn only the "incremental information aggregated by the teacher."
    - **Design Motivation**: A single student is either insufficiently expressive or inefficiently large. MoS decouples "global commonality" and "local diversity" into different branches. The authors prove in Appendix A that the expected prediction error of MoS is no worse than any single student.

2.  **Prototype-driven Soft-label Distillation (PSD)**:
    - **Function**: Avoids aligning teacher and student in instance or feature dimensions (which overfits training details) and instead aligns them in a transferable semantic prototype space.
    - **Mechanism**: For each branch $b\in\{g,\ell\}$, a prototype codebook $\mathbf{P}^b\in\mathbb{R}^{M_b\times d}$ is maintained, initialized by k-means on training features. Teacher and student calculate soft assignments relative to prototypes: $\mathbf{q}_i^b[m] = \frac{\exp(\mathrm{sim}(\mathbf{z}_i^t,\mathbf{p}_m^b)/\tau)}{\sum_{m'}\exp(\mathrm{sim}(\mathbf{z}_i^t,\mathbf{p}_{m'}^b)/\tau)}$ for the teacher and $\mathbf{s}_i^b$ for the student. Similarity is the negative squared Euclidean distance. Distillation loss is the cross-branch KL divergence: $\mathcal{L}_{\text{PSD}} = \frac{1}{|\mathcal{V}|}\sum_i\sum_b \mathrm{KL}(\mathbf{q}_i^b \| \mathbf{s}_i^b)$.
    - **Design Motivation**: Prototypes are "high-level abstract concepts" that transfer across graphs more easily than instance-level features. While instance alignment fails when node semantics differ between training and target graphs, prototype alignment preserves the "semantic roles" of nodes.

3.  **Difference-aware Commitment + Refinement (DCR)**:
    - **Function**: Prevents instability in the frozen teacher's feature space caused by cross-graph heterogeneity and prevents anomaly nodes from polluting prototypes with misleading gradients.
    - **Mechanism**: Each teacher representation $\mathbf{z}_i^t$ is quantized to the nearest prototype $\mathcal{Z}_b(\mathbf{z}_i^t) = \arg\min_m \|\mathbf{z}_i^t - \mathbf{p}_m^b\|^2$. The corresponding row of the intra-prototype relationship matrix $\mathbf{Q}^b = \mathrm{softmax}(\mathrm{sim}(\mathbf{P}^b,\mathbf{P}^b)/\tau)$ is used as a "canonical reference" to check if the node's student-teacher distribution matches the global structure, yielding reliability $w_i^b = \sigma(-\beta(\mathrm{KL}(\mathbf{q}_i^b\|\mathbf{Q}_{m_i^\star}^b) - \mu))$. The total DCR loss is $\mathcal{L}_\mathrm{DCR} = \sum_i\sum_b w_i^b(\|\mathbf{z}_i^t - \mathrm{sg}[\mathcal{Z}_b]\|^2 + \|\mathrm{sg}[\mathbf{z}_i^t] - \mathcal{Z}_b\|^2)$. The first term (commitment) pulls teacher features toward prototypes; the second (refinement via stop-gradient) lets prototypes fit the semantics of reliable nodes.
    - **Design Motivation**: Standard VQ-VAE commitment is susceptible to anomalies. Reliability weighting allows ProMoS to identify "non-normal" nodes and downweight them, performing soft anomaly exclusion unsupervised to maintain prototype purity.

### Loss & Training
The total objective is $\mathcal{L} = \mathcal{L}_\text{PSD} + \lambda \mathcal{L}_\text{DCR}$, where $\lambda$ balances distillation and prototype alignment. $\tau$ is the temperature, while $\beta$ and $\mu$ control reliability sensitivity. During inference, node anomaly scores are calculated as $s_i = \sum_b[\mathrm{KL}(\mathbf{q}_i^b\|\mathbf{s}_i^b) + \lambda(\|\Delta_h\|^2 + \|\Delta_z\|^2)]$, where $\Delta_h = \mathbf{h}_i^b - \mathcal{Z}_b(\mathbf{h}_i^b)$ and $\Delta_z = \mathbf{z}_i^t - \mathcal{Z}_b(\mathbf{z}_i^t)$. The intuition is that anomalies will either show large prototype assignment differences (distillation bias) or be far from any prototype (geometric deviation).

## Key Experimental Results

### Main Results
AUROC (selected) in zero-shot settings across 11 real-world graph datasets (Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, CS, Photo, Tolokers, T-Finance):

| Dataset | ProMoS | DOMINANT (Unsupervised) | TAM | UNPrompt (Supervised generalist) | AnomalyGFM (Supervised generalist) |
|--------|--------|-------------------|-----|------------------------------|-------------------------------|
| Cora | **84.56** | 66.53 | 62.02 | 53.19 | 47.83 |
| CiteSeer | **90.77** | 69.47 | 72.27 | 53.70 | 49.10 |
| ACM | **89.47** | 70.08 | 74.43 | 68.74 | 53.40 |
| BlogCatalog | **76.17** | 74.25 | 49.86 | 68.87 | 49.31 |
| Reddit | **60.83** | 50.05 | 55.43 | 57.10 | 52.78 |
| Photo | **72.67** | — | 58.35 | 38.60 | 49.65 |
| T-Finance | **71.62** | OOM | 56.16 | 22.14 | 64.44 |

ProMoS ranks first in AUROC on 9/11 datasets. On strongly homophilic graphs like Weibo, it is slightly behind the reconstruction-based DOMINANT but remains a solid second (91.74 vs 92.88), being the only method to simultaneously satisfy "unsupervised + zero-shot + generalist." In terms of AUPRC, ProMoS ranks first on 7/11 datasets, with significant improvements (over double) in T-Finance, CS, and Cora.

### Ablation Study

| Configuration | Approximate Performance | Description |
|------|---------|------|
| Full ProMoS | Best average across 11 graphs | Complete three components |
| w/o MoS (shared only or personalized only) | Significant drop | Loss of global/local complementarity |
| w/o Prototype Distillation (using instance KL) | Significant cross-graph drop | Reverts to instance-level alignment, fails cross-graph transfer |
| w/o DCR | Prototype collapse + anomaly drag | Unstable teacher space and polluted prototypes |
| w/o Reliability Weighting | AUROC drops 3-5 pt on some graphs | Loss of filtering for anomaly gradients |

### Key Findings
- MoS theoretical guarantees (expected error $\le$ any single student) hold in practice: single-branch performance is consistently weaker than the full MoS.
- Prototype distillation is key to cross-graph generalization: replacing it with instance KL causes cross-graph AUROC to collapse to baseline levels.
- Reliability weighting provides the largest gain for graphs with high anomaly rates (e.g., T-Finance), indicating that "adaptive exclusion of suspicious nodes" protects prototype purity.
- The self-supervised teacher can be replaced with different GNNs (e.g., GraphMAE), and ProMoS performance remains stable, showing that the framework, rather than a specific teacher, drives results.

## Highlights & Insights
- **"Do not learn normality from scratch; distill a free normality prior"**: This is the most elegant point. Numerous pre-trained self-supervised GNNs exist, but none were previously used as GAD teachers. Ours successfully brings the LLM-era paradigm of "general pre-training $\to$ downstream task" to graph anomaly detection.
- **Prototypes as both alignment anchors and inference signals**: During training, they mediate teacher-student alignment; during inference, they act as a codebook to provide quantization deviation. This dual-use of one parameter set is efficient and elegant, similar to the codebook in VQ-VAE.
- **Reliability weighting as unsupervised robust learning**: Trust is judged by whether a node's prototype distribution matches the global prototype relationship matrix. This effectively performs a soft anomaly check on each node to adjust learning weights, forming a self-consistent loop applicable to any noisy-label or self-training scenario.
- **Extensible MoS logic**: Applying the "shared global + sparse personalized" MoE concept to self-supervised distillation is a valuable takeaway for other transfer tasks like cross-domain segmentation or cross-lingual NLP.

## Limitations & Future Work
- Dependency on high-quality self-supervised teachers: If the pre-trained teacher's normality prior is biased (e.g., due to small-scale or noisy data), the performance ceiling of ProMoS will be lowered.
- Hyperparameter sensitivity: Whether parameters like $M_b$, $N$, top-$K$, $\tau$, $\beta$, and $\mu$ require per-target tuning is not entirely clear; Ours used a fixed configuration, though some graphs (e.g., Tolokers) were weaker than specialist baselines.
- Currently limited to node-level GAD: Whether this framework can be extended to graph-level GAD remains an area for exploration.
- Anomaly types are still limited to structural/attribute anomalies; extending to "rare behavioral patterns" (temporal/cross-modal) would require different teachers.

## Related Work & Insights
- **vs DOMINANT/CoLA/TAM**: Traditional unsupervised GAD needs retraining per graph and lacks cross-graph capability; ProMoS implements the "train once $\to$ universal" paradigm.
- **vs ARC / UNPrompt / AnomalyGFM**: Similar generalist GAD goals, but these require labels or few-shot samples; ProMoS is fully unsupervised and zero-shot, yet often achieves better performance.
- **vs Self-supervised GNNs (GraphMAE, etc.)**: Previously used for classification/clustering, ProMoS "inverts" them as GAD teachers, providing a new application scenario.
- **vs KD in Image Anomaly Detection (STFPM, MKD)**: While KD-AD is mature in CV, ProMoS ports the core idea (teacher–student distillation bias as anomaly score) to graphs, modifying it with prototypes to allow cross-graph functionality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)
- [\[ICML 2026\] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)

</div>

<!-- RELATED:END -->
