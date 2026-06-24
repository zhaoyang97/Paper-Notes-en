---
title: >-
  [Paper Note] EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks
description: >-
  [NeurIPS2025 Spotlight][Medical Imaging][EEG] This work proposes EvoBrain—proving theoretically for the first time that **explicit dynamic graph modeling** outperforms implicit static graphs, and that the **time-then-graph** architecture is strictly more expressive than the other two dynamic GNN paradigms (graph-then-time / time-and-graph). Guided by these findings, a dual-stream Mamba + Laplacian PE enhanced GCN model is designed, achieving significant performance improvemen…
tags:
  - "NeurIPS2025 Spotlight"
  - "Medical Imaging"
  - "EEG"
  - "Dynamic Graph"
  - "Seizure Detection"
  - "Mamba"
  - "Spatiotemporal Modeling"
  - "GCN"
  - "Expressive Power Analysis"
date: 2026-05-08
content_hash: ee311e432e27fea5
---

# EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks

**Conference**: NeurIPS2025 Spotlight  
**arXiv**: [2509.15857](https://arxiv.org/abs/2509.15857)  
**Code**: [GitHub](https://github.com/Kotoge/EvoBrain)  
**Area**: EEG Signal Analysis / Dynamic Graph Neural Networks  
**Keywords**: EEG, Dynamic Graph, Seizure Detection, Mamba, Spatiotemporal Modeling, GCN, Expressive Power Analysis

## TL;DR

This work proposes EvoBrain—proving theoretically for the first time that **explicit dynamic graph modeling** outperforms implicit static graphs, and that the **time-then-graph** architecture is strictly more expressive than the other two dynamic GNN paradigms (graph-then-time / time-and-graph). Guided by these findings, a dual-stream Mamba + Laplacian PE enhanced GCN model is designed, achieving significant performance improvements of +23% AUROC and +30% F1 in seizure detection and early prediction tasks on the TUSZ and CHB-MIT datasets, while being 17x faster in training than the state-of-the-art (SOTA).

## Background & Motivation

- **Clinical Background**: Epilepsy is a brain network disease where abnormal connections between different brain regions are important signatures of seizure onset. Electroencephalogram (EEG) is a core tool for clinical monitoring of epilepsy, making automated detection and prediction highly clinically significant.
- **Limitations of Prior Work**: In recent years, dynamic GNNs have been used to model the spatiotemporal features of multi-channel EEG, but two fundamental issues remain:
  1. **Static Graph Structure**: Although most methods claim to be "dynamic," they actually construct a fixed adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ shared across all time steps, failing to capture the evolution of brain connectivity during seizure activity.
  2. **Inadequate Spatiotemporal Modeling**: Existing dynamic GNNs can be categorized into three paradigms: graph-then-time, time-and-graph, and time-then-graph. However, there is a lack of theoretical basis regarding which paradigm is superior, leading to inconsistent performance in practical applications.
- **Motivation**: To resolve the aforementioned issues from both theoretical and experimental perspectives, designing a seizure detection model that is both theoretically sound and highly efficient.

## Core Problem

The paper formally proposes two key problems:

1. **Problem 1: Implicit vs. Explicit Dynamic Graph Modeling**
    - Implicit: Fixed adjacency matrix $\mathbf{A}_{:,:,t} = \hat{\mathbf{A}}, \forall t$, where only node features change over time.
    - Explicit: Adjacency matrix evolves over time $\mathbf{A}_{:,:,t} = f(\mathbf{x}_{:,t})$ where both nodes and edges can evolve.
    - Problem: What is the relationship between their expressive powers?

2. **Problem 2: Expressiveness Ranking of Three Dynamic GNN Architectures**
    - graph-then-time: First applies GNN independently at each time step, then processes the time series using RNN.
    - time-and-graph: Alternates GNN and RNN processing, simultaneously performing graph learning and temporal modeling at each step.
    - time-then-graph: First uses RNN to model the temporal evolution of nodes and edges, respectively, then performs GCN on the final aggregated graph.
    - Problem: Which architecture has the strongest expressive power?

## Theoretical Analysis

Based on the 1-WL (Weisfeiler-Lehman) graph isomorphism test framework, the paper provides rigorous theoretical proofs:

### Theorem 1: Explicit > Implicit

The function class of explicit dynamic graph modeling $\mathcal{F}_{\text{explicit}}$ strictly contains that of implicit dynamic graph modeling $\mathcal{F}_{\text{implicit}}$:

$$\mathcal{F}_{\text{implicit}} \subset \mathcal{F}_{\text{explicit}}$$

Proof Idea: Construct two temporal EEG graphs that have identical node features but differ in their adjacency matrix at only a single time step. Implicit models compress the adjacency into a static representation and fail to distinguish them, whereas explicit models can.

### Theorem 3: Expressiveness Ranking of Architectures

$$\text{graph-then-time} \precneqq \text{time-and-graph} \precneqq \text{time-then-graph}$$

- **Lemma 1**: In graph-then-time, the historical hidden state $\mathbf{h}_{i,t-1}$ is directly passed to the RNN cell without additional GNN encoding for cross-time step interactions, which makes its expressive power weaker than time-and-graph.
- **Lemma 2**: time-and-graph processes graph information independently at each time step, which may lead to identical representations across different time steps; on the other hand, time-then-graph first learns the complete temporal evolution of nodes and edges before performing graph reasoning, thereby distinguishing graphs with different temporal structures.

### Synthetic EEG Task Verification (Proof of Lemma 2)

The paper constructs an ingenious synthetic task to demonstrate a pair of graphs that time-and-graph cannot distinguish but time-then-graph can:

- **Setup**: Two 2-step temporal graphs, each with 8 nodes ($\mathcal{C}_{8,1}$ and $\mathcal{C}_{8,2}$). They have identical structures at the first step but different topologies at the second step.
- **time-and-graph Fails**: Since 1-WL GNN produces identical representations for $\mathcal{C}_{8,1}$ and $\mathcal{C}_{8,2}$ (due to identical node features), they cannot be distinguished despite having different structures $\rightarrow$ eventually yielding $\mathbf{Z}^{(\text{top})} = \mathbf{Z}^{(\text{btm})}$.
- **time-then-graph Succeeds**: RNN$^{\text{edge}}$ processes the edge time series of the two graphs separately. Since the edges differ at $t_2$, $\mathbf{h}^{\text{edge(top)}} \neq \mathbf{h}^{\text{edge(btm)}} \rightarrow$ after GNN, $\mathbf{Z}^{(\text{top})} \neq \mathbf{Z}^{(\text{btm})}$.

**Key Insight**: time-then-graph first uses independent RNNs to encode the complete temporal evolution of edges, preserving structural discrepancy information; whereas time-and-graph performs GNN at each step independently, losing cross-time step structural differences.

### Significance

This is the **first node-level expressive power theoretical analysis** for dynamic GNNs in the EEG analysis context. Compared to the edge/structural-level analysis of Gao & Ribeiro (2022), it is much closer to practical EEG graph construction which relies on node similarities. Furthermore, Lemma C.1 proves that using edge features alone is insufficient to distinguish certain temporal EEG graphs (pairs exist with identical edge features but different node features). Thus, **both node and edge representations must be modeled simultaneously**—which is the theoretical foundation for EvoBrain's dual-stream Mamba design.

## Method

### 1. Explicit Dynamic Brain Graph Construction

- Slice the EEG signal into $T$ snapshots.
- For each snapshot, compute the normalized cross-correlation between channels $v_i$ and $v_j$ as the edge weight:

$$a_{i,j,t} = |x_{i,t} * x_{j,t}|, \quad \text{if } v_j \in \mathcal{N}(v_i), \text{ else } 0$$

- Retain the edges with the highest correlations via a top-$\tau$ strategy to generate sparse directed weighted graphs.
- Obtain the sequence of $T$ dynamic graphs $\mathbf{G}$, where both nodes and edges evolve over time.

### 2. Dual-Stream Mamba Temporal Modeling

Based on the time-then-graph architecture, two independent Mamba streams are utilized to process node and edge time series separately:

**Input Preprocessing**: Perform STFT (Short-Time Fourier Transform) on the EEG signals, take the log magnitude of non-negative frequency components, and apply z-normalization to obtain $\mathcal{X} \in \mathbb{R}^{N \times T \times d}$.

**Mamba State Space Model** (Linear RNN + Selective State Update):

$$\mathbf{h}_t^e = \underbrace{(1 - \Delta_t^e \cdot \mathbf{D})}_{\text{Selective Forgetting}} \mathbf{h}_{t-1} + \underbrace{\Delta_t^e \cdot \mathbf{B}_t^e}_{\text{Selective Update}} \mathbf{x}_t^e$$

- The forgetting term is analogous to synaptic decay/inhibition, fading outdated information.
- The update term is analogous to neuromodulatory gating (e.g., dopamine signals), selectively enhancing important new inputs.
- $\Delta_t^e$ is constrained to be positive via softplus, regulating the trade-off of short-term/long-term memory.

The two streams output node temporal representations $\mathbf{h}_i^{\text{node}} = \mathbf{y}_{i,T}^{\text{node}}$ and edge temporal representations $\mathbf{h}_{ij}^{\text{edge}} = \mathbf{y}_{ij,T}^{\text{edge}}$, respectively.

### 3. Laplacian Position Encoding (LapPE) + GCN Spatial Modeling

- **Neuroscientific Motivation for LapPE**: Brain functions are highly localized to specific regions (e.g., neocortex, Broca's area), but standard GNNs produce identical representations for structurally equivalent nodes, losing spatial location information.
- Compute the weighted adjacency matrix $\mathbf{A}' = \tau_{\text{edge}}(f_{\text{edge}}(\mathbf{H}^{\text{edge}}))$ from edge features $\mathbf{H}^{\text{edge}}$.
- Perform eigendecomposition on the normalized Laplacian matrix $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}'\mathbf{D}^{-1/2}$.
- Concatenate the eigenvectors corresponding to the $K$ smallest eigenvalues as positional encodings $\mathbf{p}_i$ with node features:

$$\mathbf{x}_i^{\text{node}} = [\mathbf{h}_i^{\text{node}}; \mathbf{p}_i]$$

- **GCN Spatial Learning**: Multi-layer GCN aggregates neighborhood information $\rightarrow$ max pooling $\rightarrow$ fully connected layer + softmax output for classification.

## Key Experimental Results

### Datasets
- **TUSZ v1.5.2**: The largest publicly available EEG seizure database, containing 5612 records with 3050 annotated seizures across 19 channels. The train/val/test splits are partitioned by patients (530/61/45 patients). Under the 12s window setting, the training set contains ~197k samples (6.9% positive), and ~39k samples (9.3% positive) under the 60s window, exhibiting extreme class imbalance.
- **CHB-MIT**: 844 hours of 22-channel scalp EEG from 22 patients with 163 seizures. A random split of 15% patient data is used for testing.

### Tasks
- **Seizure Detection**: Distinguishing between ictal and non-ictal segments (binary classification).
- **Early Seizure Prediction**: Distinguishing between pre-ictal (1 minute prior to seizure) and normal states, which holds superior clinical value.

### Main Results (TUSZ Dataset)

| Model | Detection AUROC (60s) | Detection F1 (60s) | Prediction AUROC (12s) | Prediction F1 (12s) |
|------|----------------|-------------|----------------|-------------|
| EvolveGCN (graph-then-time) | 0.670 | 0.340 | 0.622 | 0.437 |
| DCRNN (time-and-graph) | 0.808 | 0.435 | 0.634 | 0.401 |
| GRAPHS4MER (time-then-graph) | 0.778 | 0.439 | 0.632 | 0.438 |
| GRU-GCN (time-then-graph) | 0.822 | 0.438 | 0.659 | 0.453 |
| LaBraM (Foundation) | 0.793 | 0.469 | 0.661 | 0.482 |
| **EvoBrain** (Ours) | **0.865** | **0.483** | **0.675** | **0.470** |

- Compared to EvolveGCN: AUROC +23%, F1 +30%
- Compared to LaBraM: Detection AUROC +9%, while the parameter count is only 1/30th of its size.
- EvoBrain achieves an AUC of 0.94 on the CHB-MIT dataset.

### Efficiency
- Training speed is **17×** faster than DCRNN (time-and-graph).
- Inference speed is **14×** faster (as GNN processes only once at the final step, rather than at every snapshot).

### Theoretical Analysis of Computational Complexity

The paper compares the complexity of the three architectures ($V$=number of nodes, $T$=number of time steps, $E_t$=number of edges at step $t$, $E_{\text{agg}}$=number of edges in the aggregated graph, $d$=feature dimension):

| Architecture | Complexity |
|------|--------|
| graph-then-time | $\mathcal{O}(VTd^2 + \sum_{t} E_t d)$ |
| time-and-graph | $\mathcal{O}(VTd^2 + \sum_{t} E_t d^2)$ |
| time-then-graph | $\mathcal{O}((V + E_{\text{agg}})Td^2)$ |

- Key advantage of time-then-graph: When the aggregated graph edge count $E_{\text{agg}} \ll \sum_t E_t$ (which typically holds in sparse graph scenarios), its complexity is significantly lower than time-and-graph.
- Reason: time-and-graph performs GNN at every time step (including two GNNs: $\text{GNN}_{\text{in}}^L$ and $\text{GNN}_{\text{rc}}^L$), whereas time-then-graph only performs GNN once on the final aggregated graph.

### GPU Memory Consumption

| Model | Training (MB) | Inference (MB) |
|------|-----------|----------|
| EvoBrain | 51.35 | 46.64 |
| GRU-GCN | 54.61 | 52.09 |
| GRAPHS4MER | 369.46 | 93.02 |
| DCRNN | 21.10 | 20.54 |
| EvolveGCN | 22.06 | 20.07 |

- DCRNN/EvolveGCN have the lowest memory consumption but are over 10× slower in computation.
- EvoBrain achieves the best memory efficiency among time-then-graph models (only 1/7 of GRAPHS4MER), balancing both speed and memory footprint.

### Ablation Study
- Under the same GRU + GCN setup, the time-then-graph architecture achieves the best performance, validating the theoretical analysis.
- Removing FFT preprocessing or LapPE results in decreased performance.
- Mamba shows more pronounced advantages over GRU on long sequences (60s).
- Replacing GCN with GIN yields comparable performance, while removing GNN entirely to use only RNN dramatically degrades performance.

### Implementation Details

- **Optimizer**: Adam, learning rate 1e-4, training for 100 epochs.
- **Loss Function**: Binary cross-entropy.
- **Sparsity**: top-$\tau$=3 (each node retains only the 3 edges with the highest correlation).
- **Model Scale**: Both Mamba streams have 2 layers + 2-layer GCN (64 hidden units), with a total of **114,794** parameters (~115k parameters).
- **Data Augmentation**: Random scaling of EEG amplitudes during training ($\times 0.8 \sim 1.2$).
- **Dropout**: 0 (not used).
- **Hardware**: NVIDIA A6000 GPU + Xeon Gold 6258R CPU.

### Parameter Count Comparison

| Model | Parameter Count |
|------|--------|
| EvoBrain | 114,794 |
| GRU-GCN | 183,834 |
| EvolveGCN | 200,301 |
| DCRNN | 280,769 |
| LSTM | 536,641 |
| BIOT | 3,187,201 |
| LaBraM | 5,803,137 |
| EEGPT | 51,221,121 |

- EvoBrain has the smallest parameter count, which is only ~1/50th of LaBraM and ~1/450th of EEGPT.
- Surpassing the performance of Foundation Models with orders of magnitude fewer parameters highlights the importance of incorporating appropriate inductive biases (dynamic graphs + time-then-graph).

## Clinical Analysis Highlights

The paper demonstrates the visualization and analysis of the dynamic graph structure, which holds substantial clinical value:

- **Normal State**: Weak, sparse, and widely distributed edge connections.
- **Pre-ictal State**: Connection strengths in specific regions gradually increase (enabling early warning).
- **Focal Seizures**: Sustained strong connections localized within specific brain areas (assisting in locating the seizure onset zone - SOZ).
- **Generalized Seizures**: Brain-wide strong connections.

These dynamic graph patterns align with neuroscientific observations, offering potential support for surgical planning and treatment strategies.

## Highlights & Insights

1. **Theory-Driven Design**: Rather than choosing architectures based on intuition, the authors formulate rigorous mathematical proofs before designing the model. This methodology is highly exemplary.
2. **Dual-Stream Mamba**: Modeling the temporal evolution of nodes and edges separately satisfies theoretical prerequisites (capturing distinct node and edge dynamics) and exploits the linear complexity advantages of Mamba.
3. **Neuroscience Alignment**: Using LapPE to differentiate brain region locations, leveraging FFT features to mirror clinical frequency analysis, and ensuring that dynamic graph visualizations correspond directly to clinical observations.
4. **Balanced Efficiency and Performance**: Achieving 17x acceleration alongside substantial performance leadership, thanks to the single GNN computation required in the time-then-graph architecture.
5. **Clinical Interpretability**: Dynamic graph visualizations directly map onto brain region connection changes across different stages of epilepsy.

## Limitations & Future Work

1. **Remaining Gap in Early Prediction**: Although EvoBrain outperforms existing GNN models in prediction tasks, large-scale pre-trained models like LaBraM still hold advantages in certain metrics (benefiting from large-scale pre-training and massive parameters).
2. **Task Generalization**: The framework has only been validated on seizure detection/prediction tasks and has yet to be extended to other EEG tasks (e.g., emotion recognition, motor imagery, sleep staging).
3. **Fixed Pre-ictal Definition**: The pre-ictal window is set to a fixed 1-minute period prior to seizure onset. However, in clinical practice, the duration of the pre-ictal phase varies greatly among individuals.
- **Demographic Bias in Datasets**: Training data is sourced from specific cohorts. Generalizability to other demographics remains unvalidated, potentially leading to diagnostic disparity across patient populations.
- **Sub-optimal GPU Memory Footprint**: Although the inference memory (46.64 MB) is far lower than GRAPHS4MER (93.02 MB), it is still higher than DCRNN (20.54 MB) and EvolveGCN (20.07 MB), which might restrict deployments in resource-constrained edge scenarios.
- **Possible Directions**: Combining EvoBrain with large-scale EEG pre-training; introducing adaptive pre-ictal window lengths; and extending to other neural signals (such as iEEG, MEG).

## Related Work & Insights

| Dimension | Tang et al. 2022 (DCRNN) | Tang et al. 2023 (GRAPHS4MER) | Gao & Ribeiro 2022 (GRU-GCN) | EvoBrain (Ours) |
|------|------------------------|------------------------------|------------------------------|----------|
| Architectural Paradigm | time-and-graph | time-then-graph | time-then-graph | time-then-graph |
| Graph Structure | Static (Fixed Adjacency) | Internal learning but based on the entire sequence | Static | **Explicit Dynamic** (Built per-snapshot) |
| Temporal Model | GRU | S4 (Structured State Space) | GRU | **Mamba** (Selective State Space) |
| Graph Model | GCN | GCN | GCN | **GCN + LapPE** |
| Theoretical Analysis | None | None | Yes (Edge/Structural-level) | **Yes (Node-level, closer to EEG context)** |
| Edge Feature Modeling | None | None | Yes | **Yes (Independent Mamba Stream)** |
| Efficiency | Slow (GNN at every step) | Medium | Fast | **Fastest (17x acceleration)** |

Comparison with EEG Foundation Models:
- **LaBraM** (5.8M parameters): Benefits from large-scale multi-dataset pre-training to acquire strong generalization capability. It outperforms EvoBrain on some metrics in the prediction task, but its parameter size is 30x that of EvoBrain.
- **BIOT / EEGPT**: Despite being large-scale pre-trained models, they underperform EvoBrain on the seizure detection task, demonstrating that task-specific inductive biases (dynamic graphs + separated spatiotemporal modeling) remain crucial.
- EvoBrain's advantages lie in being **lightweight, efficient**, and **highly interpretable**, making it highly suitable for clinical deployment.

Differences with General Dynamic Graph Learning Methods:
- The theoretical analysis of Gao & Ribeiro (2022) mainly focuses on edge/structural-level representations and does not explicitly consider node features. However, EEG graph construction inherently relies on similarities between channels (nodes). Thus, EvoBrain's node-level expressive power analysis is much more custom-tailored.
- General dynamic graph methods (such as FreeDyG, DeepTGC) focus on link prediction or node classification, whereas EvoBrain targets graph-level classification (seizure detection), requiring whole-graph representation capability.

## Insights & Connections

1. **Theory-First Model Design Paradigm**: The paper first proves the strict partial ordering of expressive power among architectures, and then guides the design toward the time-then-graph + explicit dynamic graph pathway. This "theory-driven design" methodology is highly generalizable to other fields—such as video understanding, where one could analyze the expressive power of different spatiotemporal fusion strategies before proceeding to model design.

2. **Dual-Stream Independent Modeling of Node/Edge Evolution**: Processing node and edge time series separately with two Mamba streams satisfies theoretical requirements while reducing computational overhead. This design can be extended to other dynamic graph scenarios, such as transportation networks (node = intersection flow, edge = road segment congestion) or social networks (node = user status, edge = interaction frequency).

3. **Application of Mamba in Graph Learning**: This work represents a successful integration of Mamba and GNNs. The analogy between Mamba's selective state update mechanism and neuromodulatory processes is highly inspiring, providing a plausible explanatory framework for its application to biological signals.

4. **Neuroscientific Motivation of LapPE**: Beyond treating LapPE merely as an engineering technique in graph representation learning, the paper endows it with neuroscientific meaning (distinguishing spatial brain regions), coupling method design deeply with domain-specific knowledge.

5. **No Trade-off Between Interpretability and Performance**: The dynamic graph visualizations in EvoBrain not only assist clinical comprehension but are also core to the model design (rather than post-hoc explanations), embodying "intrinsic interpretability" rather than "post-hoc interpretability."

6. **Potential Extensions**:
    - Applying EvoBrain's dynamic graph framework to motor imagery decoding in Brain-Computer Interfaces (BCI).
    - Integrating with large-scale EEG pre-training (e.g., LaBraM's pre-training strategy) to further boost generalization.
    - Adaptive pre-ictal window sizing: replacing the fixed 1-minute window with change-point detection.
    - Multimodal fusion: combining the high spatial resolution of fMRI with the high temporal resolution of EEG.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of theoretical analysis, explicit dynamic graphs, and dual-stream Mamba is highly novel, though the time-then-graph architecture itself has precedents.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, both detection and prediction tasks, 12 baseline models, solid ablation studies, clinical visualization analysis, and efficiency comparisons.)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical derivation, clear method description, highly informative figures and tables; though mathematically dense, imposing a slight barrier to first-time readers.)
- Value: ⭐⭐⭐⭐ (Bridges theoretical contributions and practical clinical value, serving as an important reference for dynamic GNN applications in the EEG domain.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ODEBRAIN: Continuous-Time EEG Graph for Modeling Dynamic Brain Networks](../../ICLR2026/medical_imaging/odebrain_continuous-time_eeg_graph_for_modeling_dynamic_brain_networks.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)

</div>

<!-- RELATED:END -->
