---
title: >-
  [Paper Note] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning
description: >-
  [CVPR2026][Optimization][Federated Learning] SCOPE employs a training-free vision-language geometric scorer to compress each sample into three scalars—representativeness, diversity…
tags:
  - "CVPR2026"
  - "Optimization"
  - "Federated Learning"
  - "coreset selection"
  - "long-tail distribution"
  - "Vision-Language Model"
  - "data pruning"
date: 2026-05-08
content_hash: 4bd5eb1d70d6a0a0
---

# SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning

**Conference**: CVPR2026
**arXiv**: [2603.12976](https://arxiv.org/abs/2603.12976)
**Code**: N/A
**Area**: Optimization
**Keywords**: Federated Learning, coreset selection, long-tail distribution, Vision-Language Model, data pruning

## TL;DR
SCOPE employs a training-free vision-language geometric scorer to compress each sample into three scalars—representativeness, diversity, and negative-class boundary proximity—and has the server aggregate only these lightweight statistics to form a global consensus. This consensus guides each client to first remove semantically anomalous samples and then eliminate majority-class redundancies, thereby achieving a favorable balance among accuracy, robustness, and minimal communication overhead under strongly non-IID and long-tail federated scenarios.

## Background & Motivation

Data pruning in federated learning is not a new concept, but applying it to real-world distributions driven by scientific instruments immediately makes the problem substantially more challenging. The paper addresses a representative practical scenario: each edge node holds locally collected data that is massive in volume, privacy-sensitive, severely class-imbalanced, and heterogeneous not only in label distribution but also in data quality across nodes.

The first limitation of existing approaches is their **local-only perspective**. Many coreset methods assess which samples to retain based on local density, local gradients, or local loss; however, in federated settings, a sample that appears redundant to a single client is not necessarily redundant to the entire federated system. A sample that looks "ordinary" within one node may belong to a rare tail class in the global distribution.

The second limitation is **prohibitive training cost**. Many data selection methods require a local warmup phase on the full local dataset, using signals such as loss, gradient norm, or forgetting events to identify important samples. This is economically undesirable for high-throughput scientific data, where the very motivation for pruning is to save computation—yet the dominant computational cost is incurred before pruning even begins.

The third limitation is the **fallacy that high loss implies high value**. On natural image benchmarks, high-loss samples are often treated as informative hard examples; in scientific imaging or sensor data, however, high loss frequently reflects noise, artifacts, sampling errors, or annotation inconsistencies. Retaining such samples tends not to enhance generalization but rather to inject noise further into the federated aggregation.

The fourth limitation is the **fundamental tension between global perspective and privacy/bandwidth constraints**. Acquiring a global view of the data distribution naturally motivates uploading embeddings, gradients, or proxy datasets—but these approaches either expose more semantic information than is acceptable or incur communication costs incompatible with real edge environments.

The paper therefore addresses three mutually coupled subproblems.

First, can sample utility be quantified in an approximately reliable manner solely from the semantic space of a pretrained vision-language model, without any local training?

Second, can this quantification be compressed into an extremely lightweight form, enabling the server to maintain a global statistical perspective without accessing high-dimensional features or raw samples?

Third, can data compression be performed without disrupting the long-tail structure—in particular, without allowing head classes to further crowd out the survival space of tail classes?

The authors' starting point is clear: rather than transmitting high-dimensional embeddings, project each sample into a small number of geometric indicators within a unified vision-language semantic space; rather than allowing clients to prune samples heuristically, have the server first aggregate global class-level statistics and then broadcast this "global consensus" back to guide local pruning.

The core idea of the paper can be summarized in one sentence: first use VLM semantic projections to compress sample value into communicable scalars, then use global class statistics to elevate federated pruning from "local heuristics" to "globally informed, two-stage data governance."

## Method

The method design of this paper is not complex, but the logic is clean and well-organized. The authors decompose "what makes a sample worth keeping" into three geometric problems, have the server aggregate only the statistical answers to these problems, and then have clients execute a two-stage selection procedure guided by this global consensus.

Intuitively, SCOPE's logic does not seek the "hardest samples"; instead, it partitions samples into three categories and handles each accordingly:

- Samples that are clearly semantically misaligned—resembling noise or mismatches—are removed first.
- Samples that clearly belong to the central region of head classes and exhibit high redundancy are removed next.
- Samples that genuinely support decision boundaries, maintain intra-class diversity, or protect tail classes are retained as much as possible.

The fundamental distinction from traditional loss-based pruning is that this approach does not rely on training dynamics but on the semantic geometric structure of a pretrained VLM.

### Overall Architecture

The overall pipeline can be understood in four steps.

**Step 1.** Each client encodes images into a shared semantic space using MobileCLIP-S2 and constructs text prototypes for each class via natural language prompts. Three scalar scores are then computed per sample: representativeness score $RS$, diversity score $DS$, and negative-class boundary proximity $S_{neg}$.

**Step 2.** Clients do not upload sample-level embeddings; instead, they upload class-level statistics, including the sample count per class and the mean and variance of the three scores within each class. The server constructs a global profile from these statistics, capturing both how rare each class is globally and the global distribution of each class across the three semantic indicators.

**Step 3.** Upon receiving the global profile, each client performs the **Consensus Filter**, which primarily removes semantically contradictory samples—those that resemble the negative class more than the true class.

**Step 4.** Each client then performs **Dynamic Balancing**, which does not uniformly remove samples across all classes but instead removes the most redundant samples only from classes that are locally abundant and globally non-rare, thereby compressing head classes while protecting tail classes.

The resulting coreset is used in subsequent federated training. The coreset construction itself is performed before training, is training-free, and requires only lightweight communication.

### Key Designs

1. **Three-Dimensional Semantic Scoring System ($RS$ / $DS$ / $S_{neg}$)**
   - **Function**: Characterizes the retention value of each sample from three orthogonal dimensions within the shared CLIP semantic space.
   - **Mechanism**: The representativeness score $RS_i = v_{img,i} \cdot t_{c_i}$ measures alignment between the sample and its class prototype; the diversity score captures intra-class variation beyond the class prototype via the orthogonal projection residual $DS_i = \|v_{img,i} - RS_i \, t_{c_i}\|_2$; the boundary proximity score $S_{neg,i} = \max_{j \neq c_i} v_{img,i} \cdot t_j$ characterizes the degree of confusion with the nearest negative class. The three scores are used jointly, enabling subsequent filtering to distinguish among noise, redundancy, and valuable boundary samples.
   - **Design Motivation**: No single indicator can simultaneously address anchor retention, intra-class diversity, and decision boundary support. $RS$ preserves fundamentals, $DS$ eliminates redundancy, and $S_{neg}$ identifies boundary samples. The three are orthogonally complementary, providing a complete semantic criterion for two-stage filtering.

2. **Privacy-Preserving Global Profile Construction**
   - **Function**: Establishes a unified federated semantic reference frame and class rarity without transmitting sample-level embeddings.
   - **Mechanism**: Clients upload only the sample count per class and the mean and variance of the three scores per class. The server applies the law of total variance (decomposing within-group and between-group variance) to compute global means and variances for subsequent Z-score normalization. Global class frequencies are also used to construct rarity weights $W_c \propto (1/(F_c + \epsilon))^{\gamma}$, which encode the global long-tail structure.
   - **Design Motivation**: In federated settings, data distribution heterogeneity across clients is substantial, and naive averaging underestimates this heterogeneity. Explicitly decomposing variance yields a reliable "global normal range," while class-level statistics ensure lightweight communication and privacy preservation.

3. **Two-Stage Structured Pruning (Consensus Filter + Dynamic Balancing)**
   - **Function**: Removes semantically anomalous samples first, then selectively compresses head-class redundancies, constructing a coreset subject to global fairness constraints.
   - **Mechanism**: Stage 1 computes an anomaly score $AS_i = \hat{Z}_{S_{neg},i} - \hat{Z}_{RS,i}$ (after Z-score normalization) and removes the top-$p_l$ fraction of high-anomaly samples—those that resemble the negative class more than the true class. Stage 2 defines a redundancy score $R_i = \hat{Z}_{RS,i} - \hat{Z}_{S_{neg},i} - \hat{Z}_{DS,i}$, identifying samples that are "typical, far from the boundary, and low in novelty," but applies removal of the top-$p_f$ fraction only to classes whose target index $T_c = f_c / W_c$ is high (i.e., locally frequent and globally non-rare).
   - **Design Motivation**: The two stages decouple two distinct objectives. Stage 1 identifies noise via semantic alignment rather than high loss, avoiding the erroneous removal of genuinely difficult samples. Stage 2 restricts reduction to head classes, protecting tail classes and boundary samples, thereby achieving non-uniform structured compression.

### Loss & Training

SCOPE is not a new training loss but a pre-training data selection framework. Its defining characteristic is that **selection is training-free**.

The downstream training phase follows standard federated training configurations:

- SGD with cosine decay learning rate is used for downstream training.
- Total communication rounds: 200.
- Reported results are averaged over the Top-1 accuracy of the last 10 global rounds.
- Baseline methods such as FedCS, FedCore, EL2N, Forgetting, and GradND all require local model training signals for selection.
- SCOPE requires no warmup on full local data, making the coreset selection step computationally cheaper.

From an optimization perspective, SCOPE effectively modifies the empirical distribution before training:

- Anomaly filtering reduces gradient bias terms.
- Dynamic balancing mitigates client drift.
- Global rarity constraints prevent further degradation of the long-tail distribution.

The paper also provides a theoretical justification for the non-convex setting. The conclusion is not that SCOPE modifies the optimizer, but that this data construction approach reduces both the heterogeneity term and the approximation error term in the federated objective, thereby enabling convergence to a tighter stationary point.

## Key Experimental Results

Experiments span four datasets and multiple model configurations:

- CIFAR-10 + ResNet-18
- Tiny-ImageNet + ResNet-50
- CIFAR-100 + ViT-B-16
- UHCS microstructure data + Swin-Tiny

Two types of difficulty are controlled in the federated setup:

- Local label skew induced by Dirichlet parameter $\alpha \in \{0.1, 1.0\}$.
- Global class imbalance controlled by imbalance ratio $IR \in \{2, 5, 10\}$.

Hyperparameters:

- Anomaly filter ratio $p_l = 0.1$.
- Redundancy filter ratio $p_f$ swept over $\{0.1, 0.3, 0.5, 0.7, 0.9\}$.
- Dynamic balancing threshold $\beta = 0.5$ by default; ablations in the appendix confirm this as the most robust choice.

Three conclusions are most emphasized by the authors.

**First**, SCOPE is substantially more stable at high pruning rates. Many baselines degrade sharply as $p_f$ increases, while SCOPE's accuracy curve exhibits considerably smaller fluctuations, indicating lower sensitivity to the amount of data removed.

**Second**, SCOPE can outperform Full DB training. The most direct example is on CIFAR-10 with $IR=2$, $\alpha=0.1$, and $p_f=0.1$, where SCOPE achieves 56.48%, slightly above the 55.63% obtained with full data. This demonstrates that the original data contains noise and biased samples that genuinely harm federated aggregation.

**Third**, the system efficiency advantage of SCOPE is dramatic. Because it uploads only class-level scalar statistics rather than high-dimensional feature centroids, server-side communication complexity drops from $O(K \times C \times D)$ to $O(K \times C)$, yielding bandwidth reductions of 128× to 512× across different datasets.

### Main Results

| Dataset/Setting | Metric | Ours (SCOPE) | Prev. SOTA | Full DB | Conclusion |
|---|---|---|---|---|---|
| CIFAR-10, ResNet-18, $IR=2$, $\alpha=0.1$, $p_f=0.1$ | Top-1 Acc | 56.48 | FedCore 55.96 / FedCS 53.09 | 55.63 | SCOPE marginally exceeds full-data training, demonstrating that noise removal + balancing improves the optimization trajectory |
| CIFAR-10, ResNet-18, $IR=10$, $\alpha=0.1$, $p_f=0.1$ | Top-1 Acc | 45.65 | FedCore 44.98 / FedCS 43.40 | 45.07 | Outperforms major baselines even under severe long-tail imbalance |
| Tiny-ImageNet, ResNet-50, $IR=2$, $\alpha=1.0$, $p_f=0.3$ | Top-1 Acc | 60.31 | GradND 59.49 / FedCS 58.81 | 59.85 | Achieves best result in this group at moderate pruning rates, with outstanding robustness |
| Tiny-ImageNet, ResNet-50, $IR=5$, $\alpha=0.1$, $p_f=0.9$ | Top-1 Acc | 55.38 | FedCore 52.42 / FedCS 52.57 | 54.41 | Remains competitive under extreme compression, with clear resistance to head-class bias |
| UHCS, Swin-Tiny, $IR=10$, $\alpha=0.1$, $p_f=0.9$ | Top-1 Acc | 92.62 | GradND 83.33 / FedCS 80.33 | 93.99 | Highly significant advantage on specialized scientific images, indicating that semantic geometry is more stable than gradient signals |
| CIFAR-100, ViT-B-16, $IR=2$, $\alpha=0.1$, $p_f=0.1$ | Top-1 Acc | 85.10 | FedCS 84.56 / FedCore 84.40 | 85.09 | Achieves best or tied-best performance in a complex label space |

### Ablation Study

| Configuration | CIFAR-10, $IR=10$, $p_f=0.1/0.5/0.9$ | Tiny-ImageNet, $IR=5$, $p_f=0.1/0.5/0.9$ | Notes |
|---|---|---|---|
| Full SCOPE | 45.65 / 45.04 / 42.80 | 54.65 / 54.28 / 55.28 | Complete global profile + anomaly filter + redundancy balancing |
| w/o Global Profiling | 38.68 / 31.61 / 19.04 | 53.76 / 50.19 / 38.36 | Removing global consensus causes near-collapse at high pruning rates, demonstrating the insufficiency of local heuristics |
| w/o Anomaly Filter | 43.18 / 41.87 / 39.79 | 54.46 / 54.11 / 52.25 | Skipping noise removal retains semantically anomalous samples, degrading aggregation stability |
| w/o Redundancy Filter | 42.61 / 42.45 / 42.61 | 54.07 / 54.03 / 54.78 | Without head-class redundancy removal, both long-tail protection and compression efficiency are impaired |

### Key Findings

- **Global profiling is the most critical component.** On CIFAR-10 at $p_f=0.9$, removing it causes a drop from 42.80 to 19.04, indicating that awareness of the global long-tail structure is a near-prerequisite for federated pruning to function.
- **Anomaly filtering and redundancy filtering are complementary.** The former primarily addresses noise and semantic misalignment; the latter primarily controls head-class repetition. Removing either one noticeably reduces the overall benefit.
- **SCOPE does not simply retain hard samples.** It retains samples that are "boundary-important but not clearly corrupted," while actively removing anomalous points that "resemble the negative class more than the true class." This is more stable than conventional high-loss strategies.
- **The communication and computation advantages of SCOPE stem from the score representation, not from any engineering trick.** For ResNet-50 on Tiny-ImageNet, communication volume decreases from approximately 160 MB to approximately 320 KB—a reduction of roughly 512×.
- **SCOPE significantly reduces coreset selection overhead.** Under the ViT-B-16 configuration, SCOPE's selection phase takes 283 seconds, compared to 2,186 seconds for FedCS (approximately 7.72× speedup), with peak memory usage of 1,039 MB versus 8,208 MB (approximately 7.9× reduction).
- **The scorer backbone does not need to be maximally large.** Table 7 shows that MobileCLIP-S2 with only 99M parameters achieves 58.25% on Tiny-ImageNet, comparable to larger models such as ViT-H-14 and MetaCLIP. This suggests that data selection prioritizes stable semantic structure over the strongest possible representation model.
- **The dynamic balancing threshold $\beta=0.5$ is the most robust default.** Too low a threshold leads to under-pruning; too high a threshold erroneously removes tail-class support samples, yielding a pronounced inverted-U pattern.

## Highlights & Insights

- The greatest strength is **compressing the federated global perspective into class-level scalar statistics**. This is more consistent with federated constraints than uploading embedding centroids and naturally endows the method with strong scalability.
- The **orthogonal projection step** is elegant. Rather than directly using similarity to the class prototype as the sole measure of sample value, the authors decouple the "class prototype direction" from the "orthogonal residual information," explicitly distinguishing typicality from novelty.
- **Two-stage filtering is more principled than single-score ranking.** Removing anomalies first and then redundancies corresponds to addressing noise contamination before distribution compression, avoiding the conflation of noise with genuinely difficult samples.
- The method is **decoupled from the downstream training architecture**, which is a notable strength. SCOPE scores samples using MobileCLIP-S2, yet the resulting coreset can be used with both CNNs and Transformers, without requiring the scorer and the trainer to share the same architecture.
- A key insight offered by this paper is that federated data governance need not revolve around gradients; it can instead be built around **statistical geometry in a shared semantic space**. This provides a promising direction for future lightweight federated sample selection research.

## Limitations & Future Work

- The method **relies heavily on the quality of the pretrained VLM's semantic space**. Although the authors demonstrate effectiveness on specialized data such as UHCS, the stability of $RS$/$DS$/$S_{neg}$ may degrade when class names are highly abstract or when class semantics cannot be adequately expressed through natural language prompts.
- The paper primarily compares against existing coreset/pruning baselines, without thorough comparison against more modern federated re-weighting, re-sampling, personalized FL, or class-balanced optimization methods. It therefore remains unclear whether SCOPE is better viewed as a substitute for data-level or optimization-level solutions.
- The theoretical analysis is more a plausibility argument than a tight convergence analysis. Several key assumptions—such as the Lipschitz mapping from semantic anomaly scores to gradient bias—remain relatively strong.
- The current framework requires all clients to share a **unified class vocabulary**. Extension is needed for open-world categories, inconsistent class names, or incompletely aligned label spaces across institutions.
- The diversity score is in essence derived from $RS$; although the authors reinforce its independent value through normalization and nonlinear interpretation, the two scores are not truly independent signal sources.
- One direction for improvement is to make text prompt learning a **federally updatable lightweight module** rather than a fixed template, enabling better adaptation to specialized domain labels.
- Another direction is to **combine SCOPE with federated active learning**, enabling the server to not only decide "what to remove" but also guide "what to prioritize acquiring" in future rounds.

## Related Work & Insights

- **vs. FedCS**: FedCS relies on high-dimensional feature centroids and local training information, incurring heavier communication and computation costs. SCOPE uploads only class-level scalars and grounds its pruning criterion in semantic geometry, making it especially suitable for bandwidth-constrained settings.
- **vs. FedCore**: FedCore attempts to reduce the cost of federated coreset selection but still requires a warmup training phase. SCOPE moves the selection process entirely before training, representing a more thorough design.
- **vs. EL2N / GradND / Forgetting**: These methods rely on training-dynamic signals and are prone to misidentifying scientific noise as high-value hard examples. SCOPE explicitly suppresses this misclassification through its anomaly filtering mechanism.
- **vs. traditional geometric selection**: Traditional Euclidean distance or herding methods focus primarily on covering the distributional center. SCOPE additionally considers prototype alignment, orthogonal residuals, and cross-class boundaries, operating at a richer semantic level.
- A personal takeaway is that future work on federated long-tail learning need not decompose "compression," "fairness," and "robustness" into three separately optimized modules. It may be more effective to first identify a sufficiently interpretable shared semantic space, define a small number of low-bandwidth statistics within that space, and allow global information to naturally inform local decisions.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — Using VLM semantic geometry for federated coreset selection and compressing the global consensus into scalar profiles is a novel combination, though the individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — Dataset and model coverage is broad, and system overhead is analyzed comprehensively; comparison with a wider range of federated long-tail methods could be deeper.
- **Writing Quality**: ⭐⭐⭐⭐☆ — Motivation, method, and system benefits are presented clearly and cohesively, and figures support the core conclusions well; the theoretical section leans toward intuition rather than rigor.
- **Value**: ⭐⭐⭐⭐⭐ — For practitioners working on federated data governance in real edge environments, this work is highly valuable, particularly under low-bandwidth and extreme long-tail conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[CVPR 2026\] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)

</div>

<!-- RELATED:END -->
