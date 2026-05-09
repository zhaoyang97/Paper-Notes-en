---
title: >-
  [Paper Note] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models
description: >-
  [ICLR 2026][Model Compression][Test-Time Training] Grounded in the Linear Representation Hypothesis (LRH), this paper proposes a theoretical framework termed *specialization after generalization*, providing the first systematic explanation of why TTT is effective under in-distribution settings. Foundation models suffer from concept superposition due to global underparameterization; TTT temporarily forgets irrelevant concepts to free model capacity, locally specializing to the small set of concepts relevant to the test task. The theory guarantees generalization even when the feature space is exponentially smaller than the concept space.
tags:
  - ICLR 2026
  - Model Compression
  - Test-Time Training
  - Linear Representation Hypothesis
  - Sparse Autoencoders
  - Specialization after Generalization
  - Foundation Models
date: 2026-05-08
content_hash: a7ab91bb9e4d859d
---

# Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2509.24510](https://arxiv.org/abs/2509.24510)
**Code**: None
**Area**: Model Compression / Test-Time Training
**Keywords**: Test-Time Training, Linear Representation Hypothesis, Sparse Autoencoders, Specialization after Generalization, Foundation Models

## TL;DR

Grounded in the Linear Representation Hypothesis (LRH), this paper proposes a theoretical framework termed *specialization after generalization*, providing the first systematic explanation of why TTT is effective under in-distribution settings. Foundation models suffer from concept superposition due to global underparameterization; TTT temporarily forgets irrelevant concepts to free model capacity, locally specializing to the small set of concepts relevant to the test task. The theory guarantees generalization even when the feature space is exponentially smaller than the concept space.

## Background & Motivation

**Background**: Test-Time Training (TTT) pushes fine-tuning to its extreme—adapting the model individually for each test sample. Recent years have seen substantial gains from TTT on tasks including abstract reasoning, language modeling, and video generation. The typical protocol retrieves nearest neighbors of a test point from the training set and performs a few gradient steps on the pretrained model using a supervised loss, with inference performed using the locally fine-tuned model.

**Limitations of Prior Work**: Prior explanations for TTT's effectiveness have focused on two aspects: (1) adapting to distribution shift (out-of-distribution adaptation); and (2) exploiting privileged data unseen during pretraining. However, as foundation models scale dramatically, the vast majority of test data is already in-distribution, rendering both explanations inapplicable. A critical open question emerges: **why does TTT still improve predictions under in-distribution settings?**

**Key Challenge**: Despite the massive parameter counts of contemporary foundation models, scaling law studies demonstrate that increasing model size continues to yield performance gains, indicating that models remain *effectively underparameterized*. Models must simultaneously encode a vast number of real-world concepts that far exceed the model's dimensionality, forcing multiple concepts to be superimposed onto the same activations. This superposition causes inter-concept interference during global prediction, preventing precise disentanglement of individual concept meanings.

**Goal**: (1) Theoretically characterize the mechanism by which TTT is effective in-distribution; (2) establish theoretical upper bounds on TTT error and prove their superiority over global training; (3) validate theoretical assumptions via SAE experiments and theoretical predictions via scaling studies.

**Key Insight**: The authors build on the Linear Representation Hypothesis (LRH)—models encode high-level semantic concepts as linear directions in activation space, with each input activating only a sparse set of $s$ concepts. Because the number of concepts $d_1$ far exceeds the model dimensionality $d_2$, concepts are superimposed in dense activations. TTT need only disentangle a small number of relevant concepts within a local neighborhood, rather than disentangling all concepts globally, making it substantially easier to succeed.

**Core Idea**: TTT is fundamentally *specialization after generalization*—the model first learns superimposed concept representations through global training, then at test time releases model capacity to the small set of task-relevant concepts via local fine-tuning, temporarily "forgetting" irrelevant knowledge in exchange for improved local precision.

## Method

### Overall Architecture

This paper presents a theory-driven empirical study. Rather than proposing a new TTT algorithm, it constructs a theoretical framework grounded in the LRH to explain TTT's effectiveness, and validates assumptions via sparse autoencoder (SAE) experiments and theoretical predictions via scaling studies. The work proceeds at three levels: (1) establishing three key observations (O1–O3) using SAEs on ImageNet to support theoretical assumptions; (2) conducting model- and data-scale scaling experiments on MNIST/ImageNet/Pile to validate empirical predictions of the underparameterization hypothesis; and (3) deriving TTT local error upper bounds under the LRH and comparing them theoretically against global training error.

Given a test sample $x^*$, the method retrieves its $k=50$ nearest neighbors from the training set, trains a local linear classifier (updating only the final layer) with cross-entropy loss, and predicts the label of $x^*$. The central theoretical question is: why does this strategy of "training a local model on 50 neighbors" outperform "training a global model on the full dataset"?

### Key Designs

1. **SAE Validation Framework: Constructing an Analyzable Concept Space**

   - *Function*: Decompose dense model activations into sparse concept representations via sparse autoencoders, enabling empirical examination of theoretical assumptions.
   - *Mechanism*: CLIP ViT-B/32 is used to extract $d_2=512$-dimensional features $\Psi(x)$ on ImageNet-1K. A top-$k$ SAE is trained to encode these into $d_1=4096$-dimensional, $s=16$-sparse concept vectors $\hat{\Phi}(x)$. The encoder $E \in \mathbb{R}^{d_1 \times d_2}$ retains the $s$ largest activations; the decoder $D \in \mathbb{R}^{d_2 \times d_1}$ reconstructs the original features from the sparse representation. The optimization objective is the reconstruction error $\mathbb{E}_x\|\Psi(x) - D \cdot \text{top}_s(E \cdot \Psi(x))\|_2^2$. A ghost gradient auxiliary loss mitigates dead feature problems (only 4% inactive concepts).
   - *Design Motivation*: The "true concept space" $\Phi$ in the theory is not directly accessible; an approximate concept space $\hat{\Phi}$ is learned via the SAE to validate three key assumptions. Experiments are conducted on SAE-reconstructed features $\hat{\Psi}(x)$ rather than raw CLIP features to align the experimental setup with the theoretical model, at the cost of approximately 6% reduction in global classification accuracy.

2. **Three Key Observations (O1–O3): Connecting Theory and Empirics**

   - *Function*: Establish three empirical hypotheses for TTT's effectiveness, serving as the foundation for theoretical derivations.
   - *Mechanism*:
     - **O1 (Local Geometry Preservation)**: Neighborhoods of test points are selected in each of the three spaces $\Psi$, $\hat{\Psi}$, and $\hat{\Phi}$, and the distribution of cosine similarities in concept space is measured—the three distributions are nearly identical, demonstrating that the SAE mapping preserves local angular structure.
     - **O2 (Neighborhoods Are Supported by Few Concepts)**: An adaptive binary mask $m$ is learned per neighborhood using a straight-through estimator to optimize $\hat{\Phi}_m(x) = m \odot \hat{\Phi}(x)$ with an $\ell_2$ sparsity penalty. On average, only ~40 concepts are required (out of ~180 activated across the neighborhood) without degrading TTT accuracy. More strikingly, a non-adaptive mask (retaining only the 16 concepts activated by the test point itself) achieves only 71.51%, far below the adaptive mask's 72.64%, indicating that mask learning can identify and remove spuriously correlated features.
     - **O3 (Implicit Sparsity)**: TTT performed in $\hat{\Psi}$ space and $\hat{\Phi}_m$ space yields consistent predictions on ~89% of samples, with highly similar top-10 probability distributions, indicating that feature-space TTT implicitly favors sparse solutions in concept space.
   - *Design Motivation*: Direct mathematical analysis of TTT's properties is intractable; empirical regularities are first established and then used as the conditions for theoretical proofs.

3. **Theoretical Analysis of TTT Error: Proving Local Specialization Outperforms Global Training**

   - *Function*: Theoretically quantify TTT's generalization ability and prove that TTT error is substantially smaller than that of global models under underparameterization.
   - *Mechanism*: Three formal assumptions based on O1–O3 are established: (H1) feature-space neighborhoods are contained within slightly enlarged concept-space neighborhoods (supported by the Johnson–Lindenstrauss lemma, with deviation $\delta \leq O(\sqrt{\log N / d_2})$); (H2) there exist $\Theta(s)$-sparse local concept vectors within the neighborhood that approximate the true function; (H3) TTT implicitly finds sparse solutions in concept space. Under these conditions, sparse recovery techniques are applied to prove that the TTT test error satisfies $(f(x^*) - \langle \Psi(x^*), \hat{v}_{x^*}^{\text{TTT}} \rangle)^2 \leq O(\sigma^2 s \log(d_1/s)/k)$, achieving the minimax optimal rate. As a contrast, a lower bound on the global model error is constructed: when the feature space is a random projection of the concept space, the global error is $1 - d_2/d_1$, which approaches 1 as $d_1 \to \infty$.
   - *Design Motivation*: The key contrast is that TTT error improves with neighborhood size $k$ and depends on the concept space dimension only logarithmically as $\log d_1$, whereas global model error degrades linearly as $d_2/d_1$. When $d_1$ is large (i.e., real-world concepts are extremely numerous), TTT holds an exponential advantage.

### Loss & Training

During the TTT phase, a standard cross-entropy loss is used to train a local linear classifier on $k=50$ nearest neighbors, updating only the final layer. For language modeling experiments, TTT is performed on the Qwen2.5 series using LoRA (~1% of parameters), with one gradient step taken per neighbor in descending order of similarity over 50 neighbors. Neighborhood retrieval uses $L_2$ distance (equivalent to cosine similarity for normalized CLIP features).

## Key Experimental Results

### Main Results: TTT vs. Global Training on ImageNet

| Method | Concept Space $\hat{\Phi}(x)$ Accuracy | Feature Space $\hat{\Psi}(x)$ Accuracy |
|---|---|---|
| Global Training | 71.45 ± 0.21 | 71.26 ± 0.20 |
| TTT ($k$=50 neighbors) | **72.64 ± 0.20** | **72.56 ± 0.19** |
| Adaptive Mask TTT | 72.64 ± 0.20 (~40 concepts only) | — |
| Non-adaptive Mask TTT | 71.51 | — |

TTT improves over global training by approximately 1.2–1.3 percentage points. Adaptive mask TTT achieves the same performance as dense TTT using only ~40 concepts, while the non-adaptive mask (retaining only the 16 concepts activated by the test point) yields almost no improvement, demonstrating that local concept selection requires learning rather than simple matching.

### Model/Data Scaling Experiments

| Task | Smallest Model TTT Gain | Largest Model TTT Gain | Trend |
|---|---|---|---|
| MNIST | ~0.8% error reduction | ~0.1% | Gap shrinks significantly as model scales |
| ImageNet (MLP) | ~3% accuracy gain | ~0.5% | Same |
| Pile Language Modeling (Qwen2.5) | 0.5B: ~0.07 bits/byte improvement | 7B: ~0.02 bits/byte | Same |

### Validation of TTT Locality

| Evaluation Setting | MNIST Accuracy | ImageNet Accuracy |
|---|---|---|
| Global Model | 98.57 ± 0.12 | 78.33 ± 0.19 |
| TTT evaluated on test sample | 99.01 ± 0.10 | 79.39 ± 0.18 |
| TTT evaluated on neighborhood | 100.00 ± 0.00 | 95.19 ± 0.00 |
| TTT head evaluated globally | **36.38 ± 0.16** | **77.04 ± 0.06** |

This table is particularly compelling: the TTT head achieves near-perfect performance within the neighborhood (100% on MNIST, 95.19% on ImageNet), but applying it globally to the entire test set results in catastrophic collapse (only 36.38% on MNIST), directly validating the *local specialization* mechanism—TTT trades away knowledge of irrelevant concepts in exchange for local precision.

### Key Findings

- **Model scaling trends precisely match theoretical predictions**: TTT consistently outperforms global training, but the gap narrows as model size increases. On MNIST, the smallest model achieves ~0.8% improvement, while the largest model is nearly on par; on Pile, the 0.5B model benefits most and the 7B model the least. This is consistent with the theoretical insight that "the degree of underparameterization determines TTT gains"—larger models superimpose fewer concepts, allowing the global model itself to disentangle more of them.
- **Counter-intuitive effect of data scale**: More training data leads to slightly larger TTT improvements, since a larger training set provides richer and more relevant neighborhoods for each test point, making local adaptation more effective.
- **Optimal trade-off in neighborhood size**: Experiments on ImageNet with varying $k$ show that too small a $k$ leads to high variance, while too large a $k$ introduces irrelevant concepts that violate the local sparsity assumption. The optimum is around $k \approx 50$.
- **Non-parametric baselines perform poorly**: Majority voting performs extremely poorly when the number of classes is large (1000 on ImageNet), as it cannot exploit the sparse structure of concept space and relies solely on frequency statistics.

## Highlights & Insights

- **"Forgetting as Capacity Release"—A Unified Explanation**: This work is the first to attribute TTT's in-distribution effectiveness to global model underparameterization rather than distribution shift. This perspective simultaneously explains why TTT improves smaller models more, why larger models benefit less, and why applying the TTT head globally results in collapse. The conceptual elegance is notable.
- **SAE as a Theory Validation Tool**: The authors cleverly leverage SAEs to render the otherwise unobservable "concept space" into an analyzable object. The adaptive mask experiment (O2) is particularly insightful—demonstrating that only ~40 out of 4096 concepts are truly relevant within a neighborhood, providing a quantitative explanation for why 50 neighbors suffice.
- **Theory Offers Practical Guidance**: The error bound $O(\sigma^2 s \log(d_1/s)/k)$ directly informs practitioners: TTT effectiveness depends on local concept sparsity $s$, concept space size $d_1$ (only logarithmically), and neighborhood size $k$. This provides theoretical grounding for choosing neighborhood size and for determining when TTT is likely to be beneficial.
- **Conceptual Bridge to Continual Learning**: The "temporary forgetting" mechanism in TTT is the flip side of catastrophic forgetting in continual learning—where forgetting is a problem in continual learning, it is a feature in TTT.

## Limitations & Future Work

- **Universality of LRH Assumption Unverified**: All theoretical conclusions rely on the Linear Representation Hypothesis, but whether LRH holds across all architectures and tasks remains an open question. In highly nonlinear tasks (e.g., those requiring compositional reasoning), concepts may not be encoded as linear directions.
- **Limited SAE Approximation Fidelity**: SAE reconstruction reduces global accuracy by 6% in experiments, indicating a gap between $\hat{\Phi}$ and the true concept space. Observations O1–O3 are all validated in the approximate concept space; further evidence is needed to determine whether these properties hold in the true concept space.
- **Gap Between Linear and Nonlinear TTT**: The theoretical analysis is restricted to linear classifiers (final-layer updates), whereas practical TTT typically updates multiple layers via LoRA. The authors employ LoRA TTT in language modeling experiments, but the theory does not cover this nonlinear setting.
- **Limited Experimental Scale**: Language modeling experiments are restricted to 1% of the test set and models up to 7B parameters (constrained by a single RTX 4090). Whether the scaling trends persist for much larger models (70B+) remains to be verified.
- **Neighborhood Selection Strategy Unexplored**: The paper uses a fixed $k=50$ and $L_2$ distance, but the optimal neighborhood size may depend on the test point itself (regions with high concept density may require smaller neighborhoods). Adaptive neighborhood selection is a valuable direction for future work.

## Related Work & Insights

- **vs. Classical TTT (Sun et al. 2020)**: Classical TTT uses self-supervised losses (e.g., rotation prediction) to adapt to distribution shift; this paper studies semi-parametric TTT (supervised fine-tuning on nearest neighbors) in strictly in-distribution settings. Classical TTT's theoretical analysis assumes alignment between TTT gradients and oracle label gradients; this work provides theoretical conditions under the LRH under which such alignment holds.
- **vs. Non-parametric Methods (kNN Classifiers)**: kNN/majority voting is the simplest form of "localization," but performs poorly in high-dimensional ($s$-sparse) concept spaces. This paper's theory explains why: non-parametric methods cannot exploit the sparse structure of concept space to disentangle meaning, relying solely on frequency statistics.
- **vs. Basu et al. 2023**: That work analyzes a similar TTT setup from the perspective of non-parametric estimation, assuming the target function is smooth in feature space. This paper explicitly models the underlying sparse concept space, enabling explanations for why TTT remains effective in locally high-dimensional ($s$-sparse) settings.
- **Intersection with Interpretability Research**: This paper extends SAEs from an interpretability tool to a theory validation instrument—a methodological contribution transferable to other problems requiring understanding of internal representation structure (e.g., RLHF alignment, knowledge editing).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The *specialization after generalization* framework is exceptional in both theoretical depth and explanatory power, providing the first unified account of TTT's in-distribution effectiveness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Scaling studies spanning vision and language validate theoretical predictions, and SAE experiments validate three key assumptions, though the language modeling experiments are limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is exceptionally well-structured; the logical progression from "when it works" to "why it works" is elegant, and the connection between theory and experiment is tight throughout.
- **Value**: ⭐⭐⭐⭐ — Primarily theory-oriented, with indirect practical guidance for TTT (when to use it, how to choose neighborhood size); more broadly, it offers a novel perspective on understanding the internal representations of foundation models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)

<!-- RELATED:END -->
