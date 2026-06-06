---
title: >-
  [Paper Note] GEM: Geometric Entropy Mixing for Optimal LLM Data Curation
description: >-
  [ICML 2026][Interpretability][Data mixing] GEM reformulates the LLM pre-training data partitioning problem into a variational objective combining vMF mixtures on the hypersphere with balance regularization. This objectiv…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Data mixing"
  - "Hyperspherical clustering"
  - "von Mises-Fisher"
  - "MM algorithm"
  - "Balance regularization"
date: 2026-05-08
content_hash: 18c8deb6064f2e67
---

# GEM: Geometric Entropy Mixing for Optimal LLM Data Curation

**Conference**: ICML 2026  
**arXiv**: [2605.26121](https://arxiv.org/abs/2605.26121)  
**Code**: To be confirmed  
**Area**: LLM Pre-training / Data Mixing  
**Keywords**: Data mixing, Hyperspherical clustering, von Mises-Fisher, MM algorithm, Balance regularization  

## TL;DR
GEM reformulates the LLM pre-training data partitioning problem into a variational objective combining vMF mixtures on the hypersphere with balance regularization. This objective is solved using a provably monotonic MM algorithm and distilled into a FastText classifier via Teacher-Student distillation. Experimental results on 1.1B models show an average improvement of approximately 1.2% when integrated into DoReMi, Perf, and RegMix frameworks.

## Background & Motivation

**Background**: The effectiveness of LLM pre-training increasingly depends on "data proportions." Dynamic mixing methods such as DoReMi, RegMix, Aioli, and SampleMix have become mainstream, all of which require partitioning the corpus into several "semantic buckets" before learning the weights between them.

**Limitations of Prior Work**: Existing "bucketization" solutions fall into two categories, both with inherent flaws. One is based on manual taxonomy (e.g., WebOrganizer/TnT-LLM), where LLMs label web pages according to human-written systems. This leads to ontological misalignment (mismatch between human categories and actual model semantic granularity) and unsustainable labeling costs for corpus updates. The other category comprises unsupervised methods like K-Means/HDBSCAN. While scalable, they are based on Euclidean geometry, which is mismatched with modern text embeddings (e.g., BGE, RoBERTa) trained via cosine similarity. Combined with the anisotropy/cone effect, this causes cluster collapse, where a few large buckets consume all long-tail semantics.

**Key Challenge**: While actual embeddings reside on the high-dimensional hypersphere $\mathcal{S}^{d-1}$ with signals in directions (cosine similarity), clustering objectives are built in Euclidean space. Furthermore, standard EM-learned mixture weights $\alpha_k$ suffer from "rich-get-richer" feedback, pushing mass toward dominant clusters.

**Goal**: (i) Establish semantic partitioning based on directional statistics on the hypersphere, naturally compatible with cosine similarity; (ii) explicitly suppress cluster mass collapse to achieve balanced and semantically distinct buckets; (iii) ensure low-cost deployment on trillion-token corpora.

**Key Insight**: Utilize von Mises-Fisher (vMF) mixtures as a directional generative model, where the sufficient statistics $\mu_k^\top x$ correspond to cosine similarity. Address the "collapse" problem by decoupling the generative prior $\alpha_k$ from the empirical soft mass $\boldsymbol{\pi}(\Gamma)$, adding a quadratic balance regularization $-\tfrac{\lambda}{2}\lVert\boldsymbol{\pi}-\mathbf{u}\rVert_2^2$ directly to the empirical mass.

**Core Idea**: Perform joint variational optimization of "entropy-regularized ELBO + empirical mass balance" on the hypersphere. An MM (Minorize-Maximize) algorithm is derived to provide a per-sample decomposable E-step with provable monotonic ascent, enabling stable solutions for directionally balanced semantic buckets.

## Method

### Overall Architecture
GEM consists of a two-stage pipeline. **Teacher Stage**: Variational clustering with vMF mixtures and balance regularization is performed on a sampled seed corpus $\mathcal{X}_{seed}$. The MM algorithm iteratively updates Riemannian parameters $(\mu_k, \kappa_k)$ and soft assignments $\gamma_{ik}$ to obtain $K$ directional clusters. Representative samples are selected using the Geometric Influence Score (GIS) and fed into an LLM to generate readable taxonomy labels. **Student Stage**: High-confidence, cluster-balanced pseudo-labeled sets are selected via GIS to distill a lightweight FastText linear classifier. This enables linear-time labeling of the full corpus. The resulting "buckets" are then fed into mixing algorithms like DoReMi/Perf/RegMix.

### Key Designs

1.  **Hyperspherical vMF Mixture + Empirical Mass Balance Regularization**:
    - **Function**: Describes embeddings using a generative model aligned with cosine similarity while explicitly suppressing cluster collapse.
    - **Mechanism**: Each cluster $k$ is a vMF component $f_{\text{vMF}}(x\mid\mu_k,\kappa_k)=C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x)$. The mixing prior is fixed at $\alpha_k\equiv 1/K$ to prevent "rich-get-richer" feedback. The objective is the entropy-regularized ELBO plus a quadratic penalty on the empirical soft mass $\pi_k(\Gamma)=\tfrac{1}{N}\sum_i\gamma_{ik}$, defined as $-\tfrac{\lambda}{2}\lVert\boldsymbol{\pi}(\Gamma)-\mathbf{u}\rVert_2^2$ where $\mathbf{u}=\tfrac{1}{K}\mathbf{1}$.
    - **Design Motivation**: vMF sufficient statistics are directional inner products, aligning the "geometric space" with the "embedding space." Applying regularization to the empirical mass rather than the generative prior maintains the interpretability of the generative model while supporting long-tail clusters under anisotropic embeddings.

2.  **Provably Monotonic MM Inference**:
    - **Function**: Decomposes the globally coupled regularization term into per-sample local updates for stable convergence of E/M steps.
    - **Mechanism**: The regularization $R(\boldsymbol{\pi})$ is a $\lambda$-smooth concave quadratic function with a global quadratic minorizer $R(\boldsymbol{\pi})\geq R(\boldsymbol{\pi}^{(t)})+\langle\nabla R(\boldsymbol{\pi}^{(t)}),\boldsymbol{\pi}-\boldsymbol{\pi}^{(t)}\rangle-\tfrac{\lambda}{2}\lVert\boldsymbol{\pi}-\boldsymbol{\pi}^{(t)}\rVert_2^2$. Substituting this into the E-step yields a surrogate objective $\widetilde{\mathcal{F}}_t(\Gamma)$, which is concave-decomposable for each $\gamma_i$, solved via mirror ascent. The M-step provides closed-form vMF updates: $r_k=\sum_i\gamma_{ik}x_i$, $\mu_k=r_k/\lVert r_k\rVert_2$, and $\kappa_k$ estimated via high-dimensional approximation using $\bar R_k=\lVert r_k\rVert_2/\sum_i\gamma_{ik}$.
    - **Design Motivation**: Direct maximization is difficult due to coupling via $\boldsymbol{\pi}(\Gamma)$. The MM surrogate ensures $\mathcal{F}(\Theta^{(t)},\Gamma^{(t+1)})\geq\mathcal{F}(\Theta^{(t)},\Gamma^{(t)})$, guaranteeing stable convergence on large-scale data without empirical early stopping.

3.  **GIS Sampling + Teacher-Student Distillation to FastText**:
    - **Function**: Reduces the cost of hyperspherical EM to affordable linear inference for trillion-token corpora.
    - **Mechanism**: Geometric Influence Scores are used to select high-confidence, balanced representative samples as pseudo-labels for a FastText student model. A small subset of representative samples is labeled by an LLM to produce a readable taxonomy (see Figure 3).
    - **Design Motivation**: Web-scale processing is latency-sensitive. Distillation crystallizes "geometric fidelity" into a linear classifier, reducing the categorization step to linear time while retaining GEM's balance properties.

### Loss & Training
The variational objective follows Eq.(3): $\max_{\Theta,\Gamma}\sum_i\sum_k\gamma_{ik}\log(\alpha_k f_{ik}(\Theta))+\sum_i H(\gamma_i)-\tfrac{\lambda}{2}\lVert\boldsymbol{\pi}(\Gamma)-\mathbf{u}\rVert_2^2$. In main experiments, $K=24$ and $\lambda=5000$ (aligning the logit scale with the learned vMF $\kappa\approx 900$). The backbone is a 1.1B LLaMA-style Transformer with a 25B token budget, using cleaned CommonCrawl data.

## Key Experimental Results

### Main Results
Replacing the "bucketization" module with GEM across three mixing frameworks shows performance on 9 OLMES sub-tasks (Table 1):

| Mixing Framework | Bucketization Method | Science QA | Commonsense | Logic & Ling. | Average |
|---|---|---|---|---|---|
| DoReMi | Spherical K-Means | 34.62 | 38.97 | 54.72 | 42.77 |
| DoReMi | WebOrganizer (Format) | 34.44 | 38.73 | 55.19 | 42.79 |
| DoReMi | **GEM** | **34.79** | **39.96** | **57.11** | **43.95** |
| Perf | WebOrganizer (Format) | 35.06 | 39.73 | 57.97 | 44.25 |
| Perf | **GEM** | **35.96** | **40.43** | **57.98** | **44.79** |
| RegMix | WebOrganizer (Format) | 34.12 | 33.94 | 54.26 | 40.77 |
| RegMix | **GEM** | **34.07** | **35.30** | **54.97** | **41.45** |

Under DoReMi, GEM improves over the strongest baseline (WebOrganizer) by +1.23 and +1.76 pt on Commonsense and Logic, respectively. Overall average gain is approximately 0.7–1.2 pt.

### Ablation Study
Ablations on geometry and balance (Figure 6):

| Configuration | Average | Description |
|---|---|---|
| K-Means (Euclidean + Hard) | 38.5 | Pure Euclidean; worst cluster collapse |
| Spherical K-Means (Spherical + Hard) | ↑ | Metric changed to cosine; mitigates anisotropy |
| Vanilla vMF (Spherical Soft, no Reg) | ↑↑ | Riemannian generative model used; still prone to collapse |
| **GEM (full)** | **Highest** | Balancing regularization successfully recovers long-tail clusters |

Regarding the number of clusters $K$ (Figure 5), performance peaks at $K=36$ (41.21%) and slightly decreases at $K=48$, suggesting "over-fragmentation" introduces noise.

### Key Findings
- "Spherical geometry" and "balance regularization" provide monotonic improvements (Euclidean → Spherical → vMF → GEM). Both are necessary to recover long-tail clusters; changing the metric alone is insufficient to solve cluster collapse.
- Using RegMix as a "taxonomy predictability" probe (Spearman $\rho$ across 10 splits), GEM shows a higher median and narrower IQR. This indicates that loss changes are more "linear" and predictable relative to mixing weights—a prerequisite for efficient search in mixing algorithms.
- Setting $\lambda=5000$ aligns with the vMF logit scale ($\kappa\approx 900$), proving that regularization and the generative model must be matched in magnitude to prevent either collapse or conversion to uniform noise.

## Highlights & Insights
- **Decoupling prior and mass is a transferable trick**: Many EM-like algorithms suffer from "rich-get-richer" dynamics. Decoupling the modeling distribution from the empirical distribution by fixing $\alpha_k$ and regularizing $\boldsymbol{\pi}(\Gamma)$ is applicable to MoE load balancing or retrieval category distributions.
- **Operationalizing "bucket quality" as "mixing predictability"**: Beyond standard metrics like NMI, using the RegMix Spearman $\rho$ measures whether bucketization provides a smooth optimization coordinate system for downstream algorithms, aligning directly with the data curation goal.
- **MM over EM**: For regularized ELBOs, the use of a global quadratic minorizer for $\lambda$-smooth concave functions allows for a decomposable and provably monotonic E-step. This framework can be reused for any probability model combining likelihood with global regularization.

## Limitations & Future Work
- Experiments were conducted at the 1.1B model scale with a 25B token budget. Gains might narrow at larger scales (7B+ models) or longer training durations.
- Geometric fidelity depends on the text encoder (currently BGE/RoBERTa). The robustness of the hyperspherical assumption with instruction-tuned or less isotropic encoders is unknown.
- The balance target is the uniform distribution $\mathbf{u}$. This assumes semantic classes should have equal weight, which may conflict with real-world distributions (e.g., code vs. prose). Future work could adaptively learn $\mathbf{u}$ from downstream rewards.
- FastText is a linear model and may lose fine-grained polysemy. Lightweight students based on hyperbolic or product-of-spheres spaces could improve expressiveness without violating latency constraints.

## Related Work & Insights
- **vs. WebOrganizer / TnT-LLM**: These use LLMs to label documents with human-written taxonomies. GEM "grows" labels from embedding geometry, ensuring alignment between labels and the model's perceived semantic granularity.
- **vs. Spherical K-Means / Vanilla vMFMM**: While these use hyperspherical clustering, they lack explicit suppression of cluster collapse. GEM addresses this via empirical mass balance.
- **vs. DoReMi / RegMix / Aioli / SampleMix**: These assume given buckets and learn weights. GEM solves the "input" problem for these frameworks. Using GEM as a plug-and-play categorization layer demonstrates that "better bucketization" is more effective than "intensive weight tuning on poor buckets."

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of vMF mixtures, empirical mass balance, and MM inference is a clear and well-defined direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 mixing frameworks across 9 benchmarks with sensitivity analyses ($K, \lambda$), though limited to 1.1B models.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2 clearly communicate motivation; derivations are well-supported by Lemmas/Propositions.
- Value: ⭐⭐⭐⭐ Provides a provable and deployable solution to the bucketization problem for the data mixing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance](position_lets_develop_data_probes_to_fundamentally_understand_how_data_affects_l.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)
- [\[ICML 2026\] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](diagnosing_the_reliability_of_llm-as-a-judge_via_item_response_theory.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)

</div>

<!-- RELATED:END -->
