---
title: >-
  [Paper Note] Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control
description: >-
  [ICML 2026][Recommender Systems][Sequential Recommendation] RSIR enables sequential recommendation models to generate new synthetic user interaction sequences using their own predictive capabilities, retrain a new model…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Self-Training"
  - "Data Sparsity"
  - "Fidelity Control"
  - "Implicit Regularization"
date: 2026-05-08
content_hash: 812202f57c393ab2
---

# Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control

**Conference**: ICML 2026  
**arXiv**: [2602.15659](https://arxiv.org/abs/2602.15659)  
**Code**: https://github.com/USTC-StarTeam/RSIR  
**Area**: Recommender Systems / Data Augmentation / Self-Training  
**Keywords**: Sequential Recommendation, Self-Training, Data Sparsity, Fidelity Control, Implicit Regularization

## TL;DR
RSIR enables sequential recommendation models to generate new synthetic user interaction sequences using their own predictive capabilities, retrain a new model, and filter out samples deviating from the user preference manifold via a ranking-based "fidelity check," thus preventing self-consuming model collapse. On 4 datasets × 3 mainstream backbones, it consistently improves NDCG/Recall by 4–11%, and theoretically proves this process is equivalent to implicit regularization along the tangent space of the user preference manifold.

## Background & Motivation
**Background**: Sequential recommendation mainly relies on expanding data and models, but any user only interacts with a tiny fraction of the platform catalog, resulting in inherently extremely sparse interaction signals. This leads to a rugged loss landscape, models converging to sharp minima, and poor generalization.

**Limitations of Prior Work**: (1) Data augmentation methods (Reordering / Insertion / item masking / cropping) only perturb existing data without generating new "high-fidelity user trajectories," thus offering limited improvement; (2) Data generation methods (DiffuASR, DR4SR) can create new sequences but require diffusion models/auxiliary generators, making training expensive; (3) Using LLMs as teachers for data expansion outsources the performance bottleneck to "sufficiently large external models," which is uncontrollable in deployment and introduces distribution mismatch.

**Key Challenge**: Closed-loop self-training (generating data and retraining on it) has been shown to work in LLMs and diffusion models, but is highly prone to model collapse—model biases and errors are amplified by self-generated data, causing performance to crash after a few iterations. The key is to balance "self-generated data" and "avoiding error accumulation."

**Goal**: (1) Enable recommender models to self-bootstrap like STaR/self-rewarding LLMs, without external teachers or labels; (2) Design a reliable "fidelity filtering" mechanism to prevent synthetic data from drifting off the user preference manifold; (3) Theoretically explain why this recursion does not collapse but instead regularizes.

**Key Insight**: The authors view "self-improvement" as a data-driven implicit regularization—only accepting perturbations near the tangent space of the true user interest manifold as new data, which is equivalent to imposing a gradient penalty along the manifold direction in the loss landscape, forcing optimization to converge to flatter minima.

**Core Idea**: Use the current model to generate synthetic sequences + use the same model as a "ranking judge" to filter out deviating steps, and insert this pair into the classic SFT loop, rolling recursively for multiple rounds.

## Method

### Overall Architecture
Each RSIR iteration $k$ consists of 4 steps: (1) Train model $f_{\theta_k}$ (next-item prediction) on current dataset $D_k$; (2) Use $f_{\theta_k}$ to generate $m$ synthetic sequences $D'_{k+1}$ for each user—starting from a random prefix of the user's real history, autoregressively extend; (3) Merge to obtain $D_{k+1} = D_k \cup D'_{k+1}$; (4) Train $f_{\theta_{k+1}}$ from scratch (or fine-tune the previous model). The core of generation involves two mechanisms: bounded exploration (hybrid candidate pool) + fidelity-based quality control (ranking verification).

### Key Designs

1. **Bounded Exploration Hybrid Candidate Pool**:

    - **Function**: Determines from which item pool to sample the next item at each generation step, balancing "exploiting known interests" and "exploring new interests."
    - **Mechanism**: At each step, sample from the user's history $s_u$ with probability $p$, and from the global item set $I$ with probability $1-p$, forming the candidate pool $\mathcal{C}_t \sim p \cdot \mathrm{Sample}(s_u) + (1-p) \cdot \mathrm{Sample}(I)$. The model then performs top-$k$ sampling only within $\mathcal{C}_t$. Empirically, $p \approx 0.5$ is optimal—pure exploitation ($p=1$) only reorders known items and cannot expand new interests, while pure exploration ($p=0$) easily drifts and is filtered out by quality control.
    - **Design Motivation**: Conventional autoregressive generation over a large vocabulary leads to an explosion in the generation space; introducing historical bias retains controllability and enables "extending known interests."

2. **Fidelity-Based Quality Control (Ranking Verification)**:

    - **Function**: After generating each candidate item, immediately "probe" whether it still ranks the user's real future items highly; if not, break and terminate the sequence.
    - **Mechanism**: Define $S_{tgt} = s_u \setminus S_{ctx}'$ as the set of unused items in the user's real sequence. If there exists $i_j \in S_{tgt}$ such that $\mathrm{Rank}_{f_{\theta_k}}(i_j | S_{ctx}') \leq \tau$, accept the new item and continue; otherwise, terminate the sequence. This ensures generated trajectories remain compatible with the user's true interest manifold.
    - **Design Motivation**: The authors prove that stricter $\tau$ better suppresses the "fidelity miss rate" $\tilde{p}_k$, ensuring the recursive error recurrence $\mathcal{E}(\theta_{k+1}) \leq (1-\lambda)\mathcal{E}_0 + \lambda[(1-\tilde{p}_k)\rho \mathcal{E}(\theta_k) + \tilde{p}_k \mathcal{E}_{\max}]$ satisfies contraction, preventing model collapse.

3. **Manifold Tangential Gradient Penalty (Theoretical Explanation)**:

    - **Function**: Reinterprets the "filtering + generation" loop as an implicit regularization, providing theoretical grounding.
    - **Mechanism**: Accepted perturbations are restricted to the tangent space of the user preference manifold $\mathcal{M}$, equivalent to adding a regularization term $\Omega(\theta) \propto \|\mathcal{P}_\mathcal{M} \nabla_s f_\theta\|^2$ to the original loss; this penalizes the gradient magnitude along the manifold direction, guiding solutions to "flat valleys" parallel to the true user manifold.
    - **Design Motivation**: Explains why RSIR is not simply "data expansion" but "expansion in the correct direction," and points out that the "miss noise floor" is the true reason for eventual performance saturation.

### Loss & Training
Each round uses standard next-item prediction NLL, with no loss modification; hyperparameter grid: $\tau \in \{1,3,5,10,20,50,100\}$, $m \in \{5,10,20\}$, $p \in \{0,0.2,...,1\}$. Leave-one-out evaluation, K=10/20, reporting NDCG/Recall.

## Key Experimental Results

### Main Results
On 4 datasets × 3 backbones (SASRec / CL4SRec / HSTU), compared to 5 data augmentation/generation baselines for NDCG@10, Recall@10:

| Backbone | Dataset | Best Baseline (Recall@10) | +RSIR | Gain |
|----------|---------|---------------------------|-------|------|
| SASRec   | Beauty  | 0.0557 (DR4SR)            | **0.0594** | +6.64% |
| SASRec   | Sport   | 0.0495 (DR4SR)            | **0.0512** | +3.43% |
| CL4SRec  | Beauty  | 0.0590 (DR4SR)            | **0.0649** | +10.00% |
| HSTU     | Sport   | 0.0515 (DR4SR)            | **0.0531** | +3.11% |
| HSTU     | Yelp    | 0.0386 (Insertion)        | **0.0411** | +6.48% |

Both RSIR-FT (fine-tuning previous weights) and RSIR (training from scratch) consistently outperform all baselines, with the largest improvement (~10%) on CL4SRec.

### Ablation Study
Key ablation on Amazon-Sport + SASRec:

| Configuration         | NDCG@10 | Recall@10 | Notes                  |
|-----------------------|---------|-----------|------------------------|
| Base SASRec           | 0.0271  | 0.0474    | No augmentation        |
| RSIR-1 w/o fidelity   | 0.0273  | 0.0472    | 1 round, no QC, negligible gain |
| RSIR-1 w/ fidelity    | **0.0293** | **0.0512** | 1 round + QC           |
| RSIR-2 w/o fidelity   | 0.0209  | 0.0384    | Collapses in 2nd round |
| RSIR-2 w/ fidelity    | 0.0294  | 0.0517    | Continues to improve   |
| RSIR-3 w/o fidelity   | 0.0119  | 0.0210    | Catastrophic collapse  |
| RSIR-3 w/ fidelity    | 0.0298  | 0.0528    | Still improving        |

### Key Findings
- **Fidelity filtering is critical**: Removing it leads to complete collapse after 3 rounds (Recall drops from 0.0474 to 0.0210), confirming the risk of self-consuming model collapse.
- **Multi-round iteration yields "compound interest"**: On HSTU/Sport, Recall increases by 8% in the first round, accumulating to 14% after 3 rounds, but saturates after 5–8 rounds (consistent with the theoretical noise floor).
- **Weak-to-strong transfer is feasible**: Training a strong student on data generated by a weak teacher still yields a +1.95% improvement, indicating RSIR's benefit mainly comes from implicit regularization rather than the teacher's absolute capability.
- **Data density +342% / entropy also increases**: After 8 rounds, training set density quadruples, and ApEn (approximate entropy) rises; in contrast, Insertion adds data but reduces ApEn, proving RSIR adds "informative" rather than noisy data.
- **Optimal hyperparameters $p \approx 0.5$, moderate $\tau$**—validating the exploration/exploitation trade-off.

## Highlights & Insights
- **First to rigorously transfer "self-improvement" to recommender systems**, with comprehensive theoretical analysis (manifold tangent space gradient penalty + recursive error bound), elevating the long-vague "data augmentation" in recommendation to a principled level.
- **Fidelity check design is highly ingenious**: No external critic needed; directly reuses the same model's ranking distribution for self-verification—this "model as both generator and judge" symmetry is increasingly common in LLM self-training, but using ranking as the carrier is natural in recommendation.
- The "weak model teaches strong model" experiment echoes recent LLM weak-to-strong generalization findings, suggesting that in industrial scenarios, small models can cheaply generate curricula for production models, greatly reducing deployment costs.
- The overall approach is transferable to: sequential advertising, CTR, conversational recommendation, and any next-token prediction + user behavior sequence task.

## Limitations & Future Work
- **Saturation is inevitable**: The authors acknowledge that as iterations increase, benefits diminish and the noise floor emerges. Dynamically tightening $\tau$ or introducing adaptive filtering is a future direction.
- **Fidelity only considers top-$\tau$ ranking**, lacking finer-grained "user-intent drift" detection; may fail for cold-start or single-behavior users.
- Evaluation only covers small to medium-scale datasets (Amazon × 3 + Yelp), with the largest item set still relatively small; fidelity check on industrial-scale billion-item sets would require ANN acceleration.
- No direct comparison with LLM-as-teacher augmentation (e.g., LLMRec); to fully convince readers that "self-training truly does not need LLMs," this contrast should be added.

## Related Work & Insights
- **vs DR4SR / DiffuASR**: These methods rely on diffusion or learned generators for data creation, requiring extra models and expensive training; RSIR directly reuses the backbone, with zero external models, and outperforms them from the first round.
- **vs STaR / Self-Rewarding LLM**: Conceptually similar—model self-evaluation and self-training. RSIR replaces the "self-reward" in LLMs with recommendation-specific "user real sequence ranking consistency," and adds theoretical analysis.
- **vs Insertion / Reordering**: These heuristics cannot add new items, only reorder; RSIR can expand user interest boundaries and avoids noise via fidelity.
- **vs RSIDiff / STEP (self-training generative models / video)**: Cross-modal evidence for the universality of self-improving; this work is the first to apply this paradigm to recommendation.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleanly transfers the self-improvement paradigm from LLMs/diffusion to sequential recommendation, with a manifold tangent space theoretical explanation; more original than "yet another data augmentation."
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 3 backbones × 5 baselines, multi-round iteration, ablation, weak-to-strong, runtime analysis; lacks industrial-scale datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, explicit correspondence between theorems and experiments, though some experimental details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Continuously improves recommendation without relying on external LLMs, with simple engineering (just a break condition), and is deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/recommender/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)

</div>

<!-- RELATED:END -->
