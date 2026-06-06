---
title: >-
  [Paper Note] Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control
description: >-
  [ICML 2026][Recommender Systems][Sequential Recommendation] RSIR allows sequential recommendation models to use their own predictive capabilities to generate new synthetic user interaction sequences, train a new model…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Self-Training"
  - "Data Sparsity"
  - "Fidelity Control"
  - "Implicit Regularization"
date: 2026-05-08
content_hash: ef6a2d1fc695cf9a
---

# Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control

**Conference**: ICML 2026  
**arXiv**: [2602.15659](https://arxiv.org/abs/2602.15659)  
**Code**: https://github.com/USTC-StarTeam/RSIR  
**Area**: Recommender Systems / Data Augmentation / Self-Training  
**Keywords**: Sequential Recommendation, Self-Training, Data Sparsity, Fidelity Control, Implicit Regularization

## TL;DR
RSIR allows sequential recommendation models to use their own predictive capabilities to generate new synthetic user interaction sequences, train a new model, and filter out samples that deviate from the user preference manifold using a rank-based "fidelity check," thereby preventing self-consuming model collapse. It achieves stable Gains of 4–11% in NDCG/Recall across 4 datasets × 3 mainstream backbones and theoretically proves that this process is equivalent to implicit regularization along the tangent space of the user preference manifold.

## Background & Motivation
**Background**: Sequential recommendation primarily addresses problems by scaling data and models. However, any given user only interacts with an extremely small proportion of the platform's catalog, making interaction signals naturally sparse. This results in a rugged loss landscape, models converging to sharp minima, and poor generalization.

**Limitations of Prior Work**: (1) Data augmentation methods (Reordering / Insertion / item masking / cropping) only perturb existing data and do not produce new "high-fidelity user trajectories," yielding limited effects; (2) Data generation methods (DiffuASR, DR4SR) can create new sequences but rely on diffusion models or learning auxiliary generators, which are computationally expensive; (3) Leveraging LLMs as teachers for data expansion outforces performance bottlenecks to "large enough external models," which is uncontrollable during deployment and suffers from distribution mismatch.

**Key Challenge**: Closed-loop self-training (generating data to train oneself) has been proven effective in LLMs and diffusion models but is highly susceptible to model collapse — where model biases and errors are amplified by the synthetic data, leading to performance crashes within a few iterations. The key is how to balance "self-generated data" with "avoiding error accumulation."

**Goal**: (1) Enable recommendation models to bootstrap themselves like STaR or Self-Rewarding LLMs without external teachers or labels; (2) Design a reliable "fidelity filtering" mechanism to prevent synthetic data from drifting off the user preference manifold; (3) Theoretically explain why this recursive process does not collapse but instead provides regularization.

**Key Insight**: The authors view "self-improvement" as a kind of data-driven implicit regularization — only accepting perturbations near the tangent space of real user interests as new data is equivalent to imposing a gradient penalty along the manifold direction on the loss landscape, forcing optimization to converge to flatter minima.

**Core Idea**: Use the current model to generate synthetic sequences and use the same model as a "ranking judge" to filter out deviated steps. This combination is inserted into a classic SFT loop and rolled recursively over multiple rounds.

## Method

### Overall Architecture
RSIR in iteration $k$ consists of 4 steps: (1) Train the model $f_{\theta_k}$ on the current dataset $D_k$ (next-item prediction); (2) Use $f_{\theta_k}$ to generate $m$ synthetic sequences $D'_{k+1}$ for each user — starting from a random prefix of the user's real history and expanding autoregressively; (3) Combine to obtain $D_{k+1} = D_k \cup D'_{k+1}$; (4) Train $f_{\theta_{k+1}}$ from scratch (or fine-tune the previous round's model). Two core mechanisms are used during generation: bounded exploration (mixed candidate pool) and fidelity-based quality control (ranking verification).

### Key Designs

1. **Bounded Exploration Mixed Candidate Pool**:
    - **Function**: Decides which item pool to sample the next item from at each generation step, balancing "exploiting known interests" with "exploring new interests."
    - **Mechanism**: At each step, an item is sampled with probability $p$ from the user's history $s_u$ and with probability $1-p$ from the global item set $I$, forming a candidate pool $\mathcal{C}_t \sim p \cdot \mathrm{Sample}(s_u) + (1-p) \cdot \mathrm{Sample}(I)$. The model then performs top-$k$ sampling only over $\mathcal{C}_t$. Empirically, $p \approx 0.5$ is optimal — pure exploitation ($p=1$) merely reorders known items and cannot expand interests, while pure exploration ($p=0$) easily drifts off-track and is filtered by quality control.
    - **Design Motivation**: Conventional autoregressive generation on large vocabularies would explode the generation space; introducing historical bias maintains controllability while providing the ability to "extend known interests."

2. **Fidelity-Based Quality Control**:
    - **Function**: Immediately "probes" whether each generated candidate item still ranks the user's real future items highly; if not, the sequence is terminated.
    - **Mechanism**: Define $S_{tgt} = s_u \setminus S_{ctx}'$ as the items in the user's real sequence not yet used. If $\exists i_j \in S_{tgt}$ such that $\mathrm{Rank}_{f_{\theta_k}}(i_j | S_{ctx}') \leq \tau$, the new item is accepted and expansion continues; otherwise, the sequence is terminated immediately. This ensures the generated trajectory remains compatible with the real user interest manifold.
    - **Design Motivation**: The authors prove that a stricter $\tau$ suppresses the "fidelity false positive rate" $\tilde{p}_k$, making the recursive error bound $\mathcal{E}(\theta_{k+1}) \leq (1-\lambda)\mathcal{E}_0 + \lambda[(1-\tilde{p}_k)\rho \mathcal{E}(\theta_k) + \tilde{p}_k \mathcal{E}_{\max}]$ satisfy the contraction condition, avoiding model collapse.

3. **Manifold Tangential Gradient Penalty (Theoretical Explanation)**:
    - **Function**: Reinterprets the "filter + generate" loop as a form of implicit regularization, providing the method with a theoretical footing.
    - **Mechanism**: Accepted perturbations can only move along the tangent space of the user preference manifold $\mathcal{M}$, which is equivalent to adding a regularization term $\Omega(\theta) \propto \|\mathcal{P}_\mathcal{M} \nabla_s f_\theta\|^2$ to the original loss. This term specifically penalizes the gradient magnitude along the manifold direction, causing the solution to converge to "flat valleys" parallel to the user's real manifold.
    - **Design Motivation**: Explains why RSIR is not just "expanding data" but "expanding data in the right direction," and identifies the "noise floor of false positives" as the reason performance eventually saturates.

### Loss & Training
Each round uses standard next-item prediction NLL without modifications to the loss. Hyperparameter grid: $\tau \in \{1,3,5,10,20,50,100\}$, $m \in \{5,10,20\}$, $p \in \{0,0.2,...,1\}$. Leave-one-out evaluation; NDCG/Recall are reported at K=10/20.

## Key Experimental Results

### Main Results
4 datasets × 3 backbones (SASRec / CL4SRec / HSTU) compared against 5 data augmentation/generation baselines for NDCG@10 and Recall@10:

| Backbone | Dataset | Best Baseline (Recall@10) | +RSIR | Gain |
|----------|---------|---------------------------|-------|------|
| SASRec | Beauty | 0.0557 (DR4SR) | **0.0594** | +6.64% |
| SASRec | Sport | 0.0495 (DR4SR) | **0.0512** | +3.43% |
| CL4SRec | Beauty | 0.0590 (DR4SR) | **0.0649** | +10.00% |
| HSTU | Sport | 0.0515 (DR4SR) | **0.0531** | +3.11% |
| HSTU | Yelp | 0.0386 (Insertion) | **0.0411** | +6.48% |

Both RSIR-FT (fine-tuning old weights) and RSIR (training from scratch) consistently outperform all baselines, with the largest improvement of approximately 10% seen on CL4SRec.

### Ablation Study
Key ablation on Amazon-Sport + SASRec:

| Configuration | NDCG@10 | Recall@10 | Note |
|---------------|---------|-----------|------|
| Base SASRec | 0.0271 | 0.0474 | No augmentation |
| RSIR-1 w/o fidelity | 0.0273 | 0.0472 | 1 round, no QC, almost no gain |
| RSIR-1 w/ fidelity | **0.0293** | **0.0512** | 1 round + QC |
| RSIR-2 w/o fidelity | 0.0209 | 0.0384 | Collapses by 2nd round |
| RSIR-2 w/ fidelity | 0.0294 | 0.0517 | Continues to rise |
| RSIR-3 w/o fidelity | 0.0119 | 0.0210 | Catastrophic collapse |
| RSIR-3 w/ fidelity | 0.0298 | 0.0528 | Still increasing |

### Key Findings
- **Fidelity filtering is the lifeline**: Removing it leads to complete collapse within 3 rounds (Recall dropping from 0.0474 to 0.0210), confirming the collapse risk of self-consuming models.
- **Multiple iterations produce "compound interest"**: HSTU on Sport saw an +8% Recall in the first round, accumulating to +14% after 3 rounds, but gradually saturated after 5–8 rounds (consistent with the noise floor in the theory).
- **Weak-to-strong transfer is feasible**: Training a strong student on data generated by a weak teacher also yielded a +1.95% gain, suggesting RSIR benefits primarily from implicit regularization rather than the absolute capability of the teacher.
- **Data density +342% / increased information entropy**: After 8 rounds, the training set density increased more than fourfold, and ApEn (Approximate Entropy) also rose. In contrast, while Insertion adds data, its ApEn decreases, proving RSIR adds "information-rich" content rather than just noise.
- **Hyperparameters $p \approx 0.5$ and moderate $\tau$ are optimal** — confirming the exploration/exploitation trade-off.

## Highlights & Insights
- **The first work to seriously transfer "self-improvement" to recommender systems**, accompanied by a complete theoretical analysis (manifold tangent space gradient penalty + recursive error bounds). It elevates "data augmentation" in the recommendation community from an empirical heuristic to a principled level.
- **The design of the fidelity check is highly ingenious**: It does not require an external critic; it directly reuses the same model's ranking distribution for self-verification. This symmetric "model as both generator and judge" structure is increasingly common in LLM self-training but naturally fits recommendation through ranking.
- The experiment showing "weak models can teach strong ones" echoes recent weak-to-strong generalization findings in LLMs, suggesting that in industrial scenarios, small models can be used to cheaply generate a curriculum for production models, significantly reducing deployment costs.
- The overall philosophy is transferable to sequential advertising, CTR, conversational recommendation, and any task involving next-token prediction with user behavior sequences.

## Limitations & Future Work
- **Saturation is inevitable**: The authors acknowledge that as the number of iterations increases, the benefit diminishes and the noise floor emerges. Dynamic tightening of $\tau$ or introducing adaptive filtering are future directions.
- **Fidelity only considers top-$\tau$ ranking**, lacking fine-grained "user-intent drift" detection; it might be less effective for cold-start users or users with very narrow behaviors.
- Evaluation only covers small-to-medium datasets (Amazon × 3 + Yelp), where the item set is still relatively small. Industrial-scale items (hundreds of millions) would require ANN acceleration for the fidelity check.
- No direct comparison with LLM-as-teacher augmentation (e.g., LLMRec) was reported; convincing readers that "self-training truly negates the need for LLMs" requires this control group.

## Related Work & Insights
- **vs. DR4SR / DiffuASR**: Those methods rely on diffusion or learning a generator to create data, which requires extra models and expensive training. RSIR reuses the backbone directly, requires zero external models, and outperforms them starting from the first round.
- **vs. STaR / Self-Rewarding LLM**: Philosophically linked — model self-evaluation and self-training. RSIR replaces the "self-reward" in LLMs with recommendation-specific "ranking consistency of real user sequences" and adds theoretical analysis.
- **vs. Insertion / Reordering**: Those heuristics cannot add new items but only reorder them; RSIR can expand the boundaries of user interests while avoiding noise through fidelity control.
- **vs. RSIDiff / STEP (Self-training in generative models/video)**: Cross-modal evidence for the universality of self-improvement; this paper is the first to ground this paradigm in recommendation.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleanly transfers the self-improvement paradigm from LLMs/Diffusion to sequential recommendation with manifold-based theoretical explanations. Higher originality than "just another data augmentation."
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 3 backbones × 5 baselines. Comprehensive rounds, ablation, weak-to-strong, and runtime analysis. Lacks industrial-scale datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, explicit correspondence between theorems and experiments, though some experimental details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Consistently improves recommendation performance without relying on external LLMs, and the engineering implementation is simple (just a break condition), making it deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/recommender/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[ICML 2026\] Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System](incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)

</div>

<!-- RELATED:END -->
