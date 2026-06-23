---
title: >-
  [Paper Note] In-Context Compositional Q-Learning for Offline Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] ICQL reframes Q-learning in offline RL as an "in-context inference" problem—given a query state, it retrieves the top-k similar transitions from the offline dataset and uses a linear Transformer to **infer a local Q-function on the fly** from this local context. This bypasses the difficulty of fitting a single global Q
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: a7779eb6e5113b24
---
# In-Context Compositional Q-Learning for Offline Reinforcement Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZBbKLvH0w4](https://openreview.net/forum?id=ZBbKLvH0w4)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Offline RL  
**Keywords**: Offline Reinforcement Learning, In-Context Learning, Linear Transformer, Local Q-function, Compositional Value Estimation, Retrieval Augmentation  

## TL;DR
ICQL reframes Q-learning in offline RL as an "in-context inference" problem—given a query state, it retrieves the top-k similar transitions from the offline dataset and uses a linear Transformer to **infer a local Q-function on the fly** from this local context. This bypasses the difficulty of fitting a single global Q-network to all sub-tasks, achieving improvements of up to 16.4%, 8.8%, and 6.3% on D4RL Kitchen, MuJoCo, and Adroit, respectively.

## Background & Motivation
- **Background**: The core of offline RL is learning optimal policies from fixed datasets without environment interaction. The primary obstacle is Q-value overestimation caused by distribution shift. Mainstream solutions are divided into policy constraints (TD3+BC, DT), which pull the policy toward the behavior policy, and value regularization (CQL, IQL), which impose conservative penalties on OOD actions.
- **Limitations of Prior Work**: Both approaches essentially train **a single globally shared Q-function/policy** to cover the entire state space. However, many control tasks are naturally composed of multiple sub-tasks (e.g., "walking at speed" vs. "recovering from abnormal poses" in locomotion, or different stages like opening a light/cabinet in Kitchen). Value knowledge learned in one sub-task does not necessarily transfer to another.
- **Key Challenge**: Through t-SNE visualization, the paper highlights a counter-intuitive phenomenon—**geometrically similar state clusters may correspond to behaviors with entirely different semantics and long-range returns**. Global value fitting fails to capture such local structures when data is insufficient or exploration is impossible, and forced fitting introduces errors.
- **Goal**: Instead of seeking a value approximator that accurately covers the global space, the paper acknowledges the "compositional/local" nature of the value function and learns a **family** of local value functions that adapt flexibly to different state regions.
- **Key Insight**: **[In-context Inference-based Value Learning]** The problem of "estimating the Q-value of a state" is reformulated as "inferring a local linear Q-function online, given a small set of retrieved transitions near that state." By leveraging the in-context learning capability of linear Transformers, local value estimates are adaptively provided for each query without requiring sub-task labels or predefined structures.

## Method

### Overall Architecture
The inference-training pipeline of ICQL follows a link of "Retrieval $\rightarrow$ Context $\rightarrow$ Local Q": Given a query $(s_{\text{query}}, a_{\text{query}})$, it is embedded using a feature extractor $\phi$. The top-k most similar transitions are retrieved from the offline dataset $\mathcal{D}$ to form a local context set $\Omega^{d_k}_{s_{\text{query}}}$. This set is encoded into a "prompt matrix" and fed into a linear Transformer, which **simulates an in-context TD learning process during its forward pass**, directly outputting local weights $w^L_{s_{\text{query}}}$. A linear projection then yields the local Q-value. Training follows IQL using expectile regression for value learning and advantage-weighted regression for policy extraction; during inference, the policy can be deployed independently **without further retrieval**.

```mermaid
flowchart LR
    Q["Query (s_query, a_query)"] --> PHI["Feature Extractor φ"]
    PHI --> RET["Retrieve top-k similar<br/>transitions from D"]
    RET --> CTX["Local Context Ω^dk<br/>(Prompt Matrix Z0)"]
    CTX --> LT["Linear Transformer<br/>= Implicit In-Context TD"]
    LT --> WQ["Local Weight w^L → Local Q Function"]
    WQ --> IQL["IQL-style Critic / Policy Update"]
```

### Key Designs

**1. Local Q-function: Neighborhood definition instead of global fitting.** The paper stops assuming the existence of a global weight vector and instead defines a local linear Q-function for each state $s$ determined by its neighborhood. Specifically, for a set of transitions $\Omega^{(d,\bar d)}_s$ with similar states and transitions (satisfying $\|s_i-s\|_2^2\le d^2$ and $\|s_i'-s_i\|_2^2\le\bar d^2$), there exists an optimal local weight $w^*_s$ such that $\hat Q^{\Omega}_s(\bar s,\bar a) \triangleq w^{*\top}_s \phi(\bar s,\bar a)$ approximates the true Q-value within that neighborhood (with approximation error $\le\varepsilon^s_{\text{approx}}$). This decomposes "global value approximation" into "a family of local linear approximations valid for specific state regions." Since the neighborhood radius $d$ depends on data density and cannot be tuned directly, the paper uses the retrieval set size $k$ to control "locality" in practice.

**2. Retrieval Mechanism: Composing local context from similar transitions.** Given a query state, the default is **State-Similar Retrieval**—selecting the k transitions with the smallest $\ell_2$ distance to $s_{\text{query}}$: $\Omega_{s_{\text{query}}} \triangleq \{(s_i,a_i,r_i,s_i',a_i')\in\mathcal{D} \mid s_i\in\arg\text{top-}k(-\|s_{\text{query}}-s_i\|_2^2)\}$. The paper also discusses two other strategies: random retrieval (preserves diversity but provides weak local info) and similar-plus-high-return retrieval (filters for high-quality transitions). The retrieval size k is equivalent to implicitly controlling the neighborhood radius $d_k$, serving as the bridge between the "theoretical local domain" and the "engineering context window."

**3. In-Context Inference = Linear Transformer implicitly running TD.** This is the core of ICQL. The retrieved context is constructed into a prompt matrix $Z_0$ (where each column contains transition features $\phi_i$, discounted next-step features $\gamma\phi_i'$, and reward $r_i$, with the query in the last column). This is fed into an L-layer linear Transformer using linear attention of the form $\text{LinAttn}(Z;P,G)=PZM(Z^\top G Z)$. The paper proves theoretically that under carefully constructed weight matrices $P_\ell, G_\ell$, **each layer of linear attention is exactly equivalent to one step of SARSA/TD update on the local weight**:

$$w^{l+1}_{s} = w^l_{s} + \alpha\Big(r + \gamma\, w_s^\top\phi(s',a') - w_s^{l\top}\phi(s,a)\Big)\phi(s,a)$$

Thus, the L-layer forward pass is equivalent to running L steps of in-context TD learning on the retrieved local data. Extracting $w^L_{s_{\text{query}}}$ at the end yields $\hat Q(s_{\text{query}}, a_{\text{query}}\mid\Omega^{d_k}) = w^{L\top}_{s_{\text{query}}}\phi(s_{\text{query}}, a_{\text{query}})$. In other words, the Transformer does not "memorize" values but **temporarily trains a local value estimator for each query** during the forward pass.

**4. IQL Training + Theoretical Near-Optimality.** The critic fits the local Q using expectile regression: $L_{\text{critic}}=\mathbb{E}_{\mathcal{D}}[\rho_\tau(\hat Q(s,a\mid\Omega^{d_k}_s)-y)]$, where $y=r+\gamma V(s'\mid\Omega^{d_k}_{s'})$. The policy is extracted via advantage-weighted regression $L_{\text{policy}}=\mathbb{E}[\exp(\beta(\hat Q-V))\log\pi(a\mid s)]$. Theoretically, under the assumptions of "local Q linear approximability" and "retrieval set coverage $\ge\sigma$" of the ideal local domain, the performance difference of the greedy policy is bounded: $J(\pi^*)-J(\pi)\le \frac{2}{1-\gamma}\mathbb{E}[\varepsilon^s_{\text{approx}}(1+B_\phi)+CB_\phi\sqrt{(d+\log(1/\delta))/(\sigma|\Omega^{d_k}_s|)}]$. This clearly decomposes the error into "approximation error" and "weight estimation error," the latter of which decays as context coverage increases.

## Key Experimental Results

### Main Results (D4RL, mean of 5 random seeds)

| Task Suite | BC | DT | TD3+BC | CQL | IQL | **Ours (ICQL)** | Gain |
|---|---|---|---|---|---|---|---|
| MuJoCo (Mean of 9) | 51.9 | 58.8 | 62.9 | 74.0 | 72.4 | **80.6** | **+8.8%** |
| Adroit (Mean of 6) | 17.5 | 27.9 | 24.2 | 15.5 | 33.2 | **35.3** | **+6.3%** |
| Kitchen (Mean of 3) | 51.5 | 55.8 | 52.6 | 48.2 | 52.8 | **66.8** | **+16.4%** |

Representative items: Walker2d-Medium-Replay 81.9 (Prev. SOTA CQL 77.2), HalfCheetah-Medium-Expert 89.1 (Prev. SOTA IQL 83.4), Door-Human 17.1 (IQL 9.8, +73%), Kitchen-Complete 79.3 (BC 65.0, +22%).

### Ablation Study (Retrieval Strategy, Excerpt)

| Dataset | Random | State-Similar | Similar+HighReward |
|---|---|---|---|
| Walker2d-Medium | 78.1 | 80.3 | **83.9** |
| Walker2d-Medium-Replay | 67.5 | **81.9** | 75.1 |
| Hopper-Medium-Replay | 81.0 | **96.4** | 90.8 |
| Pen-Human | 75.1 | **85.6** | 84.8 |
| Kitchen-Complete | 70.0 | **79.3** | 71.3 |

Other ablations: ① Layers (= in-context TD steps): Score increases with layers (4 $\rightarrow$ 20) in MuJoCo, confirming that more layers lead to more sufficient in-context value learning. ② Context length: Among {10, 20, 30, 40}, **20 is optimal**; excessive length increases the distance between query and context, breaking "locality" and introducing noise.

### Key Findings
- **Accurate Q Estimation as Performance Driver**: On Walker2d-Medium, the similarity between ICQL's Q-estimation distribution and online SAC's reaches 0.69, compared to 0.29 for IQL—indicating that local value modeling provides Q-values closer to "ground truth" on noisy data.
- **Maximized Gains in Compositional Tasks**: The most significant improvement (+16.4%) occurs in multi-stage long-range tasks like Kitchen, directly supporting the motivation that "value functions are inherently compositional."
- **Honest Inclusion of Failure Cases**: ICQL lags behind some baselines on Hammer-Human, which the paper attributes to small dataset size and large distances between query and retrieved states, making in-context learning difficult.

## Highlights & Insights
- **Paradigm Reformulation**: Shifting from "training a Q-network" to "inferring a local Q-function online for each query" is a clean application of in-context learning to value estimation (rather than action generation like DT). The paper notes that prior in-context RL works focused on policies, not value estimation.
- **Alignment of Theory and Mechanism**: The equivalence between linear Transformer layers and TD update steps provides a provable mechanical explanation for why Transformers can perform value inference, moving beyond a "black box."
- **Locality as a Prior**: Explicitly encoding the "local structure of value" using retrieval radius k converts a difficult continuous hyperparameter $d$ into an actionable discrete k, which is engineering-friendly and directly linked to theoretical coverage assumptions.

## Limitations & Future Work
- **Dependence on Retrieval Quality**: The core theoretical assumption for near-optimality is that retrieval set coverage $\ge\sigma$. When data is sparse or queries are outliers (e.g., Hammer-Human), insufficient coverage leads to performance drops, making the method sensitive to dataset density.
- **Ceiling of Local Linear Approximation**: The local Q assumes linear approximability; approximation error $\varepsilon_{\text{approx}}$ may become non-negligible in local domains with highly non-linear value structures.
- **Additional Computational Overhead**: Each value estimate requires retrieval plus an L-layer Transformer forward pass. While described as "moderate," it is heavier than a single network forward pass (mitigated by independent policy deployment during inference).
- **Future Directions**: Adaptive selection of k/context length, stronger feature extractors, and developing "similar+high-reward" retrieval into a learnable retriever are potential extensions.

## Related Work & Insights
- **Offline RL**: CQL (conservative Q penalty), IQL (expectile + AWR, used as the framework base), TD3+BC, ReBRAC, etc., all share global value/policy modeling. ICQL provides a contrast through local estimation.
- **In-context Learning in RL**: Decision Transformer, Gato, Algorithm Distillation, DPT, PreDeToR, etc., mostly focus on trajectory modeling or action generation. ICQL highlights its status as the first to use linear attention for **compositional value estimation**.
- **Theoretical Foundations**: The theory of linear Transformers implementing in-context (TD) learning (Von Oswald 2023, Wang 2025b) combined with coverage analysis from non-parametric regression is leveraged here to prove near-optimality.
- **Inspiration**: The pipeline of "Retrieval $\rightarrow$ In-context Inference of local model" has potential beyond Q-learning for any prediction task where global models struggle but local structures are clear (e.g., local dynamics or reward modeling).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reformulating offline RL value learning as in-context inference and proving the layer-wise TD equivalence is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three D4RL task suites, 5 baselines, and multi-dimensional ablations (retrieval/layers/length), with transparent reporting of failure cases. Lacks validation on larger/pixel-based tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation-theory-method-experiment closed loop; t-SNE visualization strongly supports core assumptions.
- **Value**: ⭐⭐⭐⭐ Significant gains in compositional long-range tasks, opening a theoretically guaranteed new path for "retrieval-augmented in-context value estimation."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](scalable_in-context_q-learning.md)
- [\[ICLR 2026\] 3D-aware Disentangled Representation for Compositional Reinforcement Learning](3d-aware_disentangled_representation_for_compositional_reinforcement_learning.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Adaptive Feature Fusion](offline_reinforcement_learning_with_adaptive_feature_fusion.md)
- [\[ICLR 2026\] Context and Diversity Matter: The Emergence of In-Context Learning in World Models](context_and_diversity_matter_the_emergence_of_in-context_learning_in_world_model.md)

</div>

<!-- RELATED:END -->
