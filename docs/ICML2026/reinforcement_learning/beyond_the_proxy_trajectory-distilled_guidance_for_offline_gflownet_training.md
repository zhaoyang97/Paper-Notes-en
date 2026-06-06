---
title: >-
  [Paper Note] Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training
description: >-
  [ICML2026][Reinforcement Learning][GFlowNet] TD-GFN is proposed as an offline GFlowNet training framework that eliminates reliance on proxy reward models. It extracts edge-level rewards from offline trajectories via inve…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "GFlowNet"
  - "Offline Training"
  - "Inverse Reinforcement Learning"
  - "DAG Pruning"
  - "Proxy-free Rewards"
date: 2026-05-08
content_hash: b95466197a08d353
---

# Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training

**Conference**: ICML2026  
**arXiv**: [2505.20110](https://arxiv.org/abs/2505.20110)  
**Code**: https://github.com/Chenruishuo/TD-GFN  
**Area**: reinforcement_learning  
**Keywords**: GFlowNet, Offline Training, Inverse Reinforcement Learning, DAG Pruning, Proxy-free Rewards  

## TL;DR
TD-GFN is proposed as an offline GFlowNet training framework that eliminates reliance on proxy reward models. It extracts edge-level rewards from offline trajectories via inverse reinforcement learning (IRL), then indirectly guides policy learning through DAG pruning and prioritized backward sampling. This ensures gradient updates rely solely on ground-truth terminal rewards, significantly outperforming existing baselines in tasks such as molecular design and sequence generation.

## Background & Motivation

**Background**: Generative Flow Networks (GFlowNets) are a class of generative models designed to sample combinatorial objects on DAG-structured environments, aiming to sample terminal nodes with probabilities proportional to rewards. They have seen extensive applications in molecule discovery, protein sequence design, and combinatorial optimization. However, in many practical scenarios, querying the reward function is extremely expensive (e.g., wet-lab experiments, human evaluation), requiring GFlowNet training from pre-collected offline datasets.

**Limitations of Prior Work**: The standard paradigm involves training a proxy reward model on the offline dataset, which the GFlowNet then queries for reward signals. However, building reliable proxies requires massive diverse data and domain expertise. More critically, when GFlowNets generate out-of-distribution (OOD) samples and query the proxy, estimation errors propagate through gradients, damaging policy quality. Existing proxy-free methods like RO-GFlowNet and COFlowNet attempt direct learning from offline trajectories but impose only coarse-grained constraints to align the policy with the data distribution, limiting generalization and exploration efficiency.

**Key Challenge**: Offline GFlowNet training faces a fundamental dilemma: proxy models introduce error propagation, while proxy-free methods' coarse constraints restrict exploration. Contributions of different edges in a DAG to learning an effective policy are unequal (e.g., critical edges leading to high-reward terminals vs. edges leading to low-reward ones), yet current methods fail to exploit these structural differences.

**Goal**: To design a proxy-free offline GFlowNet training framework capable of (1) extracting fine-grained transition-level guidance signals from trajectories and (2) effectively guiding the policy to explore high-reward regions without relying on proxy rewards for gradient updates.

**Key Insight**: Leveraging the theoretical equivalence between GFlowNet training and entropy-regularized RL, Maximum Causal Entropy IRL is applied to a rebalanced offline dataset to extract importance scores (edge rewards) for each transition. These edge rewards do not predict terminal rewards but rather quantify the structural contribution of transitions to policy learning.

**Core Idea**: Distill edge-level rewards from trajectories using IRL to indirectly guide the policy via DAG pruning and prioritized backward sampling. Since gradient updates rely only on ground-truth terminal rewards, the framework achieves efficient exploration and error isolation simultaneously.

## Method

### Overall Architecture
The training of TD-GFN consists of two stages: (1) **Edge Reward Extraction Stage**—after rebalanced sampling of the offline dataset, an adversarial IRL (based on GAIL) is used to learn an edge-level reward function $R_E(s, s')$, quantifying each edge's contribution. (2) **Policy Training Stage**—edge rewards are used to perform DAG pruning to remove inefficient transitions, followed by prioritized backward sampling to construct training trajectories. Finally, the GFlowNet policy is trained using these trajectories and ground-truth terminal rewards from the dataset. Gradient updates depend only on ground-truth rewards; edge rewards are used solely for sampling guidance.

### Key Designs

1.  **IRL-based Edge Reward Extraction**:
    - **Function**: Learns transition-level reward signals $R_E(s, s')$ from offline trajectories to quantify the importance of each DAG edge.
    - **Mechanism**: First, the dataset is rebalanced by sampling trajectories proportional to terminal rewards $P(\tau) \propto R(s_T)$ to construct an approximate expert distribution. Then, the GAIL framework is employed for adversarial training: a discriminator $D_\phi$ distinguishes rebalanced data from transitions generated by an imitation policy $\pi_\psi$. Unbiased edge rewards are extracted as $R_E(s, s') = \log D_\phi(s, s') - \log(1 - D_\phi(s, s'))$. Theoretical analysis shows that in the population limit, $R_E$ recovers the log conditional inflow scores (log of the canonical backward policy $\mathcal{P}_B^*$), and an estimation error $\|R_E - r^*\|_\infty \le \varepsilon$ guarantees the TV distance of the backward policy does not exceed $\frac{1}{2}(e^{2\varepsilon} - 1)$.
    - **Design Motivation**: Fundamentally different from proxy reward models, edge rewards capture structural preferences rather than predicting terminal rewards; thus, approximation errors do not damage the policy via gradient propagation. Experiments show edge rewards generalize well to unobserved transitions.

2.  **Reward-based DAG Pruning**:
    - **Function**: Removes inefficient transition edges to construct a more compact action space.
    - **Mechanism**: A batch of state-action pairs is sampled using the imitation policy $\pi_\psi$ to calculate the edge reward distribution. If an edge reward is below a threshold $R_E(s, s') < \text{mean}(\mathcal{D}_{R_E}) - K \cdot \text{std}(\mathcal{D}_{R_E})$ (where $K$ is a hyperparameter), it is pruned from the DAG. Isolated nodes disconnected from the root are subsequently removed. Theoretical guarantee: terminal nodes with high bottleneck scores $\beta(x) > \tau + \varepsilon$ remain reachable after pruning, and the proportional reward relationship among surviving terminals is preserved.
    - **Design Motivation**: Direct gradient shaping using edge rewards would amplify function approximation errors. Pruning serves as a robust, indirect utilization—concentrating policy attention on high-value regions while avoiding error propagation.

3.  **Prioritized Backward Sampling**:
    - **Function**: Constructs high-quality training trajectories on the pruned DAG.
    - **Mechanism**: Starting points $x$ are sampled from terminal nodes in the dataset proportional to their rewards. A full trajectory is then generated by recursively backtracking to the root using a backward policy defined by edge rewards: $\mathcal{P}_B(s_t | s_{t+1}) = \exp\{R_E(s_t, s_{t+1})\} / \sum_{(s, s_{t+1}) \in E'} \exp\{R_E(s, s_{t+1})\}$. 
    - **Design Motivation**: Reinforces GFlowNet's core inductive bias (allocating sampling resources proportional to rewards) while ensuring gradient updates rely only on ground-truth rewards recorded in the dataset, isolating potential errors from the IRL stage.

### Loss & Training
The policy can be trained using any GFlowNet objective (FM, TB, SubTB, DB). Experiments indicate that TD-GFN's pruning and sampling modules are orthogonal to the specific objective and provide consistent improvements. The main experiments utilize the Flow Matching (FM) objective for fair comparison with COFlowNet.

## Key Experimental Results

### Main Results

| Task | Method | Core Metric | Convergence Speed |
|------|------|----------|----------|
| Hypergrid $8^4$ | TD-GFN | Lowest L1 Error, 16/16 modes | <5,000 state visits (6x speedup) |
| Hypergrid $8^4$ | COFlowNet | Second lowest L1 Error | >10,000 state visits |
| Bio-sequences (AMP) | TD-GFN | Highest Top-100 Reward & Diversity | — |
| Bio-sequences (AMP) | Proxy-GFN | Reward lower than TD-GFN | — |

| Method | Reward-10 ↑ | Reward-100 ↑ | Reward-1000 ↑ | Num. Traj to Conv ↓ |
|------|-------------|--------------|---------------|-------------|
| Oracle-GFN (Ref) | 7.718 | 7.408 | 6.801 | 44.1×10⁴ |
| Proxy-GFN | 7.625 | 7.281 | 6.636 | 43.7×10⁴ |
| QM-COFlowNet | 7.611 | 7.296 | 6.638 | 4.4×10⁴ |
| FM-COFlowNet | 7.582 | 7.201 | 6.485 | 5.8×10⁴ |
| Dataset-GFN | 7.550 | 7.198 | 6.474 | 6.0×10⁴ |
| **TD-GFN** | **7.733** | **7.450** | **6.810** | **2.7×10⁴** |

### Ablation Study

| Setup | TD-GFN Performance | Note |
|----------|------------|------|
| Mixed Dataset (with random trajectories) | Maintains optimal L1 Error | Robust under noisy data |
| 1/10 Dataset (only 150 trajectories) | Still outperforms baselines | Effective in data-scarce scenarios |
| Median Behavior Policy (half-trained) | Faster convergence | Robust to suboptimal collection policies |
| Bad Behavior Policy (inverted reward) | Significantly outperforms baselines | Exceptional performance under extreme degradation |
| Molecule Diversity (Tanimoto modes) | 1.5-2× strongest baseline | Pruning does not cause overfitting; enhances exploration |
| Different GFN Objectives (TB/SubTB/DB) | Consistent gains | Orthogonal to specific objectives |

## Highlights & Insights
- **Difference between edge and proxy rewards**: Edge rewards capture the structural preference of transitions rather than predicting terminal rewards, allowing indirect use without introducing gradient error propagation.
- On molecular design tasks, TD-GFN matches the performance of online GFNs using real Oracles while using only 1/20 of the trajectories.
- Despite DAG pruning, TD-GFN discovers 1.5-2x the number of high-reward modes compared to baselines, suggesting that "reducing action space" and "enhancing exploration" are not contradictory.
- Rebalancing strategy is inherently effective: even simple rebalanced GAIL imitation learning outperforms Dataset-GFN trained directly on the dataset.

## Limitations & Future Work
- The IRL stage requires additional training of a discriminator and an imitation policy, increasing computational overhead.
- The pruning threshold $K$ is a hyperparameter that needs adjustment for different tasks.
- Theoretical guarantees depend on the approximation error $\varepsilon$ of edge rewards, which is difficult to measure directly in practice.
- Validation is limited to discrete DAG environments; extensions to continuous state spaces or non-DAG structures remain to be explored.

## Related Work & Insights
- **COFlowNet / RO-GFlowNet**: Existing proxy-free offline GFlowNet methods apply coarse-grained constraints; TD-GFN significantly outperforms them via fine-grained edge-level guidance.
- **GAIL / MaxEntIRL**: TD-GFN's edge reward extraction is built directly upon the Maximum Causal Entropy IRL framework.
- **GFlowNet-RL Equivalence (Tiapkin et al., 2024)**: Transforming GFlowNet training into entropy-regularized RL provides the theoretical foundation for this methodology.
- **Insight**: The paradigm of "distilling structural guidance from trajectories + indirect usage to isolate errors" may have broad applicability in other generative models requiring training from offline data.

## Rating
- Novelty: 9/10 — Introducing IRL to offline GFlowNet training is a brand-new paradigm; the indirect use of edge rewards is elegantly designed.
- Experimental Thoroughness: 9/10 — Very comprehensive, covering three benchmarks, various data quality settings, multiple GFN objectives, and theoretical guarantees.
- Writing Quality: 8/10 — Clear structure, well-defined motivation, and tight integration between theory and experiments.
- Value: 8/10 — Establishes a new SOTA paradigm for offline GFlowNets, with significant implications for practical applications like molecular design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas](beyond_scalar_rewards_dense_feedback_for_llm_policy_synthesis_in_sequential_soci.md)
- [\[AAAI 2026\] Know your Trajectory -- Trustworthy Reinforcement Learning Deployment through Importance-Based Trajectory Analysis](../../AAAI2026/reinforcement_learning/know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)
- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)

</div>

<!-- RELATED:END -->
