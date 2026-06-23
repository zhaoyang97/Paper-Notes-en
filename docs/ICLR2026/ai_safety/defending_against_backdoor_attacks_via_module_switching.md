---
title: >-
  [Paper Note] Defending against Backdoor Attacks via Module Switching
description: >-
  [ICLR 2026][AI Safety][Paper Note] Focusing on the post-training scenario where "suspicious pre-trained models are obtained without training data or trigger priors," this paper proposes **Module Switching Defense (MSD)**. By interchanging weights of specific layers or modules between multiple isomorphic models, MSD disrupts the "shortcut paths" that bac
tags:
  - ICLR 2026
  - AI Safety
date: 2026-05-08
content_hash: 0e7d5f1ddbcb5ab5
---
# Defending against Backdoor Attacks via Module Switching

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ieCOL2YAqv](https://openreview.net/forum?id=ieCOL2YAqv)  
**Code**: [https://github.com/weijun-l/module-switching-defense](https://github.com/weijun-l/module-switching-defense)  
**Area**: AI Security / Backdoor Defense / Model Merging  
**Keywords**: Backdoor Attack, Backdoor Defense, Model Merging, Module Switching, Evolutionary Search, Post-Training Defense  

## TL;DR
Focusing on the post-training scenario where "suspicious pre-trained models are obtained without training data or trigger priors," this paper proposes **Module Switching Defense (MSD)**. By interchanging weights of specific layers or modules between multiple isomorphic models, MSD disrupts the "shortcut paths" that backdoors rely on. The authors theoretically prove that its backdoor deviation is strictly higher than Weight Averaging (WAG). An evolutionary search is employed to find optimal switching strategies, significantly reducing the Attack Success Rate (ASR) using only two models and 20–50 clean validation samples.

## Background & Motivation
- **Background**: Backdoor attacks inject triggers into a small portion of training data, causing models to perform normally on clean samples but execute malicious behaviors when triggers appear. With the popularity of the "post-training paradigm" (HuggingFace model reuse, MoE experts, one-shot federated learning), users often adopt models from unknown sources, where the lack of transparency in training data and processes provides opportunities for attackers.
- **Limitations of Prior Work**: Traditional defenses often assume access to training resources—original data for filtering, trusted auxiliary sets for fine-tuning, or the optimization process for trigger inversion. These are unavailable in post-training scenarios. Emerging **model merging** defenses (e.g., Weight Averaging (WAG), DAM), while resource-efficient, have three constraints: ① they usually require 3–6 homologous models to effectively suppress backdoors, posing a heavy burden on defenders; ② they rely on scarce resources like trusted benchmarks, curated data, or proxy models; ③ using compromised auxiliary models as references may introduce new risks.
- **Key Challenge**: Defenders must suppress backdoors under the harsh conditions of **few models, no trigger priors, and no training data**, while maintaining model utility for downstream tasks. WAG degrades rapidly as the number of models decreases and collapses under "collusion attacks" (where multiple models share the same backdoor).
- **Goal**: Design a post-training backdoor defense that depends only on **architectural information and minimal clean validation samples**, is task-agnostic, effective in two-model scenarios, and robust against collusion attacks.
- **Core Idea**: **Backdoors are localized "shortcuts"**—they exploit spurious correlations encoded in specific modules, and different backdoor models rarely plant backdoors in the same locations. Therefore, **switching corresponding modules across models** (e.g., replacing a specific layer in Model A with the same layer from Model B) can break these fragile backdoor paths. This neutralizes vulnerabilities by replacing compromised components with benign ones while preserving utility, as models share pre-trained semantics.

## Method

### Overall Architecture
MSD formalizes "breaking backdoor shortcuts" as a discrete search problem: finding a **module source index table** for a given architecture that specifies which source model should fill each module slot (e.g., Q/K/V/O/I/P in layer $\ell$). The pipeline consists of four steps: first, establishing **theoretical and empirical foundations** on two-layer networks to prove that switching deviates from backdoors more than averaging; second, defining a set of **heuristic scoring rules** to characterize "good switching strategies"; third, using an **evolutionary algorithm** to search the vast discrete strategy space for high-scoring strategies; and finally, using **feature distance selection** to pick the candidate model most unaligned with the backdoor for deployment. The strategy relies solely on structural information, making it task-agnostic and reusable across isomorphic models (e.g., strategies found for RoBERTa can be applied to DeBERTa).

```mermaid
flowchart LR
    A[Multiple Suspicious Isomorphic Models M1..MN] --> B[Heuristic Scoring Rules<br/>Intra-layer/Cross-layer/Residual Adjacency + Balance + Diversity]
    B --> C[Evolutionary Search F·s·<br/>Obtain Module Source Index Table T]
    C --> D[Construct Candidate Merged Models<br/>Mij, Mji ...]
    D --> E[Suspect Class Detection + Feature Distance Selection<br/>20-50 Clean Samples]
    E --> F[Output Candidate Model Most Unaligned with Backdoor]
```

### Key Designs

**1. Theoretical Foundation on Two-Layer Networks: Switching deviates more from backdoors than averaging.** The authors first decompose models in a linear two-layer network $f(x;\theta)=W_2\sigma(W_1x)$ into a shared semantic term $S=W_2W_1$ and a backdoor component $B^*=W_2\Delta W_1^*+\Delta W_2^* W_1+\epsilon^*$ (where the second-order term $\epsilon^*$ is negligible). Defining the $L_2$ output distances of the weight-averaged model $M_{wag}$ and the switched model $M_{ij}=\{W_1+\Delta W_1^i,\ W_2+\Delta W_2^j\}$ relative to the original backdoor model, they prove **Theorem 1**: The total backdoor deviation of WAG is upper-bounded by the mean deviation of the switched models, i.e., $\|D_{wag,i}\|+\|D_{wag,j}\|\le \tfrac12(\|D_{ij,i}\|+\|D_{ij,j}\|+\|D_{ji,i}\|+\|D_{ji,j}\|)$. **Proposition 1** further guarantees that at least one switched model has a backdoor deviation strictly exceeding that of WAG. These conclusions provide the mathematical basis for "switching over averaging" and the motivation for selecting the most unaligned candidate. Simultaneously, the identity $L_{ij}+L_{ji}=L_i+L_j$ shows that the total utility loss of a switched pair equals the sum of the individuals, which empirically remains low, ensuring utility conservation.

**2. Heuristic Scoring Rules to Guide Search: Characterizing "what is a good switch".** Extending the two-layer results to deep Transformers, the key hypothesis is that "disrupting the propagation path of the backdoor inactivates it." Given the complexity of deep networks, the authors define five types of rules to score a switching strategy: the first three are **adjacency penalties**—penalizing three types of neighbor relationships through which a backdoor might propagate, including ① intra-layer adjacency (adjacent Q-K, K-V modules within the same layer), ② consecutive layer adjacency (between adjacent Transformer layers), and ③ residual path adjacency (propagation via skip connections). The last two include ① a **balance penalty** $B_{bal}$ to avoid over-reliance on a single source model, and ② a **diversity reward** $R_{div}$ to encourage different combinations across layers. These rules form the fitness function $F(s)=-\lambda_1 A_{intra}(s)-\lambda_2 A_{cons}(s)-\lambda_3 A_{res}(s)-\lambda_4 B_{bal}(s)+\lambda_5 R_{div}(s)$ (default $\lambda_k=1$). A higher $F(s)$ indicates better disruption of potential backdoor paths. Crucially, the scoring **requires no training or validation**, relying purely on structural info, which is why it is task-agnostic and transferable.

**3. Evolutionary Module Switching Search: Finding optimal strategies in a vast discrete space.** The search is treated as a discrete NAS problem, where a strategy $s:\{1,\dots,L\}\times M\to\{1,\dots,N\}$ assigns a source model index to each (layer, module), where $M=\{Q,K,V,O,I,P\}$. Since $F(s)$ is non-differentiable over the large discrete space, a modified **aging regularized evolution** algorithm is used: iterating via random population initialization, tournament selection of parents, mutation to generate offspring, and population truncation based on fitness. Two key modifications are: ① fitness is calculated directly using heuristic $F$ without training/validation; ② "discarding low-score strategies" replaces the original aging regularization. The search runs for 2 million generations on a single i9 CPU, taking 2.6 hours for two models and 4.3 hours for four models. **Each architecture only needs to be searched once.**

**4. Suspect Class Detection + Feature Distance Selection: Picking the stablest candidate without exhaustive Trojan detection.** Based on Theorem 1/Proposition 1, the candidate pool is better than WAG on average and contains one candidate strictly better than WAG. Thus, the final step involves selecting the most unaligned candidate. First, **Suspect Class Detection** is performed: for each model and each candidate class $c$, a random input is optimized to be predicted as $c$, yielding a dummy [CLS] feature $z^{dum}_{m,c}$. The average cosine distance between this and features of a few non-$c$ clean samples is accumulated: $S(c)=\sum_m \mathrm{avg}(1-\cos(z^{dum}_{m,c}, z^{clean}_{m,\neg c}))$. The class $c^*$ with the highest score is the suspect target class, and WAG's dummy feature $z^*$ for that class is taken as a fixed reference. Then, **Candidate Selection** is performed: for each switched candidate $m$, $d(m)=\mathrm{avg}(1-\cos(z^*, f_m(x)))$ is calculated, and the candidate $m^*$ with the maximum distance (most unlike the backdoor) is deployed. This step uses only 20–50 clean samples per class, takes less than a minute, and applies to CNNs using global average pooling of the last convolutional layer.

## Key Experimental Results

### Main Results (Text / Vision, Two-Model Merging, ASR↓)

| Scenario | Data/Model | WAG | TIES | DARE | **Ours (MSD)** |
|---|---|---|---|---|---|
| BadNet+InsertSent | SST-2 / RoBERTa-large | 31.9 | 52.9 | 47.1 | **22.0** |
| BadNet+LWS (Hidden) | SST-2 / RoBERTa-large | 62.2 | 77.1 | 61.4 | **40.4** |
| Benign+BadNet | SST-2 / RoBERTa-large | 39.3 | 69.2 | 43.2 | **12.2** |
| BadNet+WaNet | CIFAR-10 / ViT | 12.2 | 11.3 | 46.7 | **11.4** |
| BadNet+PhysicalBA | CIFAR-10 / ViT | 39.6 | 38.9 | 72.2 | **18.5** |

CACC (Clean Accuracy) remains consistent with baselines across all scenarios (~96% for text, ~98.7% for vision), indicating that backdoor suppression occurs with minimal utility loss. MSD reduces ASR for BadNet+LWS by over 21% relative to baselines, for Benign+BadNet by 27.1% compared to WAG, and for visual BadNet+PhysicalBA by at least 20.4% compared to all baselines.

### Ablation Study

| Ablation Dimension | Setting | Conclusion |
|---|---|---|
| Heuristic Rules | Remove intra-layer/consecutive/residual rules | Performance generally drops; the three rules are complementary |
| Early Stopping vs. No | Evolutionary search | No early stopping yields higher scores, ASR drops 27.2% further (fewer residual violations) |
| Cross-Arch Generalization | Reuse strategy RoBERTa→BERT→DeBERTa-v3 | Consistently outperforms WAG without re-searching |
| Cross-Arch Family | ResNet-18/50 (CNN) | ASR reduction comparable or superior to WAG, utility consistent |
| Clean Sample Size | 50→20 samples/class | Still capable of selecting low-ASR candidates |
| Poisoning Rate | 20% / 10% / 1% | Lower than WAG across all tiers |

### Key Findings
- **Stronger Defense with Fewer Models**: MSD significantly outperforms WAG with only two models, alleviating the requirement for 3–6 homologous models.
- **Robustness Against Collusion**: While WAG degrades when multiple models share the same backdoor, MSD remains effective by strategically disrupting repeated shortcuts (Table 13).
- **Extensive Structural Perturbation**: Strategies searched from three different random seeds share only 10/144 (6.94%) module positions, indicating MSD induces broad structural perturbations rather than relying on a few key layers, making it transferable.
- **Resistance to Adaptive Attacks**: Even if an attacker knows a switching strategy and retrains only those modules, using a strategy from a different random seed remains effective. MSD also maintains strong defense against complex Adaptive-Patch backdoors via transferability-based strategies.
- **Positive Correlation between Fitness and Defense**: High-scoring strategies (from no early stopping) lead to lower ASR, which the authors attribute to "fewer residual rule violations," effectively disrupting subtle spurious correlations.
- **Effective Suspect Class Detection**: The selection step correctly identifies the best-performing candidate in most cases. Even when the absolute best is missed, the selected candidate remains competitive with the best alternative and WAG.

### Three-Model / Multi-Model Scenarios
When three backdoor models are available, while WAG becomes stronger, MSD still further reduces average ASR to below 20% (Table 12). This demonstrates that module switching's advantage is not limited to the two-model setting; it extracts additional robustness gains as models and information redundancy increase, without requiring process redesign.

## Highlights & Insights
- **Shift from "Averaging" to "Switching"**: Treating backdoors as local shortcuts and "replacing compromised parts" via module switching is more precise than blurring all weights in WAG, supported by rigorous upper bounds from Theorem 1/Proposition 1.
- **Decoupling Scoring from Training**: Fitness is calculated purely through architectural adjacency rules, allowing a single search to be reused across isomorphic models. This removes the expensive "train-validate" loop from the search process.
- **Low Defender Threshold**: Requires only white-box access and 20–50 clean samples per class without prior knowledge of triggers or poisoned data, fitting realistic post-training scenarios.
- **Addressing Collusion**: Explicitly identifies that WAG degrades under shared backdoors and provides a more robust alternative.

## Limitations & Future Work
- **Dependency on Isomorphic Models**: Requires at least two models with the same architecture from related tasks/domains; heterogeneous architectures cannot be directly switched.
- **Focus on Data Poisoning**: Primarily targets data poisoning attacks; effectiveness against weight-poisoning attacks (direct weight manipulation) is not fully explored.
- **Reliability of Suspect Class Detection**: Selection relies on cosine distance heuristics of [CLS]/pooling features, whose robustness in complex or multi-target backdoor scenarios requires further validation.
- **Search Cost**: Although a one-time cost, 200 million generations of evolution still require several hours of CPU time, with costs increasing as model count grows.

## Related Work & Insights
- **Model Merging Defense Lineage**: WAG, TIES, and DARE merge weights to suppress backdoors. MSD differentiates itself by moving from "averaging" to "module switching," providing stronger theoretical deviation guarantees.
- **Backdoor-as-Shortcut Perspective**: Continues the explanation that backdoors exploit spurious correlations to create shortcuts (Gardner, He, etc.), grounding it in the actionable hypothesis that "shortcuts are localized in specific modules."
- **NAS / Evolutionary Search**: Adopts tools from discrete NAS and aging regularized evolution but replaces targets with non-differentiable structural heuristics, providing a "training-free evaluation" paradigm applicable to other weight-combination search tasks.
- **Practical Implications**: Designing methods with "architectural info + minimal samples" as primary constraints suggests that post-training security research should focus on "zero-training-resource" defenses rather than relying on hard-to-acquire trusted data.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The "module switching instead of weight averaging" perspective is novel and backed by rigorous deviation bounds.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers text/vision, Transformer/CNN, multi-model counts, collusion, adaptive attacks, and extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Logical progression from 2-layer theory to deep pipelines with clear illustrations of adjacency types and search algorithms.
- **Value**: ⭐⭐⭐⭐ — Aligns with real-world post-training constraints and addresses the overlooked threat of collusion backdoors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

*(Note: Related papers would be listed here in the original format if applicable)*

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] TrojanTO: Action-Level Backdoor Attacks Against Trajectory Optimization Models](trojanto_action-level_backdoor_attacks_against_trajectory_optimization_models.md)
- [\[ICLR 2026\] Reliable Poisoned Sample Detection against Backdoor Attacks Enhanced by Sharpness-Aware Minimization](reliable_poisoned_sample_detection_against_backdoor_attacks_enhanced_by_sharpnes.md)
- [\[CVPR 2026\] AntiStyler: Defending Object Detection Models Against Adversarial Patch Attacks Using Style Removal](../../CVPR2026/ai_safety/antistyler_defending_object_detection_models_against_adversarial_patch_attacks_u.md)
- [\[ICLR 2026\] Protection against Source Inference Attacks in Federated Learning](protection_against_source_inference_attacks_in_federated_learning.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
