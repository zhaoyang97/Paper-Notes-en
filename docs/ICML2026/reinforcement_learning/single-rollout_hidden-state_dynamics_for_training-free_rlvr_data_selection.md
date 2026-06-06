---
title: >-
  [Paper Note] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection
description: >-
  [ICML 2026][Reinforcement Learning][RLVR] SHIFT uses the "start token → end token" hidden-state difference $\Delta(x)=\mathbf{e}(x)-\mathbf{s}(x)$ from a single greedy decoding rollout as both a utility proxy and a diver…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "Data Selection"
  - "Hidden-State Dynamics"
  - "CoreSet"
  - "Training-Free"
date: 2026-05-08
content_hash: 768e73e82dc034c8
---

# Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection

**Conference**: ICML 2026  
**arXiv**: [2605.28631](https://arxiv.org/abs/2605.28631)  
**Code**: https://github.com/JianghaoWu/SHIFT  
**Area**: Reinforcement Learning / LLM Reasoning / Data Selection  
**Keywords**: RLVR, Data Selection, Hidden-State Dynamics, CoreSet, Training-Free  

## TL;DR
SHIFT uses the "start token → end token" hidden-state difference $\Delta(x)=\mathbf{e}(x)-\mathbf{s}(x)$ from a single greedy decoding rollout as both a utility proxy and a diversity feature for RLVR samples. It selects a minimal set of samples from large unlabeled pools using a quality-weighted farthest-first CoreSet without training, rewards, or ground truth answers.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) significantly enhances LLM reasoning capabilities with extreme sample efficiency—literature indicates that one or two carefully selected samples can approximate the performance achieved by RL on thousands of samples. Representative methods (e.g., Wang et al. 2025c) select high-value samples by observing the Historical Variance Score (HVS) during small-scale RL training.

**Limitations of Prior Work**: Selection methods based on training-time signals require running fine-tuning or RL on large candidate pools and necessitate verifiable rewards, which are equivalent to ground truth answers. This is costly and unfeasible in specialized domains like medical reasoning. Classical active learning criteria based on uncertainty or gradients also depend on training feedback, while pre-training signals like difficulty or PPL correlate weakly with reward-driven RLVR utility.

**Key Challenge**: RLVR sample utility is reward-driven; however, during the selection phase, neither rewards nor labels are available, and performing preliminary training is undesirable. Existing active learning signals are built upon post-training or labeled information.

**Goal**: Select $|S|=B$ most promising training samples from a large unlabeled pool in the pre-RL phase without evaluating rewards.

**Key Insight**: Theoretically, Dherin et al. 2025 equate the context effect of transformer self-attention + MLP to a rank-1 implicit weight update of the first MLP layer, providing an upper bound: $\|\Delta W(Y)\|_F \le \frac{\|W\|_2}{\|A(C\setminus Y,x)\|_2}\,\|\Delta A(Y)\|_2$. This suggests that "context-induced representation changes" can proxy the "internal learning amount" of the model. Empirically, Liang et al. 2025 confirmed that hidden-state differences before and after CoT can encode non-trivial structures of the reasoning process.

**Core Idea**: The difference between multi-layer averaged hidden states of start/end anchors in a single deterministic CoT rollout is used as the sample utility proxy $q(x)=\|\Delta(x)\|_2$. Quality-weighted farthest-first selection is then performed in the normalized space of $[\mathbf{s}(x);\Delta(x)]$.

## Method

### Overall Architecture
For each sample in the unlabeled pool $\mathcal{U}=\{x_i\}_{i=1}^{N}$: (1) A base LLM $f_\theta$ generates a CoT using greedy decoding ($T=0$) under a fixed reasoning prompt; (2) The start and end tokens of the CoT (using delimiters like `<think>`/`</think>` if supported) are used as anchors, and their multi-layer averages $\mathbf{s}(x), \mathbf{e}(x)\in\mathbb{R}^D$ are computed; (3) The RIRS $\Delta(x)=\mathbf{e}(x)-\mathbf{s}(x)$ is calculated; (4) $\tilde q(x)$ and $\phi(x)$ are processed by a quality-weighted farthest-first CoreSet to pick $B$ samples; (5) Rewards are only computed for these $B$ samples to perform RLVR. The selection process involves only one inference, zero training, and zero labels.

### Key Designs

1. **Multi-layer Averaged RIRS Representation**:
    - **Function**: Condenses "how much the internal state changed during the CoT" into a single $\mathbb{R}^D$ vector.
    - **Mechanism**: For each layer $\ell$, anchor token hidden states $\mathbf{h}^{(\ell)}_{t_s}(x)$ and $\mathbf{h}^{(\ell)}_{t_e}(x)$ are extracted and averaged across layers to get $\mathbf{s}(x)$ and $\mathbf{e}(x)$. $\Delta(x)=\mathbf{e}(x)-\mathbf{s}(x)$ is defined as the "reasoning-induced representation shift." From the perspective of rank-1 implicit weights by Dherin et al., $\|\Delta(x)\|_2$ is interpreted as an observable proxy for the trajectory-level, layer-aggregated context-induced changes—though the authors state this is motivation rather than strict derivation.
    - **Design Motivation**: This is cheaper to obtain via a single inference than "self-consistency entropy with R=32" or "running RL to observe rewards." It is more stable than single-layer anchors, preventing anomalies in one layer from biasing the results.

2. **Log-stabilized Utility Score**:
    - **Function**: Converts the RIRS norm into a numerically stable utility proxy.
    - **Mechanism**: Compute $q(x)=\|\Delta(x)\|_2$ followed by monotonic log-compression $\tilde q(x)=\log(1+q(x))$. A high $\tilde q$ signifies the sample induced a larger shift in the internal state, hypothesized to be more valuable for RLVR learning.
    - **Design Motivation**: $\|\Delta\|_2$ magnitudes vary significantly across samples of different lengths and domains; raw values would be dominated by extremes. Log transformation preserves ranking while compressing the scale, making it comparable with the diversity distance $d(x,S)$.

3. **Quality-Weighted Farthest-First CoreSet**:
    - **Function**: Performs a greedy trade-off between utility and coverage to avoid selecting redundant high-utility samples.
    - **Mechanism**: Construct $\ell_2$-normalized coverage features $\phi(x)=[\mathbf{s}(x);\Delta(x)]/\|[\mathbf{s}(x);\Delta(x)]\|_2 \in \mathbb{R}^{2D}$, containing both CoT starting context and reasoning dynamics. Initialize $S\leftarrow\{\arg\max_x \tilde q(x)\}$, then iteratively select $x^\star=\arg\max_{x\in\mathcal{U}\setminus S}\, \tilde q(x)\cdot d(x,S)$, where $d(x,S)=\min_{x'\in S}\|\phi(x)-\phi(x')\|_2$, until $|S|=B$.
    - **Design Motivation**: High $\tilde q$ samples tend to cluster; simple top-K selection would waste budget. Farthest-first selection alone tends to pick outliers. The multiplicative form requires both criteria to be met for selection and only requires an $O(NB)$ greedy scan.

### Loss & Training
SHIFT itself does not train any parameters; the selection phase consists only of greedy inference and CoreSet selection. In the RLVR phase, the same training budget and hyperparameters are applied across all methods, only changing the sample selection rule. MedQA utilizes Qwen3-1.7B, and MATH-500 utilizes Qwen2.5-Math-1.5B, both starting from public checkpoints.

## Key Experimental Results

### Main Results
| Dataset | Selection Budget | Evaluation | Full RLVR Ref. | Random | Best Baseline | SHIFT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MATH-500 (In-domain) | 2% (7/350) | Pass@1 | 66.00 | 53.73 | Cluster 44.67 / CoreSet 47.33 | Closest to Full, sign. better than CoreSet |
| AMC (OOD Math) | 2% | Pass@1 | 33.73 | 25.78 | 25.30 (Cluster) | Consistently better than training-free baselines |
| MedQA | 0.1–0.2% | Post-RLVR Acc. | — | — | — | Consistently optimal under ultra-low budgets |

> Reproduction details: MATH-500 split into 350 pool/150 test; MedQA used 10.2K pool/1.27K test, with migration to MedMCQA, PubMedQA, and MedXpertQA. Baselines include KMeans-Center (Cluster), Farthest-First (CoreSet), Q-PPL, SC-Entropy (R=32), CoT Similarity, and A-PPL.

### Ablation Study
| Configuration | Key Role | Description |
| :--- | :--- | :--- |
| Full SHIFT | RIRS Quality + RIRS Coverage | Best reported version. |
| Utility Top-K only | No farthest-first | Selects homogeneous samples, performance drops. |
| Farthest-First only | No $\tilde q$ weight | Degenerates to generic CoreSet, dominated by outliers. |
| Sentence Emb. CoreSet | No RIRS (MiniLM-L6-v2) | Cannot capture reasoning-time computation, significantly weaker. |
| Single vs Multi Rollout | Selection cost | Single greedy RIRS is sufficient; R=32 is not needed. |

### Key Findings
- The "RIRS norm" is decoupled from surface statistics like input/output length. Correlation analysis confirms $\tilde q$ gains are not explained by length factors, supporting it as a true proxy for internal updates.
- Using $\Delta(x)$ as both the utility and as part of the coverage feature $\phi(x)$ is critical; using it as only one or the other results in performance drops.
- In domains like MedQA where rewards are scarce, SHIFT compresses "labeling + reward evaluation" costs to only $B$ selected samples, making RLVR feasible in low-resource settings.
- Cross-dataset transfer (MedQA training → MedMCQA/PubMedQA/MedXpertQA evaluation) shows SHIFT's stable advantage, suggesting it learns transferable reasoning structures.
- Pure CoreSet using $[\mathbf{s}(x);\Delta(x)]$ is more effective than sentence embedding CoreSet, indicating RIRS benefits from a reasoning-aligned feature space.

## Highlights & Insights
- Replacing "training-dependent reward/gradient signals" with "inference-visible trajectory residuals" characterizes sample value from the viewpoint of in-context implicit weight updates, bridging Dherin 2025's theory to practical selection.
- Minimalist algorithm: $O(N)$ inference + $O(NB)$ CoreSet with no learning rates, no tuning, and no reward models. This design is easily portable to any model with CoT delimiters.
- Using the multiplication $\tilde q\cdot d$ instead of addition for utility and diversity is a noteworthy detail: it avoids weight tuning and ensures candidates must satisfy both conditions.

## Limitations & Future Work
- Gap between theory and method: The upper bound is for a single query position, whereas $\Delta(x)$ aggregates across layers and the entire rollout.
- Evaluated models are relatively small (1.5B, 1.7B) with very low budgets; whether "high norm = high value" remains monotonic in larger models or dense reward scenarios requires more evidence.
- Anchors depend on CoT delimiters; if a model lacks explicit CoT segments or generates unstable decoding, $\Delta(x)$ semantics may be diluted by noise.

## Related Work & Insights
- **vs. Wang et al. 2025c (HVS)**: HVS requires running RL to observe accuracy variance, which needs rewards; SHIFT acts as a zero-label alternative using a single inference pre-RL.
- **vs. Classical Active Learning**: Traditional uncertainty/distance signals from static embeddings or training loss fail to capture reasoning-time computation; SHIFT demonstrates that shifting the feature space to "reasoning dynamics" upgrades the CoreSet framework.
- **vs. Liang et al. 2025**: While they use start-end deltas to explain reasoning structures, SHIFT transforms this diagnostic signal into a selection criterion.

## Rating
- Novelty: ⭐⭐⭐⭐ Connects implicit update theory to RLVR data selection.
- Experimental Thoroughness: ⭐⭐⭐ Dual scenarios (Math/Med) + multiple baselines, but narrow model scales.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain from theory to algorithm and ablation.
- Value: ⭐⭐⭐⭐ Provides a practical zero-label selection recipe for low-resource RLVR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Probing RLVR Training Instability through the Lens of Objective-Level Hacking](probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)
- [\[ICML 2026\] EchoRL: Reinforcement Learning via Rollout Echoing](echorl_reinforcement_learning_via_rollout_echoing.md)
- [\[ICCV 2025\] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](../../ICCV2025/reinforcement_learning/mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](../../ACL2026/reinforcement_learning/learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)
- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)

</div>

<!-- RELATED:END -->
