---
title: >-
  [Paper Note] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation
description: >-
  [ICCV 2025][AI Safety][Federated Learning] This paper proposes FedVLA — the first federated learning framework for Vision-Language-Action (VLA) models — comprising three synergistic components: Instruction-Oriented Scene…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Vision-Language-Action Model"
  - "Mixture-of-Experts"
  - "Robotic Manipulation"
  - "Privacy Preservation"
date: 2026-05-08
content_hash: 048ecc78337f2456
---

# FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2508.02190](https://arxiv.org/abs/2508.02190)  
**Code**: None  
**Area**: AI Safety / Privacy Protection / Robotic Manipulation
**Keywords**: Federated Learning, Vision-Language-Action Model, Mixture-of-Experts, Robotic Manipulation, Privacy Preservation

## TL;DR

This paper proposes FedVLA — the first federated learning framework for Vision-Language-Action (VLA) models — comprising three synergistic components: Instruction-Oriented Scene-Parsing (IOSP) for task-aware feature extraction, Dual Gating Mixture-of-Experts (DGMoE) for adaptive knowledge routing, and Expert-Driven Aggregation (EDA) for effective cross-client knowledge integration, achieving task success rates comparable to centralized training while preserving data privacy.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models represent a significant advance in robotic manipulation, enabling robots to interpret natural language instructions, understand scenes, and execute tasks accordingly. Representative works such as RT-2 and OpenVLA have demonstrated strong generalization to novel instructions and unseen objects after training on large-scale datasets.

**Limitations of Prior Work**: Training VLA models requires large quantities of user-specific indoor scene data — such as videos of robot manipulation in home environments — which contain sensitive personal information including living spaces, object arrangements, and daily routines. Centralized training necessitates uploading all such data to the cloud, posing significant privacy risks and limiting the broader deployment of VLA models.

**Key Challenge**: There exists a fundamental tension between privacy preservation (data cannot be shared) and model performance (which requires large, diverse datasets). Existing federated learning methods such as FedAvg are primarily designed for unimodal settings; their simple averaging aggregation ignores task heterogeneity across clients — in VLA scenarios, robots in different households perform entirely different tasks (e.g., opening drawers vs. sweeping floors), resulting in drastically different feature distributions.

**Goal**: To design the first federated learning framework tailored for VLA models that, without exposing user data, effectively handles multi-client task heterogeneity and achieves robotic manipulation performance comparable to centralized training.

**Key Insight**: The authors observe that task heterogeneity in VLA settings can be addressed through Mixture-of-Experts (MoE), where different tasks activate different experts. Furthermore, expert activation patterns serve as a signal of inter-client task similarity and can guide federated aggregation.

**Core Idea**: Three coordinated components are proposed — IOSP decomposes scenes into object-level representations to enhance task understanding; DGMoE enables experts to autonomously determine whether to respond to each token for efficient routing; EDA performs intelligent aggregation based on expert activation similarity.

## Method

### Overall Architecture

FedVLA adopts a client-server architecture. Each client maintains a local VLA model consisting of three modules — Stem (visual encoding + IOSP), Trunk (Transformer layers with DGMoE), and Head (action prediction) — and processes local task data. In each communication round, clients perform local training for 5 epochs, then transmit Trunk parameters and expert activation statistics to the server. The server executes EDA aggregation and distributes the global Trunk back to clients. The Stem and Head remain personalized and are excluded from aggregation.

### Key Designs

1. **Instruction-Oriented Scene-Parsing (IOSP)**:

    - **Function**: Decomposes scene images into structured object-level representations conditioned on language instructions, enhancing task-aware feature extraction.
    - **Mechanism**: Given an instruction and image, named entity recognition first extracts target object names from the instruction, while YOLOv8 detects all objects in the image. CLIP then computes cosine similarity between object names and the instruction, categorizing objects into three groups: Target Objects (TOs, directly referenced by the instruction), Surrounding Objects (SOs, foreground but non-target), and Background Objects (BOs). The CLIP visual encoder assigns image tokens to each group (top-8 most relevant tokens per group), and a MoE module refines per-group features before concatenating them back into the main sequence.
    - **Design Motivation**: In federated settings, image backgrounds and object distributions vary substantially across clients. IOSP helps the model focus on task-relevant objects and filter out client-specific background interference, which is critical for cross-client generalization.

2. **Dual Gating Mixture-of-Experts (DGMoE)**:

    - **Function**: Adaptively selects an appropriate number of experts for different tokens, enabling task-aware knowledge routing.
    - **Mechanism**: Each DGMoE layer contains $K$ experts and two gating mechanisms. The token-side gate $G_t$ computes each token's preference scores over all experts via a soft router, incorporating a residual connection from the previous layer's scores (Eq. 3) to allow tokens to inherit prior expert selection preferences. The expert-side gate $G_e$ assigns each expert a learnable threshold parameter $W_e$; an expert is activated only when a token's score exceeds $\lambda W_e$ (with $\lambda = 0.5$) (Eq. 4–5). The final output is $y = \sum_{i=1}^K g_i(x) E_i(x)$, where $g_i(x) = 0$ when an expert rejects a token.
    - **Design Motivation**: Conventional MoE selects a fixed top-$k$ experts, which cannot adapt to varying task complexity — simple tasks may require only one expert while complex tasks require several. DGMoE's bidirectional selection (token-to-expert and expert-to-token) enables dynamic sparse activation, with experiments showing an average of approximately 1.22 experts activated per token, substantially reducing computation.

3. **Expert-Driven Aggregation (EDA)**:

    - **Function**: Intelligently aggregates client models on the federated server based on expert activation similarity.
    - **Mechanism**: Each client records an expert activation matrix $\mathbf{V}_i \in \mathbb{R}^{L \times K}$ ($L$ layers, $K$ experts), tracking the activation count of each expert. For layer $l$, the cosine similarity $s_{i,j}^{(l)}$ between the expert selection vectors of clients $C_i$ and $C_j$ is computed (Eq. 8), then normalized to obtain aggregation weights $w_{l,i}$ (Eq. 9). Clients with higher similarity contribute greater weight to each other's aggregation.
    - **Design Motivation**: FedAvg's simple averaging tends to cancel out task-specific knowledge in heterogeneous scenarios. EDA encourages clients performing similar tasks (with similar expert activation patterns) to learn more from one another, thereby preserving task specificity.

### Loss & Training

Clients optimize action prediction using the Huber loss. Built upon the HPT (Heterogeneous Pre-trained Transformers) backbone, the learning rate is set to $5 \times 10^{-6}$ for simulation and $2 \times 10^{-5}$ for real-world settings, with a batch size of 256 and the Adam optimizer. Training proceeds for 1000 federated communication rounds, with 5 local epochs per round.

## Key Experimental Results

### Main Results

Simulation experiments (MuJoCo + Meta-World, Sawyer robot, 4 tasks):

| Method | Avg. Success | Door Lock | Close Drawer | Sweep Into | Open Window |
|--------|-------------|-----------|-------------|-----------|-------------|
| Centralized | 65.0% | 86.7% | 73.3% | 53.3% | 46.7% |
| FedAvg | 51.7% | 66.7% | 73.3% | 40.0% | 26.7% |
| FedVLA | **63.3%** | 80.0% | **80.0%** | 53.3% | 40.0% |

Real-world experiments (UR3 robot, 4 tasks):

| Method | Avg. Success | Clean Up | Trash Collect | Open Drawer | Sorting Pills |
|--------|-------------|----------|-------------|------------|--------------|
| Centralized | 63.4% | 46.7% | 46.7% | 86.7% | 73.3% |
| FedAvg | 53.3% | 46.7% | 40.0% | 60.0% | 66.7% |
| FedVLA | **63.3%** | 53.3% | 46.7% | 80.0% | **73.3%** |

### Ablation Study

| Configuration | Avg. Success | Clean Up | Trash Collect | Open Drawer | Sorting Pills |
|--------------|-------------|----------|-------------|------------|--------------|
| w/o IOSP | 41.1% | 40.0% | 13.3% | 66.7% | 46.7% |
| w/o DGMoE | 31.7% | 20.0% | 20.0% | 46.7% | 40.0% |
| w/o EDA | 26.7% | 26.7% | 20.0% | 33.3% | 26.7% |
| FedVLA (full) | **63.3%** | 53.3% | 46.7% | 80.0% | 73.3% |

### Key Findings

- **FedVLA nearly matches centralized training**: Average success rates of 63.3% vs. 63.4% (real-world) and 63.3% vs. 65.0% (simulation), demonstrating that the federated framework achieves centralized-level performance without data sharing.
- **EDA is the most critical component**: Removing EDA causes the success rate to drop sharply from 63.3% to 26.7% (a 57.8% decrease), as naive FedAvg aggregation cancels out the task-specific knowledge encoded in different DGMoE experts.
- **IOSP is essential in multi-object scenarios**: On the Trash Collection task — which involves the largest number of objects — removing IOSP reduces success from 46.7% to 13.3%.
- **DGMoE achieves efficient sparse activation**: On average, only ~1.22 experts are activated per token (vs. a fixed $k$ in vanilla top-$k$ MoE), substantially reducing computation. Moreover, different object types (target vs. background) activate different experts, confirming the effectiveness of task-aware routing.

## Highlights & Insights

- **First federated VLA framework**: This work pioneers the intersection of privacy-preserving learning and robot learning. As domestic robots become increasingly prevalent, privacy concerns surrounding VLA models will grow in importance, and FedVLA provides a viable solution.
- **Bidirectional expert selection mechanism**: Conventional MoE relies on unidirectional token→expert selection; DGMoE introduces a reverse expert→token selection (self-aware experts), realizing a "bidirectional matching" scheme that simultaneously improves performance and efficiency. This design is transferable to other settings requiring efficient MoE architectures.
- **Expert activation as a client similarity signal**: Using DGMoE expert activation patterns to measure inter-client task similarity is more semantically meaningful than directly comparing model parameters. This "behavioral similarity"-guided aggregation strategy is an elegant contribution.

## Limitations & Future Work

- **Limited experimental scale**: Experiments involve only 4 clients/tasks with 30–80 trajectories per task; performance under large-scale heterogeneous settings (dozens of clients, more complex tasks) remains unknown.
- **Partial aggregation**: Only the Trunk module is aggregated — the Stem and Head are fully personalized — which may limit cross-client knowledge transfer.
- **Unaddressed federated learning concerns**: Communication efficiency and security attacks (e.g., model inversion attacks) are not considered, despite being critical issues in federated learning.
- **Manual threshold setting**: The self-awareness threshold $\lambda = 0.5$ in DGMoE is set manually and may not be optimal.
- **Future directions** include scaling to more clients and more complex manipulation tasks, and exploring stronger privacy guarantees such as differential privacy.

## Related Work & Insights

- **vs. FedAvg**: FedAvg applies uniform averaging aggregation with no awareness of task heterogeneity. FedVLA achieves task-aware aggregation via EDA, improving real-world success rate from 53.3% to 63.3%.
- **vs. OpenVLA/RT-2**: These centralized VLA models require pooling all data for training, posing privacy risks. FedVLA demonstrates the feasibility of distributed training without sacrificing performance.
- **vs. FedHCA2**: FedHCA2 addresses federated multi-task learning but targets unimodal settings. FedVLA handles multimodal (vision + language + action) robotic scenarios, presenting substantially greater challenges.
- **vs. DeepSeekMoE/LLaVA-MoE**: These works integrate MoE into large models but employ fixed top-$k$ selection. DGMoE's self-aware experts enable dynamic sparse activation, offering greater flexibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First federated VLA framework with novel DGMoE and EDA designs; however, individual components (federated learning, MoE, scene parsing) are not novel in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐ — Validated in both simulation and real-world settings with complete ablations, but the scale is limited (4 tasks/clients) and only FedAvg is used as a baseline.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, detailed method descriptions, and complete formulations with algorithm pseudocode.
- **Value**: ⭐⭐⭐⭐ — Establishes the federated VLA research direction at the frontier of privacy-preserving robot learning, though larger-scale validation is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](../../ICML2026/ai_safety/metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)

</div>

<!-- RELATED:END -->
