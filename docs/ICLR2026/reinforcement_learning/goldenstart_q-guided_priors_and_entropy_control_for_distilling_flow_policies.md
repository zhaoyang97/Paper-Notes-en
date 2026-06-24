---
title: >-
  [Paper Note] GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies
description: >-
  [ICLR 2026][Reinforcement Learning][flow-matching policy] GoldenStart (GS-flow) enhances single-step distilled flow-matching policies by implementing two mechanisms: relocating the generated "starting noise" to high-value regions ("Golden Start") via a Q-guided conditional VAE, and transforming the deterministic actor into a controllable stochastic distribution using entropy regularization. This addresses the challenges of "precise exploitation" and "online exploration" while…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "flow-matching policy"
  - "policy distillation"
  - "offline-to-online RL"
  - "Q-guided prior"
  - "entropy regularization"
  - "conditional VAE"
date: 2026-05-08
content_hash: da095cedcde17222
---

# GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=UqF3d6yM0S](https://openreview.net/forum?id=UqF3d6yM0S)  
**Code**: [https://github.com/ZhHe11/GSFlow-RL](https://github.com/ZhHe11/GSFlow-RL)  
**Area**: reinforcement learning  
**Keywords**: flow-matching policy, policy distillation, offline-to-online RL, Q-guided prior, entropy regularization, conditional VAE  

## TL;DR
GoldenStart (GS-flow) enhances single-step distilled flow-matching policies by implementing two mechanisms: relocating the generated "starting noise" to high-value regions ("Golden Start") via a Q-guided conditional VAE, and transforming the deterministic actor into a controllable stochastic distribution using entropy regularization. This addresses the challenges of "precise exploitation" and "online exploration" while maintaining single-step inference speed.

## Background & Motivation
**Background**: Flow-matching and diffusion-based generative policies exhibit significant potential in continuous control due to their ability to characterize complex multi-modal action distributions. However, iterative denoising results in high inference latency. Recent works like FQL have solved this latency issue using **single-step distillation** (where a student network simulates the entire denoising process in a single forward pass), establishing strong state-of-the-art (SOTA) performance in offline RL.

**Limitations of Prior Work**: Single-step distillation methods underutilize two aspects of the generative process. First, generation always begins from **fixed, uninformative standard Gaussian noise**, whereas the field of image generation has demonstrated that the initial noise itself is a critical variable for guided generation. Second, given the noise, the distilled student learns a **deterministic "point-to-point" mapping** (one noise vector $\to$ one deterministic action), which naturally lacks stochasticity and hinders online exploration.

**Key Challenge**: Single-step distillation prioritizes "speed + precision" (deterministic exploitation), while online RL fine-tuning requires "explorability" (controlled stochasticity). These two objectives conflict within a deterministic actor. Furthermore, uninformative Gaussian starting points force the policy to take a circuitous path to reach high-value actions, which is particularly detrimental in multi-modal environments prone to Q-value overestimation.

**Goal**: To enable the distilled policy to accurately hit high-value action modes and perform principled exploration during the offline-to-online phase, without sacrificing single-step inference speed.

**Core Idea**: **(1) Q-Guided Generative Prior**: Instead of starting from uninformative noise, a lightweight conditional VAE is used to learn an "advantage noise" distribution, repositioning the starting point to high-Q regions. **(2) Entropy-Regularized Distillation**: The "point-to-point" mapping is upgraded to a "point-to-distribution" mapping. The student outputs a Gaussian with mean and variance, utilizing entropy regularization to dynamically adjust stochasticity online.

## Method

### Overall Architecture
GS-flow executes a **two-stage training** process within a standard actor-critic framework. In Stage 1, **Q-Guided Prior Learning** addresses the "poor starting point" issue using an Advantage Noise Selection module to identify "advantage noise" that produces high-value actions, followed by training a conditional VAE to fit the state-conditioned distribution of this advantage noise. In Stage 2, **Entropy-Regularized Distillation** feeds the learned prior to both the teacher and student, training the student as a stochastic policy with entropy regularization. During inference, only two lightweight modules are used: the VAE decoder and the student actor. The decoder provides the advantage prior $\to$ the student produces the action distribution, which is sampled for online exploration or averaged for evaluation.

```mermaid
flowchart TD
    S[State s] --> ANS[Advantage Noise Selection<br/>Teacher generates N candidates<br/>Critic Q selects best noise x_adv]
    ANS --> CVAE[Conditional VAE<br/>Learn p&#40;x_adv | s&#41;]
    CVAE -->|Sample advantage prior x̂_adv| T[Teacher Policy π_φ<br/>Generate target action a_teacher]
    CVAE -->|x̂_adv| ST[Student Policy π_φ<br/>Dual-head output μ, σ]
    T -->|L2 Distillation| ST
    Q[Critic Q] -->|Value maximization L_Q| ST
    ST -->|Entropy Reg H| ST
    ST --> OUT[Online: Sample / Eval: Mean μ]
```

### Key Designs
**1. Advantage Noise Selection: Turning "best starting point" into an online scoring process.** The flow-matching teacher $\pi_\phi$ is a deterministic mapping; given the noise, the action is fixed. Thus, finding the best starting point is equivalent to finding the best noise. For each state $s$, the teacher first generates a set of candidate actions $A_{\text{cand}}=\{\pi_\phi(s,x_j)\mid x_j\sim\mathcal N(0,I)\}$ from $N_{\text{cand}}$ different Gaussian noises. The critic then scores these, selecting the noise that produces the action with the highest Q-value as the advantage noise: $x_{\text{adv}}(s)=\arg\max_{x_j} Q(s,\pi_\phi(s,x_j))$. This step is computed online at each training step using the latest teacher, resulting in training pairs $B_{\text{adv}}=\{(s,x_{\text{adv}}(s))\}$. While more candidates lead to better selection ($N_{\text{cand}}=15$ converges fastest), the authors balance computational cost by choosing $N_{\text{cand}}=10$. Even with 5 candidates, performance significantly exceeds FQL (the degenerate case where $N_{\text{cand}}=0$).

**2. State Conditional VAE: Fitting selected advantage noise into a samplable continuous prior.** Since state-wise noise selection only provides discrete samples, a generative model is needed to randomly generate "Golden Starts" during inference. The authors use a conditional VAE (encoder $E_{\xi_1}$, decoder $D_{\xi_2}$) to fit the state-conditioned distribution $p_{\xi_2}(x_{\text{adv}}\mid s)$. The loss consists of reconstruction and KL regularization: $L_{\text{VAE}}=L_{\text{recon}}+\lambda_{\text{KL}}L_{\text{KL}}$, where $L_{\text{KL}}=\mathbb E[D_{\text{KL}}(q_{\xi_1}(z\mid x_{\text{adv}},s)\,\|\,\mathcal N(0,I))]$ compresses the latent space toward a standard normal distribution. A VAE is chosen over a simple Gaussian because advantage noise distributions are naturally multi-modal. Visualizations in MultiCrescent show that the prior falls into high-value modes within the dataset during the offline phase, and its density adaptively shifts to newly discovered global optima during online fine-tuning.

**3. Entropy-Regularized Distillation: Converting "point-to-point" to "point-to-distribution".** The student $\pi_\phi(a\mid s,\hat x_{\text{adv}})$ utilizes a dual-head structure to output both the mean $\mu_\phi$ and standard deviation $\sigma_\phi$. Exploratory actions are generated via reparameterization: $a_\phi=\mu_\phi+\sigma_\phi\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$. The actor loss balances three terms: $L_{\text{Actor}}=\mathbb E[\alpha_1 L_{\text{L2-Distill}}+L_Q-\alpha_2 H(\pi_\phi)]$. The distillation term $L_{\text{L2-Distill}}=\mathbb E[\|\mu_\phi(s,\hat x_{\text{adv}})-\pi_\phi(s,\hat x_{\text{adv}})\|^2]$ **anchors only the deterministic mean of the student to the teacher**, with both sharing the same advantage noise $\hat x_{\text{adv}}$; these details ensure variance reduction and stable training. The value term $L_Q=-Q(s,a_\phi)$ uses sampled actions similar to SAC. The entropy temperature $\alpha_2$ is automatically learned by matching a target entropy $H_{\text{target}}$ ($L_{\alpha_2}=\mathbb E[\alpha_2(H(\pi_\phi)-H_{\text{target}})]$); exploration increases when entropy is below the target and decreases when sufficient. This allows the student to dynamically adjust stochasticity during the online phase, combining high-quality generation with controllable Gaussian exploration.

## Key Experimental Results
Evaluations were conducted on OGBench, D4RL AntMaze, and Visual Environments, covering Gaussian (BC/IQL/ReBRAC), Diffusion (IDQL/SRPO/CAC), and Flow (FAWAC/FBRAC/IFQL) policies, as well as the SOTA distillation method FQL. For offline-to-online, Cal-QL and RLPD were also compared.

### Main Results (Offline, average scores over 5 seeds)

| Benchmark | Strongest Baseline | Ours (GS-flow) |
|---|---|---|
| OGBench Avg | FQL 38.5 | **47.1** |
| D4RL AntMaze Avg | FQL 83.5 | **86.1** |
| Visual Environments Avg | FQL 65.4 | **70.9** |
| Cube Double Play (Multi-modal) | FQL 36 | **51.3** |
| Puzzle-3x3 Play | FQL 16 | **25.2** |

The advantage is most pronounced in multi-modal or tasks with multiple local optima; performance is on par with FQL in the single-modal Cube Single Play task.

### Main Results (Offline-to-Online, "Offline $\to$ Online")

| Benchmark | FQL | RLPD | Ours |
|---|---|---|---|
| OGBench Avg | 34.0 → 67.6 | 0.0 → 41.6 | **49.4 → 88.6** |
| D4RL AntMaze Avg | 74.8 → 95.2 | 0.0 → 95.7 | **86.2 → 96.8** |
| Adroit Cloned Avg | 13.2 → 110.0 | 0.8 → 88.0 | **20.5 → 111.5** |
| Puzzle-4x4 Play | 8 → 38 | 0 → 100 | **17 → 100** |

In Puzzle-4x4, a recognized hard-exploration task, GS-flow improved from 17% to 100%, matching the specialized online method RLPD, and even outperformed RLPD in tasks like AntSoccer and Cube Double.

### Ablation Study

| Setting | Observation |
|---|---|
| Candidates $N_{\text{cand}}$=5/10/15 | Performance improves with more candidates; $N_{\text{cand}}=0$ reverts to FQL. 5 already exceeds FQL; 10 is used for main experiments. |
| Ours w/o CE (No controllable entropy) | Online efficiency drops significantly; Puzzle-4x4 online performance reverts to FQL levels. |
| Inference Latency | Ours 0.51ms $\approx$ FQL 0.42ms $\ll$ IFQL 0.97ms (multi-step). |
| Training Latency | Ours 3.10ms $>$ FQL 1.90ms (due to extra inference in Advantage Noise Selection). |

### Key Findings
- The benefits of the Q-guided prior are concentrated in **multi-modal and overestimation-prone** tasks; gains are limited in unimodal tasks.
- Controllable entropy is the core component for online exploration; removing it causes the policy to revert to the limited exploration of deterministic distillation.
- Additional overhead is almost entirely restricted to the one-time training phase; inference maintains the speed of single-step distillation.

## Highlights & Insights
- **"Change the Start" not the "Network"**: Transferring the insight from image generation that initial noise is guidable to RL distillation. By using a critic to select noise and a VAE to fit it as a prior, the policy gains a "Golden Start" with nearly zero inference cost.
- **"Point-to-Point $\to$ Point-to-Distribution"**: Simply by adding a dual-head structure and entropy regularization, a deterministic distillation actor is transformed into a stochastic policy capable of exploration, filling the largest gap in distilled policies.
- **Complementary Innovations**: The prior handles offline "precise exploitation," while entropy control manages online "effective exploration." Ablations clearly decouple these benefits, demonstrating logical consistency.

## Limitations & Future Work
- **Increased Training Overhead**: Advantage Noise Selection requires the teacher to generate $N_{\text{cand}}$ candidates and pass them through the critic at each step, making training roughly 1.6x slower than FQL. This cost increases with larger action spaces or candidate counts.
- **Dependency on Critic Quality**: Advantage noise selection relies on Q-values. If the critic overestimates in unseen regions, the selected "Golden Start" might mislead the prior. While the authors mitigate this in MultiCrescent, a robust analysis of critic error sensitivity is missing.
- **Gaussian Stochasticity Assumption**: The student output is still a diagonal Gaussian. Although compensated by the multi-modal VAE prior, the expressive power of the single-step output distribution itself remains limited.
- **Evaluation Scope**: Focused on continuous control benchmarks (OGBench/D4RL/Visual); not yet validated on real-world robots or higher-dimensional tasks like VLA.

## Related Work & Insights
- **Generative Policy Acceleration**: Compared to FQL, which only performs single-step distillation and ignores the noise prior, this work treats the prior as an optimizable variable. Compared to DSRL (which learns a Gaussian prior for online adaptation), GS-flow uses a more flexible VAE prior with negligible inference overhead.
- **Online Exploration for Generative Policies**: Unlike methods that inject randomness during denoising or estimate entropy via GMMs (computationally expensive), or EXPO which trains an extra Gaussian edit policy, this work embeds entropy control directly into the distillation loss, making it more lightweight.
- **Insights**: When a generative model is distilled into a single-step policy, the "starting point design" and "output distribution form" are two often-overlooked but high-return tuning knobs. This perspective is generalizable to other single-step generative control and VLA acceleration scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Relocating guided noise from image generation to RL distillation and adding entropy control to convert point-to-point mapping are well-motivated and cleverly combined.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers OGBench/D4RL/Visual benchmarks and offline-to-online transitions. Baselines include Gaussian, Diffusion, and Flow methods. Ablations decouple the two innovations from computational overhead; lacks real-robot/high-dimensional validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and methods are clearly articulated. The MultiCrescent visualization effectively illustrates prior transfer. Formulas and algorithm flows are complete.
- **Value**: ⭐⭐⭐⭐ — Simultaneously improves exploitation and exploration while maintaining single-step speed. Highly practical for generative policy deployment (real-time control/VLA acceleration) with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints](entropy_regularizing_activation_boosting_continuous_control_large_language_model.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[ICLR 2026\] Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies](multimodal_llm-assisted_evolutionary_search_for_programmatic_control_policies.md)
- [\[ICML 2026\] Noise-Guided Transport: Imitation Learning from Random Priors](../../ICML2026/reinforcement_learning/noise-guided_transport_for_imitation_learning.md)
- [\[ICLR 2026\] Safe Exploration via Policy Priors](safe_exploration_via_policy_priors.md)

</div>

<!-- RELATED:END -->
