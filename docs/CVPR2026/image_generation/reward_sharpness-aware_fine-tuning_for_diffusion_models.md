---
title: >-
  [Paper Note] Reward Sharpness-Aware Fine-Tuning for Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Sharpness-Aware] This paper diagnoses "reward hacking" (where reward scores increase while visual quality degrades) in Reward-Directed Reinforcement Learning (RDRL) for diffusion models as a form of "adversarial attack." The core issue is that reward models lack robustness in directions where their loss surfaces are steep. To address t
tags:
  - CVPR 2026
  - Image Generation
  - Sharpness-Aware
date: 2026-05-08
content_hash: 4100b4dbbdbe9c05
---
# Reward Sharpness-Aware Fine-Tuning for Diffusion Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Reward_Sharpness-Aware_Fine-Tuning_for_Diffusion_Models_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Diffusion Models / Image Generation / RLHF Alignment  
**Keywords**: Reward hacking, Diffusion model fine-tuning, Sharpness-Aware, Adversarial robustness, Plug-and-play

## TL;DR
This paper diagnoses "reward hacking" (where reward scores increase while visual quality degrades) in Reward-Directed Reinforcement Learning (RDRL) for diffusion models as a form of "adversarial attack." The core issue is that reward models lack robustness in directions where their loss surfaces are steep. To address this, RSA-FT is proposed. Instead of retraining the reward model, it utilizes the gradient of a "smoothed" reward model. This is achieved by simultaneously applying perturbations in the **image space** (adversarial input perturbation) and **parameter space** (SAM-style weight perturbation) to find the local worst reward. This dual approach significantly mitigates reward hacking and can be integrated as a plug-and-play module into various RDRL frameworks such as ReFL, DRaFT-K, AlignProp, and DRTune across multiple diffusion backbones.

## Background & Motivation
**Background**: Inspired by the success of RLHF in Large Language Models (LLMs), text-to-image diffusion models have begun utilizing Reinforcement Learning (RL) fine-tuning to align with human preferences. This family of methods is known as Reward-Directed Reinforcement Learning (RDRL, e.g., ReFL, DRaFT-K, AlignProp, DRTune). Since collecting real-time human feedback during training is impractical, **reward models** (e.g., HPSv2) trained on human annotations serve as differentiable proxies for human preferences, allowing the diffusion model to perform gradient ascent on them.

**Limitations of Prior Work**: RDRL common suffers from **reward hacking**, where reward scores consistently rise while perceptual quality stagnates or declines (e.g., garbled text, anatomical deformities). A systematic analysis of this phenomenon has been lacking in the context of diffusion RL.

**Key Challenge**: The reward model $r$ is merely an approximation of the true human preference $r^\star$. When the generator optimizes along certain "steep directions" of $r$, it acts like an adversarial attack—tiny image perturbations can cause reward logits to skyrocket without real quality improvement, pushing the updates into isolated non-preferred regions that deviate from $r^\star$. The authors draw an analogy: "Non-robust classifiers damage sample quality under classifier guidance, whereas robust classifiers obtained via adversarial training mitigate this." However, retraining an equivalently robust reward model for human preference alignment is excessively costly (requiring larger models and more annotations).

**Goal**: To "robustify/smooth" the reward model without retraining it, thereby suppressing reward hacking.

**Key Insight**: Drawing from randomized smoothing, robustness can be enhanced by smoothing the predictions of a fixed model rather than retraining. The authors observe that "reward models are precisely non-robust where the loss surface is steep," leading them to use the gradient of a smoothed reward model. Furthermore, this smoothing naturally induces "worst-case parameter perturbations," sharing origins with Sharpness-Aware Minimization (SAM). A duality exists between SAM (parameter space) and Adversarial Training (AT) (input space).

**Core Idea**: Replace the "original reward model gradient" with a "smoothed reward model gradient," applying smoothing in both image and parameter spaces (dual robustness) to suppress reward hacking. This work is the first to unify AT and SAM within the RDRL framework.

## Method

### Overall Architecture
The essence of the method is to replace the original RDRL goal of "maximizing reward $r$" with "maximizing a smoothed reward $\tilde r^d$." The standard RDRL objective is $\mathcal{J}(\theta)=\max_\theta\mathbb{E}_{c,x_T}\big[r(x_0(x_T,c;\theta),c)\big]$, where $x_0$ is the final sample obtained by denoising from noise $x_T$ under condition $c$. This paper modifies it to $\mathcal{J}(\theta)=\max_\theta\mathbb{E}_{c,x_T}\big[\tilde r^d(x_0(x_T,c;\theta),c)\big]$, where the smoothed reward is defined as the **minimum** reward within a local neighborhood: $\tilde r^d(x,c):=\min_{d(x,x')<\rho}r(x',c)$, with $d(\cdot,\cdot)$ being a distance metric on the image manifold. The authors use two metrics (image space and parameter space) to perform one-step approximations and merge them into the joint RSA-FT objective. The method acts as a **plug-and-play objective function replacement**: it requires no changes to the diffusion backbone, no retraining of the reward model, and no additional training stages.

Since it is essentially a loss reformulation using "adversarial/SAM-style regularization" rather than a multi-stage pipeline, the mechanism is described via its algorithm (Algorithm 1): for each step, sample noise $x_T$ and condition $c$ → generate image $x_0$ → calculate image-space perturbation $x_0+\delta_{x_0}$ → calculate parameter-space perturbation $\theta+\epsilon_\theta$ → update $\theta$ using the joint objective.

### Key Designs

**1. Reward Sharpness Hypothesis & $S_1$ Metric: Quantifying Reward Hacking by Surface Steepness**

This is the diagnostic foundation. The authors hypothesize that the reward model $r$ generalizes best where its reward surface is **locally flat**, whereas deviations from true preference $r^\star$ occur at **steep** locations. Reward hacking is seen as the generator "exploiting" these steep directions. To quantify this, the reward sharpness metric is defined as $S_1=\mathbb{E}_{x\sim\mathcal{D}}\big[r(x)-\min_{\|\epsilon\|<\rho}r(x+\epsilon)\big]$, representing the "drop" in reward within a local neighborhood. This is approximated via a one-step update: $S_1\approx\mathbb{E}_x\big[r(x)-r(x-\rho\frac{\nabla_x r(x)}{\|\nabla_x r(x)\|})\big]$. Empirical results using SD1.5 fine-tuned with DRaFT-K show a strong **negative correlation** between reward sharpness and preference quality ($r_{corr}=-0.802$ for PickScore, $-0.669$ for ImageReward), confirming that "sharpness = poor generalization = reward hacking."

**2. Image-Space Smoothing: Adversarial Training on Rewards**

To prevent the generator from exploiting rewards at the image level, image-space smoothing finds the worst reward in the neighborhood: $\max_\theta\mathbb{E}\big[min_{\|\delta\|<\rho}r(x_0+\delta,c)\big]$. This is isomorphic to applying adversarial perturbations to the reward model. Using a one-step approximation, the perturbation $\delta_{x_0}=-\rho\frac{\nabla_{x_0}r(x_0,c)}{\|\nabla_{x_0}r(x_0,c)\|}$ is applied in the direction opposite to the reward gradient. The objective becomes $\max_\theta\mathbb{E}\big[r(x_0+\delta_{x_0},c)\big]$. Intuitively, this forces the generator to achieve high rewards throughout a $\rho$-ball neighborhood rather than in a single pixel direction, effectively flattening "isolated non-preferable peaks."

**3. Parameter-Space Smoothing: SAM-style Weight Perturbation**

The "worst-case local" principle is extended to the **parameter space**: $\max_\theta\mathbb{E}\big[\min_{\|\epsilon\|<\rho_\omega}r(x_0(x_T,c;\theta+\epsilon),c)\big]$. One-step approximation yields a SAM-style weight perturbation $\epsilon_\theta=-\rho_\omega\frac{\nabla_\theta r(x_0,c)}{\|\nabla_\theta r(x_0,c)\|}$, leading to the objective $\max_\theta\mathbb{E}\big[r(x_0(x_T,c;\theta+\epsilon_\theta),c)\big]$. Following SAM's practice, a stop-gradient is applied to $\epsilon_\theta$ during the outer optimization. This encourages convergence to **flat parameter minima** on the reward surface, which generalize better and suppress reward hacking from the weights side. This step marks the first application of the duality between AT (input space) and SAM (parameter space) in RDRL.

**4. RSA-FT Joint Objective: Dual Image and Parameter Smoothing**

While smoothing in either space independently can mitigate reward hacking, the authors found they are **complementary**. The final joint objective overlays both perturbations: $\max_\theta\mathbb{E}_{c,x_T}\big[r(x_0(x_T,c;\theta+\epsilon_\theta)+\delta_{x_0},c)\big]$. This enforces smoothing in both spaces, resulting in "dual-robust" reward optimization. Both perturbation radii are searched within $\{10^{-1},10^{-2},10^{-3}\}$, with $10^{-2}$ being optimal. It remains plug-and-play by replacing the original reward term with this joint objective.

### Loss & Training
Training involves gradient ascent on the diffusion parameters $\theta$ using the joint objective (Algorithm 1). Implementation details: AdamW on H100 ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$); SD1.5/SDXL sampled for 50 steps, SD3 for 28 steps; learning rate $2\times10^{-5}$, batch size 32; perturbation radii $\rho=\rho_\omega=10^{-2}$. Iterations and epochs follow the original protocols of the baselines to purely validate the gain from RSA-FT.

## Key Experimental Results

### Main Results
Using HPSv2 as the training reward signal, RSA-FT was integrated into ReFL, DRaFT-K(K=1), AlignProp, and DRTune. Evaluations were conducted on DrawBench and the HPSv2 test set using metrics HPSv2.1, PickScore, and ImageReward. For SD1.5 (512×512), **all three metrics increased simultaneously** across all baselines when adding RSA-FT:

| Method (SD1.5 / DrawBench) | HPSv2.1↑ | PickScore↑ | ImageReward↑ |
|:---|:---:|:---:|:---:|
| Vanilla | 24.02 | 21.02 | -0.147 |
| AlignProp | 25.12 | 20.98 | -0.033 |
| AlignProp + Ours | **29.59 (+4.47)** | **21.51 (+0.53)** | **0.268 (+0.30)** |
| ReFL | 31.08 | 21.57 | 0.536 |
| ReFL + Ours | 31.67 (+0.59) | 21.70 (+0.13) | 0.671 (+0.135) |
| DRTune | 30.63 | 21.34 | 0.477 |
| DRTune + Ours | 31.16 (+0.53) | 21.52 (+0.18) | 0.540 |

On HPD subsets, gains were even more pronounced; for example, AlignProp's HPSv2.1 improved from 24.93 to 32.02 (+7.09). Crucially, while AlignProp/Draft-LV originally saw HPSv2.1 rise while auxiliary rewards fell (typical reward hacking), adding RSA-FT led to a **unified increase**, indicating genuine alignment rather than metric overfitting.

### Ablation Study

| Configuration | Effect | Description |
|:---|:---:|:---|
| Image perturbation only | Independent improvement | Mitigates reward hacking and improves alignment alone. |
| Parameter perturbation only | Independent improvement | SAM-style weight smoothing is also effective alone. |
| Image + Parameter (RSA-FT) | Maximum gain | Significant complementary and synergistic effects. |

Note: Detailed ablation values are provided in Appendix E / Table 6 of the original paper.

### Key Findings
- **Reward Hacking ≈ Reward Surface Sharpness**: The sharpness metric $S_1$ shows a strong negative correlation with human preference ($-0.802$/$-0.669$), providing quantitative evidence linking intuitive phenomena to measurable metrics.
- **Robust across Backbones/Resolutions/Architectures**: RSA-FT consistently improves scores from SD1.5 (512²) to SDXL/SD3 (1024²) and the Flux.1-dev model (MMDiT), demonstrating that "smoothing rewards" is a generic strategy.
- **Human Study Evidence**: In a study with 17 annotators, RSA-FT exceeded the 50% preference threshold in most cases (e.g., SD3+ReFL visual preference 65.4% vs 34.6%), though the sample size is noted as a limitation.

## Highlights & Insights
- **A Single "Smoothing" Unifies AT and SAM**: By identifying image-space perturbation as adversarial training and parameter-space perturbation as SAM, the paper provides an elegant "dual perspective" explaining why their combination is most effective.
- **Engineering Value of No Retraining**: Constructing robust reward models usually requires large models and data; RSA-FT achieves robust gradients simply by modifying the objective, making it highly practical.
- **Formalizing Reward Hacking as an Adversarial Attack**: This maps a vague failure mode in generative alignment to established tools in adversarial robustness and randomized smoothing, allowing for the reuse of existing theories.

## Limitations & Future Work
- Evaluation relies heavily on model-based metrics (HPSv2/PickScore/ImageReward), which are imperfect proxies for human preference. The human study was small-scale.
- Reward hacking was studied only under the **single reward model** setting; complementary smoothing across multiple reward models remains to be explored.
- Smoothing uses a one-step **minimization** approximation; while Gaussian averaging might be more robust, the one-step method was chosen for efficiency.
- Currently, smoothing is applied **uniformly** to all samples. Future work could explore "selective sharpness weighting" to down-weight excessively sharp samples.

## Related Work & Insights
- **vs Adversarial Training (AT)**: AT improves robustness via worst-case loss in input space but requires retraining. Ours applies "input perturbation" to rewards to modify the generator's objective without retraining the reward model.
- **vs SAM (Sharpness-Aware Minimization)**: SAM encourages flat minima in parameter space. Ours applies this to reward optimization in RDRL and unifies it with image-space perturbations.
- **vs Randomized Smoothing / AWP**: Randomized smoothing inspires the "smoothed reward." AWP uses both input and weight perturbations in classification; RSA-FT is the first to extend this principle to the RDRL framework.
- **vs Existing RDRL (ReFL/DRaFT-K/AlignProp/DRTune)**: These frameworks maximize rewards directly and are prone to hacking. RSA-FT acts as a plug-and-play enhancement for all of them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Diagnosis of reward hacking as adversarial attack and unification of AT+SAM is a fresh, self-consistent perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 4 frameworks, 4 backbones, and 2 benchmarks, though some results are in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic from hypothesis to metrics to methodology; provides strong geometric intuition.
- Value: ⭐⭐⭐⭐⭐ Zero extra training, plug-and-play, and robust across backbones; highly practical for the diffusion alignment community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do Less, Achieve More: Do We Need Every-Step Optimization for RL Fine-tuning of Diffusion Models?](do_less_achieve_more_do_we_need_every-step_optimization_for_rl_fine-tuning_of_di.md)
- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[CVPR 2026\] Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](towards_fine-grained_attribution_instance-aware_preference_optimization_for_alig.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](../../CVPR2025/image_generation/personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)

</div>

<!-- RELATED:END -->
