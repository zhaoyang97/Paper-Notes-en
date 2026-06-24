---
title: >-
  [Paper Note] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
description: >-
  [ACL 2025][LLM (Other)][Synthetic Data] DiffLM proposes a controllable data synthesis framework based on VAE + Latent Diffusion + Frozen LLM Decoder. By introducing a diffusion process in the latent space to precisely model the real data distribution and injecting distribution information into the LLM via soft prompts, the framework achieves a synthesis quality that outperforms real data by 2%-7% across three types of structured data: tables, code, and tools.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Synthetic Data"
  - "VAE"
  - "Diffusion Models"
  - "LLM"
  - "Structured Data Generation"
date: 2026-05-08
content_hash: 1d32afe54731acfe
---

# DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models

**Conference**: ACL 2025  
**arXiv**: [2411.03250](https://arxiv.org/abs/2411.03250)  
**Code**: [bytedance/DiffLM](https://github.com/bytedance/DiffLM)  
**Area**: Data Synthesis, Generative Models  
**Keywords**: Synthetic Data, VAE, Diffusion Models, LLM, Structured Data Generation  

## TL;DR

DiffLM proposes a controllable data synthesis framework based on VAE + Latent Diffusion + Frozen LLM Decoder. By introducing a diffusion process in the latent space to precisely model the real data distribution and injecting distribution information into the LLM via soft prompts, the framework achieves a synthesis quality that outperforms real data by 2%-7% across three types of structured data: tables, code, and tools.

## Background & Motivation

- **Core Challenges of LLM Data Synthesis**: Existing LLM data synthesis methods face two major issues: (1) LLMs lack a global understanding of the target data distribution, leading to low generation diversity and susceptibility to data copying; (2) they rely on complex prompt engineering and multi-agent frameworks, making them difficult to quickly adapt to new tasks.
- **Failure of Direct VAE Application**: Although VAEs and diffusion models perform exceptionally in image synthesis, directly sampling from the latent distribution learned by VAEs for text generation yields text completely unrelated to the target distribution. This occurs because of the significant discrepancy between the encoder posterior $q_\phi(z|x)$ and the prior $p(z)$, resulting in large unutilized or vacant regions in the latent space.
- **Decoupling Paradigm**: This work decouples data distribution learning from the LLM's generative capability, enabling the LLM to retain internal knowledge while being guided by the real data distribution, achieving high-quality structured synthetic data generation.

## Method

### Overall Architecture

DiffLM consists of three core components: (1) a trainable Transformer encoder (VAE encoder) that maps text to a continuous latent space; (2) a latent diffusion module that trains a denoising network on the VAE latent space to model the latent distribution more precisely; (3) a parameter-frozen LLM decoder that receives latent features via soft prompt injection for reconstruction/generation.

### Key Designs

1. **VAE Representation Learning and Distribution Decoupling**: A trainable encoder maps the structured text $s_i$ into $x_i \in \mathbb{R}^{d \times 2}$, which is split into mean $\mu$ and standard deviation $\sigma$. The latent representation is obtained through reparameterization as $z = \mu + \sigma \odot \epsilon$. All parameters of the LLM decoder are frozen to avoid retraining and retain general knowledge, completely decoupling distribution learning from the generation objective.
2. **Latent Space Diffusion Denoising**: Addressing the core issue of poor VAE sampling quality, latent vectors $z_0$ are extracted from the trained VAE. A forward diffusion process $z_t = z_0 + \sigma(t)\epsilon$ is executed, and a denoising network is trained to learn the reverse process to recover $z_0$.
3. **Soft Prompt Latent Injection**: Through an MLP, the latent representation $z$ is projected into $k$ soft prompt embeddings $\mathbf{H}_{\text{latent}} \in \mathbb{R}^{k \times d}$. These are prepended to the BOS token as guide vectors for LLM decoding, achieving plug-and-play distribution control.

### Loss & Training

VAE training adopts a $\beta$-VAE strategy:

$$\text{ELBO}_\beta = L_{rec} - \beta L_{kl}$$

- $L_{rec}$: Reconstruction likelihood of the language model
- $L_{kl} = D_{\text{KL}}(q_\phi(z|x) \| p(z))$: KL divergence regularization
- Adopts a **decreasing $\beta$ strategy**: Initial large $\beta$ strongly constrains latent space smoothness, and after the reconstruction loss converges, $\beta$ is gradually decreased to enhance reconstruction accuracy.
- The diffusion module uses a denoising score matching loss: $\mathcal{L}_{\text{diff}} = \mathbb{E}\|\epsilon_\theta(z_t, t) - \epsilon\|^2$

## Experimental Results

### Main Results

| Method | Adult MLE↑ | Default MLE↑ | Magic MLE↑ | Shoppers MLE↑ | Beijing RMSE↓ |
|------|-----------|-------------|-----------|-------------|-------------|
| Real Data | 0.927 | 0.770 | 0.946 | 0.926 | 0.423 |
| GReaT (Fine-tuned GPT-2) | 0.913 | 0.755 | 0.888 | 0.902 | 0.653 |
| GPT-4 ICL | 0.889 | — | 0.864 | 0.835 | 0.992 |
| TabSyn (Diffusion) | 0.915 | 0.764 | 0.938 | 0.920 | 0.582 |
| **DiffLM** | **0.906** | **0.794** | **0.917** | **0.915** | **0.696** |

In code generation experiments, continuing pre-training Mistral-7B with DiffLM synthetic data achieves a HumanEval pass@1 of 35.37%, significantly exceeding the real-data version (28.58%) and CodeLLaMA-7B (33.50%). The gains are even larger on the 12B model: 42.24% vs 36.97%.

### Ablation Study: Injection Method and β Strategy

| Configuration | Reconstruction Loss | Downstream Performance |
|------|---------|---------|
| Decreasing β + Soft Prompt Injection | Lowest | Optimal |
| Cyclic β + Soft Prompt Injection | Higher | Suboptimal |
| Decreasing β + Prefix Injection | Medium | Medium |
| Decreasing β + Cross Attention | Higher | Poorer |

### Key Findings

- The downstream performance of DiffLM synthetic data on the Default dataset **surpasses real data** (MLE 0.794 vs 0.770), indicating that the synthetic data incorporates supplementary knowledge.
- In code scenarios, continuing pre-training on real data causes performance degradation on MBPP, whereas DiffLM synthetic data achieves comprehensive improvements.
- The latent diffusion module is a critical component: without it, the text sampled from the VAE is independent of the target distribution.
- In tool synthesis, the GPT-4 score of individual tools generated by DiffLM is higher than that of real data, with about 1/3 of the categories matching or exceeding in diversity.
- The decreasing $\beta$ strategy is significantly superior to the cyclic annealing strategy.

## Highlights & Insights

- First to combine VAE, latent diffusion, and LLM for high-quality structured data synthesis, presenting a framework that is both elegant and plug-and-play.
- Through a decoupled design, distribution control is achieved simply by freezing the LLM parameters, eliminating the need to fine-tune the large model.
- Comprehensively validated across tabular, code, and tool scenarios, with synthetic data outperforming real data in certain tasks.
- The decreasing $\beta$ strategy and soft prompt injection are ablation-verified as the optimal combination.

## Limitations & Future Work

- Only validates unconditional synthesis scenarios, lacking exploration into conditional generation (control by category/attribute).
- Tabular synthesis does not outperform domain-specific models like TabSyn on column distribution density metrics.
- The frozen LLM decoder may limit adaptation to data formats in specific specialized domains.
- Lacks assessments regarding the privacy and fairness of the synthetic data.

## Related Work & Insights

- **LLM Data Synthesis**: GReaT (Borisov et al., 2023) fine-tunes GPT-2 for tabular synthesis; ICL prompting methods lack diversity and perform poorly on complex structured data.
- **Latent Variable Text Generation**: Works like Optimus (Li et al., 2020) explore the integration of latent space with LMs, but have limited effectiveness on complex structured data.
- **Tabular Data Generation**: Specialized models such as CTGAN, TVAE, and TabSyn show excellent performance but lack generalizability.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Theorem Prover as a Judge for Synthetic Data Generation](theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)

</div>

<!-- RELATED:END -->
