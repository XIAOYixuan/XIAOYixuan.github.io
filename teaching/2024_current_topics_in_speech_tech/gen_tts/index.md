---
layout: post
title: Generative Models for TTS
---


# Introduction

Generative models are a class of models that aim to learn the probability distribution of training data, so that it can generate new data by sampling from the learned distribution.
More formally, a generative model estimate the data distribution $$P_{\theta}(x)$$ that matches the true data distribution $$P_{D}(x)$$ 
{% include sidenote.html id="note-prob" note="$$\theta$$ represents the parameters of the model, and $$D$$ is the training data. $$x$$ can be the acoustic feature or the waveform. e.g., if $$\theta$$ is an acoustic model, then it generates the acoustic feature $$x$$; if $$\theta$$ is a vocoder, then it generates the waveform $$x$$" %}.

One important aspect of generative models is that the generation process is inherently non-deterministic. This means that each time we sample from the model, it can produce a different result. This characteristic aligns well with the nature of TTS, which inherently involves a one-to-many mapping: for a given text input, there can be multiple valid outputs, as the text can be spoken with different voices, intonations, and prosody. 

In the following sections, we will cover four types of generative models and their applications in TTS. More specifically, we will focus on how to adjust these models such that they can generate acoustic features conditioned on text inputs, or generate waveforms conditioned on acoustic features.

1. Generative Adversarial Networks (GANs)
2. Variational Autoencoders (VAEs)
3. A Flow-based method: Normalizing Flows (NFs)
4. Denoising Diffusion Probabilistic Models (DDPMs)

# GANs

## Concept and Formulation

{% include marginfigure.html id="mp1" url="assets/img/gan.png" description="GAN diagram." %}

A raw GAN includes two networks trained in an adversarial manner: a generator network $$G(z)$$ that generate samples $$x$$ from random noise $$z$$, and a discriminator network $$D(x)$$ that evaluates whether the generated samples are real $$y$$ or fake. The training objective can be viewed as a minimax game between the generator and the discriminator. Following is the original formulation (by [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)):

$$
\min_{G_{\theta}} \max_{D_{\phi}} V(D_{\phi}, G_{\theta}) = \mathbb{E}_{x \sim p_{data}(x)}[\log D_{\phi}(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D_{\phi}(G_{\theta}(z)))]
$$

**Detailed explanation of the formula**

1. The discriminator $$D_{\phi}$$ learns a probability distribution $$D_{\phi}(x)$$ over two classes: real and fake. It assigns high probability to real data $$x$$ and low probability to fake data $$G_{\theta}(z)$$.
2. $$\mathbb{E}_{x \sim p_{data}(x)}[\log D_{\phi}(x)]$$ is the expected value of the log probability of the discriminator correctly classifying real data {% include sidenote.html id="note-gan-discriminator" note="$$x \sim p_{data}(x)$$ means the data is sampled from the true data distribution, so this part means when a real sample is given, the discriminator should identify it as real. Since it's a discriminative loss function, assigning high probability to real data means assigning low probability to fake data, focusing on the real data in this part is equivalent to focusing on the fake data at the same time. " %}, and $$\max_{D_{\phi}}$$ means the goal of training the discriminator is to maximize the probability of correctly classifying real data. In detail, since the range of $$D_{\phi}(x)$$ is [0, 1], to maximize the expected value, $$D_{\phi}(x)$$ is encouraged to be close to 1, which means it assigns high probability to real data $$x$$.
3. $$\mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D_{\phi}(G_{\theta}(z)))]$$ is another expected value term. $$z \sim p_{z}(z)$$ means the noise vector $$z$$ is sampled from a prior distribution $$p_{z}$$ (e.g., Gaussian distribution), then the generator uses it to generate a fake sample $$G_{\theta}(z)$$. The discriminator $$D_{\phi}$$ takes the sample and assigns a probability $$D_{\phi}(G_{\theta}(z))$$ to it. The goal of the generator is then to minimize the log function of $$1 - D_{\phi}(G_{\theta}(z))$$, which means it encourages the discriminator to assign high probability to the fake sample $$G_{\theta}(z)$$ (to make discriminator believe the fake sample is real).


**Training process**: Training GANs is done in an alternating manner. The discriminator $$D_{\phi}$$ is trained to better distinguish real and fake samples. 
Given the signal provided by the discriminator (i.e., the gradient from the discriminator), the generator $$G_{\theta}$$ is trained to better generate more convincing samples. To summarize, the training process includes four steps:
1. Sample a minibatch $$x$$ from the real speech set.
2. Sample a minibatch of noise vectors $$z$$ from the prior distribution $$p_{z}$$.
3. Fix the discriminator $$D_{\phi}$$, feed the noise vectors to the generator $$G_{\theta}$$, get the output $$G_{\theta}(z)$$, update the generator $$G_{\theta}$$.
4. Fix the generator $$G_{\theta}$$, feed the generated samples $$G_{\theta}(z)$$ and the real samples $$x$$ to the discriminator $$D_{\phi}$$, update the discriminator $$D_{\phi}$$.

At equilibrium, the generator $$G_{\theta}$$ successfully mimic the target data distribution $$p_{data}(x)$$, and the discriminator $$D_{\phi}$$ can no longer distinguish real and fake samples (outputting 0.5 for all samples).
In practice, training GANs can be challenging. For example, the adversarial training process is not guarantted to converge, so if not carefully designed (e.g., bad learning rates), the performance can oscillate where $$G_{\theta}$$ and $$D_{\phi}$$ continually outperform each other without reaching equilibrium. 

Another issue is called *mode collapse*, where $$G_{\theta}$$ is very "smart" and finds out that it only needs to generate one speicial sample or a subset of very convincing samples to fool $$D_{\phi}$$, then GANs can reach a stable state but the variety of generated samples is limited. 
On the other hand, $$D_{\phi}$$ can learn too fast, e.g., it always assigns very high probability to real data (or very low probability to fake data), so signal from $$D_{\phi}$$ is too weak to guide $$G_{\theta}$$. One solution is to use Wasserstein GANs (WGANs) to provide stronger signals, since it doesn't use cross-entropy loss but a distance-based loss that tells $$G_{\theta}$$ the distance between its learned distribution and the target distribution $$p_{data}(x)$$.


## Applications in TTS: GAN-TTS, MelGAN and HiFiGAN


# VAEs

## Concept and Formulation

## Applications: VITS

# NFs

## Concept and Formulation

## Applications: GlowTTS 

# DDPMs

## Concept and Formulation

## Applications: DiffTTS 

## Applications: DiffWave