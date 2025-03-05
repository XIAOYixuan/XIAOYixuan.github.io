---
layout: post 
title: Guidance of Study and Exam Prep
---

TTS is relatively complex course. Similar to ASR, it’s a task that has many sub-tasks, which are often interconnected rather than independent. Also, it’s not a classification task you might have seen many times before, but a one-to-many generation task. As a result, in prepareing this course, I believe you might have encountered the following challenges, and I tried to based on them to adjust the content. I guessed understanding the motivation of slides design might be helpful to understand the content, so I put it as follow.
{% include sidenote.html id="note-pre" note="since I’m not very familiar with curricula outside of IMS, I’m considering challenges mainly from the perspective of the IMS students."%}

**Generative Models**

The intro to deep learning course at the IMS doesn’t cover much on generative models, but in the advanced deep learning course, there’s a topic on this. Considering that 1) advanced DL is not a prerequisite 2) generative models are wildly used in current TTS systems 3) students choosing the TTS topic might need to read papers that base their models on generative models, I think it’s important to introduce these model types in more details.

**Maths**

Generative models are probabilistic models; they model uncertainty of prediction rather than deterministic values. So understanding these models makes it hard to completely avoid topics in probability, statistics, linear algebra, and calculus. Among the four models we cover, except for GANs, the other three aim for rigorous estimation of the marginal likelihood. I’m aware that some students might lack of math backgrounds, so initially, I tried to prepare slides without touching on math, but I found that, apart from broadly stating the main ideas, it was diffficult to causally relate various concepts. If I present it in this way, you’ll be forced to memorize concepts, like knowing that the objective of a VAE is ELBO or that NFs use many invertible functions. But why using ELBO models the marginal likelihood or why these functions must be invertible become confusing. Therefore, I decided to include some maths in the end.

If you want to better understand these concepts and build a solid foundation in DL, I recommend reading all the slides and exploring the external materials or links provided. However, I understand that for students who don’t have math training during their undergraduate studies, fully understanding the concepts would require a significant workload. So, if you just want to pass the exam, the slides marked as an aside won’t be asked, and the derivations won’t be examined either. Even for the formulas; if an objective is asked, you only need to intuitively explain what terms the formula has, and the motivation behind designing each term to get points. 

Of course, if you can logically write down the full formula during the exam without lengthy explanations, you’ll aslo get points and answer more quickly! Or you can try to write it down entirely based on memorization, but you’ll find that less efficient than understanding the intuition. If you memorize and misplace a subscript, the entire meaning of the formula could change, leading to lost points.

**Speech**

Speech has its unique characteristics, different from images and text. If you lack sufficient understanding of speech and signal processing, you might feel confused when trying to grasp certain model arch designs, now knowing why they’re designed that way. If you do understand the propertie of speech, some designs will seem natural, and you won’t need to memorize them explicitly. You’ll think, “ah sure, that’s the way to do it, of course” For example, why does MelGAN use window-based methods 
{% include sidenote.html id="note-meldgan" note="cuz sliding windows are common in speech processing" %}
, or why does HiFiGAN need to handle periodicity 
{% include sidenote.html id="note-hifigan" note="speech signal can be decomposed into sinosidual waves with different periods/frequencies, and the freq humans can produce are limited" %}
. Without knowing this, you might find them odd at first. I tried to include the speech properties when introducing these concepts, but as mentioned in the introduction, having a foundational knowledge of speech processing is important for this course.

Of course, if you just need enough credits to graduate, and aren’t interested in speech itself, you’ll still be able to pass the exam. Just need more effort to memorize and learn the design. However, if you’re interested in speech technology, learning about phonetics, phonology, and signal processing is crucial 
{% include sidenote.html id="note-speech" note="the other linguistics is also important for speech understanding tasks, but I assume you already know enough from the Method class or other NLP tasks" %}. So it’s worth to check the materials uploaded by Sarina in the **Additional Resources folder.**

Below are my notes from preparing the TTS part. They’re much more concise than the slides and mainly highlight the concepts I consider important . You can use these notes to prepare for the exam. All bullet points marked as *advanced* are optional. Some of these optional ones were eventually removed from the final slides, but the non-advanced parts should be fully covered. If you learn all the concepts including the advanced ones, you’ll have a better grasp of the content and better prepared for the project seminar. Skipping them won’t affect your exam score 
{% include sidenote.html id="note-skip" note="but you’ll eventually need them to understand the papers during the project seminar" %}
. Simply put, you’re not required to understand every single detail of the slides to pass the exam.  Enjoy!

Here's the original notion [link](https://tomato-bit.notion.site/CTiST-2024WS-TTS-Notes-12ff7ec8709880e19ef9d74fa249c279?pvs=4)

# Overview:

- different speech synthesis techniques
    - the difference between some speech synthesis tasks, e.g., tts, voice cloning, voice conversion
    - tts challenges
- classical tts 
- end-to-end tts

# Classicial TTS Overview

- Text Analysis, Acoustic models, vocoder
- Introduce the input and output, the motivation, the model types

# Text Analysis

- normalization
    - preprocessing: motivation of preprocessing - NSWs, how to handle NSWs 
    {% include sidenote.html id="note-preprocessing" note="example solution: rule-based, neural models" %}
    - tokenization: 
        - motivation {% include sidenote.html id="note-tokenization" note="as inputs, ease the other operations" %} and methods {% include sidenote.html id="note-tokenization-methods" note="rule-based, neural models, statistics-based methods" %}
        - summary: prepare inputs for other models, ease phonetic analysis, provide prosodic features
    - end of sentence prediction: provide prosodic features
- phonetic analysis (G2P)
    - motivation {% include sidenote.html id="note-g2p" note="e.g., pronunciation varies for different abbreviations or acronyms" %}
    - goal of G2P: turn graphemes to phonemes
    - rule-based solution → two issues {% include sidenote.html id="note-rule-based" note="e.g., large vocabulary, OOV" %}
    - WFST → use less storage
    - neural models → handle oov
- prosodic analysis
    - prosody: what are they, example, fine-grained prosodic features and high-level features
    - how to extract prosodic features: signal processing tools, two example application 
    {% include sidenote.html id="note-prosodic-features" note="how they’re used in training, how they’re used in dataset construction" %}
 
# Acoustic Models

- input and output
- the learning objective difference between ASR and TTS

## Basic DNN synthesis

- input and output representation (vectorized i/o)
    - how to prepare the input
- alignment problem
    - a naive solution: simple upsampling 
        - the problem of this solution {% include sidenote.html id="note-naive-upsampling" note="not every phone takes the same amount of output frames, so we need a duration model" %}
        - duration model: how to train this model
            - how to prepare the labels {% include sidenote.html id="note-duration-model-labels" note="forced alignment is needed" %}
            - how to prepare the inputs
            - what is the task 
    - advanced solutions: forced alignment involved
        - HMM-GMM ASR acoustic model + searching algorithm
    - model implicitly learns the alignment: attention-based mechanism or duration prediction which trained together with the TTS {% include sidenote.html id="note-attention" note="would be covered in tacotron and fastspeech later" %}
- weakness of DNN model 

## RNN, Tacotron

- overview of the structure, input/output, wave reconstruction algorithm (griffin-lim)
- what is AutoRegressive (AR)
- important designs
    - encoder-attention-decoder architecture
    - encoder
        - k-gram modeling in CBHG blocks
        - contextualized representation using RNN
    - attention mechanism to handle the alignment problem
        - bottleneck in prenet to make it easy to do attention
    - decoder
        - multiple spectrogram prediction
        - stop-flag
        - post-processing net to smoothes the output
- others, tacotron 2
    - simplifed building blocks
    - advanced vocoder rather than griffin-lim

## Transformer, FastSpeech

- motivation, input, output
- what is non-autoregress (NAR)
- FastSpeech: important designs
    - length regulator to handle the alignment problem
    - duration prediction: how to find ground truth labels {% include sidenote.html id="note-duration-model-labels" note="monotonic alignment search on the encoder-decoder attention matrix, which is provided by another AR model" %}
- FastSpeech2: important design
    - motivation, how to mitigate mode-collapse
    - variance adaptor {% include sidenote.html id="note-variance-adaptor" note="introduce each adpator's prosodic feature it focuses on, ground truth labels, training objectives" %}
        - duration predictor: no need to do stop-flag
        - pitch predictor
        - energy predictor
- weakness: they are deterministic models, so lack of change

## Generative Models

- the goal, a probabilistic view
- motivation
- intro 4 types

### GAN

likelihood-free methods, iterative training

- main architecture and probabilistic graph representation
- explain model: minimax game
- objectives of G and D
    - intro the spectial training algorithms
- how to condition GAN on text for TTS
    - use an example to show the idea: use the condition signals in the input of the generator {% include sidenote.html id="note-gan-condition" note="The condition signals could be the linguistic features for acoustic models, or mel-spectrogram for vocoder. The signals can be used directly or together with the noise vecotr." %}
    - intro why GAN is more common in vocoder but not in acoustic model
- (advanced study materials) the probabilistic view of its objective, important when we need to  make comparison to other models

### VAE

variation inference, can only approximate actual marginal likelihood

- main architecture, and probabilistic graph representation
- the intuition {% include sidenote.html id="note-vae-intuition" note="is related to disenglement" %}
    - it introduce z, and tries to model $$p(z \vert x)$$
        - $$p(x \vert z)$$ is needed for estimation due to bayes rule
    - resolve the possible conflicts: I said generative models model $$p(x)$$, now it seems that we want to model $$p(z \vert x)$$
- model comparison
    - further emphasize disenglement, and the weakness of this assumption
    - likelihood-free and MLE
- model explain
    - VAE Encoder and decoder, what they try to model
- objective: ELBO
    - problem, intractability
    - (advanced) the essense, we have ELBO because it’s variation inference, for variational inference we need Gaussian to do approximation {% include sidenote.html id="note-vae-gaussian" note="the Gaussian distribution can handle one-to-many problems" %}
    - (advanced) resolve the conflicts, we claimed we can use MLE to update VAE, why do we use ELBO, show that maximize ELBO is equivalent to MLE
    - explain the two main term of ELBO: reconstruction and regularization terms
- VAEs in TTS: applicaton and how to condition on text
    - an example to show the main method: redesign ELBO's encoder $$q_{\phi}(z \vert x)$$ to condition on linguistic context ($$z_c$$) or speaker info ($$z_s$$), then sample from it

### NF

exact likelihood representation, MLE, but limit in model selection

- architecture and probabilistic graph representation
- motivation: VAE is approximation, NF can do exact likelihood estimation
    - so we need simple transformation
    - we need a series of simple transformation to model the actual complex distribution
    - we need two directions
    - (advanced) explain why we need two direction → bayes rule, bijective models
- model comparision
    - VAE: just approximation, because given $$p(x \vert z)$$, intractable to solve $$p(z \vert x)$$
    - NF solve this problem by using invertible functions
        - (advanced) explain why using this specific funciton can solve the problem
        - because of these functions, we have normalized density, that’s why it’s called Normalized
        - because the capacity of these function is limited, we need a series of them to approx complex distribution, that’s why it’s called Flows
- (advanced) explain the determinant form from a geometric perspective
- NFs in TTS: 
    - an example Glow-TTS: not using the sampled noise vector from unconditional $$p(z)$$, but the condition distribution $$p(z \vert x)$$ {% include sidenote.html id="note-glow-tts" note="$$p(z \vert x)$$ can be learned in a similar way as VAE, then sample a vector from it, and feed that to multiple invertible functions" %}
- weakness of NFs
    - limited model selection range
    - can struggle with high-dim data due to determinant calculation

### DDPM

approximation + markov assumption for sequence modeling, also use ELBO, but wilder model selection range

- pipeline and the probabilistic graph representation
- intuition of denoising and diffusion
- method explain
    - diffusion, how to add noise and the markov assumption
    - denoising, how to reduce noise, supposed to model the distribution that’s used to add noise, but in practice, just model the noise also work; also the markov assumption
- model comparison
    - NF, how many models they need
- DDPMs in TTS (how to condition on text, example DiffTTS)
    - simply put, linguistic signal is added to the hidden states in the model that's used to learn the noise distribution

# Vocoder

- motivation
    - if STFT is lossless compression, why can’t we go fro spectrogram to wave
    - phase reconstruction problem

## WaveNet

CNN-based, autoregressive (AR)

- the conflicts between CNN and AR: CNN takes future content as input, but AR doesn’t allow using future info {% include sidenote.html id="note-wavenet-future" note="so we need to use stacked dilated causal conv (SDCV) to make it causal" %}
- main module:
    - stacked dilated causal conv (SDCV)
    - gated activation units
- model explain:
    - input, represent by SDCV
    - how to get the output (amplitude per step)
        - naive solution, regressive the amplitude → but regression is hard due to the continuous nature
        - actual solution, classification task → need quantization
            - (advanced) what is $$\mu$$-law {% include sidenote.html id="note-mu-law" note="a nonlinear transformation that compresses the dynamic range of an audio signal, which allows lower amplitudes to have more detail while higher amplitudes are quantized more coarsely. This compression is especially effective for human speech, where most important information is in the mid- to low-amplitude ranges." %}
- wavenet is an AR model, so it can produce speech by conditioning on itself, then how to condition on spectrogram (what they are and how to do it)
    - global condition
    - local condition
- weakness: slow inference speed

## GAN-based Vocder

- introduce three important ones, MelGAN, HiFiGAN, Avocode and their important discriminator’s  designs. 
    - MelGAN (what they are)
        - MSD: waveform information is hierarchical
        - window-based objective: better than making decision on the whole waveform, better at capturing more fine-grained information and high-freq artefacts
    - HiFiGAN: 
        - MPD: focus on periodicity
    - Avocodo: 
        - artefacts in differnt feature-banks, intermediate results

# End-to-end TTS

(this part is mainly explored in the seminar session)

- one-stage vs two-stage method's diffference
- example for one-stage method: NaturalSpeech series
    - (the architecture won't be asked in the exam)

# Dataset and Metrics
taken from florian’s original slides

- dataset:
    - what is needed
    - how to prepare
- metrics:
    - objective evaluation: things that can be evaluated automatically, e.g., speaker similarity, phone error rate by ASR
    - subjective evaluation: things that can be evaluated by human, e.g., naturalness, prosody, speaker similarity
        - AB preference
        - MOS
