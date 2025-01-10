# Enhancing Black-Box Adversarial Attacks on Discrete Sequential Data via Bilevel Bayesian Optimization in Hybrid Spaces

Implementation of **"[Enhancing Black-Box Adversarial Attacks on Discrete Sequential Data via Bilevel Bayesian Optimization in Hybrid Spaces]()"**, published at **KDD'25**

> **Abstract** *Black-box attacks have emerged as a significant threat to deep neural networks. This challenge is particularly difficult in discrete sequential data compared to continuous data. Recently, the Blockwise Bayesian Attack (BBA) leveraging discrete Bayesian optimization with an adapted RBF kernel has gained prominence as a cutting-edge solution. However, it relies solely on alignment information (i.e., positional differences) within the RBF kernel, which may not fully capture the information (such as statistical,  structural, and semantic information) inherent in discrete sequential data and potentially lacks the desired inductive bias necessary to approximate the target function accurately. To overcome this limitation, this paper proposes a novel bilevel Bayesian optimization approach to adaptively learn a hybrid space that better captures the similarity between discrete sequences. Specifically, we introduce a multi-kernel mechanism that incorporates multiple types of information, creating a more comprehensive similarity measure. Moreover, we develop a bilevel Bayesian optimization algorithm, where the outer-level objective determines the optimal weights of the multiple kernels, while the inner-level objective identifies the optimal adversarial sequence. Extensive experiments conducted on discrete sequential data demonstrate that 
our approach ensures secure multi-kernel selection and achieves a higher attack success rate with only a few additional queries, compared to BBA and other traditional optimization strategies.*

## Machine Information
Below are the information about machine that authors used.
* OS: "Ubuntu 22.04.4 LTS"
* CUDA Driver Version: 535.183.01
* gcc: 11.4.0
* CUDA: 12.2
* CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
* GPU: NVIDIA A40

## Contents

For the initial step, use the following command to install our git repository:

```git clone --recursive https://github.com/xcli23/BilevelBayesianOptimization.git```

Our implemented code for NLP domain can be found in [nlp\_attack folder](nlp_attack)

Our implemented code for other domain can be found in [other_domain\_attack folder](ips_attack)

## Citation
Enhancing Black-Box Adversarial Attacks on Discrete Sequential Data via Bilevel Bayesian Optimization in Hybrid Spaces
Tianxing Man, Xingchen Li, Zhaogeng Liu, Haozhen Zhang, Bin Gu*, Yi Chang*
KDD 2025 Research Track