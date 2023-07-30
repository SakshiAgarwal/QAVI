# Query-Adaptive Variational Inference

Missing Data Completion using (Hierarchical) Variational Auto-Encoders. You can find a quick explanation of QAVI [here](https://www.youtube.com/watch?v=O6iV9uOxRA4&feature=youtu.be&ab_channel=SakshiAgarwal).

## Setup

### 1. Code
`git clone https://github.com/SakshiAgarwal/QAVI.git`
### 2. Environment
`pip install numpy torch` 
### 3.Run Example
`run_mnist.ipynb` and `run_svhn.ipynb`

Find the output in `mnist/results/qavi/(Posterior)`where Posterior can be either of feat, gauss, Flow or Mix Posterior supported by QAVI.  

Note: After refactoring the code, we did not reevaluate all experiments.

## QAVI fills a missing image part using VAE
<img width="442" alt="Screen Shot 2023-07-29 at 5 38 58 PM" src="https://github.com/SakshiAgarwal/QAVI/assets/11243457/2cfa02ef-f39b-4f30-80af-9c4ad92b1cfd">
<img width="436" alt="Screen Shot 2023-07-29 at 5 37 36 PM" src="https://github.com/SakshiAgarwal/QAVI/assets/11243457/2d9a91e8-d394-4184-b793-49dceaa97a22">

**What are the gray parts?**

Those parts are missing and therefore have to be filled by QAVI.
QAVI generates the missing parts inspired by the observed parts.

**How does it work?**

QAVI starts by instantiating variational parameters per image. Then it is updated per optimization step towards a higher ELBO for observed data. Finally, post optimization it is able to infill missing parts with having some knowledge from the observed features. 

## Details on data
**Which datasets have a ready-to-use config file?**

We provide config files for MNIST and SVHN in `config.py`, their corresponding pre-trained models in `mnist/pre_trained_models/` and `svhn/pre_trained_models/` respectively. We also provide some masking options in `data.py`. 

**How to apply it for other datasets?**

If you work with other data, train a (H-)VAE model first. Note that QAVI is an inference scheme. We do not train or finetune any (H-)VAE model but condition pre-trained models.

## Code overview
**1. Data:** Test datasets can be retrieved via `data.py`

**2. Inference:** `QAVI.py` supports classes for 4 variational posteriors built on top of a VAE class in `models/vae.py`. Once a QAVI object is created, you can pass in your test data and using the `show_imputations` function, QAVI fits a variational posterior and saves imputation samples. 

**3. Evaluation:** The QAVI object can be evaluated using `.evaluate` function which computes the IWAE metric. 
