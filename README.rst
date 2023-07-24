Abstract
===========

[to be written]

Getting Started
================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper🤗.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_stars_foundation/tree/main/

This project uses `astroNN`_ and `MyGaiaDB`_ to manage `APOGEE` and `Gaia` data respectively, `PyTorch`_ as the deep learning framework.

.. _astroNN: https://github.com/henrysky/astroNN
.. _MyGaiaDB: https://github.com/henrysky/MyGaiaDB

Important⚠️: If you are using `astroNN` in the data reduction process which we did here, you have to set `magicnumber = nan` in astroNN configuration file for the code here to work properly.

Jupyter Notebooks
--------------------------------------------------------

-   | `Dataset_Reduction.ipynb`_
    | The notebook contains code to generate the dataset used by this paper. 
    | ⚠️Multi-terabytes of (mostly gaia) data need to be downloaded in the process.
-   | `Inference_Spec2Labels.ipynb`_
    | The notebook contains code to do inference on tasks of stellar spectra to stellar parameters.
-   | `Inference_Labels2Spec.ipynb`_
    | The notebook contains code to do inference on tasks of stellar parameters to stellar spectra.
-   | `Inference_Spec2Spec.ipynb`_
    | The notebook contains code to do inference on tasks of stellar spectra to stellar spectra.
-   | `Inference_Labels2Labels.ipynb`_
    | The notebook contains code to do inference on tasks of stellar parameters to stellar parameters.

.. _Dataset_Reduction.ipynb: Dataset_Reduction.ipynb
.. _Inference_Spec2Labels.ipynb: Inference_Spec2Labels.ipynb
.. _Inference_Labels2Spec.ipynb: Inference_Labels2Spec.ipynb
.. _Inference_Spec2Spec.ipynb: Inference_Spec2Spec.ipynb
.. _Inference_Labels2Labels.ipynb: Inference_Labels2Labels.ipynb

Python Script
--------------------------------------------------------

-   | `training.py`_
    | Python script to train the model.

.. _training.py: training.py


Model
--------------------------------------------------------

-   | ``model_torch`` is a trained `PyTorch`_ model
    | The model has ~8.8 millions parameters trained on ~16 millions "non-linear" tokens from ~397k stars with 118 unque "unit vector" tokens.
    | The model is trained with `PyTorch`_ 2.1 mixed precision on a consumer NVIDIA GTX 4070 Ti for ~22.5 hours with ~ 12 GB of GPU memory requirement on a Windows 11 x64 desktop computer.

.. _PyTorch: https://pytorch.org/

Graphics 
--------------------------------------------------------

-   | `model_overview.drawio`_
    | Source for Figure 1 in the paper, can be opened and edited by `draw.io`_.
    | This figure was made with icons created by the users ``imaginationlol`` and ``monkik`` on `flaticon.com`_.
-   | `model_specs.drawio`_
    | Source for Figure 2 in the paper.
    | Can be opened and edited by `draw.io`_


Example of Basic Usage
============================

Here are some examples of basic usage of the model. For the codes to work, you need to execute them at the root directory of this repository.

Get a list of vocabulary understood by the Model
--------------------------------------------------------

```python
from stellarperceptron.model import StellarPerceptron

nn_model = StellarPerceptron.load("./model_torch/", device="cpu")
print(nn_model.vocabs)
```

Give context of a star and request for information
--------------------------------------------------------

Even our model has a context window of 64 tokens, you do not need to fill up the whole length.

```python
from stellarperceptron.model import StellarPerceptron

nn_model = StellarPerceptron.load("./model_torch/", device="cpu")
# give context of two stars
# [[star1 teff, star1 logg], [star2 teff, star2 logg]]
nn_model.perceive([[4700., 2.5], [5500, 4.6]], ["teff", "logg"])
# request for information for them
print(nn_model.request(["teff"]))
```

Get an arbitrary Gaia XP spectra with source_id online and request for information
------------------------------------------------------------------------------------------

```python
from utils.gaia_utils import xp_spec_online

# Gaia DR3 source_id as integer
gdr3_source_id = 2130706307446806144

bprp_coeffs = xp_spec_online(gdr3_source_id, absolute_flux=False)
# Give the context of a star by giving XP coefficients to the NN model
nn_model.perceive(np.concatenate([bprp_coeffs["bp"][:32], bprp_coeffs["rp"][:32]]), [*[f"bp{i}" for i in range(32)], *[f"rp{i}" for i in range(32)]])
# Request for information like teff, logg, m_h
print(nn_model.request(["teff", "logg", "m_h"]))
```

Authors
===========

-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Jo: bovy [at] astro.utoronto.ca

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
