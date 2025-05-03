# PyCodeGen

## Overview
This repository contains a **from‑scratch** implementation of an encoder–decoder Transformer model in **PyTorch**, designed specifically for **Python code generation** tasks. 

The model was developed as my 3rd‑year (6th semester) solo project and features custom implementations of key Transformer components (multi‑head attention, positional encodings, layer normalization, etc.). Two size variants are provided:

This is a **75M**‑parameter **encoder-decoder** transformer model for python code generation


## Requirements
This project only requires **PyTorch** to be installed on the device and have a **GPU** (with Cuda), also install streamlit for the GUI.

    pip install torch
    pip install streamlit

However, we will also require kaggle for downloading the model so also install it using   
`pip install kaggle`


## Setup

Clone the repository into your desired folder

To download the model first make sure you to have a kaggle account and create a new token from the settings tab or just use and existing one.
A `kaggle.json` file will be downloaded, move this file into the following path `C:\Users\<YourUsername>\.kaggle\kaggle.json
`

After this much has been done open the terminal in the cloned repo folder and make sure to do `cd Components`.

To download the model run 
    `#!/bin/bash kaggle models instances versions download divyanshvishwkarma/pycodegen/pyTorch/75m/5`
A file named pycodegen.tar.gz will get installed.


This step can also be omitted by downloading the model by visiting the [model page](https://www.kaggle.com/models/divyanshvishwkarma/pycodegen)
Just click on download and then select '**Download model as .tar.gz**'. Make sure to move this file in the Components directory.

With command prompt open in the Components directory run
 `tar -xzvf pycodegen.tar.gz` and the model weights will be extracted in the directory


## Directory structure


make sure your directory structure looks like:

    PyCodeGen
        |->Components
        |   |-> model.py
        |   |-> model_weights.pt
        |   |-> Model.pt
        |   |-> src_vocb.json
        |   |-> tgt_vocab.json
        |
        |->Model.py
        |
        |->main.py


## Inference

Now you are all set to run your the model:

Run this command at the terminal with the project directory open

    -> streamlit run main.py


### Dataset
the model was trained on the dataset called [python_code_instructions_18k_alpaca]((https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca))
by Tarun Bisht, consisting on 18.6 thousand python code examples. 

### Future plans

I am planning to expand this model to larger parameter size and a larger dataset as well, so be on a lookout for it.



### Links and references

For more information about the model visit : [model page on kaggle](https://www.kaggle.com/models/divyanshvishwkarma/pycodegen)

For more information on downloading models from kaggle visit : [kaggle documentation](https://www.kaggle.com/docs/models#kagglehub-download)

For any encountered bugs or doubts feel free to contact me on [LinkedIn](https://in.linkedin.com/in/divyanshvishwkarma)
