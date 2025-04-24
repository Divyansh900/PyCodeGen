# PyCodeGen

## Overview
This repository contains a from‑scratch implementation of an encoder–decoder Transformer model in PyTorch, designed specifically for Python code generation tasks. The model was developed as my 3rd‑year (6th semester) solo project and features custom implementations of key Transformer components (multi‑head attention, positional encodings, layer normalization, etc.). Two size variants are provided:

75M‑parameter variant (trained) for python code generation


## Requirements
This project only requires **PyTorch** to be installed on the device and have a **GPU** (with Cuda), hence a single command `pip install torch==2.6.0` on the CLI.

However, we will also require kaggle for downloading the model so also install it using   
`pip install kaggle`


## Setup

Clone the repository into your desired folder

To download the model first make sure you to have a kaggle account and create a new token from the settings tab or just use and existing one.
A `kaggle.json` file will be downloaded, move this file into the following path `C:\Users\<YourUsername>\.kaggle\kaggle.json
`

After this much has been done open the terminal in the cloned repo folder and make sure to do `cd Components`.

To download the model run 
    `#!/bin/bash kaggle models instances versions download divyanshvishwkarma/pycodegen/pyTorch/75m/2`

A file named pycodegen.tar.gz will get installed.
To extract it do `tar -xzvf pycodegen.tar.gz`

make sure your directory structure looks like:

    PyCodeGen
        |->Components
        |       |-> model.py
        |       |-> Model.pt
        |       |-> src_vocb.json
        |       |-> tgt_vocab.json
        |
        |->Model.py
        |
        |->Main.py

Now you are all set to run your the model

    from Model import PyCode
    model = PyCode()
    prompt = ' ...Your code prompt...  '
    model.generate(prompt)

