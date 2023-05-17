# Description generation for code in Russian language.
## This code is very experimental and is more like a collection of some lucky experiments

This repository contains two scripts for preparation of data (translation from code to description by LLaMA-7B and translation by NLLB-3b)  
Also two scripts are use to train models on prepared data. (`finetune_descriptions_*.py`)  
Script `test_gen.py` is used to create predictions on new data. Must be used in interactive mode (cells are separated with `#%%`)  
