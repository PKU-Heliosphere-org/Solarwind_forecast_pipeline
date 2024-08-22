# Solar wind forecast tool

This tool can be used to forecast near earth solar wind speed. The. For the training method and the details of the model, see (Lin et al. 2024):

https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023SW003561

# To use

- Have Python 3.9.18, other versions are not tested. 
- Install the python modules in `requirements.txt`. 
    * A special note: here we only show the version of `torch`, but to install `torch`, you need to specify whether to use the GPU and your CUDA version, for the details see the website of torch
- Specify the epoches in `main.py`, namely the variable `date_dict`.
    * TODO: add a UI to specify the epoches.
- Run `main.py`.