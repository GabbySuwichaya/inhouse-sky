## Quickstart

This repo we adopt the content of `PhyDNetGAN.ipynb` from https://github.com/yuhao-nie/SkyGPT and modify the content for an inhouse dataset.  


- Under `PhyDNetGAN`, create a folder called `CUEE_preprocessing`  
- Put `h5files-IRR-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-15 ...` into `PhyDNetGAN/CUEE_preprocessing/`
- Also, put all the text files (`Train_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt`, `Valid_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt`, `Test_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt`) into `PhyDNetGAN/CUEE_preprocessing/`
    You should get something like this ...
    ```
    PhyDNetGAN/
        Readme.md
        python files .... 
        demo/
        save/
        CUEE_preprocessing/
            h5files-IRR-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-15/
            Train_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt
            Valid_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt
            Test_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt
    ```
- The customized dataloader is provided in `dataloader_CUEE_IRR.py`. 

- You can start run training by running
    `main_training_cuee.py`  

- After training some epoches, you can verify your results by running the inference:
    `main_inference_cuee.py`