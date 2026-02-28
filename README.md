Note:

Before running the code, ensure that Python and the required dependency packages are properly installed.
See python_version.txt and requirements.txt



If you need to run deep learning code that involves the bert-base-chinese model and find that loading it online is slow, it is recommended to download it locally for execution.
At a minimum, you need to download the following three files: vocab.txt, pytorch_model.bin, and config.json. The download link is: https://huggingface.co/google-bert/bert-base-chinese/tree/main or https://hf-mirror.com/google-bert/bert-base-chinese/tree/main
Place these files in a folder (e.g., D:/bert-base-chinese).
The code that uses this model is located in:
codes/supervised_methods/DL_methods/config.py
codes/unsupervised_methods/2CR_calculation/CR_calculation.py
In these two files, change bert-base-chinese to your local path (e.g., D:/bert-base-chinese).



To run the word clustering code, you need to obtain the pre-trained fastText word vectors first. The download link is: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz.
It is recommended to download the file locally for use. Then, replace the fasttext_model_path variable (line 22) in the code codes/unsupervised_methods/1get_attributes/2getting_word2vec_by_fasttext.py with your local file path of cc.zh.300.bin.gz.
