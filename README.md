Note:

Before running the code, ensure that Python and the required dependency packages are properly installed.
See python_version.txt and requirements.txt



If you need to run deep learning code that involves the bert-base-chinese model and find that loading it online is slow, it is recommended to download it locally for execution.
At a minimum, you need to download the following three files: vocab.txt, pytorch_model.bin, and config.json. 
The download link is: https://huggingface.co/google-bert/bert-base-chinese/tree/main or https://hf-mirror.com/google-bert/bert-base-chinese/tree/main.
Place these files in a folder (e.g., D:/bert-base-chinese).
The codes that use this model are located in:
codes/supervised_methods/DL_methods/config.py
codes/unsupervised_methods/2CR_calculation/CR_calculation.py
In these two files, change bert-base-chinese to your local path (e.g., D:/bert-base-chinese).



To run the word clustering code, you need to obtain the pre-trained fastText word vectors first. The download link is: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz.
It is recommended to download the file locally for use. Then, replace the fasttext_model_path variable (line 22) in the code codes/unsupervised_methods/1get_attributes/2getting_word2vec_by_fasttext.py with your local file path of cc.zh.300.bin.gz.


To run the code related to the large language models (LLMs), you need to configure the corresponding API keys in your system environment variables first.
Before execution, you must apply for API keys from the official platforms. You must name the environment variables exactly as ZHIPU_API_KEY, TONGYI_API_KEY, and DEEPSEEK_API_KEY respectively. 
The code will fail to run if the naming of the environment variables does not match the required format.
The application links for each platform are as follows:
Zhipu AI (GLM Series): https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys
Alibaba Cloud Tongyi (Qwen Series): https://help.aliyun.com/zh/model-studio/get-api-key
DeepSeek: https://platform.deepseek.com/api_keys
The code that uses API keys are located in:
codes/supervised_methods/LLM_methods/LLM_with_cot_5folds_for_all_prompts_with_asyncio.py
codes/supervised_methods/LLM_methods/LLM_without_cot_5folds_for_all_prompts_with_asyncio.py
