---
license: apache-2.0
language:
- zh
tags:
- legal
datasets:
- ShengbinYue/DISC-Law-SFT
---

This repository contains the DISC-LawLLM, version of Baichuan-13b-base as the base model.

<div align="center">
  
[Demo](https://law.fudan-disc.com) | [技术报告](https://arxiv.org/abs/2309.11325)
</div>

**Please note that due to the ongoing development of the project, the model weights in this repository may differ from those in our currently deployed demo.**


DISC-LawLLM is a large language model specialized in Chinese legal domain, developed and open-sourced by [Data Intelligence and Social Computing Lab of Fudan University  (Fudan-DISC)](http://fudan-disc.com), to provide comprehensive intelligent legal services. The advtantages is:
* **Legal Texts Generic Processing Capability** 
* **Legal Thinking and Reasoning** 
* **Legal knowledge Retrieval Capacity** 

In addition, the contributions include:

* **High-quality SFT datasets and effective training paradigms**
* **Chinese legal LLMs evaluation framework**
  
Check our [HOME](https://github.com/FudanDISC/DISC-LawLLM) for more information. 

# DISC-Law-SFT Dataset

we construct a high-quality supervised fine-tuning dataset, DISC-Law-SFT with two subsets, namely DISC-Law-SFT-Pair and DISC-Law-SFT-Triplet.  Our dataset converge a range of legal tasks, including legal information extraction, judgment prediction, document summarization, and legal question answering, ensuring coverage of diverse scenarios.
<img src="" alt="" width=""/>

<table>
  <tr>
    <th>Dataset</th>
    <th>Task/Source</th>
    <th>Size</th>
    <th>Scenario</th>
  </tr>
  <tr>
    <td rowspan="10">DISC-LawLLM-SFT-Pair</td>
    <td>Legal information extraction</td>
    <td>32K</td>
    <td rowspan="7">Legal professional assistant</td>
  </tr>
  <tr>
    <td>Legal event detection</td>
    <td>27K</td>
  </tr>
  <tr>
    <td>Legal case classification</td>
    <td>20K</td>
  </tr>
  <tr>
    <td>Legal judgement prediction</td>
    <td>11K</td>
  </tr>
  <tr>
    <td>Legal case matching</td>
    <td>8K</td>
  </tr>
  <tr>
    <td>Legal text summarization</td>
    <td>9K</td>
  </tr>
  <tr>
    <td>Judicial public opinion summarization</td>
    <td>6K</td>
  </tr>
  <tr>
    <td>Legal question answering</td>
    <td>93K</td>
    <td>Legal consultation services</td>
  </tr>
  <tr>
    <td>Legal reading comprehension</td>
    <td>38K</td>
    <td rowspan="2">Judicial examination assistant</td>
  </tr>
  <tr>
    <td>Judicial examination</td>
    <td>12K</td>
  </tr>
  <tr>
    <td rowspan="2">DISC-LawLLM-SFT-Triple</td>
    <td>Legal judgement prediction</td>
    <td>16K</td>
    <td>Legal professional assistant</td>
  </tr>
  <tr>
    <td>Legal question answering</td>
    <td>23K</td>
    <td>Legal consultation services</td>
  </tr>
  <tr>
    <td rowspan="2">General</td>
    <td>Alpaca-GPT4</td>
    <td>48K</td>
    <td rowspan="2">General scenarios</td>
  </tr>
  <tr>
    <td>Firefly</td>
    <td>60K</td>
  </tr>
  <tr>
    <td>Total</td>
    <td colspan="3">403K</td>
  </tr>
</table>

# Using through hugging face transformers

```python
>>>import torch
>>>>>>from transformers import AutoModelForCausalLM, AutoTokenizer
>>>from transformers.generation.utils import GenerationConfig
>>>tokenizer = AutoTokenizer.from_pretrained("ShengbinYue/DISC-LawLLM", use_fast=False, trust_remote_code=True)
>>>model = AutoModelForCausalLM.from_pretrained("ShengbinYue/DISC-LawLLM", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>>model.generation_config = GenerationConfig.from_pretrained("ShengbinYue/DISC-LawLLM")
>>>messages = []
>>>messages.append({"role": "user", "content": "生产销售假冒伪劣商品罪如何判刑？"})
>>>response = model.chat(tokenizer, messages)
>>>print(response)
```

# Disclaimer

DISC-LawLLM comes with issues and limitations that current LLMs have yet to overcome. While it can provide Chinese legal services in many a wide variety of tasks and scenarios, the model should be used for reference purposes only and cannot replace professional lawyers and legal experts. We encourage users of DISC-LawLLM to evaluate the model critically. We do not take responsibility for any issues, risks, or adverse consequences that may arise from the use of DISC-LawLLM.

# Citation

If our work is helpful for your, please kindly cite our work as follows:

```
@misc{yue2023disclawllm,
    title={DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services}, 
    author={Shengbin Yue and Wei Chen and Siyuan Wang and Bingxuan Li and Chenchen Shen and Shujun Liu and Yuxuan Zhou and Yao Xiao and Song Yun and Wei Lin and Xuanjing Huang and Zhongyu Wei},
    year={2023},
    eprint={2309.11325},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# License

The use of the source code in this repository complies with the Apache 2.0 License.
