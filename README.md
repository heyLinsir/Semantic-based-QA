# Semantic-based-QA
The code of our ACL2021 paper [*A Semantic-based Method for Unsupervised Commonsense Question Answering*](https://arxiv.org/abs/2105.14781).



## Dependencies

- Python 3.7
- torch==1.2.0
- transformers==3.1.0
- sentence-transformers==0.3.8



## Resources

- [Data](https://cloud.tsinghua.edu.cn/f/3897511ad457408ca37d/?dl=1)
  - COPA
  - ROCStory
  - SocialIQA
  - CosmosQA
- Models
  - [Sentence-RoBERTa](https://cloud.tsinghua.edu.cn/f/d86b5c8f3b6a49a485ad/?dl=1)
  - GPT2-xlarge



## Preprocessing (Rewriting)

You can directly use our [rewriting results](https://cloud.tsinghua.edu.cn/f/3897511ad457408ca37d/?dl=1), or run the following scripts to rewriting questions.

```bash
# For SocialIQA
# file_path: ./data/SocialIQA/convert_question.py
# args: valid/test (for processing valid.jsonl or test.jsonl)
# output_file: [valid/test].jsonl.convertQ
python convert_question.py valid
```

```bash
# For CosmosQA
# file_path: ./data/CosmosQA/convert_question.py
# args: valid (for processing valid.csv)
# output_file: CosmosQA-PQA.valid
python convert_question.py valid
```



## Generating

You can directly use our [generating results](https://cloud.tsinghua.edu.cn/f/3897511ad457408ca37d/?dl=1), or run the following scripts to generate *voters*.

```bash
# ./scripts/run_SEQA_generation.sh

# COPA
# output_file: copa-test.xml.gpt2xlarge.qa.1.00penalty.topP0.90.minlen2.sample500.pkl
python ./src/run_SEQA_generation.py --model_name_or_path=./models/gpt2-xlarge --model_type=gpt2 --eval_data_file=./data/COPA/copa-test.xml --repetition_penalty=1

# ROCStory
python ./src/run_SEQA_generation.py --model_name_or_path=./models/gpt2-xlarge --model_type=gpt2 --eval_data_file=./data/ROCStory/valid2018.csv --repetition_penalty=1

# CosmosQA
python ./src/run_SEQA_generation.py --model_name_or_path=./models/gpt2-xlarge --model_type=gpt2 --eval_data_file=./data/CosmosQA/CosmosQA-PQA.valid --repetition_penalty=1

# SocialIQA
python ./src/run_SEQA_generation.py --model_name_or_path=./models/gpt2-xlarge --model_type=gpt2 --eval_data_file=./data/SocialIQA/valid.jsonl.convertQ --repetition_penalty=1
```



## Voting

```bash
# ./scripts/run_SEQA_voting.sh

# COPA
python ./src/run_SEQA_voting.py --model_name_or_path=./models/sentence-robert-large-nli-mean-tokens --eval_data_file=./data/COPA/copa-test.xml

# ROCStory
python ./src/run_SEQA_voting.py --model_name_or_path=./models/sentence-robert-large-nli-mean-tokens --eval_data_file=./data/ROCStory/valid2018.csv

# CosmosQA
python ./src/run_SEQA_voting.py --model_name_or_path=./models/sentence-robert-large-nli-mean-tokens --eval_data_file=./data/CosmosQA/CosmosQA-PQA.valid

# SocialIQA
python ./src/run_SEQA_voting.py --model_name_or_path=./models/sentence-robert-large-nli-mean-tokens --eval_data_file=./data/SocialIQA/valid.jsonl.convertQ
```

