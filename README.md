## Project of PKU 2025 summer school course *Large Model: From Basic to Practice*


### Part.1

We train the gpt-2 124M model, using parameters of gpt-2 and gpt-3 paper. The architecture is just the same as the paper.

The training and data processing code is learned in karpathy's video.

You could modify the batch size, node_num, EPOCH, and repo_id.


### Part.2

We fine tune the gpt-2 124M model and distilbart-6-6-cnn model from huggingface. Then we use test_*.py to generate abstract.

The datasets used are listed in my report.

Using my finetune_*.py to reproduce my model is nearly impossible because I've modified too much operations.


### Part.3

We use check.ipynb to visualize the training and finetune process. Meanwhile we use deepseek_api.py to generate comparison.

Also I write that report, and finish some other things.



p.s. Basically, I don't think it's worth trying git clone this repo because there is nothing new. In a word, it just implemented something done by others. There is nothing really special because I don't get enough GPUs and time to turn my ideas into reality.