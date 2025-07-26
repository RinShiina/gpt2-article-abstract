#!/bin/bash

echo "数据预处理"
python fineweb.py

if [ $? -ne 0 ]; then
    echo "数据预处理失败"
    exit 1
fi

echo "训练"
torchrun --standalone --nproc_per_node=8 train_gpt2.py

if [ $? -ne 0 ]; then
    echo "训练失败"
    exit 1
fi

echo "执行完毕"