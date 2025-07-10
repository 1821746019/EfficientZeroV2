#!/bin/bash

# 1. 进入当前目录
cd "$(dirname "$0")"

# 2. 执行构建命令
python setup.py build_ext --inplace