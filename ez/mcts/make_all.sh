#!/bin/bash
cd "$(dirname "$0")"
# 查找后代目录中的所有make.sh文件
# find . -name "make.sh" -exec {} \;

# 异步执行所有make.sh文件
find . -name "make.sh" -exec bash {} \;



