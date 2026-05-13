#!/bin/bash

# ── 1. 变量展开 ${} ──────────────────────────────────────────────
a="hello"
echo ${a}           # 输出: hello
echo ${a}_world     # 输出: hello_world   (没有{}就会找变量 $a_world，结果是空)
echo $a_world       # 输出: (空，因为bash把 a_world 当成变量名)
echo "$a,你好"
echo $a
# ── 2. 变量有空格时，引号的作用 ──────────────────────────────────
path="my dir/file"
echo ${path}        # 输出: my dir/file  (但如果传给命令会被拆成两个参数)
echo "${path}"      # 输出: my dir/file  (整体作为一个参数，安全)

# ── 3. 命令替换 $() ──────────────────────────────────────────────
result=$(pwd)
echo ${result}      # 输出: 当前目录的绝对路径

# ── 4. dirname 就是去掉最后一层文件名 ───────────────────────────
echo $(dirname "pumengyu/scripts/test.sh")   # 输出: pumengyu/scripts
echo $(dirname "/a/b/c.txt")                  # 输出: /a/b

# ── 5. 组合：相对路径转绝对路径 ──────────────────────────────────
# dirname "$0" 得到相对路径，cd进去再pwd才是绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ${SCRIPT_DIR}  # 输出: 脚本所在目录的绝对路径
# &&是左边成功了,再执行右边,