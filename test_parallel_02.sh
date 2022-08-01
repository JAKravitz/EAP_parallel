#!/bin/bash
#make directory
mkdir -p run_$1
#cp commandline_eda_01.ipynb run_$1/commandline_eda_01.ipynb
cp test_single_07c.py run_$1/test_single_07c.py
cp EAP_tn.py run_$1/EAP_tn.py
cp Dmmex_R14B_4.f run_$1/Dmmex_R14B_4.f
cp Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so run_$1/Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so
cd run_$1

#set environment
source /usr/share/modules/init/bash
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda_tests/test01
source activate tf2_9

#run code
echo "Executing run $1 on" `hostname` "in $PWD"
/swbuild/analytix/tools/miniconda3_220407/envs/tf2_9/bin/python run_EAP_batch.py $1