#!/bin/bash
#make directory
mkdir -p run_$1
#cp commandline_eda_01.ipynb run_$1/commandline_eda_01.ipynb
cp run_EAP_batch.py run_$1/run_EAP_batch.py
cp EAP_tn.py run_$1/EAP_tn.py
cp Dmmex_R14B_4.f run_$1/Dmmex_R14B_4.f
cp Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so run_$1/Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so
cp run_EAP_batch.py run_$1/run_EAP_batch.py
cd run_$1

#set environment
source /usr/share/modules/init/bash
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate tf2_8

#run code
echo "Executing run $1 on" `hostname` "in $PWD"
/swbuild/analytix/tools/miniconda3_220407/envs/tf2_8/bin/python run_EAP_batch.py $1