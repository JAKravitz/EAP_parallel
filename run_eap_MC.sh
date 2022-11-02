#!/bin/bash
#make directory
mkdir -p cpu_mc
cp EAP_cyano_vacuole.py cpu_mc/EAP_cyano_vacuole.py
cp EAP.py cpu_mc/EAP.py
cp MC_k_shell.csv cpu_mc/MC_k_shell.csv
cp Dmmex_R14B_4.f cpu_mc/Dmmex_R14B_4.f
cp Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so cpu_mc/Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so
#cp {run_EAP_batch.py, EAP_tn.py, Dmmex_R14B_4.f, Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so} run_$1/
cd cpu_mc

#set environment
source /usr/share/modules/init/bash
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate tf2_9

#run code
echo "Executing run MC on" `hostname` "in $PWD"
/swbuild/analytix/tools/miniconda3_220407/envs/tf2_9/bin/python EAP_cyano_vacuole.py 