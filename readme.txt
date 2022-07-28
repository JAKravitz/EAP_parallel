Files:

parallel_test.pbs   -- PBS script to run the job
test_parallel_02.sh  -- shell script to run gnu parallel
commandline_eda_01.ipynb -- jupuyter notebook used to evaluate saved pickle files
test_single_07c.py -- python program to run dask enabled EAP code
EAP_tn.py -- my version of EAP, the original version should work fine too
Dmmex_R14B_4.f -- fortran code
Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so -- the compiled Dmmex_R14B_4.f code
phyto_data.csv -- the phytoplankton data
phyto_dict.pkl -- dictionary of lists of pytonplankton species
dict_creator.ipynb -- jupyter notebook used to create phyto_dict.pkl

Description:

Gnu parallel and dask are used to generate the data.  
Gnu parallel assignes one python script to a cpu and dask sets up workers to run on the different cores of the cpu.

Currently parallel_test.pbs will run 4 instances of the test_parallel_02.sh code.  
Each instatance will run on 1 cpu.  
To run more instances you will need to increase the number of cpus by adding more cpus to the node:

#PBS -l select=1:ncpus=4:model=bro  ---> #PBS -l select=1:ncpus=28:model=bro
(will go from 4 cpus to 28 cpus)

then in gnu parallel
-j 4  ---> -j 28
(-j flag tells gnu parallel number of processor on the node)

and or increasing the number of nodes:

#PBS -l select=1:ncpus=4:model=bro  ---> #PBS -l select=2:ncpus=28:model=bro
(will go from 4 cpus to 56 cpus since there are now 2 nodes each with 28 cpus)

then in  gnu parallel
-j 4  ---> -j 28  
(eventhough there are two nodes for a total of 56 cpus there is still only 28 cpus on each node)

You will also need to increase the seq number (seq 4) to match the number of instances running.

The dict_creator.ipynb notebook will create four lists each with 3 species names.  
It takes ~ 1hr 15mins to generate the data for all 12 species using one broadwell node with 4 cpus using the gnu parallel configuration in parallel_test.pbs.  
The dict_creator.ipynb notebook can be altered to create more lists or add more species to a list.  
However, if more list are created more cpus will need to be added to the job, since the gnu parallel code is setup to run one list per cpu.

When test_parallel_02.sh is run run_$1 directories will be created. 
The total number of directories created will equal the seq number.  The files commandline_eda_01.ipynb, test_single_07c.py, EAP_tn.py, Dmmex_R14B_4.f, Dmmex_R14B_4.cpython-39-x86_64-linux-gnu.so will be copied into each directory.

Reference:

https://www.nas.nasa.gov/hecc/support/kb/using-gnu-parallel-to-package-multiple-jobs-in-a-single-pbs-job_303.html