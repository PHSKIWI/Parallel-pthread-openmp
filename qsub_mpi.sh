#!/bin/sh
#PBS -N ann_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

# 确保 files 目录存在
mkdir -p /home/${USER}/ann/files

# 把 main 和 files 复制到所有分配到的节点
for node in $NODES; do
    scp master_ubss1:/home/${USER}/ann/main ${node}:/home/${USER}/main 1>&2
    scp -r master_ubss1:/home/${USER}/ann/files ${node}:/home/${USER}/files 1>&2
done

export OMP_NUM_THREADS=1

export PARTITION_MODE=0

export GRAPH_MODE=4

# 启动 8 个 MPI 进程
/usr/local/bin/mpiexec -np 4 -machinefile $PBS_NODEFILE /home/${USER}/main

# 把 files 目录复制回主节点
scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/ann/ 2>&1
