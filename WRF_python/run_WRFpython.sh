#!/bin/bash

source /home/earajr/anaconda3/etc/profile.d/conda.sh
conda activate wp_env

dat=$1

src_dir="/home/shared/nwr/uk/data/${dat}00"

for fil in ${src_dir}/wrfout*
do
   echo ${fil}
done
