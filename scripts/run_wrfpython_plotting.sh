#!/bin/bash 

#source /home/earajr/anaconda3/etc/profile.d/conda.sh
#conda activate ncl

script_dir="/home/earajr/cul_mor/FORCE_python/scripts"
wrf_out_dir="/home/earajr/cul_mor/output"

#project=$1
#region=$2

project="CulMor"
region="Scotland"

plot_machine="liono"
YYYYMMDD="20241022"
HH="00"

#YYYYMMDD=$( date -u --date='today' +%Y%m%d )
#HH="12"

echo "${wrf_out_dir}/${project}/${region}/data/${YYYYMMDD}${HH}"

if [ -d "${wrf_out_dir}/${project}/${region}/data/${YYYYMMDD}${HH}" ]
then
   echo "${script_dir}/read_log.sh ${project} ${region} ${YYYYMMDD} ${HH}"
#   /bin/bash ${script_dir}/plot_for_files.sh ${project} ${region} ${YYYYMMDD} ${HH}
   ssh -i /home/earajr/.ssh/thundercat_id_rsa ${plot_machine} "timeout 18h /bin/bash ${script_dir}/plot_for_files.sh ${project} ${region} ${YYYYMMDD} ${HH}"
else
   echo "There is no data directory for the ${region} domain for the date ${YYYYMMDD} at time ${HH}"
fi

