#!/bin/bash 

#source /home/earajr/anaconda3/etc/profile.d/conda.sh
#conda activate ncl

script_dir="/home/earajr/FORCE_WRF_plotting/scripts"
wrf_out_dir="/home/force-woest"

project=$1
region=$2

plotting_machine="lynxo"
YYYYMMDD=$( date -u --date='today' +%Y%m%d )
HH=12

if [ -d "${wrf_out_dir}/${project}/${region}/data/${YYYYMMDD}${HH}" ]
then
   echo "${script_dir}/read_log.sh ${project} ${region} ${YYYYMMDD} ${HH}"
   ssh -i /home/earajr/.ssh/thundercat_id_rsa ${plot_machine} "timeout 18h /bin/bash ${script_dir}/read_log.sh ${project}  ${region} ${YYYYMMDD} ${HH}"
else
   echo "There is no data directory for the ${region} domain for the date ${YYYYMMDD} at time ${HH}"
fi

