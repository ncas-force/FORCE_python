#!/bin/bash 

#source /home/earajr/anaconda3/etc/profile.d/conda.sh
#conda activate ncl

script_dir="/home/force-nwr/nwr/FORCE_python/scripts"
#script_dir="/home/earajr/FORCE_WRF_plotting/scripts"
wrf_out_dir="/home/force-nwr"

project=$1
region=$2
run_len=$3

plot_machine="jaga"
YYYYMMDD=$( date -u --date='today' +%Y%m%d )
HH=00

if [ "${region}" == "uk" ]
then
   plot_machine="jaga"
elif [ "${region}" == "iceland" ]
then
   plot_machine="lynxo"
elif [ "${region}" == "cape_verde" ]
then
   plot_machine="liono"
fi
echo "$region"

if [ -d "${wrf_out_dir}/${project}/${region}/data/${YYYYMMDD}${HH}" ]
then
   echo "${script_dir}/read_log.sh ${project} ${region} ${YYYYMMDD} ${HH}"
   ssh -i /home/force-nwr/.ssh/thundercat_id_rsa ${plot_machine} "timeout 18h /bin/bash ${script_dir}/read_log.sh ${project}  ${region} ${YYYYMMDD} ${HH} ${run_len}"
else
   echo "There is no data directory for the ${region} domain for the date ${YYYYMMDD} at time ${HH}"
fi

