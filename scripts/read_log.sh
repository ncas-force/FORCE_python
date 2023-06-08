#!/bin/bash

source /home/earajr/.bashrc
source /home/earajr/anaconda3/etc/profile.d/conda.sh
conda activate wp_env

project=$1
region=$2
strt_date=$3
strt_hour=$4

forecast_len=60

end_time=$( date -u -d "${strt_hour}:00:00 ${strt_date:0:4}-${strt_date:4:2}-${strt_date:6:2} +${forecast_len}hours" +"%Y-%m-%d_%H:%M:%S" )

script_dir="/home/earajr/FORCE_WRF_plotting/scripts"
namelist_vars="${script_dir}/namelist.vars"
namelist_locs="${script_dir}/sounding.locs"
src_dir="/home/force-woest/woest/${region}/data/${strt_date}${strt_hour}"
base_dest_dir="/home/earajr/FORCE_WRF_plotting/images/${project}/${region}/${strt_date}/${strt_hour}"
log_file="${src_dir}/nwr_log"
command_list="${script_dir}/command_list_${project}_${region}_${strt_date}_${strt_hour}"
fil_list="${script_dir}/fil_list_${region}_${strt_date}_${strt_hour}"

while true
do
   if grep -q "Starting wrf.exe:" ${log_file}
   then
      break
   fi
done

if [ -f "${command_list}" ]
then
   rm -rf ${command_list}
fi
touch ${command_list}

if [ -f "${fil_list}" ]
then
   rm -rf ${fil_list}
fi
touch ${fil_list}

count=0 
while true;
do
   date_time=$( date -u '+%Y%m%d%H%M%S' )
   touch ${command_list}_${date_time}

   if compgen -G "${src_dir}/wrfout*" > /dev/null;
   then
      for fil in ${src_dir}/wrfout*;
      do
         if grep -q ${fil} ${fil_list};
         then
  	    echo "############################################################################################################################"
            echo "File \"${fil}\" already processed."
            echo "############################################################################################################################"
         else
            base_fil=$( basename ${fil} ) 
            dom=$( echo ${base_fil} | awk -F "_" '{print $2}' )

            while IFS= read -r var_line; do
               var_line_head=$( echo ${var_line} | awk -F ":" '{print $1}' )
               if [ "${var_line_head}" == "s_lev_vars" ]
               then
                  for var in $( echo ${var_line} | awk -F ":" '{print $2}' )
                  do
                     var1=$( echo ${var} | tr -d , )
#	             dest_dir="${base_dest_dir}/${dom}/${var1}/"
		     dest_dir="${base_dest_dir}/maps/${dom}/${var1}/"
	             if [ ! -d ${dest_dir} ]
                     then
                        mkdir -p ${dest_dir}
	             fi
		     echo "python ${script_dir}/../WRF_python/map_${var1}.py ${fil} ${dest_dir}" >> ${command_list}_${date_time}
#         	     echo "ncl 'dom=\"${dom}\"' 'dest=\"${dest_dir}\"' 'a=addfile(\"${fil}\", \"r\")' ${script_dir}/ncl/${var1}.ncl"# >> ${script_dir}/command_list_${region}_${strt_date}_${strt_hour}_${date_time}
                  done
#               elif [ "${var_line_head}" == "m_lev_vars" ]
#               then
#                  IFS=',' read -ra vars_plevs  <<< "$( echo ${var_line} | awk -F ":" '{print $2}' )"
#                  for var_plevs in "${vars_plevs[@]}"; do
#                     var=$( echo ${var_plevs} | awk '{print $1}' )
#	             IFS=' ' read -ra plevs  <<< "$( echo ${var_plevs} | awk '{$1 = ""; print $0}' )"
#	             for plev in "${plevs[@]}"; do
#	                dest_dir="${base_dest_dir}/${dom}/${var}/${plev}/"
#	                if [ ! -d ${dest_dir} ]
#                        then
#                           mkdir -p ${dest_dir}
#                        fi
#                        echo "ncl 'dom=\"${dom}\"' 'dest=\"${dest_dir}\"' 'plevs=${plev}' 'a=addfile(\"${fil}\", \"r\")' ${script_dir}/ncl/${var}.ncl" >> ${script_dir}/command_list_${region}_${strt_date}_${strt_hour}_${date_time}
#	             done
#   	          done
#               elif [ "${var_line_head}" == "loc_vars" ]
#               then
#                  for var in $( echo ${var_line} | awk -F ":" '{print $2}' )
#                  do
#                     var1=$( echo ${var} | tr -d , )
#                     while IFS= read -r loc_line; do
#                        loc_name=$( echo ${loc_line} | awk -F "," '{print $1}' )
#     	                loc_stat=$( echo ${loc_line} | awk -F "," '{print $2}' )
#	                loc_lat=$( echo ${loc_line} | awk -F "," '{print $3}' )
#	                loc_lon=$( echo ${loc_line} | awk -F "," '{print $4}' )
#
#   		        if [ "${region}" == "iceland" ]
#		        then
#
#                           loc_name2=$( echo ${loc_name//À/A~H-15V6F35~A~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//à/a~H-13V2F35~A~FV-2H3~})
#                           loc_name2=$( echo ${loc_name2//Á/A~H-15V6F35~B~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//á/a~H-13V2F35~B~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Â/A~H-15V6F35~C~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//â/a~H-13V2F35~C~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ã/A~H-15V6F35~D~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//â/a~H-13V2F35~D~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ä/A~H-15V6F35~H~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ä/a~H-13V2F35~H~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//È/E~H-15V6F35~A~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//è/e~H-13V2F35~A~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//É/E~H-15V6F35~B~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//é/e~H-13V2F35~B~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ê/E~H-15V6F35~C~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ê/e~H-13V2F35~C~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ë/E~H-15V6F35~H~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ë/e~H-13V2F35~H~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ì/I~H-10V6F35~A~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ì/i~H-10V2F35~A~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Í/I~H-08V6F35~B~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//í/i~H-08V2F35~B~FV-2~})
#	                   loc_name2=$( echo ${loc_name2//Î/I~H-09V6F35~C~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//î/i~H-09V2F35~C~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ï/I~H-09V6F35~H~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ï/i~H-09V2F35~H~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ò/O~H-15V6F35~A~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ò/o~H-13V2F35~A~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ó/O~H-15V6F35~B~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ó/o~H-13V2F35~B~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ô/O~H-16V6F35~C~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ô/o~H-14V2F35~C~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Õ/O~H-15V6F35~D~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//õ/o~H-13V2F35~D~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ö/O~H-16V6F35~H~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ö/o~H-14V2F35~H~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ù/U~H-15V6F35~A~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ù/u~H-13V2F35~A~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ú/U~H-13V6F35~B~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ú/u~H-13V2F35~B~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Û/U~H-15V6F35~C~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//û/u~H-13V2F35~C~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ü/U~H-15V6F35~H~FV-6H3~})
#	                   loc_name2=$( echo ${loc_name2//ü/u~H-13V2F35~H~FV-2H3~})
#	                   loc_name2=$( echo ${loc_name2//Ð/D~H-22V-8F35~E~FV8H10~})
#	                   loc_name2=$( echo ${loc_name2//ð/d~H-10F35~E~FH5~~H-4~})
#                           loc_name2=$( echo ${loc_name2//Æ/A~H-10~E})
#	                   loc_name2=$( echo ${loc_name2//æ/a~H-6~e})
#                        fi
#
#	                if [ "${loc_stat}" == " " ]
#                        then
#                           loc_stat=${loc_name}
#                        fi
#
#                        dest_dir="${base_dest_dir}/${dom}/${var1}/${loc_name}/"
#	                if [ ! -d ${dest_dir} ]
#                        then
#                           mkdir -p ${dest_dir}
#                        fi
#                        echo "ncl 'dom=\"${dom}\"' 'dest=\"${dest_dir}\"' 'ids=\"${loc_name}\"' 'stat=\"${loc_stat}\"' 'name=\"${loc_name2}\"' 'lats=${loc_lat}' 'lons=\"${loc_lon}\"' 'a=addfile(\"${fil}\", \"r\")' ${script_dir}/ncl/${var1}.ncl" >> ${script_dir}/command_list_${region}_${strt_date}_${strt_hour}_${date_time}
#                     done < ${namelist_locs}
#	          echo "ncl ${var1}"
#	          done
	       fi
            done < ${namelist_vars}
            echo ${fil} >> ${fil_list}
         fi
      done
      echo "Attempting to run plotting in parallel!"
      parallel -j 10 < ${command_list}_${date_time}
      cat ${command_list}_${date_time} >> ${command_list}
   fi
   rm -rf ${command_list}_${date_time}
   sleep 30s

   if grep -q ${end_time} ${fil_list}
   then
      ((count+=1))
      echo ${count}
      if (( ${count} > 2 ))
      then
         break
      fi
   fi
   break
done

rm -rf ${fil_list}
rm -rf ${command_list}

