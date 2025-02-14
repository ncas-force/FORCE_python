#!/bin/bash

source /home/earajr/.bashrc
source /home/earajr/anaconda3/etc/profile.d/conda.sh
conda activate wp_env

project=$1
region=$2
strt_date=$3
strt_hour=$4

script_dir="/home/earajr/cul_mor/FORCE_python/scripts"
map_info="${script_dir}/map_info"
crosssection_info="${script_dir}/crosssection_info"
profile_namelist="${script_dir}/profile.namelist.vars"
profile_info="${script_dir}/profile_info"
src_dir="/home/earajr/cul_mor/output/${project}/${region}/data/${strt_date}${strt_hour}"
base_dest_dir="/home/earajr/cul_mor/out_images/${project}/${region}/${strt_date}/${strt_hour}"
job_list="${script_dir}/job_list_${project}_${region}_${strt_date}_${strt_hour}"

if [ -f "${job_list}" ]
then
   rm -rf ${job_list}
fi
touch ${job_list}

#for fil in ${src_dir}/wrfout*
#do
#    echo ${fil}
#done

if compgen -G "${src_dir}/wrfout*" > /dev/null;
then
    echo "WRF files are present!"
    for fil in ${src_dir}/wrfout*
    do

	echo ${fil}
        
        base_fil=$( basename ${fil} ) 
        dom=$( echo ${base_fil} | awk -F "_" '{print $2}' )

# MAP PLOTS
        while IFS= read -r map_info_line; do
            map_name=$( echo ${map_info_line} | awk -F "," '{print $1}' )
            limit_lat1=$( echo ${map_info_line} | awk -F "," '{print $2}' )
            limit_lat2=$( echo ${map_info_line} | awk -F "," '{print $3}' )
            limit_lon1=$( echo ${map_info_line} | awk -F "," '{print $4}' )
            limit_lon2=$( echo ${map_info_line} | awk -F "," '{print $5}' )

            namelist_vars=${script_dir}/${map_name}.namelist.vars

            while IFS= read -r var_line; do
                var_line_head=$( echo ${var_line} | awk -F ":" '{print $1}' )
                if [ "${var_line_head}" == "s_lev_vars" ]
                then
                    for var in $( echo ${var_line} | awk -F ":" '{print $2}' )
                    do
                        var1=$( echo ${var} | tr -d , )
                        dest_dir="${base_dest_dir}/${dom}/maps/${var1}/"
                        if [ ! -d ${dest_dir} ]
                        then
                             mkdir -p ${dest_dir}
                        fi
                        echo "python ${script_dir}/../WRF_python/map_${var1}.py ${fil} ${dest_dir} ${map_name} ${limit_lat1} ${limit_lat2} ${limit_lon1} ${limit_lon2}" >> ${job_list}
                    done
                elif [ "${var_line_head}" == "m_lev_vars_p" ]
                then
                    IFS=',' read -ra vars_plevs  <<< "$( echo ${var_line} | awk -F ":" '{print $2}' )"
                    for var_plevs in "${vars_plevs[@]}"; do
                        var=$( echo ${var_plevs} | awk '{print $1}' )
                        IFS=' ' read -ra plevs  <<< "$( echo ${var_plevs} | awk '{$1 = ""; print $0}' )"
                        for plev in "${plevs[@]}"; do
                            dest_dir="${base_dest_dir}/${dom}/maps/${var}/p${plev}"
                            if [ ! -d ${dest_dir} ]
                            then
                                mkdir -p ${dest_dir}
                            fi
                                echo "python ${script_dir}/../WRF_python/map_${var}.py ${fil} ${dest_dir} p${plev} ${map_name} ${limit_lat1} ${limit_lat2} ${limit_lon1} ${limit_lon2}" >> ${job_list}
                        done
                    done
                elif [ "${var_line_head}" == "m_lev_vars_a" ]
                then
                    IFS=',' read -ra vars_alevs  <<< "$( echo ${var_line} | awk -F ":" '{print $2}' )"
                    for var_alevs in "${vars_alevs[@]}"; do
                        var=$( echo ${var_alevs} | awk '{print $1}' )
                        IFS=' ' read -ra alevs  <<< "$( echo ${var_alevs} | awk '{$1 = ""; print $0}' )"
                        for alev in "${alevs[@]}"; do
                            dest_dir="${base_dest_dir}/${dom}/maps/${var}/a${alev}"
                            if [ ! -d ${dest_dir} ]
                            then
                                mkdir -p ${dest_dir}
                            fi
                            echo "python ${script_dir}/../WRF_python/map_${var}.py ${fil} ${dest_dir} a${alev} ${map_name} ${limit_lat1} ${limit_lat2} ${limit_lon1} ${limit_lon2}" >> ${job_list}
                        done
                    done
                fi
            done < ${namelist_vars}
        done < ${map_info}

# PROFILE PLOTS
        while IFS= read -r var_line; do
            var_line_head=$( echo ${var_line} | awk -F ":" '{print $1}' )
            if [ "${var_line_head}" == "profile_vars" ]
            then
                IFS=',' read -ra vars_domains  <<< "$( echo ${var_line} | awk -F ":" '{print $2}' )"
                for var_domains in "${vars_domains[@]}"; do
                    var=$( echo ${var_domains} | awk '{print $1}' )
                    IFS=' ' read -ra domains  <<< "$( echo ${var_domains} | awk '{$1 = ""; print $0}' )"
                    for domain in "${domains[@]}"; do
                        if [[ "$domain" == "${dom}" ]];
                        then
                            while IFS= read -r profile_info_line; do
                                profile_name=$( echo ${profile_info_line} | awk -F "," '{print $1}' )
                                profile_lat=$( echo ${profile_info_line} | awk -F "," '{print $2}' )
                                profile_lon=$( echo ${profile_info_line} | awk -F "," '{print $3}' )

                                dest_dir="${base_dest_dir}/${domain}/profiles/${profile_name}/${var}"

                                if [ ! -d ${dest_dir} ]
                                then
                                    mkdir -p ${dest_dir}
                                fi
      
                                echo "python ${script_dir}/../WRF_python/profile_${var}.py ${fil} ${dest_dir} ${profile_lat} ${profile_lon} ${profile_name}" >> ${job_list}

                                if [ ! -f "${base_dest_dir}/${domain}/profiles/${profile_name}/profilelocation_${dom}_${profile_name}.png" ]
                                then
                                    echo "python ${script_dir}/../WRF_python/map_profilelocation.py ${fil} ${base_dest_dir}/${domain}/profiles/${profile_name} ${profile_name} ${profile_lat} ${profile_lon}" >> ${job_list}
                                fi
                            done < ${profile_info}
                        fi
                    done
                done
            fi
        done < ${profile_namelist}

# CROSSSECTION PLOTS

        while IFS= read -r crosssection_info_line; do
            crosssection_name=$( echo ${crosssection_info_line} | awk -F "," '{print $1}' )
            lats=$( echo ${crosssection_info_line} | awk -F "," '{print $2}' )
            lons=$( echo ${crosssection_info_line} | awk -F "," '{print $3}' )
            base_alt=$( echo ${crosssection_info_line} | awk -F "," '{print $4}' )
            top_alt=$( echo ${crosssection_info_line} | awk -F "," '{print $5}' )

            dest_dir="${base_dest_dir}/${dom}/crosssections/${crosssection_name}"
            if [ ! -f "${dest_dir}/crosssectionlocation_${dom}_${crosssection_name}.png" ]
            then
                echo "python ${script_dir}/../WRF_python/map_crosssectionlocation.py ${fil} ${dest_dir} ${crosssection_name} $( echo ${lats} | tr : , ) $( echo ${lons} | tr : , )" >> ${job_list}
            fi

            namelist_vars=${script_dir}/${crosssection_name}.namelist.vars

            while IFS= read -r var_line; do
                var_line_head=$( echo ${var_line} | awk -F ":" '{print $1}' )
                if [ "${var_line_head}" == "crosssection_vars" ]   
                then
                    IFS=',' read -ra vars_domains  <<< "$( echo ${var_line} | awk -F ":" '{print $2}' )"
                    for var_domains in "${vars_domains[@]}"; do
                        var=$( echo ${var_domains} | awk '{print $1}' )
                        IFS=' ' read -ra domains  <<< "$( echo ${var_domains} | awk '{$1 = ""; print $0}' )"
                        for domain in "${domains[@]}"; do
                            if [[ "$domain" == "${dom}" ]];
                            then
                                dest_dir="${base_dest_dir}/${domain}/crosssections/${crosssection_name}/${var}"
                                if [ ! -d ${dest_dir} ]
                                then
                                    mkdir -p ${dest_dir}
                                fi
                                echo "python ${script_dir}/../WRF_python/crosssection_${var}.py ${fil} ${dest_dir} $( echo ${lats} | tr : , ) $( echo ${lons} | tr : , ) 0,0 ${base_alt} ${top_alt} ${crosssection_name}" >> ${job_list}
                            fi
                        done
                    done
                fi
            done < ${namelist_vars}
        done < ${crosssection_info}
    done
fi

echo "Attempting to run plotting in parallel!"
#parallel -j 40 < ${job_list}

