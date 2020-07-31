#!/bin/bash

#Script to create sets (varying muw) of UNDIRECTED WEIGHTED lfr benchmarks with <iter> realizations each of them. 
#Parameters: -N 5000 -k 20 -maxk 50 -minc 20 -maxc 100 -mut 0.8 -beta 1.5

start_time=`date +%s`
script_full_path=$(dirname "$0")

mut=0.8
for muw in $(seq 0.0 0.1 1.0)
do
	for iter in $(seq 0 9)	
	do
		$script_full_path/lfrbench_udwov.exe -N 5000 -k 20 -maxk 50 -minc 20 -maxc 100 -muw ${muw} -mut ${mut} -name lfr_ut_${mut}_uw_${muw}_rep_${iter} -beta 1.5 -cnl 1 
	done
done

#Moving output files to corresponding sub-folder

dest_path="../LFR_benchmarks_for_testing/undirected_weighted/lfr_n_5k_sz_big_ut_08"

mkdir -v -p $dest_path
zip a -r -sdel $dest_path/cnl_files.zip *.cnl
zip a -r -sdel $dest_path/nse_files.zip *.nse
rm -v *.nst

end_time=`date +%s`
echo 
echo Execution time : `expr $end_time - $start_time` s.