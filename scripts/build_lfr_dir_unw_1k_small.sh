#!/bin/bash

#Script to create sets (varying mu) of lfr benchmarks with <iter> realizations each of them. 

start_time=`date +%s`
script_full_path=$(dirname "$0")

for mu in $(seq 0.0 0.1 1.0)
do
	for iter in $(seq 0 9)	
	do
		$script_full_path/lfrbench_udwov.exe -N 1000 -k 20 -maxk 50 -minc 10 -maxc 50 -muw ${mu} -name lfr_ut_${mu}_rep_${iter} -beta 1 -cnl 1 -a 1
	done
done

#Moving output files to corresponding sub-folder

dest_path="../LFR_benchmarks_for_testing/directed_unweighted/lfr_n_1k_sz_small"

mkdir -v -p $dest_path
zip a -r -sdel $dest_path/cnl_files.zip *.cnl
zip a -r -sdel $dest_path/nse_files.zip *.nsa
rm -v *.nst

end_time=`date +%s`
echo 
echo Execution time : `expr $end_time - $start_time` s.