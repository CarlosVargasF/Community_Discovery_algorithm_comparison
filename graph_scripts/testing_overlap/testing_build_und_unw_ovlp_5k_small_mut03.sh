#!/bin/bash

#Script to create a small test set of UNDIRECTED UNWEIGHTED WITH OVERLAPPING lfr benchmarks with one realization each of them. 
#(varying the fraction of overlaping nodes between comms)

start_time=`date +%s`
script_full_path=$(dirname "$0")

mut=0.3
for on in {0..5000..500}
do
	for iter in 0	
	do
		$script_full_path/lfrbench_udwov.exe -N 5000 -k 20 -maxk 50 -minc 10 -maxc 50 -muw ${mut} -name lfr_ut_${mut}_on_${on}_rep_${iter} -beta 1 -cnl 1 -on ${on} -om 2
	done	
done


#Moving output files to corresponding sub-folder

dest_path="../LFR_benchmarks_for_testing/testing_overlap/lfr_n_5k_sz_small_ut_03"

mkdir -v -p $dest_path
zip a -r -sdel $dest_path/cnl_files.zip *.cnl
zip a -r -sdel $dest_path/nse_files.zip *.nse
rm -v *.nst

end_time=`date +%s`
echo 
echo Execution time : `expr $end_time - $start_time` s.