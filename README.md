# Description
This work consist of four parts. The first one is a general-purpose function to apply Community Detection (CD) algorithms (Crisp partitions / Overlapping communities) over provided benchmark graphs and calulate their performances using one or more evaluation metrics. The second one is a standardization function that make use of the first function to perform a comparison between multiple algorithms using several pre-defined sets of benchmarks graphs generated with an [Extended version of the Lancichinetti-Fortunato-Radicchi Benchmark](https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp/tree/1ccbbc38c0aa363ca88d67fe6787cd78bb93d9ff) code and the research paper [Community Detection Algorithms: A Comparative Analysis](https://www.researchgate.net/publication/43020118_Community_Detection_Algorithms_A_Comparative_Analysis). The third part are two visualization functions, one made with classical plotting librairies (matplotlib/seaborn) and the other with [Bokeh](https://bokeh.org/) which allow interactive plots. The final part is a test performed over all CDlib algorithms used into the functions to check implementation issues.

Lyon, July 30 2020.

# Content
* [Technologies](#technologies)
* [Generic_Function](#generic-function)
* [Standardization Function](#standardization-function)
* [Visualization](#visualization)
* [Algorithm's Test Results](#test-results)
* [Author](#author)

# Technologies
The main tool used to develop this code was Python using Jupyter notebooks in Google Colab.

The libraries used:

Python                   3.6.9,
google-colab             1.0.0,
CDlib                    0.1.8,
networkx                 2.4,
numpy                    1.18.5,
pandas                   1.0.5,
matplotlib               3.2.2,
seaborn                  0.10.1,
tqdm                     4.41.1,
wurlitzer                2.0.1,
urllib3                  1.24.3,
bokeh                    2.1.1,
infomap                  1.1.3,
leidenalg                0.8.1,
angel-cd                 1.0.2


# Generic Function 

`evaluate()` is a general-purpose function to apply a (or few) Community Discovery algorithm(s) over a list of benchmark graphs with community structure.
The list of true communities must be provided for each graph to evaluate the performance of the algorithm.
According to algorithm type it is possible to choose one or more evaluation metrics.

**Parameters**
```
graph_comms_list   : list, Each element is a list/tuple specifying the graphs (networkx/igraph object), the communities and a graph attribute dictionary.
algorithm_dict     : dictionary, Each key correspond to a CD function/algorithm to apply. The value could also be a list/dict with the user-defined function(s) to apply.
eval_method_dict   : dictionary, Contains the metric name (key) and the corresponding evaluation function name (value)
benchmark          : str, Name of the type of benchmarks.
case_name          : str, optional, Name of the specific study case, i.e. if there are several configurations for the same benchmark.
show_eval_progress : boolean, optional, If true the function will show a progress bar for the evaluation of each graph.
```

**Supported CD algorithms and evaluation metrics**

The function was thought to work with any community discovery algorithm taking a graph as input, so if an algorithm takes any additional parameters the user must provide an user-defined list of functions which take care of that parameters with respect to each graph.

The function was thought to work with any community evaluation method.

# Standardization Function 

`evaluate_and_compare()` is a function that compares Community Detection algorithms using pre-defined benchmark graphs with several case studies.
It is possible to compare multiple algorithms, with multiple metrics over multiple case studies.

**Parameters**
```
algos_list     : list or str, Algorithm's names to compare. If str, it must be a space-separated string containing the algorithms. 
methods_list   : list or str, Evaluation metric's names to use. If str, it must be a space-separated string containing the metrics.
benchmark_type : str, The type of benchmark graphs to use.
case_list      : list or str, optional, Specific benchmark cases to use. If str, it must be a space-separated string containing the cases.
```

**Supported CD algorithms and evaluation metrics**

The function supports almost all node clustering community discovery algorithms implemented in CDlib including Crisp communities and Overlapping communities algorithms. However, for some algorithms using extra parameters some random values were used to generate the functions that allow them to be launched into the function. These random implementations are included in the code and need to be carefully checked. The concerned algorithms are: agdl, scan, kclique, angel, lemon, lfm, multicom, node_perception, overlapping_seed_set_expansion.  

All partition comparisons scores implemented in CDlib are supported.

**benchmark_type and case_list options**

There are several possible case studies regarding selected benchmark. 

Benchmark |  Case
-----     |  ------ 
'undir_unwei' |'n_1k_sz_small'   
'undir_unwei'|'n_1k_sz_big'    
'undir_unwei'|'n_5k_sz_small'   
'undir_unwei'|'n_5k_sz_big'  
'undir_wei'|'n_5k_sz_small_ut_05' 
'undir_wei'|'n_5k_sz_small_ut_08'     
'undir_wei'|'n_5k_sz_big_ut_05'    
'undir_wei'|'n_5k_sz_big_ut_08'
'dir_unwei'|'n_1k_sz_small'
'dir_unwei'|'n_1k_sz_big'   
'dir_unwei'|'n_5k_sz_small' 
'dir_unwei'|'n_5k_sz_big'
'undir_unwei_ovlp'|'n_5k_sz_small_ut_01' 
'undir_unwei_ovlp'|'n_5k_sz_small_ut_03' 
'undir_unwei_ovlp'|'n_5k_sz_big_ut_01'  
'undir_unwei_ovlp'|'n_5k_sz_big_ut_03'

**Benchmark Notes**

The provided set of benchmark graphs were generated using external C++ code from this [source](https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp/tree/1ccbbc38c0aa363ca88d67fe6787cd78bb93d9ff) and the parameter values as well as the case studies considered were taken from this [paper](https://www.researchgate.net/publication/43020118_Community_Detection_Algorithms_A_Comparative_Analysis). However, the resulting benchmarks were reduced in quantity due to practical reasons, so instead of 100 graphs we generate only 10 graphs per mixing parameter (ut) value/point. 
Another diference with respect the paper is the distance between mixing parameter points, we consider a distance of 0.1 instead 0.05, the results is the half of points considered in the paper.

The `shell(bash)` scripts created for this task are available in the `Graph_scripts` folder of this repository.

**Output**

This is an example of the expected output.

![output1](/img/exec1.png)

![output2](/img/df.png)

# Visualization

Both plotting functions `plot_and_compare()`  and  `plot_and_compare_bokeh()` take the same input parameters and produce the same plots, the only difference is the first one uses classical plotting librairies while the second use Bokeh which allow some interactivity. 

**Parameters**

```
dataframe         : dataframe, A dataframe produced with previous functions
algos_to_compare  : list or str, optional, If str it must be a space-separated string containing the algorithms.
cases_to_compare  : list or str, optional, If str it must be a space-separated string containing the cases.
heat_map          : boolean, optional, If true the output is a heatmap.```
```

**Output**

This is an example of the expected output for `plot_and_compare()` function.

![plot1](/img/plot1.png)

![plot2](/img/plot2.png)

This is an example of the expected output for `plot_and_compare_bokeh()` function.

![plot3](/img/plot3.png)

![plot4](/img/plot4.png)

# Test Results

Each algorithm considered in this work was tested using `evaluate_and_compare()` to check possibles incompatibilities and implementation issues. Some of them (Overlapping communities) were tested using a smaller graph set due to high execution times. These are the results.

![tab1](/img/table1.png)

![tab2](/img/table2.png)

![tab3](/img/table3.png)

# Author
Code performed by Carlos Vargas Figueroa under supervision of Rémy Cazabet, Phd. 
