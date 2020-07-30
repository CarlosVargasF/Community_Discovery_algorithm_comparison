# Internship_Liris_Vargas
Code and results of internship work performed by Carlos Vargas Figueroa under supervision of Rémy Cazabet as part of the bachelor degree (L3) in Informatics at Université de Lyon.  
# Description
This work consist of three parts. The first one is a general-purpose function to apply Community Detection (CD) algorithms (Crisp partitions / Overlapping communities) over provided benchmark graphs and calulate their performances using one or more evaluation metrics. The second one is a standardization function that make use of the first function to perform a comparison between multiple algorithms using several pre-defined sets of benchmarks graphs generated with an [Extended version of the Lancichinetti-Fortunato-Radicchi Benchmark](https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp/tree/1ccbbc38c0aa363ca88d67fe6787cd78bb93d9ff) code and the research paper [Community Detection Algorithms: A Comparative Analysis](https://www.researchgate.net/publication/43020118_Community_Detection_Algorithms_A_Comparative_Analysis). The third part are two visualization functions, one made with classical plotting librairies (matplotlib/seaborn) and the other with [Bokeh](https://bokeh.org/) which allow interactive plots.

Lyon, July 30 2020.

# Content
* [Generic function](#generic-function)
*
*

# Technologies


# Functions
## Generic Function (evaluate())

This is a general-purpose function to apply a (or few) Community Discovery algorithm(s) over a list of benchmark graphs with community structure.
The list of true communities must be provided for each graph to evaluate the performance of the algorithm.
According to algorithm type it is possible to choose one or more evaluation metrics.

**Parameters**
```
graph_comms_list   : list, Each element is a list/tuple specifying the graphs (networkx/igraph object), the communities and a graph attribute dictionary.
algorithm_dict     : dictionary, Each key correspond to a CD function/algorithm to apply. The value could also be a list/dict with the user-defined function(s) to apply.
eval_method_dict   : dictionary, Contains the metric name (key) and the corresponding function name (value)
benchmark          : str, Name of the type of benchmarks.
case_name          : str, optional, Name of the specific study case, i.e. if there are several configurations for the same benchmark.
show_eval_progress : boolean, optional, If true the function will show a progress bar for the evaluation of each graph.
```

**Supported CD algorithms**

The function supports almost all node clustering community discovery algorithms implemented in CDlib including Crisp communities and Overlapping communities algorithms. However, if an algorithm takes additional parameters besides the original graph then it must provide an user-defined list of functions which take care of that parameters.    

## Standardization Function (evaluate_and_compare())

Function that compares Community Detection algorithms using pre-defined benchmark graphs with several case studies.
It is possible to compare multiple algorithms, with multiple metrics over multiple case studies.

**Parameters**
```
- algos_list: list or str, Algorithm's names to compare. If str, it must be a space-separated string containing the algorithms. 
- methods_list: list or str, Evaluation metric's names to use. If str, it must be a space-separated string containing the metrics.
- benchmark_type: str, The type of benchmark graphs to use.
- case_list: list or str, Specific benchmark cases to use. If str, it must be a space-separated string containing the cases.
```

****
**Benchmark Notes**

The provided set of benchmark graphs were generated using external C++ code from this [source](https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp/tree/1ccbbc38c0aa363ca88d67fe6787cd78bb93d9ff) and the parameter values as well as the case studies considered were taken from this [paper](https://www.researchgate.net/publication/43020118_Community_Detection_Algorithms_A_Comparative_Analysis). However, the resulting benchmarks were reduced in quantity due to practical reasons, so instead of 100 graphs we generate only 10 graphs per mixing parameter (ut) value/point. 
Another diference with respect the paper is the distance between mixing parameter points, we consider a distance of 0.1 instead 0.05, the results is the half of points considered in the paper.

The shell scripts built for this task are available in the Scripts folder of this repository.
