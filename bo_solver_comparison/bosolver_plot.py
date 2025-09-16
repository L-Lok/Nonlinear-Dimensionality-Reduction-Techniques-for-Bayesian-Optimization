from objective_functions.utils.plotting import *
from pathlib import Path

# File with information about problem suite
problem_info_files = ['./objective_functions/utils/problem_info_bayesopt.csv','./objective_functions/utils/problem_info_botorch.csv',
                     './objective_functions/utils/problem_info_gpyopt.csv']

### 1. Process files ###
# Build list of input and output files
# Input your results files path here
infiles = ["./objective_functions/bayesopt_results.csv", "./objective_functions/botorch_results.csv", "./objective_functions/gpyopt_results.csv"]
outfiles = [infile.replace('results', 'clean_results') for infile in infiles]

# Process each file
for problem_info_file, infile, outfile in zip(problem_info_files, infiles, outfiles):
    solved_times = get_solved_times_for_file(problem_info_file, infile)  # process
    solved_times.to_csv(outfile)  # save to csv

### 2. Create plots ###
# Plot information is a list of tuples (filename_stem, label, color, linestyle, [marker], [markersize])
#   - filename_stem is a string defining the results files
#         This allows multiple runs for the same solver (useful when have noisy problems, for instance).
#         Result files must be available at "filename_stem*.csv", where filename_stem can have folder references.
#   - label is the legend entry for this solver. For LaTeX formatting, use as: r"Result $x=1$"
#   - color, linestsyle, marker and markersize have the standard definitions
#         See here: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
solver_info = []  # each entry is tuple = (filename_stem, label, colour, linestyle, [marker], [markersize]), where marker info is optional
solver_info.append(('./objective_functions/bayesopt_clean_results.csv', r'BayesOpt', 'b', '-', '.', 12))
solver_info.append(('./objective_functions/botorch_clean_results.csv', r'Botorch', 'r', '--'))
solver_info.append(('./objective_functions/gpyopt_clean_results.csv', r'GPyOpt', 'k', '-.'))

#  A list of accuracy levels to generate plots for
tau_levels = [1, 2, 3, 4]

# Output format for graphs (e.g. png, eps, ...)
fmt = "png"
    
# Whether to generate data/performance profiles
generate_data_profiles = True
generate_perf_profiles = True

# Maximum budget (in simplex gradients) to plot results for
budget = 50
    
# For data profiles, should the x-axis be on a log scale?
dp_with_logscale = True

# Output name for plots
# Format is "outfile_stem_[data|perf][tau].fmt"
outfile_stem = Path('./objective_functions')

# Expected number of problems in input files
expected_nprobs = 12

# Generate plots
create_plots(outfile_stem, solver_info, tau_levels, budget, 
             data_profiles=generate_data_profiles, perf_profiles=generate_perf_profiles, dp_with_logscale=dp_with_logscale, fmt=fmt, expected_nprobs=expected_nprobs)

print('Done')


    