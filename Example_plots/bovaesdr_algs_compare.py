# Adjust the Path
from bo_solver_comparison.utils.plotting import *
from pathlib import Path

# example usage of the plotting functions to compare different algorithms

encoder_path = "encoders_100_30_2" # input the encoder structure here

# File with information about problem suite
problem_info_files = [
    f"./output/bovaesdr_algs_compare/{encoder_path}/bo_sdr/problem_info.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_bo_sdr/problem_info.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_dml_bo/problem_info.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/vae_bo_sdr/problem_info.csv",
]

### 1. Process files ###
# Build list of input and output files
infiles = [
    f"./output/bovaesdr_algs_compare/{encoder_path}/bo_sdr/in_file.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_bo_sdr/in_file.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_dml_bo/in_file.csv",
    f"./output/bovaesdr_algs_compare/{encoder_path}/vae_bo_sdr/in_file.csv",
]
outfiles = [infile.replace("in_file", "out_file") for infile in infiles]

# Process each file
# for problem_info_file, infile, outfile in zip(problem_info_files, infiles, outfiles):
#     solved_times = get_solved_times_for_file(problem_info_file, infile)  # process
#     solved_times.to_csv(outfile)  # save to csv

### 2. Create plots ###
# Plot information is a list of tuples (filename_stem, label, color, linestyle, [marker], [markersize])
#   - filename_stem is a string defining the results files
#         This allows multiple runs for the same solver (useful when have noisy problems, for instance).
#         Result files must be available at "filename_stem*.csv", where filename_stem can have folder references.
#   - label is the legend entry for this solver. For LaTeX formatting, use as: r"Result $x=1$"
#   - color, linestsyle, marker and markersize have the standard definitions
#         See here: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
solver_info = (
    []
)  # each entry is tuple = (filename_stem, label, colour, linestyle, [marker], [markersize]), where marker info is optional
solver_info.append(
    (f"./output/bovaesdr_algs_compare/{encoder_path}/bo_sdr/out_file.csv", r"BO-SDR", "b", "-", ".", 5)
)
solver_info.append(
    (
        f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_bo_sdr/out_file.csv",
        r"R-BOVAE",
        "r",
        "--",
    )
)
solver_info.append(
    (
        f"./output/bovaesdr_algs_compare/{encoder_path}/retrain_vae_dml_bo/out_file.csv",
        r"S-BOVAE",
        "k",
        "-.",
    )
)
solver_info.append(
    (f"./output/bovaesdr_algs_compare/{encoder_path}/vae_bo_sdr/out_file.csv", r"V-BOVAE", "g", "dotted",'*',5)
)
#  A list of accuracy levels to generate plots for
tau_levels = [3]

# Output format for graphs (e.g. png, eps, ...)
fmt = "png"

# Whether to generate data/performance profiles
generate_data_profiles = True
generate_perf_profiles = True

# Maximum budget (in simplex gradients) to plot results for
budget = 117

# For data profiles, should the x-axis be on a log scale?
dp_with_logscale = True

# Output name for plots
# Format is "outfile_stem_[data|perf][tau].fmt"
outfile_stem = Path(f"./output/bovaesdr_algs_compare/{encoder_path}")

# Expected number of problems in input files
expected_nprobs = 10

# Generate plots
create_plots(
    outfile_stem,
    solver_info,
    tau_levels,
    budget,
    data_profiles=generate_data_profiles,
    perf_profiles=generate_perf_profiles,
    dp_with_logscale=dp_with_logscale,
    fmt=fmt,
    expected_nprobs=expected_nprobs,
)

print("Done")
