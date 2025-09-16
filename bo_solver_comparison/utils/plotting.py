#!/usr/bin/env python

"""
Plotting routines - generate data and performance profiles
in the style of [1]. 

Lindon Roberts, 2017.

References:
[1]  J. J. More' and S. M. Wild, Benchmarking Derivative-Free Optimization Algorithms,
     SIAM J. Optim., 20 (2009), pp. 172-191.

Obtained from
https://lindonroberts.github.io/opt/resources.html#methodology-for-comparing-solvers
https://lindonroberts.github.io/opt/plotting/plotting.py


"""

import os
from math import log
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# Setting the rcParams individually
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
__all__ = ['get_solved_times_for_file', 'create_plots']


DEFAULT_TAU_LEVELS = list(range(1, 11))  # [1, 2, ..., 10]


def get_problem_info(problem_info_file, tau_levels=DEFAULT_TAU_LEVELS):
    testing_info = pd.read_csv(problem_info_file)
    testing_info = testing_info.set_index('Setting')

    cutoffs = pd.DataFrame(columns=(
        'Setting', 'n', 'f0', 'fmin') + tuple(['tau'+str(y) for y in tau_levels]))
    for setting in testing_info.index.tolist():
        f0 = testing_info.at[setting, 'f0']
        m = testing_info.at[setting, 'm']
        n = testing_info.at[setting, 'n']
        fmin = testing_info.at[setting, 'fmin (approx)']
        this_cutoffs = []
        for lvl in tau_levels:
            tau = 10.0**(-lvl)
            this_cutoffs.append(tau*f0 + (1-tau)*fmin)
        cutoffs.loc[cutoffs.shape[0]] = [setting, n, f0, fmin] + this_cutoffs

    cutoffs['Setting'] = cutoffs['Setting'].astype('int64')
    cutoffs = cutoffs.set_index('Setting')
    return cutoffs


def get_solved_times_for_file(problem_info_file, logfile, tau_levels=DEFAULT_TAU_LEVELS):
    # Go through each output file within 'logfile' and save solved_times information
    # Info is: num evals to get solution to within accuracy 10^(-tau_levels), plus total runtime for full budget

    cutoffs = get_problem_info(problem_info_file, tau_levels=tau_levels)

    df = pd.read_csv(logfile, header=None, names=['Setting', 'nf', 'fval'])
    df['fval'] = df['fval'].astype(float)
    this_result = []  # columns are: setting, n, fmin, nf, [tau*]
    for setting in cutoffs.index.tolist():
        fvals = df[df['Setting'] == setting]
        fmin = fvals['fval'].min()
        neval = fvals['nf'].max()
        n = cutoffs.at[setting, 'n']
        solved_times = []
        for lvl in tau_levels:
            cutoff = cutoffs.at[setting, 'tau'+str(lvl)]
            if cutoff < fmin:  # i.e. never solved
                solved_time = -1.0
            else:
                solved_time = fvals['nf'][fvals['fval'] <= cutoff].min()
            solved_times.append(solved_time)
        this_result.append([setting, n, fmin, neval] + solved_times)

    # Clean up these results
    tau_cols = tuple(['tau%g' % y for y in tau_levels])
    this_df = pd.DataFrame(this_result, columns=(
        'Setting', 'n', 'fmin', 'nf') + tau_cols)
    this_df['Setting'] = this_df['Setting'].astype('int64')
    this_df['n'] = this_df['n'].astype('int64')
    this_df['nf'] = this_df['nf'].astype('int64')
    for lvl in tau_levels:
        this_df['tau%g' % lvl] = this_df['tau%g' % lvl].astype('int64')
    this_df = this_df.set_index('Setting')

    return this_df


def get_all_results(run_id):
    # Split run_id into folder and file
    if '/' in run_id:
        results_folder = run_id[:run_id.rfind('/')]
        results_stem = run_id[run_id.rfind('/')+1:]
    else:
        results_folder = os.getcwd()
        results_stem = run_id

    result_files = []
    for myfile in os.listdir(results_folder):
        if myfile.startswith(results_stem) and myfile.endswith('.csv'):
            result_files.append(myfile)

    all_results = []
    for infile in result_files:
        df = pd.read_csv(results_folder + '/' + infile)
        df['Setting'] = df['Setting'].astype('int64')
        df = df.set_index('Setting')
        all_results.append(df)
    return all_results


def get_all_results_csv(file_path):
    all_results = []
    df = pd.read_csv(file_path)
    df['Setting'] = df['Setting'].astype('int64')
    df = df.set_index('Setting')
    all_results.append(df)
    return all_results


def get_data_profile(results, xvals, tau_level, expected_nprobs=None, xvals_in_gradients=True):
    # Get data profile as np.array for a *single run*
    if expected_nprobs is not None:
        assert len(results) == expected_nprobs, "Results has %g settings (expected %g)" % (
            len(results), expected_nprobs)

    nvals = len(xvals)
    dp = np.zeros((nvals,))

    # Get solved budget in terms of gradients
    col = 'tau%g' % tau_level
    if xvals_in_gradients:
        solved_budget = results[col] / (results['n'] + 1)
        solved_budget[results[col] < 0] = -1.0
    else:
        solved_budget = results[col]

    for i in range(nvals):
        budget = xvals[i]
        nsolved = len(
            solved_budget[(solved_budget >= 0) & (solved_budget <= budget)])
        dp[i] = float(nsolved) / float(len(results))

    return dp


def get_average_data_profile(all_results, xvals, tau_level, expected_nprobs=None, xvals_in_gradients=True):
    nruns = len(all_results)
    nvals = len(xvals)
    dp = np.zeros((nvals, nruns))

    for i in range(nruns):
        dp[:, i] = get_data_profile(all_results[i], xvals, tau_level,
                                    expected_nprobs=expected_nprobs, xvals_in_gradients=xvals_in_gradients)
    # average over columns (i.e. average over each run)
    return np.mean(dp, axis=1), nruns


def plot_data_profile(all_results_list, xvals, colours, linestyles, labels, markers, tau_level, expected_nprobs=None, fmt="png",
                      save_to_file=False, out_folder_path=None, dp_with_logscale=False):
    nplot = len(all_results_list)
    assert len(colours) == nplot, "colours has wrong length (expect %g)" % nplot
    assert len(linestyles) == nplot, "colours has wrong length (expect %g)" % nplot
    assert len(labels) == nplot, "colours has wrong length (expect %g)" % nplot

    if save_to_file:
        font_size = 'large'  # x-large for presentations
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        assert out_folder_path is not None, "Need to specify outfile_stem if saving plots"
    else:
        font_size = 'small'

    plt.ioff()  # non-interactive mode
    plt.figure(1)

    plt.clf()
    ax = plt.gca()  # current axes
    plot_fun = ax.semilogx if dp_with_logscale else ax.plot
    results = {}
    results['xval'] = xvals
    for i in range(nplot):
        dp, nruns = get_average_data_profile(
            all_results_list[i], xvals, tau_level, expected_nprobs=expected_nprobs, xvals_in_gradients=True)
        if nruns > 0:
            results[labels[i]] = dp
            # Plot details
            col = colours[i]
            ls = linestyles[i]
            lbl = labels[i]
            mkr = markers[i][0]
            ms = markers[i][1]
            if mkr != '':
                # If using a marker, only put the marker on a subset of points (to avoid cluttering)
                skip_array = np.mod(np.arange(len(dp)), len(dp)//10) == 0
                # Line 1: the subset of points with markers
                plot_fun(xvals[skip_array], dp[skip_array], label='_nolegend_',
                         color=col, linestyle='', marker=mkr, markersize=ms)
                # Line 2: a single point with the correct format, so the legend label can use this
                plot_fun(xvals[0], dp[0], label=lbl, color=col,
                         linestyle=ls, marker=mkr, markersize=ms)
                # Line 3: the original line with no markers (or label)
                plot_fun(xvals, dp, label='_nolegend_', color=col,
                         linestyle=ls, linewidth=2.0, marker='', markersize=0)
            else:
                plot_fun(xvals, dp, label=lbl, color=col, linestyle=ls,
                         linewidth=2.0, marker='', markersize=0)

    results_df = pd.DataFrame.from_dict(results)
    
    ax.set_xlabel(r"Simplex Gradient [Budget/($n_p$ + 1)]", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)

    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([0, np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()

    if save_to_file:
        results_df.to_csv(out_folder_path /
                          f"data{tau_level}_raw.csv", index=False)
        print(out_folder_path / f"data{tau_level}_raw.{fmt}")
        plt.savefig(out_folder_path /
                    f"data{tau_level}_raw.{fmt}", bbox_inches='tight')
    else:
        plt.show()

    return


def get_min_solved_times(all_results_list, tau_level):
    # For each problem, get the smallest solve time of *any* run (the baseline for performance profiles)
    nprobs = len(all_results_list[0][0])
    col = 'tau%g' % tau_level
    # nf in absolute terms (not in gradients)
    min_solved_times = -np.ones((nprobs,), dtype=np.int8)
    for all_results in all_results_list:
        # For each solver
        for results in all_results:
            # For each run for the given solver
            solved_times = results[col].to_numpy()
            assert len(solved_times) == nprobs, "solved_times has wrong length (got %g, expecting %g)" % (
                len(solved_times), nprobs)
            for i in range(nprobs):
                if solved_times[i] >= 0:  # if this run solved the problem
                    if min_solved_times[i] < 0 or min_solved_times[i] > solved_times[i]:
                        min_solved_times[i] = solved_times[i]
    return min_solved_times


def get_perf_profile(results, xvals, min_solved_times, tau_level, max_ngradients, expected_nprobs=None):
    # Get performance profile as np.array for a *single run*
    if expected_nprobs is not None:
        assert len(results) == expected_nprobs, "Results has %g settings (expected %g)" % (
            len(results), expected_nprobs)

    nvals = len(xvals)
    pp = np.zeros((nvals,))

    # Get solved budget in terms of gradients
    col = 'tau%g' % tau_level
    solved_budget = results[col].to_numpy().astype(
        np.float32) / min_solved_times.astype(np.float32)
    solved_budget[results[col] < 0] = -1.0
    # don't allow problems beyond max_ngradients
    solved_budget[results[col] / (results['n']+1) > max_ngradients] = -1.0

    for i in range(nvals):
        ratio = xvals[i]
        nsolved = len(
            solved_budget[(solved_budget >= 0) & (solved_budget <= ratio)])
        pp[i] = float(nsolved) / float(len(results))

    return pp


def get_average_perf_profile(all_results, xvals, min_solved_times, tau_level, max_ngradients, expected_nprobs=None):
    nruns = len(all_results)
    nvals = len(xvals)
    pp = np.zeros((nvals, nruns))

    for i in range(nruns):
        pp[:, i] = get_perf_profile(all_results[i], xvals, min_solved_times,
                                    tau_level, max_ngradients, expected_nprobs=expected_nprobs)
    # average over columns (i.e. average over each run)
    return np.mean(pp, axis=1), nruns


def plot_perf_profile(all_results_list, xvals, colours, linestyles, labels, markers, tau_level, max_ngradients, expected_nprobs=None, fmt="png",
                      save_to_file=False, out_folder_path=None):
    nplot = len(all_results_list)
    assert len(colours) == nplot, "colours has wrong length (expect %g)" % nplot
    assert len(linestyles) == nplot, "colours has wrong length (expect %g)" % nplot
    assert len(labels) == nplot, "colours has wrong length (expect %g)" % nplot

    min_solved_times = get_min_solved_times(
        all_results_list, tau_level=tau_level)

    if save_to_file:
        font_size = 'large'  # x-large for presentations
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        assert out_folder_path is not None, "Need to specify out_folder_path if saving plots"
    else:
        font_size = 'small'

    plt.ioff()  # non-interactive mode
    plt.figure(1)

    plt.clf()
    ax = plt.gca()  # current axes
    plot_fun = ax.semilogx
    results = {}
    results['xval'] = xvals
    for i in range(nplot):
        dp, nruns = get_average_perf_profile(
            all_results_list[i], xvals, min_solved_times, tau_level, max_ngradients, expected_nprobs=expected_nprobs)
        if nruns > 0:
            results[labels[i]] = dp
            # Plot details
            col = colours[i]
            ls = linestyles[i]
            lbl = labels[i]
            mkr = markers[i][0]
            ms = markers[i][1]
            if mkr != '':
                # If using a marker, only put the marker on a subset of points (to avoid cluttering)
                skip_array = np.mod(np.arange(len(dp)), len(dp)//10) == 0
                # Line 1: the subset of points with markers
                plot_fun(xvals[skip_array], dp[skip_array], label='_nolegend_',
                         color=col, linestyle='', marker=mkr, markersize=ms)
                # Line 2: a single point with the correct format, so the legend label can use this
                plot_fun(xvals[0], dp[0], label=lbl, color=col,
                         linestyle=ls, marker=mkr, markersize=ms)
                # Line 3: the original line with no markers (or label)
                plot_fun(xvals, dp, label='_nolegend_', color=col,
                         linestyle=ls, linewidth=2.0, marker='', markersize=0)
            else:
                plot_fun(xvals, dp, label=lbl, color=col, linestyle=ls,
                         linewidth=2.0, marker='', markersize=0)

    results_df = pd.DataFrame.from_dict(results)

    ax.set_xlabel(r"Performance Ratio = Budget / min budget of any solver", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([np.min(xvals), np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()

    # Nicely format x-axis labels
    log_xmax = int(round(log(np.max(xvals), 2.0)))
    # 1, 2, 4, 8, ..., max(xvals)
    xticks = [2 ** y for y in range(log_xmax + 1)]
    ax.set_xticks(xticks)
    # in newer matploblib versions, minor ticks break label changes for log-scale axes
    ax.minorticks_off()
    # ax.set_xticks(range(1, xticks[-1] + 1), minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if save_to_file:
        results_df.to_csv(out_folder_path /
                          f"perf{tau_level}_raw.csv", index=False)
        plt.savefig(out_folder_path /
                    f"perf{tau_level}.{fmt}", bbox_inches='tight')
    else:
        plt.show()

    return


def create_plots(out_folder_path, solver_info, tau_levels, budget, max_ratio=32.0,
                 data_profiles=True, perf_profiles=True, save_to_file=True, fmt="png",
                 dp_with_logscale=False, expected_nprobs=None):
    all_results_list = []
    colours = []
    linestyles = []
    labels = []
    markers = []
    for s in solver_info:
        filename = s[0]
        lbl = s[1]
        col = s[2]
        ls = s[3]
        if len(s) > 4:
            mkr = s[4]
            ms = s[5]
        else:
            mkr = ''
            ms = 0
        labels.append(lbl)
        colours.append(col)
        linestyles.append(ls)
        markers.append((mkr, ms))
        all_results_list.append(get_all_results_csv(filename))

    for i in range(len(all_results_list)):
        print("Found %g runs for solver [%s]" %
              (len(all_results_list[i]), labels[i]))

    if data_profiles:
        print("Generating data profiles")
        if dp_with_logscale:
            xvals = 10.0 ** np.linspace(0.0, log(budget, 10.0), 101)
            xvals = np.pad(xvals, [(1, 0)], 'constant',
                           constant_values=0)  # prepend zero
        else:
            xvals = np.linspace(0.0, budget, 101)

        for lvl in tau_levels:
            plot_data_profile(all_results_list, xvals, colours, linestyles, labels, markers, lvl, expected_nprobs=expected_nprobs,
                              save_to_file=save_to_file, out_folder_path=out_folder_path, dp_with_logscale=dp_with_logscale, fmt=fmt)

    if perf_profiles:
        print("Generating performance profiles")
        xvals = 2.0 ** np.linspace(0.0, log(max_ratio, 2.0), 101)
        for tau_level in tau_levels:
            plot_perf_profile(all_results_list, xvals, colours, linestyles, labels, markers, tau_level, budget, expected_nprobs=expected_nprobs,
                              save_to_file=save_to_file, out_folder_path=out_folder_path, fmt=fmt)

    return

