import matplotlib.pyplot as plt
import matplotlib
import argparse
import pandas as pd
import os
import numpy as np

matplotlib.use('agg')
legend_keywords = {'prop': {'size': 24}}
plot_keywords = {'markersize': 24}
title_keywords = {'fontsize': 24}
xlabel_keywords = {'fontsize': 24}
ylabel_keywords = {'fontsize': 24}

test_markers = ['o', "v", "^", "s", "*", "D"]


def create_REORDER_plots(df, output_dir=None):
    small_df = df.filter(items=['name', 'cycles', 'test_name'])

    test_dfs = {}

    reorder_tests = ['ijk',
                     'jik',
                     'ikj',
                     'jki',
                     'kij',
                     'kji']

    for test in reorder_tests:

        t_df = small_df[small_df['test_name'] == test]
        test_dfs[test] = t_df

    # First example graph - fixed split factor, vary sparsity
    plot_reorder(test_dfs=test_dfs, test_names=reorder_tests, legend=reorder_tests, output_dir=output_dir)


def plot_reorder(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test]
        # plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
        plt.bar(t_df['test_name'], t_df['cycles'])
    plt.legend(legend, **legend_keywords)
    plt.xlabel('Reordering', **xlabel_keywords)
    plt.ylabel('Cycles', **ylabel_keywords)
    plt.title('Cycles vs Loop Reordering', **title_keywords)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim([10000, 1000000])
    plt.yscale('log')
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('reorder', output_dir)


def create_FUSION_plots(df, output_dir=None):
    # print(df.to_string())
    small_df = df.filter(items=['name', 'cycles', 'test_name', 'kdim'])

    test_dfs = {}

    fusion_names = ['sddmm_coiter_fused',
                    'sddmm_locate_fused',
                    'sddmm_unfused']

    fusion_legend = ['sddmm_coiter_fused',
                     'sddmm_locate_fused',
                     'sddmm_unfused']

    for test in fusion_names:

        # t_df = small_df.filter(like='test')
        # t_df = small_df[small_df['test_name'] == test]
        t_df = small_df[small_df['test_name'] == test]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_fusion(test_dfs=test_dfs, test_names=fusion_names, legend=fusion_legend, output_dir=output_dir)


def plot_fusion(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    unfused = []
    coiterate = []
    locate = []
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test]
        print(t_df)
        for kdim in [1, 10, 100]:
            t_df_ = t_df[t_df["kdim"] == kdim]
        # plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
            print(t_df_)
            print("__________")
            if "unfused" in test:
                print(t_df_["cycles"].values[0])
                unfused.append(t_df_["cycles"].values[0])
            if "coiter" in test:
                coiterate.append(t_df_["cycles"].values[0])
            if "locate" in test:
                locate.append(t_df_["cycles"].values[0])
    print(unfused)
    print(coiterate)
    print(locate)
    width = 0.25
    x = np.arange(3)
    plt.bar(x - width, unfused, width, label="Unfused")
    plt.bar(x, coiterate, width, label="Coiteration")
    plt.bar(x + width, locate, width, label="Iterate-Locate")
    # plt.bar(t_df['test_name'], t_df['cycles'])
    labels = ['1', '10', '100']
    plt.legend(legend, **legend_keywords)
    plt.xticks(x, labels)
    plt.yscale('log')
    plt.xlabel('Increasing K dimention for Fusion (for sddmm)', **xlabel_keywords)
    plt.ylabel('Cycles', **ylabel_keywords)
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('fusion', output_dir)


def create_ACCEL_plots(df, output_dir=None):
    # print(df.to_string())
    small_df = df.filter(items=['cycles', 'vectype', 'sparsity', 'split_factor', 'test_name', 'block_size', 'run_length'])

    test_dfs = {}

    test_names = ['test_unit_vec_elemmul_c_c_c',
                  'test_unit_vec_elemmul_u_u_u',
                  'test_vec_elemmul_skip_c_c_c',
                  'test_vec_elemmul_split',
                  'test_vec_elemmul_bv_split',
                  'test_vec_elemmul_bv']

    test_names_legend = ['Compressed Coordinates (Crd)',
                         'Uncompressed (dense)',
                         'Crd w/ skip',
                         'Crd w/ split',
                         'Bitvector w/ split',
                         'Bitvector']

    for test in test_names:

        # t_df = small_df.filter(like='test')
        t_df = small_df[small_df['test_name'] == test]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_urandom_const_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend, output_dir=output_dir)
    plot_urandom_sf_const_sp(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend, output_dir=output_dir)
    plot_runs_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend, output_dir=output_dir)
    plot_blocks_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend, output_dir=output_dir)


def plot_urandom_const_sf(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        # plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
        # Have percent of 0's in sparsity
        plt.plot(2000 - (t_df['sparsity'] * 2000), t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('# nonzeros', **xlabel_keywords)
    plt.ylabel('Cycles', **ylabel_keywords)
    plt.title('Cycles vs Sparsity (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('urandom_const_sf', output_dir)


def plot_urandom_sf_const_sp(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') & ((test_dfs[test]['sparsity'] == 0.9))]
        print(t_df)
        plt.plot(t_df['split_factor'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    # plt.xscale('log')
    plt.xlabel('Split Factor', **xlabel_keywords)
    plt.yscale('log')
    plt.ylabel('Cycles', **ylabel_keywords)
    plt.title('Cycles vs Split Factor (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('urandom_const_sp', output_dir)


def plot_runs_sf(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'runs') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        print(t_df)
        plt.plot(t_df['run_length'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.xscale('log')
    plt.xlabel('Run Length', **xlabel_keywords)
    plt.yscale('log')
    plt.ylabel('Cycles', **ylabel_keywords)
    plt.title('Cycles vs Split Factor (runs)', **title_keywords)
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('runs_sf', output_dir)


def plot_blocks_sf(test_dfs, test_names, legend, output_dir=None):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'blocks') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        print(t_df)
        plt.plot(t_df['block_size'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.xscale('log')
    plt.xlabel('Block Size', **xlabel_keywords)
    plt.yscale('log')
    plt.ylabel('Cycles', **ylabel_keywords)
    plt.title('Cycles vs Split Factor (blocks)', **title_keywords)
    fig.set_size_inches(16, 12)
    save_figure_ASPLOS('blocks_sf', output_dir)


def create_ASPLOS_plots(csv_path, name=None, output_dir=None):

    assert output_dir is not None
    assert name is not None
    df = pd.read_csv(csv_path)

    print(df)

    if name == "ACCEL":
        create_ACCEL_plots(df, output_dir=output_dir)
    elif name == "REORDER":
        create_REORDER_plots(df, output_dir=output_dir)
    elif name == "FUSION":
        create_FUSION_plots(df, output_dir=output_dir)
    else:
        raise NotImplementedError

    # plt.show()


def save_figure_ASPLOS(name, output_dir, file_formats=None):
    if file_formats is None:
        file_formats = ['pdf', 'svg']

    output_path = os.path.join(output_dir, name)

    for file_format in file_formats:
        plt.savefig(f'{output_path}.{file_format}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot synthetics results csv')
    # parser.add_argument("--csv_path", type=str, default=None)
    # parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./OUTPUT_DIR")
    args = parser.parse_args()

    # csv_path = args.csv_path
    # name = args.name

    od_ = args.output_dir

    if not os.path.isdir(od_):
        os.mkdir(od_)

    # assert csv_path is not None
    # assert name is not None

    create_ASPLOS_plots(csv_path="./SYNTH_OUT_ACCEL.csv", name="ACCEL", output_dir=od_)
    create_ASPLOS_plots(csv_path="./SYNTH_OUT_FUSION.csv", name="FUSION", output_dir=od_)
    create_ASPLOS_plots(csv_path="./SYNTH_OUT_REORDER.csv", name="REORDER", output_dir=od_)
