import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib.patheffects as path_effects

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from utils.train_utils import return_data_ncl_imgsize, mkdir

import copy
from copy import deepcopy
# https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values-into-matplotlib

# https://stackoverflow.com/questions/44941082/plot-multiple-columns-of-pandas-dataframe-using-seaborn
# sns.set_theme(style="ticks", palette="pastel")
plt.rcParams['font.family'] = 'Arial'

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.4)
# sns.set_theme(font_scale=1.4)
# sns.set_theme(style="whitegrid", font_scale=1.4)
sns.set_style("whitegrid")
# colors = ["#FF0B04", "#4374B3"]
# sns.set_palette(sns.color_palette(colors)) # 'deep', 'colorblind', 'Paired'

# color_pelette = sns.color_palette("Blues")
color_pelette = sns.color_palette("rocket_r")
# color_pelette_hex = ["#ffbaba", "#ff7b7b", "#ff5252", "#ff0000", "#a70000", "#eeeeee"]
# color_pelette_hex = ["#ffbaba", "#ff7b7b", "#ff5252", "#ff0000", "#a70000", "#eeeeee"]
color_pelette[5] = "#eeeeee"

sns.set_palette(color_pelette) # 'deep', 'colorblind', 'Paired'
 

# sns.set_palette(sns.color_palette("Paired")) # 'deep', 'colorblind', 'Paired'

# Initialize the figure with a logarithmic x axis
# https://stackoverflow.com/questions/27698377/how-do-i-make-sans-serif-superscript-or-subscript-text-in-matplotlib
# update $$ font is not italic 
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

# ax.set_xscale("log")
# ORDER = {'Corrupted', 'MARC','cycleGAN','BSA', 'Our'}

# BOX_PAIRS = [("csmri1", "asl"), ("csmri2", "asl"),("csmtl", "asl"),("csl", "asl")]

# Load the dataset and preprocessing
def read_excel_data(filename):
    # if filename is None:
        # excel_abs_path = r"./_0.15_random_20220114-214856.xlsx"
    
    pd_data = pd.read_excel(open(filename, 'rb'), sheet_name=None, index_col=0) # sheet_name=None is to read all sheets. get a dict.
    return pd_data

def change_key_name(dict):
    # https://stackoverflow.com/questions/16475384/rename-a-dictionary-key
    # new_keys = ['Baseline','CS-MTL','MD-Recon-Net', 'LOUPE', 'SemuNet']
    # new_dict = {'$CSMRI_{Baseline}$': None,'$CSMTL_{Liu}$': None,'$CSMRI_{MDReconNet}$': None, '$CSL_{LOUPE}$':None, '$ASL_{SemuNet}$':None}

    new_dict = {'Step1': None,'Step2': None,'Step3': None}

    for key, n_key in zip(dict.keys(), new_dict.keys()):
        new_dict[n_key] = dict[key]

    return new_dict

def multi_sheets_to_one_withwarning(pd_data):
    df_all = None
    new_seg_name = {"Cortical GM": "Cortical Gray Matter", "Basal ganglia":"Basal Ganglia", "WM":"White Matter", 
    "CSF":"Cerebrospinal Fluid\nin the Extracerebral Space"}



    for key in pd_data.keys():

        pd_data[key] = pd_data[key].rename(columns=new_seg_name)

        rows = len(pd_data[key]) # 48
        pd_data[key]['Methods']= [key] * rows
        # pd_data[key].rename(columns = {'Unnamed: 0':'id'}, inplace = True)
        if df_all is None:
            df_all = pd_data[key]
        else:
            df_all = df_all.append(pd_data[key], ignore_index = True)

    return df_all

def multi_sheets_to_one(pd_data):
    df_all = None
    new_seg_name = {"Cortical GM": "Cortical Gray Matter", "Basal ganglia":"Basal Ganglia", "WM":"White Matter", 
    "CSF":"Cerebrospinal Fluid\nin the Extracerebral Space"}

    dataframes = []

    for key in pd_data.keys():

        pd_data[key] = pd_data[key].rename(columns=new_seg_name)

        rows = len(pd_data[key]) # 48
        pd_data[key]['Methods']= [key] * rows
        # pd_data[key].rename(columns = {'Unnamed: 0':'id'}, inplace = True)
        dataframes.append(pd_data[key])

    df_all = pd.concat(dataframes, ignore_index=True)
    return df_all

def multi_dictto_one_df(all_df_dicts):
    dfs = all_df_dicts
    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat([df.assign(key=key) for key, df in dfs.items()], ignore_index=True)
    return combined_df

def add_median_labels_orig(ax, fmt='.1f', x_adj=0, y_adj=0):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for i, median in enumerate(lines[4:len(lines):lines_per_box]):
        x, y = (data.mean() for data in median.get_data())
        # calculate maximum value in box plot
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x + x_adj, y + y_adj, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white', fontsize=5)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def bar_plot_psnr_and_ssim(df, name, pdf_or_png, axes):
    
    df1 = copy.deepcopy(df)

    df1.drop(df[df['Methods'] =='supre'].index, inplace = True)  # remove NaN of supremum for segmentation 

    i = 0
    for key in ['PSNR', 'SSIM']:
        # f, ax = plt.subplots(figsize=(3, 6))
        # titles = 'DC'
        # if key == 'SSIM':
            # titles = 'HD'
        # ax = sns.boxplot(ax=axes[i], x="Methods", y=key, showfliers = True,
                        # data=df1, linewidth=2.5)  # linewidth 是 方块的线宽。
        ax = sns.barplot(ax=axes[i], x="key", y=key, edgecolor=".5",
                        data=df1,
                        hue="Methods",
                        #  palette="vlag", 
                        linewidth=2.5)
        # sns.stripplot(x="method", y=key, data=df,
        #           size=4, color=".3", linewidth=0)
        # sns.despine()
        # add_stat_annotation(ax, data=df, x="Methods", y=key, order=ORDER,
                            # box_pairs=BOX_PAIRS,
                            # test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        # ax.set(xticklabels=[])
        yticks_fontsize = 20
        ax.set(title=key)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        if key in ['SSIM']:
            ax.set_ylim(0, 100)
            # plt.ylabel('SSIM', fontsize=ylabel_fontsize)
            ax.axes.set_title('SSIM', fontsize=22, fontweight='bold')
        elif key in ['PSNR']:
            ax.set_ylim(15, 45)
            # plt.ylabel('PSNR', fontsize=ylabel_fontsize)
            ax.axes.set_title('PSNR', fontsize=22, fontweight='bold')       
        # plt.setp(ax.get_xticklabels(), rotation=55, horizontalalignment='center', fontweight='light')
        ax.set_xticks([])
        plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
        # plt.xticks([])
        plt.tight_layout()
        # plt.savefig('{}_{}.{}'.format(name, key, pdf_or_png))
        i += 1

def box_plot_psnr_and_ssim(df, name, pdf_or_png, axes):
    df1 = copy.deepcopy(df)

    # df1.drop(df[df['Methods'] =='supre'].index, inplace = True)  # remove NaN of supremum for segmentation 

    i = 0
    sns.set(style='ticks', font='sans-serif', font_scale=1.2, rc={'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 0.5})

    for key in ['PSNR', 'SSIM']:
        # f, ax = plt.subplots(figsize=(3, 6))
        # titles = 'DC'
        # if key == 'SSIM':
            # titles = 'HD'
        # ax = sns.boxplot(ax=axes[i], x="Methods", y=key, showfliers = True,
        #                 data=df1, linewidth=2.5)  # linewidth 是 方块的线宽。
        flierprops = dict(markerfacecolor='0.75', markersize=1,
              linestyle='none')
        ax =sns.boxplot(ax=axes[i], x="key", y=key,
                    hue="Methods", 
                    # palette=['#5F9EA0', '#DB7093'],
                    # hue="smoker", 
                    # palette=color_pelette,
                    flierprops=flierprops, # outlier customize
                    width=.55,
                    palette="vlag",
                    data=df1)
        
        ax.get_legend().remove()
        # sns.despine(offset=10, trim=True)
        # sns.stripplot(ax=axes[i], x="key", y=key, data=df,
                #   size=4, color=".3", linewidth=0)
        # sns.despine()
        # add_stat_annotation(ax, data=df, x="Methods", y=key, order=ORDER,
                            # box_pairs=BOX_PAIRS,
                            # test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        # ax.set(xticklabels=[])
        # https://seaborn.pydata.org/examples/horizontal_boxplot.html
        # Add in points to show each observation
        # sns.stripplot(x="distance", y="method", hue="Methods", data=df1,
                    # size=4, color=".3", linewidth=0)

        yticks_fontsize = 8
        y_adj=4.2
        ax.set(title=key)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        if key in ['SSIM']:
            ax.set_ylim(70, 100)
            # plt.ylabel('SSIM', fontsize=22)
            ax.axes.set_title('SSIM', fontsize=yticks_fontsize, fontweight='bold')
            # ax.axes.set_title('SSIM', fontweight='bold')
            # add_median_labels(ax)
            add_median_labels_orig(ax, y_adj=y_adj)


        elif key in ['PSNR']:
            ax.set_ylim(15, 40)
            # plt.ylabel('PSNR', fontsize=22)
            ax.axes.set_title('PSNR', fontsize=yticks_fontsize, fontweight='bold')  
            # ax.axes.set_title('PSNR', fontweight='bold') 
            # add_median_labels(ax)
            add_median_labels_orig(ax, y_adj=y_adj)
        # plt.setp(ax.get_xticklabels(), rotation=55, horizontalalignment='center', fontweight='light')
        # Add mean values to the plot
        plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=yticks_fontsize)


        # ax.set_xticks([])
        # ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        # ax.set_xlim(-0.5, len(df1['key'].unique()) - 0.5)
        # plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
        # plt.xticks([])
        plt.tight_layout()
        # plt.savefig('{}_{}.{}'.format(name, key, pdf_or_png))
        # sns.despine(left=True)
        i += 1

def pandas_group_by(all_df, name, shot_shot_filename, keys_list):
    # # https://www.statology.org/pandas-mean-by-group/
    # and chatgpt
    # https://stackoverflow.com/questions/72368434/group-pandas-dataframe-and-calculate-mean-for-multiple-columns
    df = deepcopy(all_df)
    # key is (target domain) linear, nonlinear, sudden....
    # Methods is methods.
    # Save the DataFrame to a CSV file
    formatted_data = process_data(df, keys_list)
    formatted_data.to_csv('{}_{}.csv'.format(name, shot_shot_filename), index=True)
    
    latex_table = formatted_data.style.to_latex()
    print(latex_table)

    # df.groupby('Methods')[['key', 'assists']].mean()

# promp learning using chatGPT to csv file or latex file, ./results/tab1/tab1_a_tab1.csv

def process_data(df, keys_list):
    # Read the data from a string, replace this with the actual file path if needed
    
    # Group the data by 'Methods' and 'key' and calculate the mean for 'PSNR' and 'SSIM' columns
    grouped_data = df.groupby(['Methods', 'key']).agg({'PSNR': 'mean', 'SSIM': 'mean'}, numeric_only=False).reset_index()
    
    # Define custom orders
    # method_order = ['Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours', 'Supervised']
    method_order = keys_list
    key_order = ['Linear', 'Nonlinear', 'Sudden', 'Single-shot', 'RL', 'Moderate', 'Heavy']
    
    # Convert the 'Methods' and 'key' columns to categorical data types with custom orders
    grouped_data['Methods'] = pd.Categorical(grouped_data['Methods'], categories=method_order, ordered=True)
    grouped_data['key'] = pd.Categorical(grouped_data['key'], categories=key_order, ordered=True)
    
    # Sort the data by the custom orders
    sorted_data = grouped_data.sort_values(['Methods', 'key'])
    
    # Pivot the DataFrame
    pivoted_data = sorted_data.pivot_table(index='Methods', columns='key', values=['PSNR', 'SSIM']).round(2)
    
    # Combine the PSNR and SSIM values for each method and key
    combined_data = pivoted_data['PSNR'].astype(str) + '/' + pivoted_data['SSIM'].astype(str)
    
    # Calculate the mean value of PSNR and SSIM for each method
    psnr_mean = pivoted_data['PSNR'].mean(axis=1).round(2)
    ssim_mean = pivoted_data['SSIM'].mean(axis=1).round(2)

    # Combine the PSNR and SSIM mean values for each method
    mean_values = psnr_mean.astype(str) + '/' + ssim_mean.astype(str)

    # Add mean values as a new column
    combined_data['Mean'] = mean_values


    return combined_data

def process_data_withwarning(df):
    # Read the data from a string, replace this with the actual file path if needed
    
    # Group the data by 'Methods' and 'key'
    grouped_data = df.groupby(['Methods', 'key']).mean().reset_index()
    
    # Define custom orders
    method_order = ['Corrupted', 'MARC', 'CycleGAN', 'BSA', 'Ours']
    key_order = ['Linear', 'Nonlinear', 'Sudden', 'Single-shot', 'RL', 'Moderate', 'Heavy']
    
    # Convert the 'Methods' and 'key' columns to categorical data types with custom orders
    grouped_data['Methods'] = pd.Categorical(grouped_data['Methods'], categories=method_order, ordered=True)
    grouped_data['key'] = pd.Categorical(grouped_data['key'], categories=key_order, ordered=True)
    
    # Sort the data by the custom orders
    sorted_data = grouped_data.sort_values(['Methods', 'key'])
    
    # Pivot the DataFrame
    pivoted_data = sorted_data.pivot_table(index='Methods', columns='key', values=['PSNR', 'SSIM']).round(2)
    
    # Combine the PSNR and SSIM values for each method and key
    combined_data = pivoted_data['PSNR'].astype(str) + '/' + pivoted_data['SSIM'].astype(str)
    
    return combined_data


def box_plot_seg(df, name, pdf_or_png, axes): 
    pd_data = df.drop(columns=['PSNR', 'SSIM', 'slice'])
    pd_data.drop(pd_data[pd_data['Methods'] =='supre'].index, inplace = True)  # remove NaN of supremum for segmentation 
    # convert to long (tidy) form
    dfm = pd_data.melt('Methods', var_name='Brain Organs', value_name="Dice Similarity Coefficient $\\times100\%$")
    # f, ax = plt.subplots(figsize=(3, 6))
    # g = sns.catplot(x="X_Axis", y="vals", hue='cols', data=dfm, kind='point')
    ax = sns.barplot(ax=axes[2], x="Methods", y="Dice Similarity Coefficient $\\times100\%$", edgecolor=".5",
                    data=dfm,
                    #  palette="vlag", 
                    linewidth=2.5)

    # ax = sns.boxplot(ax=axes[2], x="Methods", y="Dice Similarity Coefficient $\\times100\%$", showfliers = True, linewidth=2.5,
                    # data=dfm)
    # add_stat_annotation(ax, data=df, x="Methods", y="Dice Similarity Coefficient $\\times100\%$", order=ORDER,
                        # box_pairs=BOX_PAIRS,
                        # test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # ax.set(title='DSC')  # AVD
    ax.axes.set_title('DSC', fontsize=22, fontweight='bold')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set_xticks([]) # make x ticks is none
    # plt.setp(ax.get_xticklabels(), rotation=55, horizontalalignment='center', fontweight='light')
    ax.set_ylim(0, 100)
    yticks_fontsize = 20
    
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
    plt.tight_layout()
    # plt.savefig('{}_seg.{}'.format(name, pdf_or_png))
    # plt.savefig('output_seg.svg')

def bar_eachorgan_plot_seg(df, name, pdf_or_png): 
    pd_data = df.drop(columns=['PSNR', 'SSIM', 'WM lesions'])
    # convert to long (tidy) form
    dfm = pd_data.melt('Methods', var_name='Brain Organs', value_name="Dice Similarity Coefficient $\\times100\%$")
    f, ax = plt.subplots(figsize=(20, 6))
    ax = sns.barplot(x="Brain Organs", y="Dice Similarity Coefficient $\\times100\%$", hue="Methods", data=dfm)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    ax.set_ylim(10, 100)
    plt.tight_layout()
    plt.savefig('{}_seg.{}'.format(name, pdf_or_png))

def bar_plot_psnr_and_ssim(df, name, pdf_or_png):
    for key in ['PSNR', 'SSIM']:
        f, ax = plt.subplots(figsize=(3, 6))
        ax = sns.barplot(x="Methods", y=key,
                        data=df, palette="vlag", linewidth=2.5)
        # sns.stripplot(x="method", y=key, data=df,
        #           size=4, color=".3", linewidth=0)
        plt.setp(ax.get_xticklabels(), rotation=45)
        if key == 'PSNR':
            ax.set_ylim(0, 40)
        elif key == 'SSIM':
            ax.set_ylim(0, 100)
        # plt.tight_layout()
        plt.savefig('{}_{}.{}'.format(name, key, pdf_or_png))

def bar_plot_seg(df, name, pdf_or_png): 
    pd_data = df.drop(columns=['PSNR', 'SSIM', 'WM lesions'])
    # convert to long (tidy) form
    dfm = pd_data.melt('Methods', var_name='Brain Organs', value_name="Dice Similarity Coefficient $\\times100\%$")
    f, ax = plt.subplots(figsize=(7, 6))
    ax = sns.barplot(x="Methods", y="Dice Similarity Coefficient $\\times100\%$", data=dfm)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    ax.set_ylim(10, 100)
    plt.tight_layout()
    plt.savefig('{}_seg.{}'.format(name, pdf_or_png))

def slice_to_volume(pd_data):
    """https://stackoverflow.com/questions/30328646/python-pandas-group-by-in-group-by-and-average

    Args:
        pd_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # new_dict = {'csmri1': None,'csmri2': None,'csmtl': None, 'csl':None, 'asl':None}

    for key in pd_data.keys():
        df = pd_data[key]
        mean_col = df.groupby(['shotname']).mean()
        pd_data[key] = mean_col
    return pd_data

def read_excel_and_plot(tabs, name, pdf_or_png, keys_list):

    # merge multi-excels to one (multi-sheets).

    all_df_dicts = {}

    for key, value in tabs.items():
        # filename = key
        pd_data = read_excel_data(value)

        # pd_data = change_key_name(pd_data)
        # pd_data = slice_to_volume(pd_data)

        df_all = multi_sheets_to_one(pd_data)
        all_df_dicts[key] = df_all

    all_df = multi_dictto_one_df(all_df_dicts)


    # fig, axes = plt.subplots(2, 1, figsize=(6, 3))

    fig, axes = plt.subplots(2, 1)
    box_plot_psnr_and_ssim(all_df, name, pdf_or_png, axes)
        
    # box_plot_seg(df_all, name, pdf_or_png, axes)

    # bar_plot_psnr_and_ssim(df_all, name, pdf_or_png)
    # bar_plot_seg(df_all, name, pdf_or_png)
    shot_filename, extension = os.path.splitext(filename)
    shot_shot_filename = shot_filename.split("/")[-1]

    pandas_group_by(all_df, name, shot_shot_filename, keys_list)

    plt.savefig('{}_{}_all.{}'.format(name, shot_shot_filename, pdf_or_png), dpi=300)

# https://www.statology.org/seaborn-legend-position/
# https://stackoverflow.com/questions/58476654/how-to-remove-or-hide-x-axis-labels-from-a-seaborn-matplotlib-plot/58476779

if __name__ == "__main__":

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    conditional_filename =  './results/ablation/ablation3_EIP/{}/'.format(filename) 
    mkdir(conditional_filename)

    tab1_a = {
                
                "Linear": './results/ablation/ablation3_EIP/ixi_t1_linear_moderate_Axial_None20230625-095306.xlsx',
                "Nonlinear": './results/ablation/ablation3_EIP/ixi_t1_nonlinear_moderate_Axial_None20230625-095556.xlsx',
                "Sudden": './results/ablation/ablation3_EIP/ixi_t1_sudden_moderate_Axial_None20230625-095845.xlsx',
                "Single-shot": './results/ablation/ablation3_EIP/ixi_t1_singleshot_moderate_Axial_None20230625-100133.xlsx',
                "RL": './results/ablation/ablation3_EIP/ixi_t1_periodic_slight_rl_Axial_None20230625-100422.xlsx',
                "Moderate": './results/ablation/ablation3_EIP/ixi_t1_periodic_moderate_Axial_None20230625-100709.xlsx',
                "Heavy": './results/ablation/ablation3_EIP/ixi_t1_periodic_heavy_Axial_None20230625-100958.xlsx',
    }

    keys_list = ['csmri', 'ei', 'rotcsmri', 'csmri_ei', 'csmri_rotcsmri', 'ei_rotcsmri', 'our']


    name = conditional_filename + 'tab1_a'
    pdf_or_png = 'png'

    read_excel_and_plot(tab1_a, name, pdf_or_png, keys_list)

