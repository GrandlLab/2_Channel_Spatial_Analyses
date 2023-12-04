
### libraries used in this script ###
import os
import statistics
import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
import sklearn
from math import sqrt
from statistics import mean
from statistics import stdev
import cProfile
import pstats
from scipy.spatial import KDTree
import re


### Color Pallete for Images 

colors=['aqua','darkmagenta','teal', 'purple','fuchsia', 'cyan', 'darkturquoise', 'hotpink']
colormap1=(sns.color_palette(colors))

def NN(DF1,DF2,radius): #x and y coordinates need to be in column 0 and column 1 of the dataframe
    
    #Generate KDtree For Each Population 
    tree1=KDTree(DF1[['XMnm', 'YMnm']].values)  # generate the KD-tree
    tree2=KDTree(DF2[['XMnm', 'YMnm']].values)
    
    # Nearest Neighbors Between Population 1, 2
    
    ### Pop 2 to 1
    d1_2, i1_2 = tree1.query(DF2[['XMnm', 'YMnm']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF2['NND_other'] = d1_2[:, 0]  # make new column containing the distance to the nearest neighbour
    DF2['NNI_other'] = i1_2[:, 0]  # make new column containing the index of the nearest neighbour
     
    ### Pop 1 to 2
    d2_1, i2_1 = tree2.query(DF1[['XMnm', 'YMnm']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_other'] = d2_1[:, 0]  # make new column containing the distance to the nearest neighbour
    DF1['NNI_other'] = i2_1[:, 0]  # make new column containing the index of the nearest neighbour
     
    
    #Nearest Self Neighbor in Population 1 and Population 2
    d1_1, i1_1 = tree1.query(DF1[['XMnm', 'YMnm']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_self'] = d1_1[:, 1]  # make new column containing the distance to the nearest neighbour
    DF1['NNI_self'] = i1_1[:, 1]  # make new column containing the index of the nearest neighbour
    
    d2_2, i2_2 = tree2.query(DF2[['XMnm', 'YMnm']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF2['NND_self'] = d2_2[:, 1]  # make new column containing the distance to the nearest neighbour
    DF2['NNI_self'] = i2_2[:, 1]  # make new column containing the index of the nearest neighbour
    
    
    #Ball Queries
    #Ball-Other
    Ball_12=tree2.query_ball_tree(tree1, r=radius)
    c_12 = [len(b) for b in Ball_12]
    
    Ball_21=tree1.query_ball_tree(tree2, r=radius)
    c_21 = [len(b) for b in Ball_21]
    
    #Ball-Self
    Ball_11=tree1.query_ball_tree(tree1, r=radius)
    c_11 = [len(b) for b in Ball_11]
    
    Ball_22=tree2.query_ball_tree(tree2, r=radius)
    c_22 = [len(b) for b in Ball_22]
   
    #Append to DF
    DF1['ball_counts']=c_21
    DF1['ball_selfcounts']=c_11
    
    DF2['ball_counts']=c_12
    DF2['ball_selfcounts']=c_22

 
    return DF1, DF2

def var(Piezo_df, TREK_df, roi):
    area_micron=(roi.loc[0][2]) #get area from ROI df, 
    
    sqr_pixel_conversion=(roi.loc[0][15])
    area_pixel=(area_micron/sqr_pixel_conversion)
    
    x_pixel_conversion=(roi.loc[0][13])
    y_pixel_conversion=(roi.loc[0][14])
 
    
    print(sqr_pixel_conversion)

    arena=area_pixel+(area_pixel*3) ## change to min and max of the mask population 

    TREK_count=TREK_df['XMnm'].count() #number of puncta TREK

    Piezo_count=Piezo_df['XMnm'].count() #number of puncta Piezo

    root_area_micron=(sqrt(area_micron))*1000 # area for RNG modeling cell as a square for the same total area,convert to nm

    root_area_pixel=(sqrt(area_pixel))

    root_arena_pixel=(sqrt(arena))

    density_TREK_um=TREK_count/area_micron #density TREK- not in use as of 3/10/23

    density_TREK_pix=TREK_count/area_pixel

    density_Piezo_um=Piezo_count/area_micron

    sim_overexpressedPiezo=15*area_micron

    density_overexpressedPiezo_pixel=sim_overexpressedPiezo/area_pixel

    density_Piezo_pix=Piezo_count/area_pixel #density Piezo- not in use as of 3/10/23
    

    t1p1_real_nnd_avg=TREK_df['NND_other'].mean()
    t1p1_real_nnd_med=TREK_df['NND_other'].median()
    t1p1_real_nnd_quantile_1=TREK_df['NND_other'].quantile(0.25)
    t1p1_real_nnd_quantile_3=TREK_df['NND_other'].quantile(0.75)

    p1t1_real_nnd_avg=Piezo_df['NND_other'].mean()
    p1t1_real_nnd_med=Piezo_df['NND_other'].median()
    p1t1_real_nnd_quantile_1=Piezo_df['NND_other'].quantile(0.25)
    p1t1_real_nnd_quantile_3=Piezo_df['NND_other'].quantile(0.75)

    t1t1_real_nnd_avg=TREK_df['NND_self'].mean()
    t1t1_real_nnd_med=TREK_df['NND_self'].median()
    t1t1_real_nnd_quantile_1=TREK_df['NND_self'].quantile(0.25)
    t1t1_real_nnd_quantile_3=TREK_df['NND_self'].quantile(0.75)

    pz1pz1_nnd_avg=Piezo_df['NND_self'].mean()
    pz1pz1_nnd_med=Piezo_df['NND_self'].median()
    pz1pz1_nnd_quantile_1=Piezo_df['NND_self'].quantile(0.25)
    pz1pz1_nnd_quantile_3=Piezo_df['NND_self'].quantile(0.75)


    
    
   #'P1T1 Mean NND (empirical)(nm)', 'P1T1 Median NND (empirical)(nm)', 'P1T1 Quartile1 NND (empirical)','P1T1 Quartile3 NND (empirical)(nm)', 'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 
 # 'T1T1 Mean NND(nm)', 'T1T1 Median NND(nm)', 'T1T1 Quantile 1 NND(nm)', 'T1T1 Quantile 3 NND (nm)', 
    variable_list=area_micron, area_pixel, arena, root_area_micron, root_area_pixel, root_arena_pixel, TREK_count, density_TREK_um, density_TREK_pix, Piezo_count, density_Piezo_um, density_Piezo_pix,t1p1_real_nnd_avg, t1p1_real_nnd_med,t1p1_real_nnd_quantile_1,t1p1_real_nnd_quantile_3,p1t1_real_nnd_avg,p1t1_real_nnd_med,p1t1_real_nnd_quantile_1,p1t1_real_nnd_quantile_3,t1t1_real_nnd_avg,t1t1_real_nnd_med,t1t1_real_nnd_quantile_1,t1t1_real_nnd_quantile_3,pz1pz1_nnd_avg,pz1pz1_nnd_med,pz1pz1_nnd_quantile_1,pz1pz1_nnd_quantile_3,x_pixel_conversion, y_pixel_conversion, sqr_pixel_conversion #Vars list
    col_names=['Area', 'Area Mask (pix)', 'Area for Arena','square root Area', 'square root Area (nm)','square root Area (pix)',' Trek Count','Trek Density','Trek Density (pix)','Piezo Count','Piezo Density (um)', 'Piezo Density(pix)','T1P1 Mean NND (empirical)(nm)', 'T1P1 Median NND (empirical)(nm)', 'T1P1 Quartile1 NND (empirical)','T1P1 Quartile3 NND (empirical)(nm)', 'P1T1 Mean NND (empirical)(nm)', 'P1T1 Median NND (empirical)(nm)', 'P1T1 Quartile1 NND (empirical)','P1T1 Quartile3 NND (empirical)(nm)',  'T1T1 Mean NND(nm)', 'T1T1 Median NND(nm)', 'T1T1 Quantile 1 NND(nm)', 'T1T1 Quantile 3 NND (nm)',  'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 'x_pixel_conversion', 'y_pixel_conversion','sqr_pixel_conversion'] #Headers
    variables = pd.DataFrame(columns=col_names)
    variables.loc[0]=variable_list
    print(variables)
 
    
    return variables


def bin_solo(DF):
    
    bin_edges_tc= np.linspace(0, 4000.0, 81) 
    
    binned_data_other= np.histogram(DF['NND_other'], bins=bin_edges_tc)
    df_bin_other=pd.DataFrame(binned_data_other).transpose()
    df_bin_other.rename({0:'count',1:'bin'}, axis=1, inplace=True)
    df_bin_other['norm']=df_bin_other['count']/(len(DF))
    

    return df_bin_other

