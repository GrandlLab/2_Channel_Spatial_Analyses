
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

def NN_one(DF1, radius): #x and y coordinates need to be in column 0 and column 1 of the dataframe
    
    #Generate KDtree For Each Population 
    tree1=KDTree(DF1[['XMnm', 'YMnm']].values)  # generate the KD-tree
    #tree2=KDTree(DF2[['XMnm', 'YMnm']].values)
    
    # Nearest Neighbors Between Population 1, 1 

    #Nearest Self Neighbor in Population 1 and Population 2
    d1_1, i1_1 = tree1.query(DF1[['XMnm', 'YMnm']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_self'] = d1_1[:, 1]  # make new column containing the distance to the nearest neighbour
    DF1['NNI_self'] = i1_1[:, 1]  # make new column containing the index of the nearest neighbour
    
    #Ball-Self
    Ball_11=tree1.query_ball_tree(tree1, r=radius)
    c_11 = [len(b) for b in Ball_11]
    

    #Append to DF

    DF1['ball_selfcounts']=c_11

    return DF1


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

def NN_mod(DF1,DF2,radius): #x and y coordinates need to be in column 0 and column 1 of the dataframe
    
    #Generate KDtree For Each Population 
    tree1=KDTree(DF1[['X', 'Y']].values)  # generate the KD-tree
    tree2=KDTree(DF2[['X', 'Y']].values)
    
    # Nearest Neighbors Between Population 1, 2
    
    ### Pop 2 to 1
    d1_2, i1_2 = tree1.query(DF2[['X', 'Y']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF2['NND_other'] = d1_2[:, 0]  # make new column containing the distance to the nearest neighbour
    DF2['NNI_other'] = i1_2[:, 0]  # make new column containing the index of the nearest neighbour
     
    ### Pop 1 to 2
    d2_1, i2_1 = tree2.query(DF1[['X', 'Y']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_other'] = d2_1[:, 0]  # make new column containing the distance to the nearest neighbour
    DF1['NNI_other'] = i2_1[:, 0]  # make new column containing the index of the nearest neighbour
     
    
    #Nearest Self Neighbor in Population 1 and Population 2
    d1_1, i1_1 = tree1.query(DF1[['X', 'Y']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_self'] = d1_1[:, 1]  # make new column containing the distance to the nearest neighbour
    DF1['NNI_self'] = i1_1[:, 1]  # make new column containing the index of the nearest neighbour
    
    d2_2, i2_2 = tree2.query(DF2[['X', 'Y']].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
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

def var_yoda(Piezo_df, roi):
    area_micron=(roi.loc[0][1]) #get area from ROI df, 
    
    sqr_pixel_conversion=roi.loc[0][9]
    x_pixel_conversion=(roi.loc[0][8])
    y_pixel_conversion=(roi.loc[0][7])
    print(x_pixel_conversion)
    
    area_pixel=(area_micron/sqr_pixel_conversion)
    print(area_pixel)

    arena=area_pixel+(area_pixel*3) ## change to min and max of the mask population 

    #TREK_count=TREK_df['XMnm'].count() #number of puncta TREK

    Piezo_count=Piezo_df['XMnm'].count() #number of puncta Piezo

    root_area_micron=(sqrt(area_micron))*1000 # area for RNG modeling cell as a square for the same total area,convert to nm

    root_area_pixel=(sqrt(area_pixel))

    root_arena_pixel=(sqrt(arena))

    #density_TREK_um=TREK_count/area_micron #density TREK- not in use as of 3/10/23

    #density_TREK_pix=TREK_count/area_pixel

    density_Piezo_um=Piezo_count/area_micron

    sim_overexpressedPiezo=15*area_micron

    density_overexpressedPiezo_pixel=sim_overexpressedPiezo/area_pixel

    density_Piezo_pix=Piezo_count/area_pixel #density Piezo- not in use as of 3/10/23
    

#     t1p1_real_nnd_avg=TREK_df['NND_other'].mean()
#     t1p1_real_nnd_med=TREK_df['NND_other'].median()
#     t1p1_real_nnd_quantile_1=TREK_df['NND_other'].quantile(0.25)
#     t1p1_real_nnd_quantile_3=TREK_df['NND_other'].quantile(0.75)

#     p1t1_real_nnd_avg=Piezo_df['NND_other'].mean()
#     p1t1_real_nnd_med=Piezo_df['NND_other'].median()
#     p1t1_real_nnd_quantile_1=Piezo_df['NND_other'].quantile(0.25)
#     p1t1_real_nnd_quantile_3=Piezo_df['NND_other'].quantile(0.75)

#     t1t1_real_nnd_avg=TREK_df['NND_self'].mean()
#     t1t1_real_nnd_med=TREK_df['NND_self'].median()
#     t1t1_real_nnd_quantile_1=TREK_df['NND_self'].quantile(0.25)
#     t1t1_real_nnd_quantile_3=TREK_df['NND_self'].quantile(0.75)

    pz1pz1_nnd_avg=Piezo_df['NND_self'].mean()
    pz1pz1_nnd_med=Piezo_df['NND_self'].median()
    pz1pz1_nnd_quantile_1=Piezo_df['NND_self'].quantile(0.25)
    pz1pz1_nnd_quantile_3=Piezo_df['NND_self'].quantile(0.75)


    
    
   #'P1T1 Mean NND (empirical)(nm)', 'P1T1 Median NND (empirical)(nm)', 'P1T1 Quartile1 NND (empirical)','P1T1 Quartile3 NND (empirical)(nm)', 'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 
 # 'T1T1 Mean NND(nm)', 'T1T1 Median NND(nm)', 'T1T1 Quantile 1 NND(nm)', 'T1T1 Quantile 3 NND (nm)', 
    variable_list=area_micron, area_pixel, arena, root_area_micron, root_area_pixel, root_arena_pixel, Piezo_count, density_Piezo_um, density_Piezo_pix,pz1pz1_nnd_avg,pz1pz1_nnd_med,pz1pz1_nnd_quantile_1,pz1pz1_nnd_quantile_3,x_pixel_conversion, y_pixel_conversion, sqr_pixel_conversion #Vars list
    col_names=['Area', 'Area Mask (pix)', 'Area for Arena','square root Area', 'square root Area (nm)','square root Area (pix)','Piezo Count','Piezo Density (um)', 'Piezo Density(pix)', 'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 'x_pixel_conversion', 'y_pixel_conversion','sqr_pixel_conversion'] #Headers
    variables = pd.DataFrame(columns=col_names)
    variables.loc[0]=variable_list
    print(variables)
      
    
    
    return variables
    
  
    return variables

# def var_one(Piezo_df, roi):
#     area_micron=(roi.loc[0][1]) #get area from ROI df, 
#     sqr_pixel_conversion=roi.loc[0][15]
#     x_pixel_conversion=(roi.loc[0][15])
#     y_pixel_conversion=(roi.loc[0][16])
    
#     area_pixel=(area_micron/sqr_pixel_conversion)
#     root_area_pixel=(sqrt(area_pixel))
    
#     arena=area_pixel+(area_pixel*3)
    
#     #arena=area_pixel+(area_pixel*3) ## change to min and max of the mask population 

#     #TREK_count=TREK_df['XMnm'].count() #number of puncta TREK

#     Piezo_count=int(Piezo_df['XMnm'].count()) #number of puncta Piezo
#     print(Piezo_count)
#     print(type(Piezo_count))
#     #root_area_micron=(sqrt(area_micron))*1000 # area for RNG modeling cell as a square for the same total area,convert to nm

#     root_area_pixel=(sqrt(area_pixel))
#     root_arena_pixel=(sqrt(arena))
#     #root_arena_pixel=(sqrt(arena))

#     #density_TREK_um=TREK_count/area_micron #density TREK- not in use as of 3/10/23

#     #density_TREK_pix=TREK_count/area_pixel

#     density_Piezo_um=Piezo_count/area_micron

#     #sim_overexpressedPiezo=15*area_micron

#     #density_overexpressedPiezo_pixel=sim_overexpressedPiezo/area_pixel

#     #density_Piezo_pix=Piezo_count/area_pixel #density Piezo- not in use as of 3/10/23
    

# #     t1p1_real_nnd_avg=TREK_df['NND_other'].mean()
# #     t1p1_real_nnd_med=TREK_df['NND_other'].median()
# #     t1p1_real_nnd_quantile_1=TREK_df['NND_other'].quantile(0.25)
# #     t1p1_real_nnd_quantile_3=TREK_df['NND_other'].quantile(0.75)

# #     p1t1_real_nnd_avg=Piezo_df['NND_other'].mean()
# #     p1t1_real_nnd_med=Piezo_df['NND_other'].median()
# #     p1t1_real_nnd_quantile_1=Piezo_df['NND_other'].quantile(0.25)
# #     p1t1_real_nnd_quantile_3=Piezo_df['NND_other'].quantile(0.75)

# #     t1t1_real_nnd_avg=TREK_df['NND_self'].mean()
# #     t1t1_real_nnd_med=TREK_df['NND_self'].median()
# #     t1t1_real_nnd_quantile_1=TREK_df['NND_self'].quantile(0.25)
# #     t1t1_real_nnd_quantile_3=TREK_df['NND_self'].quantile(0.75)

#     pz1pz1_nnd_avg=Piezo_df['NND_self'].mean()
#     pz1pz1_nnd_med=Piezo_df['NND_self'].median()
#     pz1pz1_nnd_quantile_1=Piezo_df['NND_self'].quantile(0.25)
#     pz1pz1_nnd_quantile_3=Piezo_df['NND_self'].quantile(0.75)


    
    
#    #'P1T1 Mean NND (empirical)(nm)', 'P1T1 Median NND (empirical)(nm)', 'P1T1 Quartile1 NND (empirical)','P1T1 Quartile3 NND (empirical)(nm)', 'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 
#  # 'T1T1 Mean NND(nm)', 'T1T1 Median NND(nm)', 'T1T1 Quantile 1 NND(nm)', 'T1T1 Quantile 3 NND (nm)', 
#     #variable_list=area_micron, area_pixel, arena, root_area_micron, root_area_pixel, root_arena_pixel, TREK_count, density_TREK_um, density_TREK_pix, Piezo_count, density_Piezo_um, density_Piezo_pix,t1p1_real_nnd_avg, t1p1_real_nnd_med,t1p1_real_nnd_quantile_1,t1p1_real_nnd_quantile_3,p1t1_real_nnd_avg,p1t1_real_nnd_med,p1t1_real_nnd_quantile_1,p1t1_real_nnd_quantile_3,t1t1_real_nnd_avg,t1t1_real_nnd_med,t1t1_real_nnd_quantile_1,t1t1_real_nnd_quantile_3,pz1pz1_nnd_avg,pz1pz1_nnd_med,pz1pz1_nnd_quantile_1,pz1pz1_nnd_quantile_3 #Vars list
#     variable_list=area_micron, Piezo_count, density_Piezo_um, pz1pz1_nnd_avg, pz1pz1_nnd_med, pz1pz1_nnd_quantile_1, pz1pz1_nnd_quantile_3, x_pixel_conversion, y_pixel_conversion
#     #col_names=['Area', 'Area Mask (pix)', 'Area for Arena','square root Area', 'square root Area (nm)','square root Area (pix)',' Trek Count','Trek Density','Trek Density (pix)','Piezo Count','Piezo Density (um)', 'Piezo Density(pix)','T1P1 Mean NND (empirical)(nm)', 'T1P1 Median NND (empirical)(nm)', 'T1P1 Quartile1 NND (empirical)','T1P1 Quartile3 NND (empirical)(nm)', 'P1T1 Mean NND (empirical)(nm)', 'P1T1 Median NND (empirical)(nm)', 'P1T1 Quartile1 NND (empirical)','P1T1 Quartile3 NND (empirical)(nm)',  'T1T1 Mean NND(nm)', 'T1T1 Median NND(nm)', 'T1T1 Quantile 1 NND(nm)', 'T1T1 Quantile 3 NND (nm)',  'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)' ] #Headers
#     col_names=['Area','Piezo Count','Piezo Density (um)', 'P1P1 Mean NND(nm)', 'P1P1 Median NND(nm)', 'P1P1 Quantile 1 NND(nm)', 'P1P1 Quantile 3 NND (nm)', 'x_pixel_conversion', 'y_pixel_conversion']
#     variables = pd.DataFrame(columns=col_names)
#     variables.loc[0]=variable_list
#     print(variables)
    
  
    return variables

def bin_solo(DF):
    
    bin_edges_tc= np.linspace(0, 4000.0, 81) 
    
    binned_data_other= np.histogram(DF['NND_other'], bins=bin_edges_tc)
    df_bin_other=pd.DataFrame(binned_data_other).transpose()
    df_bin_other.rename({0:'count',1:'bin'}, axis=1, inplace=True)
    df_bin_other['norm']=df_bin_other['count']/(len(DF))
    

    return df_bin_other

def ThomasCluster_kd_one(Piezodensity_tc,Piezo_count,percentclustered,sigma,arenasize,mask,seed,graph=True): 
 
    ##Note, the points will all plot by default; this is turned off for the repeating case.


    # Simulation window parameters
    xMin_tc = 0;
    xMax_tc = arenasize;
    yMin_tc = 0;
    yMax_tc = arenasize;

    # Parameters for the parent and daughter point processes
    #lambdaPiezo = 1;  # density of Piezos
    #TREKdensity = 10
    #percentclustered = 90
    lambdaPz1 = (Piezodensity_tc)*(percentclustered/100);  # mean number of points in each cluster ###check
    #print(lambdaTREK, "lambdaTREK")
    #sigma = .25;  # sigma for normal variables (ie random locations) of daughters

    # Extended simulation windows parameters
    #rExt=6*sigma; # extension parameter 
# for rExt, use factor of deviation sigma eg 5 or 6
#     xMinExt = xMin - rExt;
#     xMaxExt = xMax + rExt;
#     yMinExt = yMin - rExt;
#     yMaxExt = yMax + rExt;
#     # rectangle dimensions
#     xDeltaExt = xMaxExt - xMinExt;
#     yDeltaExt = yMaxExt - yMinExt;
#     areaTotalExt = xDeltaExt * yDeltaExt;  # area of extended rectangle
    #print(areaTotalExt,"area total")
    areaTotalExt = xMax_tc * yMax_tc;
    
    # Simulate Poisson point process for the parents
    numbPointsPiezo = Piezo_count;# Poisson number of points
    numbPointsPz1rand = np.random.poisson(areaTotalExt * Piezodensity_tc* ((100-percentclustered)/100))
        #print(numbPointsPiezo,"Piezo points")
        #print(numbPointsTREKrand,"random TREK points")


        # x and y coordinates of Poisson points for the parent
    #xxPiezo=Piezo_df['xPix']
    #yyPiezo=Piezo_df['yPix']

        #x and y coordinates of random TREK points (the not clustered ones)
#     xxTREKrand = xMinExt + xDeltaExt * np.random.uniform(0, 1, numbPointsTREKrand);
#     yyTREKrand = yMinExt + yDeltaExt * np.random.uniform(0, 1, numbPointsTREKrand);
    
    xxPz1rand = xMax_tc * np.random.uniform(0, 1, numbPointsPz1rand);
    yyPz1rand = yMax_tc * np.random.uniform(0, 1, numbPointsPz1rand);
        #print(len(xxTREKrand))

        # Simulate Poisson point process for the daughters (ie final poiint process)
    numbPointsPz1 = np.random.poisson(lambdaPz1, numbPointsPz1rand); ###check
    numbPoints = sum(numbPointsPz1);  # total number of points
        #print(numbPoints, "non-random TREK points")

        # Generate the (relative) locations in Cartesian coordinates by
        # simulating independent normal variables
    xx0 = np.random.normal(0, sigma, numbPoints);  # (relative) x coordinaets
    yy0 = np.random.normal(0, sigma, numbPoints);  # (relative) y coordinates

#         # replicate parent points (ie centres of disks/clusters)
    xxPz1 = np.repeat(xxPz1rand, numbPointsPz1);
    yyPz1 = np.repeat(yyPz1rand, numbPointsPz1);

    # translate points (ie parents points are the centres of cluster disks)
    xxPz1 = xxPz1 + xx0;
    yyPz1 = yyPz1 + yy0;

    xxPz1full = np.append(xxPz1, xxPz1rand) ##rename as xxTREKclustered-- check 
    yyPz1full = np.append(yyPz1, yyPz1rand)


        # thin points if outside the simulation window
    #booleInside = ((xxTREKfull >= xMin) & (xxTREKfull <= xMax) & (yyTREKfull >= yMin) & (yyTREKfull<= yMax));
        # retain points inside simulation window
    #xxTREKfull = xxTREKfull[booleInside];  
    #yyTREKfull = yyTREKfull[booleInside]; ## take 
    
    Pz1xy=pd.DataFrame({'XMpix':xxPz1full, 'YMpix':yyPz1full}, columns=['XMpix', 'YMpix'])
    #remove points from mask arena
    
    misky=np.array(mask)
    #print(misky)
    
    #kd tree to search for nn
    twee=KDTree(misky[:,0:2])
    dit, iti=twee.query(Pz1xy[['XMpix','YMpix']].values)
    ini_outo_list=misky[iti][:,2].astype(bool)
    
    #sns.scatterplot(data=TREKxy, x='XMpix', y='YMpix')
    
    cell_points = Pz1xy[ini_outo_list]
    merged_xy=cell_points
    #print(len(merged_xy))
    #sns.scatterplot(data=cell_points, x='XMpix', y='YMpix', color='red')
#     xxTREK_clustered = merged_xy['XMpix'];  
#     yyTREK_clustered = merged_xy['YMpix']; 
    
    
#     Pz1xy_int=Pz1xy.astype(int)
#     merged_xy=pd.merge(Pz1xy_int, mask_int, on=coordinate)
    
    xxPz1_clustered = merged_xy['XMpix'];  
    yyPz1_clustered = merged_xy['YMpix'];  
    
        #turning xy coordinates into dataframes for NN calculations
    Pz1stack = np.row_stack((xxPz1_clustered,yyPz1_clustered))
    Pz1_TC_df=pd.DataFrame(Pz1stack).transpose()
    #Piezostack = np.row_stack((xxPiezo,yyPiezo))
    #Piezo_TC_df=pd.DataFrame(Piezostack).transpose()  

    #Plotting raw data if flag==true
    #if graph == True:
        #ax = Pz1_TC_df.plot.scatter(x=0,y=1,c='blue')
    #print(len(xxTREK))
    #print(len(xxTREKrand))
    #print(len(xxTREK_clustered))
    #print(percentclustered)
    return Pz1_TC_df

def NN_sim_one(Pz1_TC_df, radius, percentclustered,x_pixel_conversion): #x and y coordinates need to be in column 0 and column 1 of the dataframe

    tree_Pz1_tc1=KDTree(Pz1_TC_df[[0,1]].values)  # generate the KD-tree
    tree_Pz1_tc2=KDTree(Pz1_TC_df[[0,1]].values)
    
    #Count=Ttree.count_neighbors(tree_Pz_tc, 28000, p=2.0, weights=None, cumulative=True) #counts NN within a specified distance
    dP_tc, iP_tc = tree_Pz1_tc2.query(Pz1_TC_df[[0,1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    # d is the distance to the nearest neighbour (shortest to longest)
    # i is the index of the nearest neighbour in the original dataset
    #print(dP)
        
    Pz1_TC_df['tc_NND'] = dP_tc[:, 1]  # make new column containing the distance to the nearest neighbour
    Pz1_TC_df['tc_NNI'] = iP_tc[:, 1]  # make new column containing the index of the nearest neighbour
 
    Pz1_TC_df['tc_NNDnm']=(((((Pz1_TC_df['tc_NND']))*x_pixel_conversion)*1000)) #convert pixels back to nm
    Pz1_TC_df['%TC'] = percentclustered
    #dT_tc, iT_tc = tree_Pz1_tc2.query(Piezo_TC_df[[0, 1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    # d is the distance to the nearest neighbour (shortest to longest)
    # i is the index of the nearest neighbour in the original dataset
    #print(len(dT))
   

    #Piezo_TC_df['tc_p1NND'] = dT_tc[:, 0]  # make new column containing the distance to the nearest neighbour
    #Piezo_TC_df['tc_p1NNI'] = iT_tc[:, 0]  # make new column containing the index of the nearest neighbour
    
    Pz1_TC_df['tc_p1NNDnm'] = (((((Pz1_TC_df['tc_NND']))*x_pixel_conversion)*1000))
    #print(Piezo_TC_df)
    
        
    Ball_11=tree_Pz1_tc1.query_ball_tree(tree_Pz1_tc2, r=(radius/(x_pixel_conversion*1000)))
    c_11 = [len(b) for b in Ball_11]
    
   
    #Append to DF
    Pz1_TC_df['ball_selfcounts']=c_11
    
    return Pz1_TC_df

def RepeatThomasClusterNN_sim_one(Pz1density_tc, Piezo_count,percentclustered, sigma,arenasize,mask,radius,pixel_conversion,times=100):


    Pz1_TC_df_a= ThomasCluster_kd_one(Pz1density_tc,Piezo_count,percentclustered,sigma,arenasize,mask,seed=1)
    Pz1_TC_df_a = NN_sim_one(Pz1_TC_df_a,radius,percentclustered,pixel_conversion)
    
    for i in range(1,times):
        Piezodfnew=ThomasCluster_kd_one(Pz1density_tc,Piezo_count,percentclustered,sigma,arenasize,mask,seed=i+1)
        Piezodfnew=NN_sim_one(Piezodfnew,radius,percentclustered,pixel_conversion)
        #TREK_TC_df_a = pd.concat([TREK_TC_df_a,TREKdfnew],ignore_index=True)
        Pz1_TC_df_a = pd.concat([Pz1_TC_df_a,Piezodfnew],ignore_index=True)
    
    sns.histplot(
        data=Pz1_TC_df_a,
        x='tc_NNDnm',
        stat='proportion',
        color='green',
        binwidth=50
    )
    plt.xlabel('distance (nm)')
    
    print(Pz1_TC_df_a)
    return Pz1_TC_df_a

def plot_yoda(DF1, DF2):
    sns.set_style('darkgrid')
    #sns.set(rc={"text.color": "white"}) 
    ax1=sns.histplot(data=DF1['tc_NNDnm'],
                 binwidth=50,
                 stat='proportion',
                 color='grey'
                )
    sns.histplot(
        data=DF2['NND_self'],
        #x='Pz1-Pz1_NN',
        stat='proportion',
        color='aqua',
        binwidth=50, 
    )
    plt.legend(title='', loc='upper right')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # sns.histplot(
    #         data=TREKfinal_sim,
    #         x='tc_NNDnm',
    #         stat='proportion',
    #         color='aqua',
    #         binwidth=50)


    handles, labels = ax1.get_legend_handles_labels()

    # Create custom legend entries for color boxes
    # #custom_handles = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Simulated'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Empirical')
    # ]

    sns.set(style="ticks")
    
    plt.gca().yaxis.set_ticks([])
    # # Combine the handles and labels
    # all_handles = custom_handles + handles
    # all_labels = ['Simulated Random', 'Empircal'] + labels
    new_yticks = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4]
    plt.yticks(new_yticks)
    # Add a legend with color boxes
    #plt.legend(handles=all_handles, labels=all_labels, title="Legend")
    #plt.title('TREK1 Nearest Piezo Neighbor', fontsize=20)
    plt.xlim(right=800)
    plt.ylim(top=0.35)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=14)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)
    #plt.gca().yaxis.set_ticks([])
    plt.xlabel('Distance (nm)', fontsize=20)
    plt.ylabel('Ratio', fontsize=20)
    plt.tight_layout()
    
    return 


#TREK_NND_tc_sim.loc[TREK_NND_tc_sim['%TC'] == 0]
    
