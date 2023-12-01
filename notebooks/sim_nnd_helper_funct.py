
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

def kd_ThomasCluster(Piezodensity_tc,TREKdensity_tc, Piezo_count, Piezo_df, percentclustered,sigma,arenasize,mask,seed): 
  
    np.random.seed(seed)

    
    # Simulation window parameters
    xMin_tc = 0;
    xMax_tc = arenasize;
    yMin_tc = 0;
    yMax_tc = arenasize;

  
    lambdaTREK = (TREKdensity_tc/Piezodensity_tc)*(percentclustered/100);  # mean number of points in each cluster

    areaTotalExt = xMax_tc * yMax_tc;
    
    # Simulate Poisson point process for the parents
    numbPointsPiezo = Piezo_count;# Poisson number of points
    
    #random.seed(10)
    
    numbPointsTREKrand = np.random.poisson(areaTotalExt * TREKdensity_tc* ((100-percentclustered)/100))
        #print(numbPointsPiezo,"Piezo points")
        #print(numbPointsTREKrand,"random TREK points")


        # x and y coordinates of Poisson points for the parent
    xxPiezo=Piezo_df['xPix']
    yyPiezo=Piezo_df['yPix']
    sns.scatterplot(data=Piezo_df, x='xPix', y='yPix')
  
    xxTREKrand = xMax_tc * np.random.uniform(0, 1, numbPointsTREKrand);
    yyTREKrand = yMax_tc * np.random.uniform(0, 1, numbPointsTREKrand);
        #print(len(xxTREKrand))

        # Simulate Poisson point process for the daughters (ie final poiint process)
    #random.seed(10)
    numbPointsTREK = np.random.poisson(lambdaTREK, numbPointsPiezo);
    numbPoints = sum(numbPointsTREK);  # total number of points
        #print(numbPoints, "non-random TREK points")

        # Generate the (relative) locations in Cartesian coordinates by
        # simulating independent normal variables
    #random.seed(10)
    xx0 = np.random.normal(0, sigma, numbPoints);  # (relative) x coordinaets
    yy0 = np.random.normal(0, sigma, numbPoints);  # (relative) y coordinates
    
    
    
#         # replicate parent points (ie centres of disks/clusters)
    xxTREK = np.repeat(xxPiezo, numbPointsTREK);
    yyTREK = np.repeat(yyPiezo, numbPointsTREK);
 
    
    # translate points (ie parents points are the centres of cluster disks)
    xxTREK = xxTREK + xx0;
    yyTREK = yyTREK + yy0;
    
  
          
    xxTREKfull = np.append(xxTREK, xxTREKrand) ##rename as xxTREKclustered-- check 
    yyTREKfull = np.append(yyTREK, yyTREKrand)
    
    
    
    TREKxy=pd.DataFrame({'XMpix':xxTREKfull, 'YMpix':yyTREKfull}, columns=['XMpix', 'YMpix'])
    #remove points from mask arena

    misky=np.array(mask)
    #print(misky)
    
    #kd tree to search for nn
    twee=KDTree(misky[:,0:2])
    dit, iti=twee.query(TREKxy[['XMpix','YMpix']].values)
    ini_outo_list=misky[iti][:,2].astype(bool)
    
    #sns.scatterplot(data=TREKxy, x='XMpix', y='YMpix')
    
    cell_points = TREKxy[ini_outo_list]
    merged_xy=cell_points
    #print(len(merged_xy))
    sns.scatterplot(data=cell_points, x='XMpix', y='YMpix', color='red')
    xxTREK_clustered = merged_xy['XMpix'];  
    yyTREK_clustered = merged_xy['YMpix'];  
    
        #turning xy coordinates into dataframes for NN calculations
    TREKstack = np.row_stack((xxTREK_clustered,yyTREK_clustered))
    TREK_TC_df=pd.DataFrame(TREKstack).transpose()
    Piezostack = np.row_stack((xxPiezo,yyPiezo))
    Piezo_TC_df=pd.DataFrame(Piezostack).transpose()
    TREK_TC_df['seed']=seed
    
    return Piezo_TC_df, TREK_TC_df


def NearestNeighbors(Piezo_TC_df,TREK_TC_df,radius,percentclustered,pixel_conversion): #x and y coordinates need to be in column 0 and column 1 of the dataframe
    DF1=Piezo_TC_df
    DF2=TREK_TC_df
    
    #Generate KDtree For Each Population 
    tree1=KDTree(DF1[[0,1]].values) # generate the KD-tree
    tree2=KDTree(DF2[[0,1]].values)
    
    # Nearest Neighbors Between Population 1, 2
    
    ### Pop 2 to 1
    d1_2, i1_2 = tree1.query(DF2[[0,1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF2['NND_other'] = d1_2[:, 0]  # make new column containing the distance to the nearest neighbour
    DF2['NND_other'] = (((((DF2['NND_other']))*pixel_conversion)*1000))
    DF2['NNI_other'] = i1_2[:, 0]  # make new column containing the index of the nearest neighbour
     
    ### Pop 1 to 2
    d2_1, i2_1 = tree2.query(DF1[[0,1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_other'] = d2_1[:, 0]  # make new column containing the distance to the nearest neighbour
    DF1['NND_other'] = (((((DF1['NND_other']))*pixel_conversion)*1000))
    DF1['NNI_other'] = i2_1[:, 0]  # make new column containing the index of the nearest neighbour
   
     
    
    #Nearest Self Neighbor in Population 1 and Population 2
    d1_1, i1_1 = tree1.query(DF1[[0,1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF1['NND_self'] = d1_1[:, 1]  # make new column containing the distance to the nearest neighbour
    DF1['NND_self'] = (((((DF1['NND_self']))*pixel_conversion)*1000))
    DF1['NNI_self'] = i1_1[:, 1]  # make new column containing the index of the nearest neighbour
  
    
    d2_2, i2_2 = tree2.query(DF2[[0,1]].values, k=2)  # query the tree for nearest neighbours, k is the number of nearest neighbours to find
    DF2['NND_self'] = d2_2[:, 1]  # make new column containing the distance to the nearest neighbour
    DF2['NND_self']= (((((DF2['NND_self']))*pixel_conversion)*1000))
    DF2['NNI_self'] = i2_2[:, 1]  # make new column containing the index of the nearest neighbour

    
    #Ball Queries
    #Ball-Other
    Ball_12=tree2.query_ball_tree(tree1, r=(radius/(pixel_conversion*1000)))
    c_12 = [len(b) for b in Ball_12]
    
    Ball_21=tree1.query_ball_tree(tree2, r=(radius/(pixel_conversion*1000)))
    c_21 = [len(b) for b in Ball_21]
    
    #Ball-Self
    Ball_11=tree1.query_ball_tree(tree1, r=(radius/(pixel_conversion*1000)))
    c_11 = [len(b) for b in Ball_11]
    
    Ball_22=tree2.query_ball_tree(tree2, r=(radius/(pixel_conversion*1000)))
    c_22 = [len(b) for b in Ball_22]
   
    #Append to DF
    DF1['ball_counts']=c_21
    DF1['ball_selfcounts']=c_11
    
    DF2['ball_counts']=c_12
    DF2['ball_selfcounts']=c_22

 
    return DF1, DF2


### Repeat TC Functions 
def RepeatThomasClusterNN(Piezodensity_tc,TREKdensity_tc, Piezo_count, Piezo_df, percentclustered, pixel_conversion,sigma,arenasize,mask,radius,times=100):

    
    Piezo_TC_df_a, TREK_TC_df_a= kd_ThomasCluster(Piezodensity_tc,TREKdensity_tc, Piezo_count, Piezo_df, percentclustered,sigma,arenasize,mask,seed=1)
    Piezo_TC_df_a, TREK_TC_df_a = NearestNeighbors(Piezo_TC_df_a,TREK_TC_df_a,radius,percentclustered,pixel_conversion)
    
    for i in range(1,times):
        Piezodfnew,TREKdfnew=kd_ThomasCluster(Piezodensity_tc,TREKdensity_tc, Piezo_count, Piezo_df, percentclustered,sigma,arenasize,mask,seed=i+1)
        Piezodfnew,TREKdfnew=NearestNeighbors(Piezodfnew,TREKdfnew, radius, percentclustered,pixel_conversion)
        TREK_TC_df_a = pd.concat([TREK_TC_df_a,TREKdfnew],ignore_index=True)
        Piezo_TC_df_a = pd.concat([Piezo_TC_df_a,Piezodfnew],ignore_index=True)
    
    #print(Piezo_TC_df_a)
    return TREK_TC_df_a, Piezo_TC_df_a

def var_sim(DF):
    t1p1_sim0_nnd_avg=DF['NND_other'].mean()
    t1p1_sim0_nnd_med=DF['NND_other'].median()
    t1p1_sim0_nnd_quantile_1=DF['NND_other'].quantile(0.25)
    t1p1_sim_nnd_quantile_3=DF['NND_other'].quantile(0.75)
    
    t1t1_sim_nnd_avg=DF['NND_self'].mean()
    t1t1_sim_nnd_med=DF['NND_self'].median()
    t1t1_sim_nnd_quantile_1=DF['NND_self'].quantile(0.25)
    t1t1_sim_nnd_quantile_3=DF['NND_self'].quantile(0.75)
    
    
    v_list= t1p1_sim0_nnd_avg, t1p1_sim0_nnd_med, t1p1_sim0_nnd_quantile_1, t1p1_sim_nnd_quantile_3, t1t1_sim_nnd_avg, t1t1_sim_nnd_med, t1t1_sim_nnd_quantile_1, t1t1_sim_nnd_quantile_3
    col_names= 'T1P1 Mean NND (sim0)(nm)', 'T1P1 Median NND (sim0)(nm)', 'T1P1 Quartile1 NND (sim0)(nm)','T1P1 Quartile3 NND (sim0)(nm)', 'simT1T1 Mean NND(nm)', 'simT1T1 Median NND(nm)', 'simT1T1 Quartile 1 NND(nm)', 'simT1T1 Quartile NND(nm)'
    sum_values_df = pd.DataFrame(columns=col_names)
    
    sum_values_df.loc[0]=v_list   
    
    return sum_values_df 

def bin_one(DF):
    
    bin_edges_tc= np.linspace(0, 4000.0, 81) 
    

    binned_data_other= np.histogram(DF['NND_other'], bins=bin_edges_tc)
    df_bin_other=pd.DataFrame(binned_data_other).transpose()
    df_bin_other.rename({0:'count',1:'bin'}, axis=1, inplace=True)
    df_bin_other['norm']=df_bin_other['count']/(len(DF))
    
    binned_data_self= np.histogram(DF['NND_self'], bins=bin_edges_tc)
    df_bin_self=pd.DataFrame(binned_data_self).transpose()
    df_bin_self.rename({0:'count',1:'bin'}, axis=1, inplace=True)
    df_bin_self['norm']=df_bin_self['count']/(len(DF))
    
    return df_bin_other, df_bin_self
