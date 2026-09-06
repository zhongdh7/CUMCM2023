import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr,spearmanr

#设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def data_preprocessing():
    
    df1=pd.read_excel('./data/附件1.xlsx')
    df2=pd.read_csv('./data/附件2.csv')

    
    # print(df1.head())
    # print(df2.head())
    
    merge_table=pd.merge(df1,df2,on='单品编码',how='left')
    # print(merge_table)
    
    
    #按照每一天的销量划分，研究每一天和每一天的销量之间的关系
    data=merge_table.pivot_table(index='销售日期',columns='分类名称',values='销量(千克)',aggfunc=np.sum)
    # print(data)
    data.to_excel('./data/problem1_每个种类每日销量表.xlsx')
    
    return data

def relation_analyze(data: pd.DataFrame):
    # plt.figure(figsize=(12,6))
    fig,ax=plt.subplots(1,2,figsize=(12,6))
    
    sns.heatmap(ax=ax[0],data=data.corr(),cmap='coolwarm',annot=True)
    ax[0].set_title('皮尔逊相关系数')
    
    sns.heatmap(ax=ax[1],data=data.corr(method='spearman'),cmap='coolwarm',annot=True)
    ax[1].set_title('斯皮尔逊相关系数')
    plt.tight_layout()
    fig.suptitle('两种相关系数热力图')
    plt.savefig('./figure/相关系数热力图.jpg',dpi=400)
    plt.savefig('./figure/相关系数热力图.pdf',dpi=400)
    plt.show()
    
    to_analyze_more_specific=input('请输入需要进行相关性假设检验的两个种类(用空格分隔):').split(' ')
    
    i,j=to_analyze_more_specific
    
    spearmanr_corr,spearmanr_p=spearmanr(data[i],data[j])
    pearsonr_corr,pearsonr_p=pearsonr(data[i],data[j])
    
    print(f'皮尔逊相关系数:{pearsonr_corr},p值:{pearsonr_p}')
    print(f'斯皮尔逊相关系数:{spearmanr_corr},p值:{spearmanr_p}')
    
    sns.pairplot(data)
    # plt.title('散点图矩阵')
    plt.savefig('./figure/散点图矩阵.jpg',dpi=400)
    plt.savefig('./figure/散点图矩阵.pdf',dpi=400)
    plt.show()    
    


if __name__=="__main__":
    #获得可以进行相关性分析的表格
    data=data_preprocessing()
    print(data.describe())
    relation_analyze(data)
    pass