#!/usr/bin/env python
# coding: utf-8


import pickle
import pymysql
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os 
path_name=os.path.dirname(os.path.abspath('__file__'))
import warnings
warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
plt.rc('font',family='SimHei',size='12')# 解决中文不能正常显示的问题

# 获取前一天日期
from datetime import timedelta, datetime
def get_lastday(num):
    dt=(datetime.today() + timedelta(-num)).strftime('%Y-%m-%d')
    return dt
last0d = get_lastday(0)
last1d = get_lastday(1)
last2d = get_lastday(2)
last7d = get_lastday(7)

# last1d = (datetime.today() + timedelta(-1)).strftime('%Y-%m-%d')
# last2d = (datetime.today() + timedelta(-2)).strftime('%Y-%m-%d')
# last7d= (datetime.today() + timedelta(-7)).strftime('%Y-%m-%d')

# # V2数据库
# v2_db = pymysql.connect(host='172.28.12.187',
#                         port=6668,
#                         user='root',
#                         password='wGW2gv6k9ejq',
#                         database='station')


# V2数据库 线上
v2_db = pymysql.connect(host='127.0.0.1',
                        port=13306,
                        user='readonly-ap1',
                        password='06[p%S5MP7^OoExy1K~O',
                        database='station')
# V3数据库
v3_db = pymysql.connect(host='172.28.12.187',
                        port=6666,
                         user='admin',
                         password='oFbfwieFH1fn75QvFsHo',
#                          database='station'
                       )


# 储存数据
def pkl_dump(name,list_data):
    with open('./1-store/'+name+'.pkl', 'wb') as fo:
        pickle.dump(list_data, fo)
        
# 加载数据
def pkl_load(name):
    with open('./1-store/'+name+'.pkl', 'rb') as fo:
        list_data = pickle.load(fo, encoding='bytes')
    return list_data


# # V2

# In[32]:


# 链和币
sql_chain_token_v2='''SELECT a.chain_id,
                       token_id,
                       out_token_id
                FROM
                  (SELECT chain_id
                   FROM station.chains
                   WHERE status='enabled'
                     AND chain_id IS NOT NULL
                     AND chain_id!=''
                   GROUP BY chain_id) a
                JOIN
                  (SELECT chain_id,
                          token_id,
                          out_token_id
                   FROM station.coins
                   WHERE status='enabled')b ON a.chain_id=b.chain_id'''
data_chain_token_v2=pd.read_sql(sql_chain_token_v2,v2_db)
data_chain_token_v2['ins_table_name_v2']=[x+'_ins' for x in data_chain_token_v2['token_id']]
data_chain_token_v2['out_table_name_v2']=[x+'_outs' for x in data_chain_token_v2['token_id']]


# In[33]:


# 充值
data_v2_ins=pd.DataFrame()
# 提现
data_v2_outs=pd.DataFrame()
not_found_tb=[]

for i in range(len(data_chain_token_v2['token_id'])):
    # 表名 V2 充值
    ins_table_name_v2=data_chain_token_v2['ins_table_name_v2'][i]
#     ins_table_name_v2='eth_ins'
    try:
        # 充值
        sql_ins_v2='''SELECT token_id,
                               amount,
                               DATE_FORMAT(created_at,'%%Y-%%m-%%d') created_dt,
                               created_at
                        FROM station.%s
                        WHERE TYPE='default'
                          AND amount>0
                          AND created_at>= '%s'
                       '''%(ins_table_name_v2,last7d)
        data_ins_add_v2=pd.read_sql(sql_ins_v2,v2_db)
        data_v2_ins=pd.concat([data_ins_add_v2,data_v2_ins],axis=0)
    except:
        not_found_tb.append(ins_table_name_v2)

    # 表名 V2 提现
    out_table_name_v2=data_chain_token_v2['out_table_name_v2'][i]
    try:
        # 提现
        sql_outs_v2='''SELECT token_id,
                              amount,
                              DATE_FORMAT(created_at,'%%Y-%%m-%%d') created_dt,
                              created_at
                        FROM station.%s
                        WHERE TYPE='default'
                          and amount>0
                          AND created_at>= '%s'
                       '''%(out_table_name_v2,last7d)
        data_outs_add_v2=pd.read_sql(sql_outs_v2,v2_db)
        data_v2_outs=pd.concat([data_outs_add_v2,data_v2_outs],axis=0)
    except:
        not_found_tb.append(out_table_name_v2)

# 转换为数值
data_v2_ins['amount']=data_v2_ins['amount'].map(float)
data_v2_outs['amount']=data_v2_outs['amount'].map(float)

# 关联充值数据和提现数据
data_v2_ins_merge=pd.merge(data_chain_token_v2.loc[:,['chain_id','token_id','out_token_id']],data_v2_ins,how='inner',on='token_id')
data_v2_out_merge=pd.merge(data_chain_token_v2.loc[:,['chain_id','token_id','out_token_id']],data_v2_outs,how='inner',on='token_id')


# # V3

# In[34]:


# 链和币种
sql_chain_token_v3="""
         SELECT ch.chain_id,
                  token_id,
                  out_token_id
           FROM
             (SELECT CONVERT(id USING utf8) COLLATE utf8_general_ci AS chain_id
              FROM wallet.chain
              WHERE enabled=1)ch
           JOIN
             (SELECT chain_id,
                     CONVERT(id USING utf8) COLLATE utf8_general_ci AS token_id,
                     out_token_id
              FROM wallet.token
              WHERE enabled=1)tok ON ch.chain_id=tok.chain_id
"""
data_chain_token_v3=pd.read_sql(sql_chain_token_v3,v3_db)

sql_ins=''' SELECT 
                  token_id,
                  biz_amount amount,
                  DATE_FORMAT(created_at,'%%Y-%%m-%%d') created_dt,
                  created_at
           FROM wallet.deposit
           WHERE STATE!='CANCELED'
             AND STATE!='DISCARDED'
             and biz_amount>0
             AND created_at>='%s'
'''%last7d
data_v3_ins=pd.read_sql(sql_ins,v3_db)
data_v3_ins['amount']=data_v3_ins['amount'].map(float)

sql_out='''SELECT 
                      token_id,
                      biz_amount amount,
                      DATE_FORMAT(created_at,'%%Y-%%m-%%d') created_dt,
                      created_at
               FROM wallet.orders
               WHERE STATE='CONFIRMED'
                 and biz_amount>0
                 AND created_at>='%s'
'''%last7d
data_v3_outs=pd.read_sql(sql_out,v3_db)
data_v3_outs['amount']=data_v3_outs['amount'].map(float)

# 关联充值数据和提现数据
data_v3_ins_merge=pd.merge(data_chain_token_v3.loc[:,['chain_id','token_id','out_token_id']],data_v3_ins,how='inner',on='token_id')
data_v3_out_merge=pd.merge(data_chain_token_v3.loc[:,['chain_id','token_id','out_token_id']],data_v3_outs,how='inner',on='token_id')


# # 合并数据

# In[35]:


data_ins_concat=pd.concat([data_v2_ins_merge,data_v3_ins_merge],axis=0) 
data_out_concat=pd.concat([data_v2_out_merge,data_v3_out_merge],axis=0) 


# # Get Price

# In[36]:


import requests
import json
URL = "https://api.kucoin.com/api/v1/prices"
# fetching the json response from the URL
req = requests.get(URL)
text_data= req.json()
data = text_data['data']
price_df=pd.DataFrame(data,index=[0]).T.reset_index(drop=False)
price_df.rename(columns={'index':'out_token_id',0:'price'},inplace=True)


# In[37]:


def _kur(x):# 峰度（Kurtosis）衡量实数随机变量概率分布的峰态。峰度高就意味着方差增大是由低频度的大于或小于平均值的极端差值引起的
    return scipy.stats.kurtosis(x, axis=0, fisher=True, bias=True, nan_policy='propagate')

def vari(x):# 正确应该是变异系数越大，变异（偏离）程度越大，风险越大，就是越分散。
    mean=np.mean(x) #计算平均值
    std=np.std(x,ddof=0) #计算标准差
    cv=std/mean
    return cv

def _get_threshold(data):
    IQR = data.loc[:,'75%']-data.loc[:,'25%']  #四分位数间距
    QL = data.loc[:,'25%']  #下四分位数
    QU = data.loc[:,'75%']  #上四分位数
    data['threshold_Lower_aomunt'] = QL - 1.5 * IQR #下阈值
    data.loc[data['threshold_Lower_aomunt']<0,'threshold_Lower_aomunt']=0
    data['threshold_Upper_aomunt']  = QU + 1.5 * IQR #上阈值
    return data

def _rename(data,strs):
    data.rename(columns={k:k+strs for k in data.columns},inplace=True)
    return data

def _data_handle(data):
    # 补充价格信息
    price_df['out_token_id']=[x.lower() for x in price_df['out_token_id']]
    data['out_token_id']=[x.lower() for x in data['out_token_id']]
    data_all=pd.merge(data,price_df,how='left',on='out_token_id')
    data_all['price'].fillna(value=0,inplace=True)
    data_all['price']=data_all['price'].map(float)
#     # 充值笔数
#     data_all['amount_con']=(data_all['created_at'].isnull()==False).map(int)
    # 充值金额
    data_all['amount_u']=data_all['amount']*data_all['price']
    # 日期格式
    data_all['created_dt']=[pd.to_datetime(x)  for x in data_all['created_dt']]

    #求和 变异系数
    df1_amount=data_all.groupby(['created_dt','chain_id','token_id'])['amount'].agg([np.sum,vari])
    df1_amount=_rename(df1_amount,'_amount')
    df2_amount=data_all.groupby(['created_dt','chain_id','token_id'])['amount'].describe()
    df2_amount=_rename(df2_amount,'_amount')
    df3_amount=data_all[data_all['price']>0].groupby(['created_dt','chain_id','token_id'])['amount_u'].agg([np.sum,vari])
    df3_amount=_rename(df3_amount,'_amount_U')
    
    # 上下阈值
    df=data_all.groupby(['chain_id'])['amount'].describe()
    df_amount_threshold=_get_threshold(df).reset_index()
    
    df_amount=pd.concat([df1_amount,df2_amount,df3_amount],axis=1).reset_index()
    df_amount_all=pd.merge(df_amount,df_amount_threshold[['chain_id','threshold_Lower_aomunt','threshold_Upper_aomunt']],how='left',on='chain_id')

    return data_all,df_amount_all

data_ins_all,df_ins_amount=_data_handle(data_ins_concat)
df_ins_amount.rename(columns={k:k+"_ins" for k in df_ins_amount.columns[3:]},inplace=True)
data_out_all,df_out_amount=_data_handle(data_out_concat)
df_out_amount.rename(columns={k:k+"_outs" for k in df_out_amount.columns[3:]},inplace=True)


# In[39]:


# 所有token
chain_token_id_list=pd.concat([data_chain_token_v2[['chain_id','token_id']],data_chain_token_v3[['chain_id','token_id']]],axis=0).drop_duplicates().reset_index(drop=True)
# 生成时间列表
def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l
chain_token_id_list['created_dt']="/".join(datelist(last7d,last1d))

# 将一列炸裂成多列
cols_name=['created_dt'+str(x) for x in list(range(0,(pd.to_datetime(last1d)-pd.to_datetime(last7d)).days+1))]
chain_token_id_list[cols_name] = chain_token_id_list["created_dt"].str.split("/",expand=True)
del chain_token_id_list['created_dt']
# 将行转列
df_final_list= chain_token_id_list.melt(id_vars=["chain_id","token_id"],value_name="created_dt")
del df_final_list['variable']
df_final_list['created_dt']=[pd.to_datetime(x) for x in df_final_list['created_dt']]


# In[40]:


need_cols=['sum_amount',
           'sum_amount_U',
           'count_amount',
           'vari_amount',
           'threshold_Upper_aomunt'
           ]
on_cols=['token_id','created_dt']
df_merge1=pd.merge(df_final_list,df_ins_amount[on_cols+[x+'_ins' for x in need_cols]],how='left',on=on_cols)
df_result=pd.merge(df_merge1,df_out_amount[on_cols+[x+'_outs' for x in need_cols]],how='left',on=on_cols)
df_result.fillna(value=0,inplace=True)


# In[41]:


df_result_last1day=df_result[df_result['created_dt']==pd.to_datetime(last1d)]
del df_result_last1day['created_dt']
# 净充值
df_result_last1day['net_ins']=df_result_last1day['sum_amount_U_ins']-df_result_last1day['sum_amount_U_outs']

df_result_last2day=df_result[df_result['created_dt']==pd.to_datetime(last2d)]
del df_result_last2day['created_dt']
df_result_last1day=df_result_last1day.set_index(['chain_id','token_id'])
df_result_last2day=df_result_last2day.set_index(['chain_id','token_id'])


# In[42]:


df_result_last1day['count_amount_ins_reduce']=df_result_last1day['count_amount_ins']-df_result_last2day['count_amount_ins']
df_result_last1day['count_amount_ins_cplastday']=df_result_last1day['count_amount_ins_reduce']/df_result_last2day['count_amount_ins']
df_result_last1day['count_amount_outs_reduce']=df_result_last1day['count_amount_outs']-df_result_last2day['count_amount_outs']
df_result_last1day['count_amount_outs_cplastday']=df_result_last1day['count_amount_outs_reduce']/df_result_last2day['count_amount_outs']


# In[43]:


def get_head20(cols):
    strs='sum_amount_U_'+cols
    hotcoins_amount_U=df_result_last1day.loc[df_result_last1day[strs]>0,[strs]]
    hotcoins_amount_U_20=hotcoins_amount_U.sort_values(by=strs,ascending=False).head(20)
    hotcoins_amount_U_20['net_ins']=df_result_last1day['net_ins']
    hotcoins_amount_U_20=hotcoins_amount_U_20.reset_index()
    return hotcoins_amount_U_20
hotcoins_amount_U_ins_20=get_head20('ins')
hotcoins_amount_U_outs_20=get_head20('outs')
hotcoins_amount_U_net_ins_top20=df_result_last1day.sort_values(by='net_ins',ascending=False).head(10).reset_index()
hotcoins_amount_U_net_ins_bottom20=df_result_last1day.sort_values(by='net_ins',ascending=True).head(10).reset_index()


# pd.concat([df_result_last1day.sort_values(by='net_ins',ascending=False).head(10),df_result_last1day.sort_values(by='net_ins',ascending=True).head(10)],axis=0)
# hotcoins_amount_U_net_ins_20=hotcoins_amount_U_net_ins_20.reset_index()


# In[1095]:


# hotcoins_amount_U_ins_20[['token_id','sum_amount_U_ins']].sort_values(by='sum_amount_U_ins',ascending=True).plot(kind='barh')
# plt.yticks(range(len(hotcoins_amount_U_ins_20['token_id'])),list(hotcoins_amount_U_ins_20['token_id']))

# for x,y in zip(hotcoins_amount_U_ins_20['token_id'],hotcoins_amount_U_ins_20['sum_amount_U_ins']):
#         plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
# ax1.set_title('充值—U（TOP20）',fontsize=32)  


# In[44]:


plt.rcParams['figure.figsize']=(15,(10/16)*15)
ax1=plt.subplot(2, 1, 1)
hotcoins_amount_U_net_ins_top20=hotcoins_amount_U_net_ins_top20.sort_values(by='net_ins',ascending=True)

# plt.bar(hotcoins_amount_U_net_ins_top20.sort_values(by='net_ins',ascending=True)['token_id'],
#         hotcoins_amount_U_net_ins_top20.sort_values(by='net_ins',ascending=True)['net_ins'],color='CadetBlue')
plt.bar(hotcoins_amount_U_net_ins_top20['token_id'],
        hotcoins_amount_U_net_ins_top20['net_ins'],color='CadetBlue')

for x,y in zip(hotcoins_amount_U_net_ins_top20['token_id'],hotcoins_amount_U_net_ins_top20['net_ins']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
ax1.get_yaxis().set_visible(False)
plt.xticks(rotation=30)
ax1.set_title('净充值—U（TOP10）',fontsize=20) 

ax2=plt.subplot(2, 1, 2)
hotcoins_amount_U_net_ins_bottom20=hotcoins_amount_U_net_ins_bottom20.sort_values(by='net_ins',ascending=False)

plt.bar(hotcoins_amount_U_net_ins_bottom20['token_id'],
        hotcoins_amount_U_net_ins_bottom20['net_ins'],color='LightCoral')

# plt.bar(hotcoins_amount_U_net_ins_bottom20.sort_values(by='net_ins',ascending=True)['token_id'],
#         hotcoins_amount_U_net_ins_bottom20.sort_values(by='net_ins',ascending=True)['net_ins'],color='LightCoral')
for x,y in zip(hotcoins_amount_U_net_ins_bottom20['token_id'],hotcoins_amount_U_net_ins_bottom20['net_ins']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
plt.xticks(rotation=30)
ax2.get_yaxis().set_visible(False)
ax2.set_title('净充值—U（Bottom10）',fontsize=20) 
plt.tight_layout()
plt.savefig(path_name+'/净充值（U）.png',dpi=100,bbox_inches = 'tight')


# In[55]:


gs = gridspec.GridSpec(3,2) 

x_major_locator=MultipleLocator(1)
plt.rcParams['figure.figsize']=(30,(10/16)*30)

# ax1 = plt.subplot(gs[0,0])
ax1=plt.subplot(2, 2, 1)
plt.bar(hotcoins_amount_U_ins_20['token_id'],hotcoins_amount_U_ins_20['sum_amount_U_ins'],color='CadetBlue')
# max_value=str(hotcoins_amount_U_ins_20['sum_amount_U_ins'].max())[0]
# lengh=len(str(hotcoins_amount_U_ins_20['sum_amount_U_ins'].max()).split('.')[0])
# max_ylim=(int(first)+1)*(10**(lengh-1))
# plt.yticks(np.arange(0, max_ylim, (10**(lengh-1))))
# plt.axis('off')
plt.xticks(rotation=30)
ax1.get_yaxis().set_visible(False)
for x,y in zip(hotcoins_amount_U_ins_20['token_id'],hotcoins_amount_U_ins_20['sum_amount_U_ins']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
ax1.set_title('充值—U（TOP20）',fontsize=32)   


ax2=plt.subplot(2, 2, 2)
plt.bar(hotcoins_amount_U_ins_20['token_id'],hotcoins_amount_U_ins_20['net_ins'],color='LightCoral')
plt.xticks(rotation=30)
ax2.get_yaxis().set_visible(False)
for x,y in zip(hotcoins_amount_U_ins_20['token_id'],hotcoins_amount_U_ins_20['net_ins']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
ax2.set_title('净充值—U',fontsize=32) 


ax3=plt.subplot(2, 2, 3)
plt.bar(hotcoins_amount_U_outs_20['token_id'],hotcoins_amount_U_outs_20['sum_amount_U_outs'],color='CadetBlue')
# max_value=str(hotcoins_amount_U_outs_20['sum_amount_U_outs'].max())[0]
# lengh=len(str(hotcoins_amount_U_outs_20['sum_amount_U_outs'].max()).split('.')[0])
# max_ylim=(int(first)+1)*(10**(lengh-1))
# plt.yticks(np.arange(0, max_ylim, (10**(lengh-1))))
plt.xticks(rotation=30)
ax3.get_yaxis().set_visible(False)
for x,y in zip(hotcoins_amount_U_outs_20['token_id'],hotcoins_amount_U_outs_20['sum_amount_U_outs']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
ax3.set_title('提现—U（TOP20）',fontsize=32)   
        
ax4=plt.subplot(2, 2, 4)
plt.bar(hotcoins_amount_U_outs_20['token_id'],hotcoins_amount_U_outs_20['net_ins'],color='LightCoral')
plt.xticks(rotation=30)
ax4.get_yaxis().set_visible(False)
for x,y in zip(hotcoins_amount_U_outs_20['token_id'],hotcoins_amount_U_outs_20['net_ins']):
        plt.text(x, y, str('%s' %round(y)), ha='center', va='bottom', fontsize=10,rotation=0)
ax4.set_title('净充值—U',fontsize=32) 
plt.savefig(path_name+'/充值提现(U)_TOP20.png',dpi=100,bbox_inches = 'tight')


# In[49]:


cons1=(df_result_last1day['count_amount_ins']>100)
cons2=(df_result_last1day['count_amount_ins_cplastday']>0.5)
cons3=(df_result_last1day['count_amount_ins_cplastday']<-0.5)
indexs=df_result_last1day[cons1&(cons2|cons3)].index

df_result_cp=df_result.set_index(['chain_id','token_id']).loc[indexs].reset_index()

def get_plot(data,cols):
    plt.rc('font',family='SimHei',size='12')

    plt.rcParams['figure.figsize']=(40,10)
    fig=plt.figure()
    ax1=plt.subplot(121)
    x1=range(len(data['created_dt']))
    y1=round(data['count_amount_ins'],0)
    y2=round(data['sum_amount_U_ins'],1)

    ax1.bar(x1,y1,label='充值件数',alpha=0.8,color='CadetBlue')
    plt.xticks(x1,[datetime.strftime(x,'%Y%m%d') for x in data['created_dt']],rotation=90,fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('充值件数',fontsize=20)  
    for a,b in zip(x1,y1):   #柱子上的数字显示
        plt.text(a,b,'%s'%round(b),ha='center',va='bottom',fontsize=20)
    plt.legend(fontsize=20)#显示图例
    
    ax2=plt.subplot(122)
#     ax2=ax1.twinx()
    ax2.plot(x1,y2,color='r',label='充值金额—U')
    plt.xticks(x1,[datetime.strftime(x,'%Y%m%d') for x in data['created_dt']],rotation=90,fontsize=20)
#     plt.ylim([0,100])
    plt.yticks(fontsize=20)
    for x,y in zip(x1,y2):
        plt.text(x,y+0.3,'%s'%y,fontsize=20)
    plt.suptitle("token_id:"+str(cols),fontsize=40,weight='bold')

    plt.legend(fontsize=20)#显示图例
    plt.savefig(path_name+'/'+cols+'_充值波动较大.png',dpi=100,bbox_inches = 'tight')
    
cols_list=[]
for i in list(set(df_result_cp['token_id'])):
    df=df_result_cp.loc[df_result_cp['token_id']==i]
    get_plot(df,i)
    cols_list.append(i+"_充值波动较大")


# In[50]:


# 最近一天的数据明细
data_ins_all_last1day=data_ins_all.loc[data_ins_all['created_dt']==last1d].set_index(['chain_id','token_id'])

# 上线阈值
data_ins_all_last1day['threshold_Upper_aomunt_ins']=df_result_last1day['threshold_Upper_aomunt_ins']

# 异常数据
dffff=data_ins_all_last1day[data_ins_all_last1day['amount']>data_ins_all_last1day['threshold_Upper_aomunt_ins']].reset_index()

# 异常数据汇总
dffff_g=dffff.groupby(['chain_id','token_id']).agg(amount=('amount','sum'),
                                      amount_con=('amount','count'),
                                      amount_u=('amount_u','sum'))

# 异常币种的充值明细
dffff_g_detail=data_ins_all_last1day.loc[dffff_g[(dffff_g['amount_con']>100)|(dffff_g['amount_u']>1000)].index]


# In[51]:


# 补充数据
df_result_last1day['sum_amount_ins_Outliers']=dffff_g['amount']
df_result_last1day['count_amount_ins_Outliers']=dffff_g['amount_con']
df_result_last1day['sum_amount_U_ins_Outliers']=dffff_g['amount_u']


# In[52]:


# 附件文件
df_result_last1day_attachments=df_result_last1day.drop(['vari_amount_outs','threshold_Upper_aomunt_outs','count_amount_outs_reduce','count_amount_outs_cplastday'],axis=1).reset_index()


# In[53]:


# 重命名
df_2={'sum_amount_ins':'充值金额', 
      'sum_amount_U_ins':"充值金额(u)",
       'count_amount_ins':"充值笔数",
      'vari_amount_ins':"充值金额变异系数（越高则离散度越高）",
      'threshold_Upper_aomunt_ins':"充值金额异常阈值",
       'sum_amount_outs':"提现金额",
      'sum_amount_U_outs':"提现金额(U)",
      'count_amount_outs':"提现笔数",
      'net_ins':"净充值(U)",
       'count_amount_ins_reduce':"充值笔数较前一日波动值",
      'count_amount_ins_cplastday':"充值笔数环比",
       'sum_amount_ins_Outliers':"充值异常金额", 
      'count_amount_ins_Outliers':"充值异常笔数",
      'sum_amount_U_ins_Outliers':"充值异常金额(U)"}
df_result_last1day_attachments.rename(columns=df_2,inplace=True)

cols_rank=['chain_id', 'token_id','净充值(U)', '充值金额', '充值金额(u)', '充值笔数', '提现金额', '提现金额(U)', '提现笔数','充值金额变异系数（越高则离散度越高）',
        '充值笔数较前一日波动值',
       '充值笔数环比', '充值金额异常阈值', '充值异常金额', '充值异常笔数', '充值异常金额(U)']

df_result_last1day_attachments=df_result_last1day_attachments[cols_rank].sort_values(by='净充值(U)',ascending=False)

with pd.ExcelWriter(path_name+'/每日明细.xlsx') as writer:
    df_result_last1day_attachments.to_excel(writer,sheet_name ="数据明细", index = False)
#     df_2.to_excel(writer,sheet_name ="字段释义", index = False)


# # 自动邮件

# In[57]:


import win32com.client as win32

outlook = win32.Dispatch('outlook.application')

mail = outlook.CreateItem(0)
mail.To = 'allyn.liu@flsdex.com'  #收件人
mail.Subject = '充提日报:'+str(last1d)  #邮件主题
# mail.Body = '这是一封测试邮件'  #邮件正文
mail.Attachments.Add(path_name+r'/每日明细.xlsx') # 添加附件
mail.Attachments.Add(path_name+'/充值提现(U)_TOP20.png') # 先把要插入的图片当作一个附件添加
strs=""
for i in cols_list:
    mail.Attachments.Add(path_name+'/'+i+'.png')
    strs=strs+"<div><img src='%name.png' /></div>".replace('%name',i)

mail.Attachments.Add(path_name+'/净充值（U）.png')   

mail.BodyFormat = 2  # 2表示使用Html format，可以调整格式等
# mail.HtmlBody = "<div><img src='充值提现_TOP20.jpg' /></div>"
mail.HTMLBody  = '''<H2>以下为重点监控内容</H2><BR>
    	<FONT SIZE=4><BR>
    	1、充值/提现 TOP20
        <BR>
        <div><img src='充值提现(U)_TOP20.png' /></div>
        
        <FONT SIZE=4><BR>
        
        <FONT SIZE=4><BR>
    	2、净充值（U）
        <BR>
        <div><img src='净充值（U）.png' /></div>
        
        <FONT SIZE=4><BR>
        
    	3、典型波动异常的近一周数据
        <BR>
        <Font Face=Times Roman Size=3 Color=blue>
    	统计维度：统计tokend_id当日充值笔数>100且较前一日波动在(-inf,50%)或(50%,inf)范围的近一周充值笔数及充值金额(U)明细
            </font>
            <BR>
        <BR>
        %s
        '''.replace("%s",strs)
mail.Display()  #显示发送邮件界面
mail.Send()   #发送

