# coding: utf-8
import os

from sklearn.decomposition import PCA
from pandas.core.frame import DataFrame
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import re
import logging
import time
from gensim.models import Word2Vec


SEED = 2021
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow":10 ,
                      "favorite": 10}



PROJECT_DATAPATH = '../data/'
DATA_PATH = PROJECT_DATAPATH + 'wedata/'
TEMP_PATH = PROJECT_DATAPATH + 'temp/'
RESULT_PATH = PROJECT_DATAPATH + 'result/'
MODEL_PATH = PROJECT_DATAPATH + 'model/'

user_action = pd.read_csv(DATA_PATH + 'user_action.csv')
feed_info = pd.read_csv(DATA_PATH + 'feed_info.csv')
feed_embeddings = pd.read_csv(DATA_PATH + 'feed_embeddings.csv')
    


'''
特征工程部分:
    feed_embeddings的降维特征
    manual_tag
    manual_keyword
    machine_keyword
    userid的w2v特征
    feedid的w2v特征
    authorid的w2v特征
'''
#视频的embedding向量是512维的，以字符串形式存储，现将每个视频的embedding存储在512维的数组中
def process_embed(feed_embeddings):
    feed_embed_array = np.zeros((feed_embeddings.shape[0], 512))#生成一个（视频个数*每个视频embedding维数）的全零数组
    for i in tqdm(range(feed_embeddings.shape[0])):
        x = feed_embeddings.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    feed_embeddings = pd.concat((feed_embeddings, temp), axis=1)
    del feed_embeddings['feed_embedding']
    return feed_embeddings

def pca(n,feed_embeddings):
    '''
    n:降到n维
    '''
    feed_embed_df = process_embed(feed_embeddings)

    feed_data = feed_embed_df.copy()

    del feed_data['feedid']

    dataMat = np.array(feed_data)

    pca_sk = PCA(n_components=n)

    newMat = pca_sk.fit_transform(dataMat)

    feed_embed_pca = DataFrame(newMat)

    feed_embed_pca['feedid'] = feed_embed_df['feedid']
    
    feed_embed_pca.rename(columns={i:'feed_embedding'+str(i) for i in range(n)},inplace=True)
    
    feed_embed_pca.to_csv(TEMP_PATH + 'feed_embeddings_{}.csv'.format(n),index = False)
    
    return feed_embed_pca

#'manual_tag_list':'manual_tag'
#'manual_keyword_list':'manual_keyword'
#'machine_keyword_list':'machine_keyword'
def get_tag_keyword_feature(col,feature):
    '''
    col:feed_info中的列名，有'manual_tag_list'，'manual_keyword_list','machine_keyword_list'
    feature:'manual_tag','manual_keyword','machine_keyword',与col相对应
    下面注释以manual_tag为例
    '''
    feature_dict = {feature + '_1':[],feature + '_2':[],feature + '_3':[],feature + '_4':[]}#用4个列表分别保存feed的第1，2，3，4个manual_tag
    for i in tqdm(range(feed_info.shape[0])):
        i_feature = []#用来存储第i个视频的前4个manual_tag

        if type(feed_info[col][i])==type(0.0):#空值，则全部设置为'0'

            i_feature = ['0','0','0','0']
        else:

            i_feature = re.findall(r'\d+\.?\d*',feed_info[col][i])#利用正则表达式将字符串中的所有数字找出来

        while(len(i_feature)<4):#少于4个，则补'0'直到长度为4
            i_feature.append('0')

        #第i个视频的第一个manual_tag存储到feature_1列表中，其余类似
        feature_dict[feature + '_1'].append(i_feature[0])
        feature_dict[feature + '_2'].append(i_feature[1])
        feature_dict[feature + '_3'].append(i_feature[2])
        feature_dict[feature + '_4'].append(i_feature[3])
    
    df = pd.DataFrame(feature_dict)
    df['feedid'] = feed_info['feedid']
    df.to_csv(TEMP_PATH + f'{feature}.csv', index = False)
    return df

def get_embedding(feature_1,feature_2,feature_embed,size,user_action):
    '''
    feature_1,feature_2:根据feature_1来做feature_2的w2v特征,例如feature_1='userid',feature_2='feedid'时，那么一个用户看的n个视频就可以当作
    一个长度为n的句子，一共可以获得‘用户总数’个句子
    feature_embed:对应于feature_2的列名，例如feature_2='feedid',feature_embed就可以设为'feed_embed'
    size:词向量维数
    user_action:用户行为信息
    '''
    user_action = user_action.sort_values('date_')
    
    user_action[feature_2] = user_action[feature_2].astype(str)
    
    docs = user_action.groupby([feature_1])[feature_2].apply(lambda x: list(x)).reset_index()
    
    docs = docs[feature_2].values.tolist()#获得句子
    
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    
    w2v = Word2Vec(docs, vector_size=size, sg=1, window=10, seed=2020, workers=24, min_count=1, epochs=1)
    
    w2v_dict = {k: w2v.wv[k] for k in user_action[feature_2]}#获得每个feature_2的词向量，键为feature_2,值为对应的词向量
    
    embed_df = pd.DataFrame(w2v_dict).T#转化成DataFrame形式
    
    for i in range(size):
        embed_df.rename(columns = {i:feature_embed + str(i)},inplace=True)
        
    embed_df = embed_df.reset_index()
    
    embed_df.rename(columns = {'index':feature_2},inplace = True)
    
    embed_df.to_csv(TEMP_PATH + f'{feature_embed}_{str(size)}.csv',index = False)
    
    return embed_df

def create_dir():
    """
    创建所需要的目录
    """
    # 创建data目录
    if not os.path.exists(PROJECT_DATAPATH + 'temp'):
        os.mkdir(PROJECT_DATAPATH + 'temp')
    if not os.path.exists(PROJECT_DATAPATH + 'model'):
        os.mkdir(PROJECT_DATAPATH + 'model')
        os.mkdir(PROJECT_DATAPATH + 'model/deepfm')
        os.mkdir(PROJECT_DATAPATH + 'model/lgb')
        os.mkdir(PROJECT_DATAPATH + 'model/mmoe')
    if not os.path.exists(PROJECT_DATAPATH + 'result'):
        os.mkdir(PROJECT_DATAPATH + 'result')

create_dir()
#获得feed_embeddings的PCA降维特征
feed_embed_pca = pca(64,feed_embeddings)
#获得'manual_tag','manual_keyword','machine_keyword'特征
manual_tag = get_tag_keyword_feature('manual_tag_list','manual_tag')
manual_keyword = get_tag_keyword_feature('manual_keyword_list','manual_keyword')
machine_keyword = get_tag_keyword_feature('machine_keyword_list','machine_keyword')
#获得'userid','feedid','authorid'w2v特征
feed_embed = get_embedding('userid','feedid','feed_embed',8,user_action)
user_action = pd.merge(user_action,feed_info[['feedid','authorid']],on='feedid',how='left')
author_embed = get_embedding('userid','authorid','author_embed',8,user_action)
user_embed = get_embedding('feedid','userid','user_embed',8,user_action)

'''
deepfm模型的数据准备:
'''
train = pd.merge(user_action, feed_info[FEA_FEED_LIST], on='feedid', how='left')
# test = pd.merge(test_b, feed_info[FEA_FEED_LIST], on='feedid', how='left')
# test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
# test.to_csv(TEMP_PATH + 'test_data_b.csv', index=False)

for action in tqdm(ACTION_LIST):
    print(f"prepare data for {action}")
    tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
    df_neg = tmp[tmp[action] == 0]
    df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
    df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])

    df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)

    df_all.to_csv(TEMP_PATH + f'/train_data_for_{action}.csv', index=False)
