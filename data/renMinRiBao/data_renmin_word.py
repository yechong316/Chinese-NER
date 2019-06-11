# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
import pandas as pd
import numpy as np
import collections
def originHandle():
    with open('./renmin.txt','r', encoding='utf-8') as inp,open('./renmin2.txt','w', encoding='utf-8') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i<len(line)-1:
                if line[i][0]=='[':
                    outp.write(line[i].split('/')[0][1:])
                    i+=1
                    while i<len(line)-1 and line[i].find(']')==-1:
                        if line[i]!='':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1]=='nr':
                    word = line[i].split('/')[0] 
                    i+=1
                    if i<len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')           
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1
            outp.write('\n')
def originHandle2():
    with codecs.open('./renmin2.txt','r','utf-8') as inp,codecs.open('./renmin3-0606.txt','w','utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag=='nr' or tag=='ns' or tag=='nt':
                    outp.write(word[0]+"/B_"+tag+" ")
                    for j in word[1:len(word)-1]:
                        if j!=' ':
                            outp.write(j+"/M_"+tag+" ")
                    outp.write(word[-1]+"/E_"+tag+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/O ')
                i+=1
            outp.write('\n')
def sentence2split():
    with open('./renmin3.txt','r',encoding='utf-8') as inp,codecs.open('./renmin4.txt','w','utf-8') as outp:
        texts = inp.read()
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip()+'\n')

def data2pkl():
    datas = list()
    labels = list()
    linedata=list()
    linelabel=list()
    tags = set()
    tags.add('')
    input_data = codecs.open('renmin4.txt','r','utf-8')
    for line in input_data.readlines():
        line = line.split()
        linedata=[]
        linelabel=[]
        numNotO=0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1]!='O':
                numNotO+=1
        if numNotO!=0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    print(len(datas))
    print(len(labels))
    import collections
    def flatten(x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
        return result

    print(flatten(["junk", ["nested stuff"], [], [[]]]))
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = list(range(1, len(set_words)+1))

    
    tags = [i for i in tags] # <class 'list'>: ['', 'E_ns', 'B_nt', 'B_ns', 'B_nr', 'M_nr', 'E_nr', 'E_nt', 'O', 'M_ns', 'M_nt']
    tag_ids = list(range(len(tags)))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"] = len(word2id) + 1 # 给未知符号添加标记
    id2word[len(word2id)] = "unknow"
    print(tag2id)
    max_len = 60
    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:  # 超出最大程度的截断
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len: 
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=list(range(len(datas))))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))


    # def train_test_split(x, y, test_size=0.2, random_state=40):
    #
    #     length = len(x)
    #     test_data_size = length / test_size
    #     train_data_size = 1 - length / test_size
    #
    #
    #     return x_train,x_test, y_train, y_test
    #     pass
    # from sklearn.model_selection import train_test_split


    split_1 = int(x.shape[0] * 0.8)
    split_2 = int(split_1 * 0.8)
    
    x_train, x_test = x[:split_1], x[split_1:]
    y_train, y_test = y[:split_1], y[split_1:]
    x_train, x_valid = x_train[:split_2], x_train[split_2:split_1]
    y_train, y_valid = y_train[:split_2], y_train[split_2:split_1]
    


    import pickle
    import os


    pkl_path = '../renmindata-full.pkl'
    with open(pkl_path, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)
    print('** Finished saving the data.')
    

if __name__ == '__main__':

    # originHandle()
    # originHandle2()
    # sentence2split()
    data2pkl()
