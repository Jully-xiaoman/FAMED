import pandas as pd
import dill
import numpy as np
from collections import defaultdict
import ast
from rdkit import Chem
from rdkit.Chem import BRICS

########################## 紫薇添加：子结构相关 ################################
def list_functions(py_file):
    with open(py_file, "r", encoding="utf-8") as f:
        node = ast.parse(f.read())
    funcs = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    return funcs


def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    # for atc3, drugname in med_pd[["ATC3", "DRUG"]].values:
    for atc3, drugname in med_pd[["ndc", "drug"]].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)
    return atc3toDrugDict

def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    # 把drug_info表给读进来
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"): # 过滤非字符串的情况
            drug2smiles[drugname] = smiles
    # 根据ATC3toDrugDict{ATC3:drugname} 和 drug2smiles{drugname:smiles}两个字典来获取ATC3:smiles的映射，1个ATC3只保留3个smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3] # 选取一个ATC3对应的3个SMILES
    return atc3tosmiles

# 将每个药物(ATC3/ATC4)分解成它的分子子结构集合，然后构造一个”药物 × 子结构“的矩阵，矩阵中1表示该药物包含该子结构；
# 先为每个药物生成它对应的 BRICS 子结构集合；
# 再收集所有药物的子结构并去重，得到一个 全局子结构列表；
# 最后把每个药物是否包含某个子结构编码成一个 0/1 矩阵。
def get_ddi_mask(atc42SMLES, med_voc):
    # ATC3_List[22] = {0}
    # ATC3_List[25] = {0}
    # ATC3_List[27] = {0}
    fraction = [] # 初始化一个列表 fraction，用来存放每个药物对应的 BRICS 片段集合（set）
    for k, v in med_voc.idx2word.items(): # k是idx v是ATC3
        tempF = set() # 临时集合
        for SMILES in atc42SMLES[v]: # atc42SMLES[v] 就是用键 v 取对应的值。ATC3:smiles是1 ：3
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES)) # 用 RDKit 将 SMILES 解析成分子，再做 BRICS 分解：
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF) # 把当前药物的"子结构集合tempF"作为一个整体，追加到列表fraction的末尾。
    fracSet = []
    for i in fraction:
        fracSet += i      # fracSet += i 等价于 fracSet.extend(i)：把集合 i 里的元素逐个追加到列表里。
    # 构建全局子结构表
    fracSet = list(set(fracSet))  # set of all segments set(fracSet)将列表转为set去重
    dill.dump(list(set(fracSet)), open('substructure_smiles.pkl', 'wb'))

    # 新建一个二维矩阵 ddi_matrix，形状为 (药物数 M, 片段数 F)，初始全 0。
    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    # 将全局的”子结构片段列表“映射到我当前的药物数据中，标记出这个药物具体包含哪些片段。
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1 # 通过 fracSet.index(frac) 找到该片段在全局片段表中的列号；
    return ddi_matrix

##### process medications #####
# load med data
def med_process(med_file):
    """读取MIMIC原数据文件，保留pid、adm_id、data以及NDC，以DF类型返回"""
    # 读取药物文件，NDC（National Drug Code）以类别类型存储
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'}) # 它让 NDC 字段在读取时直接存储为 category 类型。

    # 删除不相关列，只留下核心的ID,时间,NDC;
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    # 去除无效记录
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)

    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    # 按 病人ID → 住院ID → ICU住院ID → 开始时间 排序，保证药物事件的时间顺序。
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)  # 重置索引，同时drop原索引

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# medication mapping
def ndc2atc4(med_pd):
    """将NDC映射到ATC4"""
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())

    # NDC -> RXCUI，ndc2rxnorm_mapping.txt
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    # RXCUI -> ATC4，ndc2atc_level4.csv
    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)  # 删除特定的RXCUI
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)

    # NDC -> RXCUI -> ATC4
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])  # 合并两个表

    # 保留ATC4并重命名为NDC
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})

    # 保留ATC前4位，此时为ATC3级
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    """筛除admission次数小于两次的患者数据"""
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a


# most common medications
def filter_300_most_med(med_pd):
    # 按照NDC出现的次数降序排列，取前300
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    # 定义了一个内嵌函数：筛选出频次最高的2000个ICD9诊断码记录
    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    total_diag_count = pro_pd['ICD9_CODE'].nunique()
    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    """药物、症状、proc的数据结合"""

    # 从每个数据表中提取患者ID(SUBJECT_ID)和住院ID(HADM_ID)的唯一组合，并去除重复记录，确保每个(患者，住院)组合只出现一次
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    # 时间
    adm = pd.read_csv(admission_file)
    adm = adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    adm.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True) # 这里结束后有ADMITIME列，用于后续排序。
    adm_key = adm[['SUBJECT_ID', 'HADM_ID']].drop_duplicates() # adm_key这里只有 SUBJECT_ID 和 HADM_ID

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(adm_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 三个集合的交集
    # 内连接就是"只保留大家都有的记录"
    # 外连接保留所有记录，并在缺失的位置填充空值
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    adm = adm.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    ##### 以上我认为都是在去重和清洗数据 #####

    # flatten and merge
    # 将一个患者一次住院的多个记录聚合成单个记录，把多个代码值合并为数组
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  

    # 将numpy数组转换为Python列表
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    # 将诊断、药物、手术按照(SUBJECT_ID,HADM_ID)合成一个数据表;
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    # data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    data = adm.merge(data, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    a = data[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1] # 选择访问次数大于1的
    data = data.merge(a[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)
    
    data = data.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    # 输出唯一患者数量
    print('#clinical events ', len(data))
    # 输出总的临床事件记录数

    diag = data['ICD9_CODE'].values # 获取所有诊断代码（ICD9标准）
    med = data['NDC'].values # 获取所有药品代码（国家药品编码）
    pro = data['PRO_CODE'].values # 获取所有手术代码
    
    unique_diag = set([j for i in diag for j in list(i)]) # 提取所有唯一的诊断代码
    unique_med = set([j for i in med for j in list(i)]) # 提取所有唯一的药物代码
    unique_pro = set([j for i in pro for j in list(i)]) # 提取所有的手术代码
    
    print('#diagnosis ', len(unique_diag)) # 输出诊断数量
    print('#med ', len(unique_med)) # 输出药物数量
    print('#procedure', len(unique_pro)) # 输出手术数量
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0 # 就诊次数计数器
        for index, row in item_data.iterrows(): # 遍历该患者的每次就诊
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        # 转换为集合去重
        x, y, z = set(x), set(y), set(z)
        # 累加统计值
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of visits ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    # 分别创建三个词表（vocabulary）对象，用来维护字符串<->整数token的双向映射
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    # 逐行遍历DataFrame
    for index, row in df.iterrows():
        # 把该行的三个字符串分别“加入”到对应词表
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])

    # 把三个词表对象序列化到磁盘，文件名voc_final.pkl
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open('voc_final.pkl','wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def get_season(month):
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:
        return 4


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """
    保存list类型的记录
    每一项代表一个患者，患者中有多个visit，每个visit包含三者数组，按顺序分别表示诊断、proc与药物
    存储的均为编号，可以通过voc_final.pkl来查看对应的具体word
    """
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    # 遍历所有患者ID.unique()给出出现过的去重顺序列表。但是unique()不保证患者内部行是按时间排序的。
    for subject_id in df['SUBJECT_ID'].unique():
        # 取出该患者的所有行，得到该患者的"纵向病程表"
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        begin = 0
        for index, row in item_df.iterrows():
            # 第一次满足条件时，begin会被设置为pd.Timestamp,其值等于当前这行row['ADMITIME']解析后的时间
            if begin == 0:
                begin = pd.to_datetime(row['ADMITTIME'])
            timestamp = (pd.to_datetime(row['ADMITTIME']) - begin).days # 更改时间戳类型
            # 初始化当前就诊的容器
            admission = []
            # 把该次就诊的诊断/手术/药物代码映射为整数ID后，作为三个列表依次加入admission
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            admission.append(timestamp)
            #admission.append(get_season(pd.to_datetime(row['ADMITTIME']).month))
            # 把该次就诊加入该患者的序列中
            patient.append(admission)
        # 一个患者循环结束后，把该患者的完整就诊记录加入总记录records.
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records

# get ddi matrix
# ddi_file = './input/drug-DDI.csv'
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set) # 建立CID-> ATC3集合的映射（值用set防重复）
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]    # 所有的药物的ATC3
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)   # 其实这个地方能很好的控制ATC4和ATC3的转换

    # 把cid_atc的外部映射表(每行：CID,ATC1,ATC2,...)读进来，并建立cid2atc_dic:CID -> 一组ATC3(用前4位)
    # 这里也需要控制ATC4和ATC3的转换
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    #print(cid2atc_dic)

    # 读取DDI CSV并按照副作用频次做Top-K过滤
    # 将最严重的40种DDI相互作用给筛选出来
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect，也是采取topK的形式
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)


    # weighted ehr adj
    # 构建EHR共现邻接矩阵(对称、加权)
    # 遍历每一位病人的每次就诊
    # 双重循环枚举此就诊内的药物无序对（通过 if j <=i :continue 仅保留上三角），把对应共现计数+1，并对称加到[med_j,med_i];
    # 注释掉的两行 = 1是"二值化贡献"，现在采用"计数加权的共现"
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] += 1
                    ehr_adj[med_j, med_i] += 1
    dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))

    # ddi adj，DDI表是CID编码的，因此需要将CID映射到ATC编码，才能记录数据集中药物之间的冲突信息
    # 构建DDI邻接矩阵
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        # 注意这个cid2atc_dic是上文构建好的字典
        # 这段代码不是“从我的数据里找出哪些药算冲突”，而是“从外部DDI知识库里查出哪些药本来就有冲突”，然后把它们映射到我数据集的药物索引空间上，做出一个药物冲突图。
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                # 相等就说明是同一个ATC药物了，不相等就直接产生冲突，因为cid本就是筛选过的TOP40的冲突药物
                if med_voc.word2idx[atc_i] != med_voc.word2idx[atc_j]:
                        # 对称添加；
                        ddi_adj[med_voc.word2idx[atc_i], med_voc.word2idx[atc_j]] = 1
                        ddi_adj[med_voc.word2idx[atc_j], med_voc.word2idx[atc_i]] = 1
                # atc_level3 -> atc_level4
                # for i in atc3_atc4_dic[atc_i]:
                #     for j in atc3_atc4_dic[atc_j]:
                #         if med_voc.word2idx[i] != med_voc.word2idx[j]:
                #             ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                #             ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb')) 

    return ddi_adj

def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm[2]
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

if __name__ == '__main__':
    # 使用示例

    # MIMIC数据文件，分别包括药物、诊断和proc
    med_file = './input/PRESCRIPTIONS.csv'
    diag_file = './input/DIAGNOSES_ICD.csv'
    procedure_file = './input/PROCEDURES_ICD.csv'
    admission_file = './input/ADMISSIONS.csv'

    drugbankinfo = "./input/drugbank_drugs_info.csv"

    # 药物信息
    med_structure_file = './input/atc3toSMILES.pkl'   # 药物到分子式的映射

    # drug code mapping files
    ndc2atc_file = './input/ndc2atc_level4.csv'   # NDC code to ATC-4 code mapping file，用于读取xnorm到ATC
    cid_atc = './input/drug-atc.csv'              # drug（CID） to ATC code mapping file，用于处理DDI表
    ndc_rxnorm_file = './input/ndc2rxnorm_mapping.txt'    # NDC to xnorm mapping file

    # ddi information
    ddi_file = './input/drug-DDI.csv'

    # 处理MIMIC中的药物数据
    med_pd = med_process(med_file) # 处方

    # visit_counts = med_pd.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    # print('less than two visit: ', len(visit_counts[visit_counts<2]), len(med_pd))

    med_pd = ndc2atc4(med_pd) # 只有atc4

    # med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)   # 注意这里仅仅是针对med表中出现了两次以上admission的patient
    # med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True) # 只要出现了两次以上admission的patient

    # 紫薇添加 :顺溜获取atc->smiles的文档
    # atc3toDrug = ATC3toDrug(med_pd)  # 将 ATC3码和药物名称映射 ATC3:drugname 1对多
    # druginfo = pd.read_csv(drugbankinfo)
    # med_structure_file = atc3toSMILES(atc3toDrug, druginfo)  # 将 ATC3码和药物SMILES映射 ATC3:SMILES 1对3
    # dill.dump(atc3toSMILES, open(med_structure_file, "wb"))  # atc3toSMILES是字典


    # 为了公平比较
    # 筛选出来有SMILES的ATC
    NDCList = dill.load(open(med_structure_file, 'rb'))
    med_pd = med_pd[med_pd.NDC.isin(list(NDCList.keys()))]

    med_pd = filter_300_most_med(med_pd) # 出现次数排名前300的药物
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file) # 出现次数排名前2000的诊断集合
    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)

    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)

    data.to_pickle('data_final.pkl')

    print ('complete combining')
    print(len(data))

    # ddi_matrix
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)   # diag,proc,medication按顺序存储
    print(len(records))

    ## check
    visit_counts = data.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    print('less than two visit: ', len(visit_counts[visit_counts<2]), len(data))
    # for user in records:
    #     if len(user) <=1:
    #         print('yes!!!',user)

    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    ddi_rate = ddi_rate_score(records, "ddi_A_final.pkl")
    print("ddi_rate", ddi_rate)

