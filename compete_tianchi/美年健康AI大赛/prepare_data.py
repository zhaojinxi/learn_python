import pandas
import numpy

# part1=open('data/meinian_round1_data_part1_20180408.txt',encoding='utf-8')
# k=[]
# for i in part1.readlines():
#     ii=i.split('$')
#     k.append(ii)
# part1.close()
# part1=pandas.DataFrame(k,columns=['vid','table_id','field_result'])
# part1.to_pickle('data/part1')
# part2=open('data/meinian_round1_data_part2_20180408.txt',encoding='utf-8')
# k=[]
# for i in part2.readlines():
#     ii=i.split('$')
#     k.append(ii)
# part2.close()
# part2=pandas.DataFrame(k,columns=['vid','table_id','field_result'])
# part2.to_pickle('data/part2')

# part1=pandas.read_pickle('data/part1')
# part2=pandas.read_pickle('data/part2')
# part1=part1.drop(0)
# part2=part2.drop(0)
# part1['field_result']=part1['field_result'].str.replace('\n','')
# part2['field_result']=part2['field_result'].str.replace('\n','')
# part1.to_pickle('data/part1p')
# part2.to_pickle('data/part2p')

part1=pandas.read_pickle('data/part1p')
part2=pandas.read_pickle('data/part2p')
table_id1=part1['table_id'].drop_duplicates().sort_values().as_matrix().tolist()
table_id2=part2['table_id'].drop_duplicates().sort_values().as_matrix().tolist()
all=numpy.zeros([part1['vid'].drop_duplicates().shape[0], len(table_id1)])
new_part1=pandas.DataFrame(all, index=part1['vid'].drop_duplicates().sort_values().as_matrix(), columns=table_id1)
gb=part1.groupby(by=['vid'])
t=1
for x in gb:
    for k in range(x[1].shape[0]):
        new_part1.loc[x[0], x[1].iloc[k]['table_id']]=x[1].iloc[k]['field_result']
    print(t)
    t=t+1
new_part1.to_pickle('data/new_part1')
all=numpy.zeros([part2['vid'].drop_duplicates().shape[0], len(table_id2)])
new_part2=pandas.DataFrame(all, index=part2['vid'].drop_duplicates().sort_values().as_matrix(), columns=table_id2)
gb=part2.groupby(by=['vid'])
t=1
for x in gb:
    for k in range(x[1].shape[0]):
        new_part2.loc[x[0], x[1].iloc[k]['table_id']]=x[1].iloc[k]['field_result']
    print(t)
    t=t+1
new_part2.to_pickle('data/new_part2')