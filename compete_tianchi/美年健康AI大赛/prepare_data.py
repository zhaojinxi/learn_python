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
gb=part1.groupby(by=['vid'])
new_part1=pandas.DataFrame(columns=['vid']+table_id1)
for x in gb:
    for k in range(x[1].shape[0]):
        print(x[1].iloc[k]['table_id'])