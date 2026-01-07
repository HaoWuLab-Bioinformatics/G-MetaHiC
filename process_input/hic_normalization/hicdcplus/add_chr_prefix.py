import pandas as pd

import os
path = '/home/user_home/zhouxiangfei/chengxianjin/ChromaFold-main/chromafold/datasets/hicdc'


chrs = [str(i) for i in range(1, 23)] + ['X']
# print(chrs)

for chr1 in chrs:
    # 1. 读取原始 txt，假设以空白字符分隔
    df = pd.read_csv(os.path.join(path,'CD4_TCells_10kb_{}_matrix.txt'.format(chr1)),
                     sep='\t',      # 或者 sep='\t' 取决于文件实际分隔符
                     engine='python')

    # 2. 给第一列和第二列（此处假设列名分别是 'binI' 和 'binJ'）加前缀
    df['binI'] = 'chr' + df['binI'].astype(str)
    df['binJ'] = 'chr' + df['binJ'].astype(str)
    df['chrI'] = 'chr' + df['chrI'].astype(str)
    df['chrJ'] = 'chr' + df['chrJ'].astype(str)



    # 3. （可选）将结果写回文件
    df.to_csv(os.path.join(path,'CD4_TCells_10kb_chr{}_matrix.txt'.format(chr1)),
              sep='\t',       # 保持制表符分隔
              index=False)
    os.remove(os.path.join(path,'CD4_TCells_10kb_{}_matrix.txt'.format(chr1)))
    print('chr',chr1, 'already processed!')




#前两列已经加上chr前缀的情况下，在第三列和第四列加前缀
# strchrs = ['chr' + item for item in chrs]
# print(strchrs)
# for chr1 in strchrs:
#     # 1. 读取原始 txt，假设以空白字符分隔
#     df = pd.read_csv(os.path.join(path, 'gm12878_2_10kb_{}_matrix.txt'.format(chr1)),
#                          sep='\t',  # 或者 sep='\t' 取决于文件实际分隔符
#                          engine='python')
#
#     # 2. 给第一列和第二列（此处假设列名分别是 'binI' 和 'binJ'）加前缀
#     df['chrI'] = 'chr' + df['chrI'].astype(str)
#     df['chrJ'] = 'chr' + df['chrJ'].astype(str)
#
#     # 3. （可选）将结果写回文件
#     df.to_csv(os.path.join(path, 'gm12878_2_10kb_{}_matrix.txt'.format(chr1)),
#                   sep='\t',  # 保持制表符分隔
#                   index=False)
#
#     print(chr1, 'already processed!')

