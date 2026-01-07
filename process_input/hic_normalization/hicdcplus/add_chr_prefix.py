import pandas as pd

import os
path = './gmetahic/datasets/hicdc'


chrs = [str(i) for i in range(1, 23)] + ['X']
# print(chrs)

for chr1 in chrs:
    
    df = pd.read_csv(os.path.join(path,'CD4_TCells_10kb_{}_matrix.txt'.format(chr1)),
                     sep='\t',      
                     engine='python')

    df['binI'] = 'chr' + df['binI'].astype(str)
    df['binJ'] = 'chr' + df['binJ'].astype(str)
    df['chrI'] = 'chr' + df['chrI'].astype(str)
    df['chrJ'] = 'chr' + df['chrJ'].astype(str)



    
    df.to_csv(os.path.join(path,'CD4_TCells_10kb_chr{}_matrix.txt'.format(chr1)),
              sep='\t',      
              index=False)
    os.remove(os.path.join(path,'CD4_TCells_10kb_{}_matrix.txt'.format(chr1)))
    print('chr',chr1, 'already processed!')

