import os
import zipfile
import csv 
import json
import glob
import gzip
import json
import datetime
import demoji
import re
from collections import Counter
import pandas as pd
import datetime 

all_categories = []
emojis = open("list.txt", "rt").read()
all_categories = list(emojis)
dict_emojis = { i : 0 for i in all_categories}
res = pd.DataFrame(columns  = ["Date"]+  all_categories)

lst_occur = []
directory = glob.glob(os.path.expanduser('~/project/json_emoji/geoTwitter20-*.json'))
for filename in sorted(directory):
    with open(filename, 'r') as file:
        name = os.path.basename(filename)
        name = name[:-5]
        name = name[10:]
        name = "20" + name
        date1 = datetime.datetime.strptime(name, "%Y-%m-%d")
        data = json.load(file)
        for i in data:
            for key, value in i.items():
                lst_occur.append(key)
        c = Counter(lst_occur)
        dict_emojis.update(c)
        data = dict(dict_emojis)
        df = pd.DataFrame([data])
        df.insert(0,"Date",date1)
        res =pd.concat([res, df])
        dict_emojis = { i : 0 for i in all_categories}
        df = pd.DataFrame([])
        lst_occur.clear()
        #lst_return =[(i, c[i] / len(lst_occur) * 100.0) for i in c]
        #print(lst_return)
        #print(dict_emojis)
res['anticipation'] = res.loc[:,['🙏']].sum(axis=1,numeric_only = int)
res['anger'] = res.loc[:,['😡','😤','😈','😠','😼','😾']].sum(axis=1,numeric_only = int)
res['disgust'] = res.loc[:,['😒','😫','😪','😣','😑','😐','😖','😶']].sum(axis=1,numeric_only = int)
res['sadness'] = res.loc[:,['😭','😔','😩','😢','😞','😥','😓','🙁','😟','😿']].sum(axis=1,numeric_only = int)
res['suprise'] = res.loc[:,['😳','😜','😝','😛','🙊','😲','😮','😨','😯']].sum(axis=1,numeric_only = int)
res['fear'] = res.loc[:,['😅','😁','😱','😬','😕','😰','😵','😧','🙀','😦']].sum(axis=1,numeric_only = int)
res['trust'] = res.loc[:,['😘','🙌','😻','😍','😚','😗','😙','😽']].sum(axis=1,numeric_only = int)
res['joy'] = res.loc[:,['😂','😊','😉','😎','😆','😋','😌','🙂','😃','😀','😄','😇','😹','😸','😺']].sum(axis=1,numeric_only = int)
res['mask'] = res.loc[:,['😷']].sum(axis=1,numeric_only = int)
res['other'] = res.loc[:,['🙇','🙋','🙆','🙅','🙍','🙎','🙄','😏','🙃','🙈','😴','🙉']].sum(axis=1,numeric_only = int)
res['total'] = res.loc[:,['anticipation','anger','disgust','sadness','suprise','fear','trust','joy','mask','other']].sum(axis=1,numeric_only = int)

res['anticipation_p'] = res['anticipation']/res['total']
res['anger_p'] = res['anger']/res['total']
res['disgust_p'] = res['disgust']/res['total']
res['sadness_p'] = res['sadness']/res['total']
res['suprise_p'] = res['suprise']/res['total']
res['fear_p'] = res['fear']/res['total']
res['trust_p'] = res['trust']/res['total']
res['joy_p'] = res['joy']/res['total']
res['mask_p'] = res['mask']/res['total']
res['other_p'] = res['other']/res['total']
#res['praying_p'] = res['🙏']/res['total']

res.to_csv("count_emojis_praying.csv", index=False)