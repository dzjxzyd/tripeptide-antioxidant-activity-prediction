
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'https://www.genome.jp/ftp/db/community/aaindex/aaindex1'
html = urlopen(url).read()

soup = BeautifulSoup(html, "html.parser")
soup.get_text()
soup.text
value= soup.text.split('\n')
value



# extract the row name
aa_name= []
aa_name
for i in range(10576):
    if len(value[i])>1:
        if value[i][0] == 'H':
            mm = value[i].split(' ')
            aa_name.append(mm[1])

# extract the amino acid Indices
aaindex = collections.defaultdict(list)
aaindex
j=0
for i in range(10576):
    if len(value[i])>1:
        if value[i][0] == 'I':
            aa1 = value[i+1].split(' ')
            a1=[]
            a2=[]
            for word in aa1:
                if len(word) > 1:
                    a1.append(word)
            aa2 = value[i+2].split(' ')
            for word in aa2:
                if len(word) > 1:
                    a2.append(word)
            aaindex[aa_name[j]]= a1+a2
            j=j+1
aaindex['VENT840101']
# export data
import pandas as pd
df1 = pd.DataFrame.from_dict(aaindex,  orient='index')
df1
pd.DataFrame(df1).to_csv('newaaindex566data.csv', header=False, index=True)
