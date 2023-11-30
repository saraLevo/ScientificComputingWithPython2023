##Text files
import numpy as np 
import numpy.random as npr
import csv
import pandas as pd

rvec = list(npr.randint(1,9,size=(1,10)))

file_name = "data/data_int.txt"
with open(file_name, 'w') as f1:
    for el in rvec:
        f1.write(str(el)+'\n')

!cat data/data_int.txt

print("\n")

rmatrix = npr.rand(5,5)

file_name = "data/data_float.txt"
with open(file_name, 'w') as f2:
    for row in rmatrix:
        #f2.write('[ ')
        for el in row:
            f2.write(str(el)+' ')
        #f2.write(']')
        f2.write('\n')

!cat data/data_float.txt

f3 = np.loadtxt('data/data_float.txt')
with open('data/data_float.csv', 'w') as csvf3:
    for line in f3:
        s = str(line).replace('  ', ' ').replace(" ]", "]").replace(' ',',').replace('[','').replace(']','')
        csvf3.write(s + "\n")

print()
!cat data/data_float.csv


##JSON files
import pandas as pd
import json
import csv

data = json.load(open('data/user_data.json'))
#!cat data/user_data.json

data_filter = [i for i in data if i['CreditCardType']=='American Express']
data_json = json.dumps(data_filter)

df = pd.read_json(data_json)
df.to_csv('data/user_data.csv')
df


##CSV files with Pandas
import pandas as pd
import numpy as np

file_name = "data/mushrooms_categorized.csv"
data = pd.read_csv(file_name)
#data

columns_means = data.groupby('class').mean()
columns_means.to_json("data/mushrooms_categorized.json")
columns_means

##Reading a database
import sqlite3 as sql 
import pandas as pd

link = sql.connect('data/sakila.db')
cursor = link.cursor()

query = "SELECT * FROM actor"
results = cursor.execute(query).fetchall()

df = pd.DataFrame(results)

cursor.close()
link.close()

names = df[1]
temp = names.str.startswith('A')
counter = 0
for i in range(len(names)):
    if temp[i] == True:
        counter += 1

print('The number of actors that have their first name starting with A is', counter)


##Reading credit card numbers
#!hexdump data/credit_card.dat
import struct
import pandas as pd

cc_data = {}

with open('data/credit_card.dat', 'rb') as file:
    file_content = file.read()
    cc_counter = 0
    char_size = 6 #credit card number length 
    char_str = []
    i = 0
    while i < len(file_content):
        cc_counter += 1
        if cc_counter < 20: 
            char_str.append(chr(int(file_content[i:i+char_size],2)))
            i = i + char_size
        else:
            i = i + (char_size-1)
            cc_counter = 0



