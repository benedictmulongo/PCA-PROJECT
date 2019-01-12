
import pandas as pd
import csv
import json


#Initialization of the dictionary
countries = {"names":[],"data":[],"readme":[]}


def read_tsv(date):
    print(" ***************  ********************** ")
    country_data = []
    country_name = []
    with open(str(date)+'.csv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        count = 0

        for row in reader:
            row.pop(1)
            # if count != 0 :
            if count == 0 :
                countries['readme'] = row
            else :
                name = row[0]
                
                rest = row[1:]
                rest = [float(x) for x in rest]
                rest[0] = int(rest[0])
                rest[1] = int(rest[1])
                
                country_data.append(rest)
                country_name.append(name)
                # print(name )
                # print(rest)
                    
                
            count += 1
            
    print(country_name)
    
    
    
    countries['names'] = country_name
    countries['data'] = country_data
    return country_data, country_data
    
date = 2015
read_tsv(date)

f = open('countries_data_' + str(date )+'scores_int.json', 'w')
json.dump(countries, f, indent=2)
f.close()
