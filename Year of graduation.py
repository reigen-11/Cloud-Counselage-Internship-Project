#!/usr/bin/env python
# coding: utf-8


import pandas as pd


data=pd.read_excel("Final Lead Data.xlsx")


data.columns

cols = ['Academic Year', 'Email']
data = data[cols]

data.dropna(subset=['Academic Year'], inplace=True)
data.drop_duplicates(['Email'],inplace=True)


data['Academic Year'].value_counts()


course_duration = 4

data['Remaining Years'] = course_duration - data['Academic Year']

current_year = pd.Timestamp.now().year
data['Year_of_Graduation '] = current_year + data['Remaining Years']+1


data['Year_of_Graduation'].to_excel('Year_of_Graduation_calculated.xlsx', index=False)

