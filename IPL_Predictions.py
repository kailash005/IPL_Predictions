#The Py file is only for reference as it is not functional as it was developed in a python note book environment please refer to the ipynb file for the functional code.

import pandas as pd
import numpy as np

matches = pd.read_csv("matches.csv")

matches

matches.isnull().sum()

matches.drop(columns = 'umpire3' , inplace=True )

teams = pd.concat([matches['team1'], matches['team2']]).unique()  
team_ids = {team: idx for idx, team in enumerate(teams)}

matches['team1'] = matches['team1'].map(team_ids)
matches['team2'] = matches['team2'].map(team_ids)
matches['toss_winner'] = matches['toss_winner'].map(team_ids)

matches

city = matches['city'].unique()
city_ids = {city:idx for idx , city  in enumerate(city)}
matches['city'] = matches['city'].map(city_ids)

matches['date'] = pd.to_datetime(matches['date'], format='mixed')

toss_result = matches['toss_decision'].unique()
toss_result_id = {toss_result:idx for idx , toss_result in enumerate(toss_result)}

toss_result_id

matches['toss_decision'] = matches['toss_decision1']

matches

matches['winner1'] = matches['winner'].map(team_ids)

matches['winner'] = matches['winner1']

matches.drop(columns = ['winner1'], inplace = True)

matches.drop(columns = ['venue' , 'umpire1' , 'umpire2' ], inplace = True)

matches.drop(columns = ['player_of_match' , 'result'], inplace = True)

matches.dropna()

matches.dropna(subset=['winner'], inplace=True)

matches['winner'] = matches['winner'].astype('int64')

matches.corr()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = matches.drop(columns=['id', 'season', 'date', 'toss_decision1'])

X = data.drop(columns=['winner', 'dl_applied', 'win_by_runs', 'win_by_wickets'])  
y = data['winner']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
