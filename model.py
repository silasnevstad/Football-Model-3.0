# [-- Imports --]
import numpy as np
import pandas as pd
# from IPython.display import display
# import soccerdata as sd
import sys

# correlation tools
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import seaborn as sb

# plotting tools
import matplotlib.pyplot as plt

# regression tools
from sklearn.model_selection import train_test_split
from scipy import stats as stats
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from regressors import stats as statsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import classification_report


# data scraping tools
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests

import matplotlib.pyplot as plt
from functools import reduce

# texting tools
from twilio.rest import Client

# backtesting tools
from itertools import combinations, permutations

# twitter tools
import tweepy

# string manipulation tools
import re

#firebase tools
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

sys.path.append('~/Desktop/Everything/Programming/CurrentProjects/AlgoPicks')

# [-- Global Variables --]
coefficients = []

bestMSE = 1 # mean squared error to keep track of the best model
MSEHistory = [] # list to keep track of the MSE of each model
ScoreHistory = [] # list to keep track of the accuracy of each model
R2History = [] # list to keep track of the score of each model

actualCurrentGA = []
actualCurrentGoals = []
actualPastGoals = []
actualPastGA = []

columns_to_remove = []

# actualCurrentTeams = squadStats2021['Squad']
# actualCurrentGoals = squadStats2021['Gls']
# actualPastGoals = squadStats2020['Gls']
# actualPastGoalsColumn = squadStats2020['Gls']
# actualCurrentGA = goalkeepingStats2021['GA']
# actualPastGA = goalkeepingStats2020['GA']
# # a = goalkeepingStats2021['GA']
# currentSquadsColumn = squadStats2019['Squad']

# teamsInEPL = ["Arsenal",
#               "Aston Villa",
#               "Brentford",
#               "Brighton and Hove Albion",
#               "Burnley",
#               "Chelsea",
#               "Crystal Palace",
#               "Everton",
#               "Leeds United",
#               "Leicester City",
#               "Liverpool",
#               "Manchester City",
#               "Manchester United",
#               "Newcastle United",
#               "Norwich City",
#               "Southampton",
#               "Spurs",
#               "Tottenham Hotspur",
#               "Watford",
#               "West Ham United",
#               "Wolverhampton Wanderers"]

shouldPrint = True


# Current Data
currentData = pd.DataFrame()
currentDataGF = pd.DataFrame()
currentDataGA = pd.DataFrame()

curr_columns_to_drop = ['Age', 'Min', 'xAG', '# Pl'] # Squad

# Final Data
# finalDataframe = []

# # Merging initial data
# df2019 = [advancedKeeping2019, goalkeepingStats2019, goalShotCreation2019, squadDefense2019, squadPlayingTime2019, squadPossession2019, squadPassing2019, squadStats2019, shootingStats2019]
# df2020 = [advancedKeeping2020, goalkeepingStats2020, goalShotCreation2020, squadDefense2020, squadPlayingTime2020, squadPossession2020, squadPassing2020, squadStats2020, shootingStats2020]
# df2021 = [advancedKeeping2021, goalkeepingStats2021, goalShotCreation2021, squadDefense2021, squadPlayingTime2021, squadPossession2021, squadPassing2021, squadStats2021, shootingStats2021]
# dfMerged2019 = reduce(lambda  left,right: pd.merge(left,right,on=['Squad'], how='outer'), df2019)
# dfMerged2020 = reduce(lambda  left,right: pd.merge(left,right,on=['Squad'], how='outer'), df2020)
# dfMerged2021 = reduce(lambda  left,right: pd.merge(left,right,on=['Squad'], how='outer'), df2021)
# frames = [dfMerged2019, dfMerged2020, dfMerged2021]
# dataDataframe = pd.concat(frames)
# #dataDataframe = dataDataframe.loc[:,~dataDataframe.T.duplicated(keep='first')].copy()

# dataDataframe.drop(columns = "Squad", inplace = True)
# actualPastGoals = dataDataframe['Gls'].values.tolist()
# actualPastGoalsColumn = dataDataframe['Gls']
# actualPastGA = dataDataframe['GA'].values.tolist()
# actualPastGAColumn = dataDataframe['GA']
# dataframeGF = dataDataframe.copy()
# dataframeGA = dataDataframe.copy()

# ====================================================================================================
#                                   [-- Data Scraping Step --]
# ====================================================================================================
def get_spi_data(url):
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    
    table = soup.find('table', {'class': 'forecast-table'})
    df = pd.read_html(str(table))[0]
    
    spiColumn = df[('Team rating', 'spi')]
    nameColumn = df[('Unnamed: 0_level_0', 'team')]
    nameColumn = nameColumn.apply(lambda x: re.split('(\d+)', x)[0]) # rename teams cutting out pts number at end
    
    # rename teams to match FBRef (Man. City -> Manchester City, Man. United -> Manchester United, Newcastle -> Newcastle United, Leicester -> Leicester City, Nottm Forest -> Nott'ham Forest)
    # how would i find the index of the team name and replace it with the new name?
    # nameColumn
    renames = [['Man. City', 'Manchester City'],
               ['Man. United', 'Manchester Utd'],
               ['Newcastle', 'Newcastle Utd'],
               ['Leicester', 'Leicester City'],
               ['Nottm Forest', 'Nott\'ham Forest'],
               ['Athletic Bilbao', 'Athletic Club'],
               ['Real Betis', 'Betis'],
               ['FC Augsburg', 'Augsburg'],
               ['VfL Bochum', 'Bochum'],
               ['Eintracht', 'Eint Frankfurt'],
               ['SC Freiburg', 'Freiburg'],
               ['', 'Köln'],
               ['Gladbach', 'M\'Gladbach'],
               ['Mainz', 'Mainz 05'],
               ['VfB Stuttgart', 'Stuttgart'],
               ['AC Ajaccio', 'Ajaccio'],
               ['Clermont', 'Clermont Foot'],
               ['PSG', 'Paris S-G'],
               ['Verona', 'Hellas Verona'],
               ['Inter Milan', 'Inter'],
               ['AC Milan', 'Milan'],
               ['Schalke ', 'Schalke 04']]
    for i in renames:
        nameColumn = nameColumn.replace(i[0], i[1])
    df = pd.concat([spiColumn, nameColumn], axis=1) # merge the two columns (SPI and team names)
    df.columns = ['SPI', 'Teams'] # rename the columns
    df.set_index('Teams', inplace = True) # set the index to the team names
    
    return df

def soccerDataFBRef():
    print("Scraping FBRef data...")
    seasonsToScrape = ['2017', '2018', '2019', '2020', '2021']
    statsToScrape = ['standard', 'keeper', 'keeper_adv', 'shooting', 'passing', 'passing_types', 'goal_shot_creation', 'defense', 'possession', 'playing_time', 'misc']
    dataframesSeperate = []
    fbref = sd.FBref(leagues=['ENG-Premier League'], seasons = seasonsToScrape)
    
    for i in statsToScrape:
        currentStats = fbref.read_team_season_stats(stat_type=i)
        currentStats.reset_index(inplace=True)
        currentStats = currentStats.rename(columns = {'index':'team'})
        currentStats.columns = currentStats.columns.map('|'.join).str.strip('|')
        currentStats.drop(columns = ['league', 'season', 'url'], inplace = True, axis = 1)
        dataframesSeperate.append(currentStats)
        
    currentData = pd.concat(dataframesSeperate, axis=1) # concats all dataframes into one
    currentData = removeSecondOccurenceColumns(currentData) # remove second occurence of columns
    first_column = currentData.pop('team') # pops the team column to the front column variable
    currentData.insert(0, 'team', first_column) # inserts the team column to the front of the dataframe
    print(currentData.columns)
    print(getDuplicateColumnnsAsList(currentData))
    display(currentData) # displays the dataframe
    
def soccerDataFifa():
    print("Getting FIFA data...")
    
def soccerDataFiveThirtyEight():
    print("Scraping Five-Thirty-Eight data...")
    

def removeSecondOccurenceColumns(df):
    return df.loc[:, ~df.columns.duplicated(keep = "first")]

def removeDuplcaitesList(lst): 
    return list(set(lst))

# ====================================================================================================
#                                   [-- Current Data Step --]
# ====================================================================================================
opts = Options()
# opts.add_argument("--headless")
# opts.add_argument("--disable-extensions")
# opts.add_argument("--disable-popup-blocking")
opts.add_experimental_option("excludeSwitches", ["enable-automation"])
opts.add_experimental_option('useAutomationExtension', False)

def downloadCurrentData(url, saveFileName, save):
    global currentData, currentDataGF, currentDataGA, curr_columns_to_drop, actualCurrentGoals, actualCurrentGA, actualCurrentTeams
    if (shouldPrint): print("Downloading data...")

    # url = f"https://fbref.com/en/comps/9/Premier-League-Stats" # EPL url using fbref
        
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = opts) # downloads and sets newest chromedriver
    params = {'behavior': 'allow', 'downloadPath': r'data'}
    driver.execute_cdp_cmd('Page.setDownloadBehavior', params) # download behaviour set to this directory

    driver.get(url) # driver launches the given url (fbref EPL)
    driver.maximize_window()
    
    closeAdButton = driver.find_element(By.CSS_SELECTOR, '[class*="fs-close-button fs-close-button-sticky"]')
    squadStatsTogglePerMatch = driver.find_element(By.ID, "stats_squads_standard_for_per_match_toggle") # squad
    keeperStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_keeper_for_per_match_toggle") # goalkeeper
    keeperAdvStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_keeper_adv_for_per_match_toggle") # advanced keeper
    shootingStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_shooting_for_per_match_toggle") # shooting
    passingStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_passing_for_per_match_toggle") # passing
    goalShotCreationStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_gca_for_per_match_toggle") # goal & shot creation
    defenseStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_defense_for_per_match_toggle") # defensive stats
    possStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_possession_for_per_match_toggle") # possession
    playingTimeStatsTogglePerMatch = driver.find_element(By.ID ,"stats_squads_playing_time_for_per_match_toggle") # playing time

    toggles = [squadStatsTogglePerMatch,
               keeperStatsTogglePerMatch,
               keeperAdvStatsTogglePerMatch,
               shootingStatsTogglePerMatch,
               passingStatsTogglePerMatch,
               goalShotCreationStatsTogglePerMatch,
               defenseStatsTogglePerMatch,
               possStatsTogglePerMatch,
               playingTimeStatsTogglePerMatch
               ]
    
    closeAdButton.click()
    
    for toggle in toggles:
        if (shouldPrint): print(f"toggling {toggle}")
        toggled = False
        while (not toggled):
            try:
                toggle.click()
                toggled = True
            except:
                driver.execute_script("arguments[0].scrollIntoView();", toggle) 
        
    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')
    
    tableIndexes = [15, 11, 17, 23, 13, 25, 29, 27, 19]
    # 11 = squad stats
    # 13 = keeper stats
    # 15 = adv goal keeper stats
    # 17 = squad shooting
    # 19 = squad passing
    # 23 = goal and shot creation
    # 25 = squad defensive actions
    # 27 = squad possession
    # 29 = squad playing time
    seperateData = []
    
    for index in tableIndexes:
        rawData = soup.find_all("table")[index]
        statsTable = pd.read_html(str(rawData))
        statsTable = statsTable[0]
        statsTable.columns = statsTable.columns.droplevel()
        if (index == 11):
            statsTable.columns = ["Squad", "# Pl", "Age", "Poss", "MP","Starts","Min", "90s","Gls","Ast","G+A", "G-PK", "PK", "PKatt", "CrdY","CrdR", "xG", "npxG", "xGA", "npxG+xGA", "PrgC", "PrgP", "Gls", "Ast","G+A","G-PK", "G+A-PK", "xG", "xAG", "xG+xGA","npxG", "npxG+xAG"]
            statsTable.drop(columns = "Poss", inplace = True)
        if (index == 13):
            statsTable.rename(columns = {"PKatt" : "GPKatt"}, inplace = True)
        if (index == 15):
            statsTable.columns = ["Squad", "# Pl", "90s", "GA","PKA","FK","CK","OG","PSxG","PSxG/SoT","PSxG+/-","/90","LGCmp","LGAtt","LGCmp%","PAtt","Thr","PLaunch%","PAvgLen","GAtt","GLaunch%","GAvgLen","Opp","Stp","Stp%","#OPA","#OPA/90","AvgDist"]
        if (index == 19):
            statsTable.columns = ["Squad", "# Pl", "90s", "TCmp","TAtt","TCmp%","PTotDist","PPrgDist","SCmp","SAtt","SCmp%","MCmp","MAtt","MCmp%","LCmp","LAtt","LCmp%", "Ast", "xAG", "xA", "A-xA", "KP", "1/3", "PPA", "CrsPA", "PProg"]
        if (index == 23):
            statsTable.columns = ["Squad", "# Pl", "90s", "SCA▼","SCA90","SCAPassLive","SCAPassDead","SCADrib","SCASh","SCAFld","SCADef","GCA","GCA90","GCAPassLive","GCAPassDead","GCADrib","GCASh","GCAFld","GCADef"]
        if (index == 25):
            statsTable.columns = ["Squad", "# Pl", "90s", "TKTkl","TklW","TK Def 3rd","TK Mid 3rd","TK Att 3rd","VSTkl","DAtt","Tkl%","Past","Blocks","DSh","Pass","Int","Tkl+Int","Clr","Err"]
        if (index == 27):
            statsTable.columns = ["Squad", "# Pl", "Poss", "90s", "Touches","Def Pen","Def 3rd","Mid 3rd","Att 3rd","Att Pen","Live","Att","Succ","Succ%", "Tkld","Tkld%","Carries","TotDist","ProgDist","PrgC","1/3","CPA","Mis","Dis","Rec","ProgR"]
        seperateData.append(statsTable)
    
    # merges all different types of stats into a single dataframe
    # renaming all repeat duplicate columns with a _dup at the end.
    currentData = reduce(lambda left,right: pd.merge(left,right,on=['Squad'], how='outer', suffixes=['', '_dup']), seperateData) #
    
    currentData = currentData.loc[:, ~currentData.columns.duplicated()]

    for col in currentData.columns:
        if (col[-3:] == "dup"):
            if (col not in curr_columns_to_drop):
                curr_columns_to_drop.append(col)
    
    for col in curr_columns_to_drop:
        currentData.drop(columns = col, inplace = True)
        
    if save: currentData.to_csv("data/" + saveFileName, index = False)
    
    actualCurrentTeams = currentData['Squad'].values.tolist()
    actualCurrentGoals = currentData['Gls'].values.tolist()
    actualCurrentGA = currentData['GA'].values.tolist()
    
    currentData.drop(columns = "Squad", inplace = True)
    
    driver.close()
    driver.quit()
    
    if (shouldPrint): ("Uploading current data...")
    if (shouldPrint): print("\nData downloaded.\n")
    
    return currentData
    
    
def downloadPastData(url1, url2, saveFileName):
    global currentData, currentDataGF, currentDataGA, curr_columns_to_drop, actualCurrentGoals, actualCurrentGA, actualCurrentTeams
    if (shouldPrint): print("Downloading the old data...")
    years = ['2021-2022/2021-2022-', '2020-2021/2020-2021-', '2019-2020/2019-2020-', '2018-2019/2018-2019-', '2017-2018/2017-2018-']  # '2016-2017/2016-2017-'
    dataframes = []
    for i in range(len(years)):
        df = downloadCurrentData(url1 + years[i] + url2, "", False) # set to false to not save the data locally (by year individually)
        dataframes.append(df)
    
    merged = pd.concat(dataframes)
    
    merged.to_csv("data/" + saveFileName, index = False)
    
    print(merged)
    
def getPastDataFromLocal(filename):
    if (shouldPrint): print("Uploading old data...")
    data = pd.read_csv("data/" + filename)
    
    return data
    
def getCurrentDataLocally(filename):
    global currentData, currentDataGF, currentDataGA, actualCurrentGoals, actualCurrentGA, actualCurrentTeams
    
    currentData = pd.read_csv("data/" + filename)
    
    for col in curr_columns_to_drop:
        try:
            currentData.drop(columns = col, inplace = True)
        except:
            pass
    
    actualCurrentTeams = currentData['Squad'].values.tolist()
    actualCurrentGoals = currentData['Gls'].values.tolist()
    actualCurrentGA = currentData['GA'].values.tolist()
    
    currentData.drop(columns = "Squad", inplace = True)
    
    return currentData

# verifies that two given dataframes have the same columns (or are compatible)
def verifyColumnLengths(df1, df2):
    if (shouldPrint): ("Verifying current and past data compatibitly")
    
    currColumns = list(df1.columns)
    regColumns = list(df2.columns)
    
    nonMatches = non_match_between_lists(currColumns, regColumns)
    if (len(nonMatches) == 0):
        firstLen = len(df1.columns)
        secondLen = len(df2.columns)
        
        print(f"Length of the first dataframe: {firstLen}")
        print(f"Length of the second dataframe: {secondLen}")
        
        if (firstLen == secondLen): # math.isclose(firstLen, secondLen, abs_tol=1)
            if (shouldPrint): print("Data compatabile.")
        else:
            list1 = list(df1.columns)
            list2 = list(df2.columns)
            output = [x for x in list2 if not x in list1 or list1.remove(x)]
            print(output)
            print("Failed, dataframes' column lengths do not equal each other.")
            exit()
    else:
        print("Data incompatible, columns do not match.")
        print(nonMatches)
        exit()

# gets non macthing variables between two lists
def non_match_between_lists(list_a, list_b):
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append([i, 1])
    for i in list_b:
        if (i not in list_a) and (i not in non_match):
            non_match.append([i, 2])
    return non_match

# finds the duplicate columns of a given dataframe
def getDuplicateColumnnsAsList(df):
    columns_found = []
    duplicates = []
    for col in df.columns:
        if (col in columns_found):
            duplicates.append(col)
        else:
            columns_found.append(col)
    return duplicates

# ====================================================================================================
#                                   [-- Correlation Step --]
# ====================================================================================================
# returns the highest correlation in a data set, given variable to avoid
def highestCorrelation(df, avoid):
    df = df.drop(columns = avoid)
    corr = df.corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    # upper_tri = upper_tri.drop(columns = avoid)
    return highestCorrelationHelper(upper_tri, findMaxInDataframe(upper_tri)[0], findMaxInDataframe(upper_tri)[1])

# returns the highest value found in a given dataframe
def findMaxInDataframe(df):
    maxVal = 0.0
    maxCol = ""
    maxRow = ""
    for col in df.columns: # loops through columns
        for row in df.columns: # loops through rows
            if (df.loc[col, row] > maxVal): # determines if value is higher than previous maximum
                maxCol = col
                maxRow = row
                maxVal = df.loc[col, row]
    return[maxCol, maxRow]

# determines, given two data points, the one with the lowest correlation related to other data points. 
def highestCorrelationHelper(df, col, row):
    sumDfVertical = df.sum(axis = 0)
    sumDfHorizaontal = df.sum(axis = 1)
    totalDf = sumDfVertical.add(sumDfHorizaontal)
    return [col, totalDf.loc[col]] if totalDf.loc[col] > totalDf.loc[row] else [row, totalDf.loc[row]] # returns the data point with highest overall correlation 


# removes columns with high correlation given a dataframe
def correlation(df, avoid):
    if (shouldPrint): print("--- Removing Correlation ---")
    global dataframeGF, dataframeGA, columns_to_remove
    
    while (len(df.columns) > 125):
        highestCorrelatedItem = highestCorrelation(df, avoid)[0] # out of the two data points with highest correlation, this gets the one with the higher overall correlation
        correlationValue = round(highestCorrelation(df, avoid)[1], 2) # gets the correlation value of the single most correlated data point
        df.drop(columns = highestCorrelatedItem, inplace = True) # removes the highest correlated data point
        columns_to_remove.append(highestCorrelatedItem)
        if (shouldPrint): print(f"Removed {highestCorrelatedItem} [{correlationValue}]")

# ====================================================================================================
#                               [-- Regression Step --]
# ====================================================================================================
def regression(df, targetVariable, targetVariableColumn):
    if (shouldPrint): print("--- Regressing Data (Ridge Regression) ---")
    global dataframeGF, dataframeGA, coefficients, columns_to_remove, bestMSE, MSEHistory
    
    df = df.fillna(0)
    
    y = targetVariableColumn
    n = len(df.columns)
    dfDirty = df.copy()+0.000001*np.random.rand(60, n)
    
    # check if target variable column has as many rows as the dataframe
    if (len(targetVariableColumn) != len(df)):
        print(f'Error: Target variable lengeth ({len(targetVariable)}) does not match dataframe length ({len(df)}).')
        exit()
    
    while (len(df.columns) > 20):
        try:
            model = Ridge(alpha=0.001) # ridge regression
            
            try:
                X = df.drop([targetVariable], axis=1) # all variables but target
            except:
                X = df
            
            X_train, x_test = train_test_split(X, test_size = 0.2, random_state = 9) # split data to avoid overfitting the model
            Y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 9)

            model.fit(X_train, Y_train)
            
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error on Training Data:", mse)
            MSEHistory.append(mse)
            
            params = np.append(model.intercept_, model.coef_)
        
            p_values = statsRegressor.coef_pval(model, X_train, Y_train)
            real_values = p_values[1:]
            
            if (mse < bestMSE):
                bestMSE = mse
                print("new best MSE: ", bestMSE)
            
        except:
            model = Ridge(alpha=0.001) # ridge regression
            
            try:
                X = dfDirty.drop([targetVariable], axis=1) # all variables but target
            except:
                X = dfDirty
                
            
            X_train, x_test = train_test_split(X, test_size = 0.2, random_state = 9) # split data to avoid overfitting the model
            Y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 9)

            model.fit(X_train, Y_train) # use regression to predict with test data
            
            params = np.append(model.intercept_, model.coef_)
        
            p_values = statsRegressor.coef_pval(model, X_train, Y_train)
            real_values = p_values[1:]
            
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error on Training Data:", mse)
            MSEHistory.append(mse)
            
            if (mse < bestMSE):
                bestMSE = mse
                print("new best MSE: ", bestMSE)
            
        
        maximum = max(real_values)
        
        # print(statsRegressor.summary(lr, X_train, Y_train))
        
        for i, j in enumerate(real_values):
            if j == maximum:
                index = i

        droppedColumn = df.columns[index]
        df.drop(columns = droppedColumn, axis = 1, inplace = True)
        # if (droppedColumn == "Gls"):
        #     print("foo")
        #     exit()
        maximum = round(maximum, 4)
        if (shouldPrint): print(f"Removed {droppedColumn} [{maximum}]")
        columns_to_remove.append(droppedColumn)
        
        coefficients = np.copy(params)

def randomForestRegression(df, targetVariable, targetVariableColumn):
    if (shouldPrint): print("--- Regressing Data (Random Forest Regression) ---")
    global dataframeGF, dataframeGA, coefficients, columns_to_remove, bestMSE, MSEHistory
    
    df = df.fillna(0)
    
    y = targetVariableColumn
    
    # check if target variable column has as many rows as the dataframe
    if (len(targetVariableColumn) != len(df)):
        print(f'Error: Target variable lengeth ({len(targetVariable)}) does not match dataframe length ({len(df)}).')
        exit()
        
    while (len(df.columns) > 50):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            X = df.drop([targetVariable], axis=1) # all variables but target
        except:
            X = df
        
        X_train, x_test = train_test_split(X, test_size = 0.2, random_state = 9) # split data to avoid overfitting the model
        Y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 9)

        model.fit(X_train, Y_train)
        
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error on Training Data:", mse)
        MSEHistory.append(mse)
        ScoreHistory.append(model.score(x_test, y_test))
        R2History.append(metrics.r2_score(y_test, y_pred))
        
        importances = model.feature_importances_
        sorted_importances = sorted(zip(importances, X.columns), reverse=True)
        least_important = sorted_importances[-1]
        least_import_column = least_important[1]
        print(f"Least important: {least_important}")
        
        
        if (mse < bestMSE):
            bestMSE = mse
            print("new best MSE: ", bestMSE)
            
        df.drop(columns = least_import_column, axis = 1, inplace = True)
        columns_to_remove.append(least_import_column)


def ensembleLearning(df, targetVariable, targetVariableColumn, actualTargetColumn, currentX, columnNames):
    if (shouldPrint): print("--- Stacking Models (Random Forest and Ridge Regressions) ---")
    global dataframeGF, dataframeGA, coefficients, columns_to_remove, bestMSE, MSEHistory
    
    df = df.fillna(0)
    
    y = targetVariableColumn
    n = len(df.columns)
    
    # check if target variable column has as many rows as the dataframe
    if (len(targetVariableColumn) != len(df)):
        print(f'Error: Target variable lengeth ({len(targetVariable)}) does not match dataframe length ({len(df)}).')
        exit()

    try:
        X = df.drop([targetVariable], axis=1) # all variables but target
    except:
        X = df
    
    X_train, x_test = train_test_split(X, test_size = 0.2, random_state = 9) # split data to avoid overfitting the model
    Y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 9)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(x_test)
    
    model_Random_Forest = RandomForestRegressor(n_estimators=100, random_state=42)
    model_Ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    
    model_Random_Forest.fit(X_train_scaled, Y_train)
    model_Ridge.fit(X_train, Y_train)
    
    predictions_Random_Forest = model_Random_Forest.predict(X_test_scaled)
    predictions_Ridge = model_Ridge.predict(x_test)
    
    predictions = np.concatenate((predictions_Random_Forest.reshape(-1,1), predictions_Ridge.reshape(-1,1)), axis=1)
    
    model = Ridge()
    model.fit(predictions, y_test)
    
    currentX = currentX.drop(columns = targetVariable, axis = 1)
    
    # Make an empty list to store the predictions
    predictions_current = []

    # Loop over the rows of the dataframe
    for index, row in currentX.iterrows():
        # Make predictions on the current row using the base models
        predictions_1 = model_Random_Forest.predict(row.values.reshape(1, -1))
        predictions_2 = model_Ridge.predict(row.values.reshape(1, -1))

        # Combine the predictions of the base models
        predictions = np.concatenate((predictions_1.reshape(-1, 1), predictions_2.reshape(-1, 1)), axis=1)

        # Use the second-level model to make predictions on the current row
        prediction_current = model.predict(predictions)
        
        if (columnNames[0] == 'xGF'): prediction_current = prediction_current * 0.9
        # Add the prediction to the list
        predictions_current.append(prediction_current[0])

        # Add the predictions to the dataframe as a new column
    currentX[columnNames[0]] = predictions_current # predictions
    currentX[columnNames[1]] = actualTargetColumn # actual data
    currentX['Teams'] = actualCurrentTeams # team names
    
    currentX.set_index('Teams', inplace=True)
    
    currentX.to_csv("data/" + columnNames[0] + '.csv')
    
    return currentX
    
    

def selectBestFeatures(df, targetVariable):
    if (shouldPrint): print("--- Selecting Best Features ---")
    global dataframeGF, dataframeGA, columns_to_remove
    
    # the independent variables set
    listOfCols = list(df.columns)
    listOfCols.remove(targetVariable)
    X = df[listOfCols]
    
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                            for i in range(len(X.columns))]
    
    # print(vif_data.sort_values(by=["VIF"], ascending=False))
    
    while (len(df.columns) > 25):
        highestVIF = vif_data.sort_values(by=["VIF"], ascending=False).iloc[0]
        # print(highestVIF)
        df.drop(columns = highestVIF[0], inplace = True)
        columns_to_remove.append(highestVIF[0])
        # if (shouldPrint): print(f"Removed {highestVIF[0]} [{highestVIF[1]}]")
        
        # the independent variables set
        listOfCols = list(df.columns)
        listOfCols.remove(targetVariable)
        X = df[listOfCols]
        
        # VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                                for i in range(len(X.columns))]
        
        # print(vif_data.sort_values(by=["VIF"], ascending=False))
        
    print(vif_data.sort_values(by=["VIF"], ascending=False))
    
def selectNBestChiFeatures(df, targetVariableColumn, n):
    y = targetVariableColumn
    X = df
    
    best_features= SelectKBest(score_func = chi2, k = 3)
    fit = best_features.fit(X, y)
    
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    
    features_scores = pd.concat([df_columns, df_scores], axis=1)
    features_scores.columns = ['Features', 'Score']
    features_scores.sort_values(by = 'Score')
    
    print(features_scores)
    
def graphModelPerfomanceHistory():
    global MSEHistory
    plt.plot(MSEHistory, color = 'red', label = 'MSE History')
    # how would i plot the best MSE?
    plt.scatter(MSEHistory.index(min(MSEHistory)), min(MSEHistory), color = 'green', label = 'Best MSE')
    plt.title("Model MSE Perfomance History")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()
    plt.close()
    plt.plot(ScoreHistory, color = 'blue', label = 'Accuracy History')
    plt.scatter(ScoreHistory.index(max(ScoreHistory)), max(ScoreHistory), color = 'green', label = 'Best Accuracy')
    plt.title("Model Score Perfomance History")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()
    plt.close()
    plt.plot(R2History, color = 'orange', label = 'Score History')
    plt.scatter(R2History.index(max(R2History)), max(R2History), color = 'green', label = 'Best Score')
    plt.title("Model Score Perfomance History")
    plt.legend(loc = 'best')
    plt.show()
    plt.clf()
    plt.close()
    
    
    
# ====================================================================================================
#                                   [---- Model Step ----]
# ====================================================================================================
# standardizes a given dataframe using sklearn's StandardScaler
def standardizeData(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler.transform(df)

# cleans current data by removing unnecessary data points (determined by correlation & regression earlier)
def cleanData(df):
    if (shouldPrint): print("Cleaning current data...")
    global currentData, currentDataGF, currentDataGA
    
    unwanted_cols_no_duplicates = [*set(columns_to_remove)] # removes duplicates
    df.drop(columns = unwanted_cols_no_duplicates, inplace = True)
    # for col in unwanted_cols_no_duplicates:
    #     if (shouldPrint): print(f"removed unwanted column: {col}, from current data")
    #     df.drop(columns = col, inplace = True)

# orders to given dataframes to be the same
def orderTwoDataFrames(df1, df2):
    df1 = df1.reindex(sorted(df1.columns), axis=1)
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    return df1, df2

def cleanUpData(df1, df2):
    global currentData, currentDataGA, currentDataGF, dataframe, dataframeGA, dataframeGF, columns_to_remove
    cleanData(df1)
    cleanData(df2)
    
    verifyColumnLengths(df1, df2)
    return orderTwoDataFrames(df1, df2)
    
# get coefficients from the given dataframe, using the given target variable
def getRegressionCoefficientsFromData(df, targetVariableColumn):
    global coefficients
    
    y = targetVariableColumn
    
    X = df # all variables but target
    
    X_train, x_test = train_test_split(X, test_size = 0.2, random_state = 9) # split data to avoid overfitting the model
    Y_train, y_test = train_test_split(y, test_size = 0.2, random_state = 9)
    
    model = Ridge(alpha=0.001) # ridge regression
    model.fit(X_train, Y_train)
    
    params = np.append(model.intercept_, model.coef_)
    
    coefficients = np.copy(params)
    
def setUpDataframeforDatabase(gf, ga):
    columns_to_keep_for = ['xGF', 'acGF']
    columns_to_keep_against = ['xGA', 'acGA']
    # only keep the columns we want
    gf = gf[columns_to_keep_for]
    ga = ga[columns_to_keep_against]
    # merge the two dataframes on the index
    merged = pd.merge(gf, ga, left_index=True, right_index=True)
    meanGF = merged['xGF'].mean()
    meanGA = merged['xGA'].mean()
    # loop through the dataframe and calculate the attack and defensive strength for each team to then be saved as a column\
    for i in merged.itertuples():
        # merged.loc[i.Index, 'Attack Strength'] = i.xGF / (i.xGF + meanGF)
        # merged.loc[i.Index, 'Defensive Strength'] = i.xGA / (i.xGA + meanGA)
        merged.loc[i.Index, 'Attack Strength'] = i.xGF / meanGF
        merged.loc[i.Index, 'Defensive Strength'] = i.xGA / meanGA
    
    return merged
    
# get the predicted values for the given dataframe, either for goals for or against (uses global coefficients)
# def prediction(df, expectedColumnName, actualColumnName, actualExpected):
#     global currentDataGA, currentDataGF
#     setterArray = []
    
#     for i in range(1, len(teamsInEPL)):
#         setterArray.append("")
        
#     df[expectedColumnName] = setterArray
#     coeff = coefficients.tolist()
#     # df['Squad'] = currentSquadsColumn
#     df.set_index('Squad', inplace = True, drop = True)
    
#     for j in df.itertuples():
#         predicted = float(coeff[0])
#         for i in range(1, len(coeff) - 1):
#             predicted += (float(coeff[i]) * float(j[i]))
#         predicted = round(predicted, 2)
#         df.loc[j.Index, expectedColumnName] = predicted
    
#     #df = df.join(actualExpected)
#     df[actualColumnName] = actualExpected


def resetParameters():
    global coefficients, columns_to_remove, MSEHistory, ScoreHistory, R2History
    coefficients = [] # resetting the coefficients and columns_to_remove (unwanted columns from the current data)
    columns_to_remove = []
    MSEHistory = []
    ScoreHistory = []
    R2History = []

# builds the model
def buildData(currentURL, spiURL, saveFinalDataframeName, pastDataFileName, currentFilename, downloadRequired):
    global shouldPrint, currentData, currentDataGF, currentDataGA, dataframeGF, dataframeGA, columns_to_remove, coefficients
    pastData = getPastDataFromLocal(pastDataFileName) # gets the past data from the local files
    dataframeGA = pastData.copy() # copies the past data to a new dataframe (dataframeGA)
    dataframeGF = pastData.copy() # copies the past data to a new dataframe (dataframeGF)
    actualPastGoals = dataframeGF["Gls"].values # gets the actual goals scored from the past data
    actualPastGA = dataframeGA["GA"].values # gets the actual goals conceded from the past data
    
    currentDataGF = pd.DataFrame()
    currentDataGA = pd.DataFrame()
    resetParameters()
    
    shouldPrint = True
    if (shouldPrint): print("Constructing model...\n")
    
    if downloadRequired:
        currentData = downloadCurrentData(currentURL, currentFilename, True) # downloads the current data from online and stores it in a singular dataframe (currentData) (time consuming)
    else:
        currentData = getCurrentDataLocally(currentFilename) # gets the current data from the local files (can be used if already downloaded recently to save a lot of time) (currentData.csv)
    currentDataGF = currentData.copy() # copies the current data to a new dataframe (currentDataGF)
    currentDataGA = currentData.copy() # copies the current data to a new dataframe (currentDataGA)
    
    actualCurrentGoals = currentDataGF["Gls"].values # gets the actual goals scored from the current data
    actualCurrentGA = currentDataGA["GA"].values # gets the actual goals allowed from the current data
    
    #                                            [-- Offensive --]
    
    if (shouldPrint): print("\nOffensive Analysis...")
    
    verifyColumnLengths(currentDataGF, dataframeGF) # verifies that current dataframe and past dataframe are compatible (same columns)
    resetParameters() # resets the parameters for the offensive analysis
    
    randomForestRegression(dataframeGF, "Gls", actualPastGoals) # removing statistics based on random forest regression (feature importance)
    # graphModelPerfomanceHistory() # graphs the MSE, R2, and Score history for the model
    
    currentDataGF, dataframeGF = cleanUpData(currentDataGF, dataframeGF) # cleans the current data (removing unnecessary columns, ordering the columns, and verifying that the dataframes are compatible)
    currentDataGF = ensembleLearning(dataframeGF, "Gls", actualPastGoals, actualCurrentGoals, currentDataGF, ['xGF', 'acGF']) # stacks two models together to create a third model (ensemble learning)
    
    #                                          [-- Defensive --]
    
    if (shouldPrint): print("\nDefensive Analysis...")
    
    verifyColumnLengths(currentDataGA, dataframeGA) # verifies that current dataframe and past dataframe are compatible (same columns)
    resetParameters() # resetting the parameters for defensive analysis
    
    randomForestRegression(dataframeGA, "GA", actualPastGA) # removing statistics based on random forest regression (feature importance)
    # graphModelPerfomanceHistory() # graphs the model performance history
    
    currentDataGA, dataframeGA = cleanUpData(currentDataGA, dataframeGA) # cleans the current data (removing unnecessary columns, ordering the columns, and verifying that the dataframes are compatible)
    currentDataGA = ensembleLearning(dataframeGA, "GA", actualPastGA, actualCurrentGA, currentDataGA, ['xGA', 'acGA']) # stacks two models together to create a third model (ensemble learning)
    
    SPIDataframe = get_spi_data(spiURL) # gets the SPI data from online and stores it in a dataframe
    
    finalDataframe = setUpDataframeforDatabase(currentDataGF, currentDataGA) # sets up the dataframes for the database

    # merge the SPI data with the final dataframe (by the team name)
    finalDataframe = finalDataframe.merge(SPIDataframe, how = 'left', left_on = 'Teams', right_on = 'Teams')
    finalDataframe.sort_values(by = ['SPI'], ascending = False, inplace = True)
    finalDataframe.to_csv("data/" + saveFinalDataframeName)

def initFirebaseDatabase():
    if shouldPrint: print("Initializing Firebase Database...")
    cred = credentials.Certificate('functions/algopick-acc98-firebase-adminsdk-mwc8r-f53eed233a.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://algopick-acc98-default-rtdb.firebaseio.com'
    })
    # verify that the database is initialized
    if shouldPrint: print("Database Initialized: " + str(db.reference().get() != None))

def uploadToDatabase(dataframe, databaseName):
    # cred = credentials.Certificate('algopick-acc98-firebase-adminsdk-mwc8r-f53eed233a.json')
    # firebase_admin.initialize_app(cred, {
    #     'databaseURL': 'https://algopick-acc98-default-rtdb.firebaseio.com'
    # })

    # Get a reference to the Realtime Database
    ref = db.reference()
    
    data = dataframe.to_dict()
    print(data)
    
    ref.child(databaseName).set(data)
    
    if shouldPrint: print("Data uploaded to database: " + databaseName)

def predictingResults(awayTeam, homeTeam, finalDf):
    try:
        finalDf.set_index('Teams', inplace = True)
    except:
        pass
    
    awayTeam_SPI = finalDf.loc[awayTeam, 'SPI']
    homeTeam_SPI = finalDf.loc[homeTeam, 'SPI']
    awayTeam_att_strength = finalDf.loc[awayTeam, 'Attack Strength']
    homeTeam_att_strength = finalDf.loc[homeTeam, 'Attack Strength']
    awayTeam_def_strength = finalDf.loc[awayTeam, 'Defensive Strength']
    homeTeam_def_strength = finalDf.loc[homeTeam, 'Defensive Strength']
    
    # calculate the expected goals for each team
    awayTeam_expected_goals = awayTeam_att_strength * homeTeam_def_strength * finalDf['xGF'].mean() * (awayTeam_SPI / homeTeam_SPI) # SPI ratio is used to adjust the expected goals
    homeTeam_expected_goals = homeTeam_att_strength * awayTeam_def_strength * finalDf['xGF'].mean()
    
    # calculate the poisson distribution for each team
    awayTeam_poisson = poisson.pmf(np.arange(0, 10), awayTeam_expected_goals)
    homeTeam_poisson = poisson.pmf(np.arange(0, 10), homeTeam_expected_goals)
    
    # calculate the probability of each team winning
    awayTeam_win_prob = np.sum(np.tril(np.outer(awayTeam_poisson, homeTeam_poisson), -1))
    homeTeam_win_prob = np.sum(np.triu(np.outer(awayTeam_poisson, homeTeam_poisson), 1))
    draw_prob = np.sum(np.diag(np.outer(awayTeam_poisson, homeTeam_poisson)))
    
    home_odds = 1 / homeTeam_win_prob - 1 # calculate the odds for each team
    away_odds = 1 / awayTeam_win_prob - 1 # calculate the odds for each team
    draw_odds = 1 / draw_prob - 1 # calculate the odds for each team
    confidence_factor = max(home_odds, away_odds, draw_odds) - 1 # calculate the confidence factor
    
    return awayTeam_win_prob, homeTeam_win_prob, draw_prob, confidence_factor



def cleanOddsJson(league):
    url = f"https://api.the-odds-api.com/v4/sports/{league}/odds/?apiKey=d9098eff0d09ba0aad0190ab431fe018&regions=us&markets=h2h"
    odds_response = requests.get(url)
    odds_json = odds_response.json()
    simple_json = []
    
    for game in odds_json: # for each game
        awayID = 1
        homeID = 0
        
        away_team = rename(game['away_team']) # rename the away team
        home_team = rename(game['home_team']) # rename the home team
        
        siteIdx = 0 # index of the bookmaker site
        
        if game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['name'] != away_team: # if the awayID is not the away team
            awayID = 0 # set the away team to the first outcome
            homeID = 1 # set the home team to the second outcome
        away_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['price'] # get the away odds
        home_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][homeID]['price'] # get the home odds
        draw_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][2]['price'] # get the draw odds
        
        while (away_odds == 1.0 or home_odds == 1.0): # if away or home odds are 1.0, then use a different site
            siteIdx += 1 # increment the site index
            if (siteIdx == len(game['bookmakers'])): # if there are no more sites, then skip the game
                break
            if game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['name'] != away_team: # if the awayID is not the away team
                awayID = 0
                homeID = 1
            away_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['price']
            home_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][homeID]['price']

        if (away_odds == 1.0 or home_odds == 1.0): # if away or home odds are 1.0, then skip the game
            continue
        
        g = [away_team, home_team, away_odds, home_odds, draw_odds] # create a list of the game data
        
        simple_json.append(g) # add the game to the list of games
    
    return simple_json

def rename(team):
    if team == "Wolverhampton Wanderers":
        team = "Wolves"
    if team == "Nottingham Forest":
        team = "Nott'ham Forest"
    if team == "West Ham United":
        team = "West Ham"
    if team == "Tottenham Hotspur":
        team = "Tottenham"
    if team == "Newcastle United":
        team = "Newcastle Utd"
    if team == "Brighton and Hove Albion":
        team = "Brighton"
    if team == "Manchester United":
        team = "Manchester Utd"
        
    return team

def getPicks(leagues, finalDf):
    for league in leagues:
        odds_json = cleanOddsJson(league)
        picks_json = []
        for game in odds_json:
            # if game already happened, skip it
            for odds in picks_json:
                if game[0] in odds and game[1] in odds:
                    break
            away_team = game[0]
            home_team = game[1]
            away_odds = decimalOddsToAmericanOdds(game[2])
            home_odds = decimalOddsToAmericanOdds(game[3])
            draw_odds = decimalOddsToAmericanOdds(game[4])
            
            away_prob, home_prob, draw_prob, confidence = predictingResults(away_team, home_team, finalDf)
            
            predicted_result = getFTR(away_prob, home_prob, draw_prob)
            
            if confidence < 3.5:
                if predicted_result == "A":
                    if away_odds > 200:
                        picks_json.append([away_team, away_odds, home_team, confidence])
                    else:
                        continue
                elif predicted_result == "H":
                    if home_odds > 200:
                        picks_json.append([home_team, home_odds, away_team, confidence])
                    else:
                        continue
                elif predicted_result == "D":
                    if draw_odds > 200:
                        picks_json.append([draw_odds, away_team, home_team, confidence])
                    else:
                        continue
            
            if predicted_result == "A":
                picks_json.append([away_team, away_odds, home_team, confidence])
            elif predicted_result == "H":
                picks_json.append([home_team, home_odds, away_team, confidence])
            elif predicted_result == "D":
                picks_json.append([draw_odds, away_team, home_team, confidence])
    
    print("Total picks: " + str(len(odds_json)))
    print("Number of picks: " + str(len(picks_json)))
    return picks_json

def writePicksToDB(picks):
    # write picks to firebase database
    ref = db.reference()
    
    # remove duplicates
    picks = [list(x) for x in set(tuple(x) for x in picks)]
    
    # delete old picks
    ref.child("picks").delete()
    
    # write to database
    ref.child("picks").set(picks)
    
    if (shouldPrint): print("Picks written to database")

def backtest(finalDf):
    # get matches from PremierLeagueMatches.csv
    matches = pd.read_csv("data/PremierLeagueMatches.csv")
    correct = incorrect = 0
    finalDf = finalDf.set_index('Teams')
    for index, row in matches.iterrows():
        # print("Match " + str(index + 1) + " of " + str(len(matches)))
        # print("Away Team: " + row['Away'] + " vs. Home Team: " + row['Home'])
        if not row['Home'] in finalDf.index or not row['Away'] in finalDf.index: # if the team is not in the finalDf
            continue
        
        away_team_prob, home_team_prob, draw_prob, confidence = predictingResults(row['Away'], row['Home'], finalDf) # get the odds for the match
        
        my_prediction = getFTR(away_team_prob, home_team_prob, draw_prob) # get the FTR from my prediction
        print(confidence)
        if confidence < 3: # if the confidence is less than 50%
            print("Confidence is less than 50%")
            continue
        
        actual_result = row['FTR'] # get the FTR from the actual match
        
        # print("My Prediction: " + my_prediction)
        # print("Actual Result: " + actual_result)
        
        if my_prediction == actual_result:
            correct += 1
        else:
            incorrect += 1
            
    print("Correct: " + str(correct))
    print("Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct / (correct + incorrect)))
    
def decimalOddsToAmericanOdds(decimalOdds):
    if decimalOdds < 2:
        return int(-100 / (decimalOdds - 1))
    else:
        return int((decimalOdds - 1) * 100)

def backtestWithOdds(finalDf):
    
    matches = pd.read_csv("data/EPLOddsAll.csv") # get matches from EPLOddsAll.csv
    
    correct = incorrect = 0
    total_bal = 500
    # lazy_bal = 1000
    
    finalDf = finalDf.set_index('Teams')
    for index, row in matches.iterrows():
        if not row['HomeTeam'] in finalDf.index or not row['AwayTeam'] in finalDf.index: # if the team is not in the finalDf
            continue
        # print("Match " + str(index + 1) + " of " + str(len(matches)))
        # print("Away Team: " + row['Away'] + " vs. Home Team: " + row['Home'])
        
        away_team_prob, home_team_prob, draw_prob, confidence = predictingResults(row['AwayTeam'], row['HomeTeam'], finalDf)
        
        predicted_result = getFTR(away_team_prob, home_team_prob, draw_prob) # get the FTR from my prediction
        
        actual_result = row['FTR'] # get the FTR from the actual match
        
        # get odds from B365H, B365D, B365A
        away_oods = decimalOddsToAmericanOdds(row['B365A'])
        home_odds = decimalOddsToAmericanOdds(row['B365H'])
        draw_odds = decimalOddsToAmericanOdds(row['B365D'])
        
        if confidence < 3.5:
            if predicted_result == actual_result:
                if predicted_result == 'A':
                    lazy_bal += getWinnings(away_oods, 100)
                elif predicted_result == 'H':
                    lazy_bal += getWinnings(home_odds, 100)
                else:
                    lazy_bal += getWinnings(draw_odds, 100)
            else:
                print(confidence)
                incorrect += 1
                if predicted_result == 'A':
                    lazy_bal -= 100
                elif predicted_result == 'H':
                    lazy_bal -= 100
                else:
                    lazy_bal -= 100
            
            if predicted_result == "A":
                if away_oods > 200:
                    print("")
                else:
                    continue
            if predicted_result == "H":
                if home_odds > 200:
                    print("")
                else:
                    continue
            if predicted_result == "D":
                if draw_odds > 200:
                    print("")
                else:
                    continue
        
        # calculate the total bal after betting
        if predicted_result == actual_result:
            if predicted_result == 'A':
                total_bal += getWinnings(away_oods, 100)
                lazy_bal += getWinnings(away_oods, 100)
            elif predicted_result == 'H':
                total_bal += getWinnings(home_odds, 100)
                lazy_bal += getWinnings(home_odds, 100)
            else:
                total_bal += getWinnings(draw_odds, 100)
                lazy_bal += getWinnings(draw_odds, 100)
            correct += 1
        else:
            print(confidence)
            incorrect += 1
            if predicted_result == 'A':
                total_bal -= 100
                lazy_bal -= 100
            elif predicted_result == 'H':
                total_bal -= 100
                lazy_bal -= 100
            else:
                total_bal -= 100
                lazy_bal -= 100
        
    print("Total Balance: " + str(total_bal))
    print("Lazy Balance: " + str(lazy_bal))
    print("Correct: " + str(correct))
    print("Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct / (correct + incorrect)))
    

def getWinnings(odds, value):
    if odds < 0:
        return (100 / abs(odds)) * value
    else:
        return (odds / 100) * value

def getFTR(away_team_prob, home_team_prob, draw_prob):
    # get the FTR from my prediction
    if away_team_prob > home_team_prob and away_team_prob > draw_prob:
        return 'A'
    elif home_team_prob > away_team_prob and home_team_prob > draw_prob:
        return 'H'
    else:
        return 'D'

def daily(past, current):
    initFirebaseDatabase()
    
    if past:
        downloadPastData("https://fbref.com/en/comps/9/", "Premier-League-Stats", "pastDataPremierLeague.csv")
    
    buildData("https://fbref.com/en/comps/9/Premier-League-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/premier-league/", "finalDataframePremierLeague.csv", "pastDataPremierLeague.csv", "currentDataPremierLeague.csv", current)
    # buildData("https://fbref.com/en/comps/12/La-Liga-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/la-liga/", "finalDataframeLaLiga.csv", "pastDataLaLiga.csv", "currentDataLaLiga.csv", True)
    # buildData("https://fbref.com/en/comps/11/Serie-A-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/serie-a/", "finalDataframeSerieA.csv", "pastDataSerieA.csv", "currentDataSerieA.csv", True)
    # buildData("https://fbref.com/en/comps/20/Bundesliga-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/bundesliga/", "finalDataframeBundesliga.csv", "pastDataBundesliga.csv", "currentDataBundesliga.csv", True)
    # buildData("https://fbref.com/en/comps/13/Ligue-1-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/ligue-1/", "finalDataframeLigue1.csv", "pastDataLigue1.csv", "currentDataLigue1.csv", True)

    bigFive = [['finalDataframePremierLeague.csv', 'PremierLeagueAnalysis']]
            #    ['finalDataframeLaLiga.csv', 'LaLigaAnalysis'],
            #    ['finalDataframeSerieA.csv', 'SerieAAnalysis'],
            #    ['finalDataframeBundesliga.csv', 'BundesligaAnalysis'],
            #    ['finalDataframeLigue1.csv', 'Ligue1Analysis']]
    
    for i in bigFive:
        print(f"Uploading {i[0]} to database...")
        df = pd.read_csv("data/" +  i[0]).T
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace = True)
        uploadToDatabase(df, i[1])

# downloadPastData("https://fbref.com/en/comps/9/", "Premier-League-Stats", "pastDataPremierLeague.csv")
# downloadPastData("https://fbref.com/en/comps/12/", "La-Liga-Stats", "pastDataLaLiga.csv")
# downloadPastData("https://fbref.com/en/comps/11/", "Serie-A-Stats", "pastDataSerieA.csv")
# downloadPastData("https://fbref.com/en/comps/20/", "Bundesliga-Stats", "pastDataBundesliga.csv")
# downloadPastData("https://fbref.com/en/comps/13/", "Ligue-1-Stats", "pastDataLigue1.csv")
#buildData("https://fbref.com/en/comps/9/Premier-League-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/premier-league/", "finalDataframePremierLeague.csv", "pastDataPremierLeague.csv", "currentDataPremierLeague.csv", True)
#buildData("https://fbref.com/en/comps/12/La-Liga-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/la-liga/", "finalDataframeLaLiga.csv", "pastDataLaLiga.csv", "currentDataLaLiga.csv", True)
#buildData("https://fbref.com/en/comps/11/Serie-A-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/serie-a/", "finalDataframeSerieA.csv", "pastDataSerieA.csv", "currentDataSerieA.csv", True)
#buildData("https://fbref.com/en/comps/20/Bundesliga-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/bundesliga/", "finalDataframeBundesliga.csv", "pastDataBundesliga.csv", "currentDataBundesliga.csv", True)
#buildData("https://fbref.com/en/comps/13/Ligue-1-Stats", "https://projects.fivethirtyeight.com/soccer-predictions/ligue-1/", "finalDataframeLigue1.csv", "pastDataLigue1.csv", "currentDataLigue1.csv", True)
# soccerDataFBRef()

# initFirebaseDatabase()

# bigFive = [['finalDataframePremierLeague.csv', 'PremierLeagueAnalysis'],
#            ['finalDataframeLaLiga.csv', 'LaLigaAnalysis'],
#            ['finalDataframeSerieA.csv', 'SerieAAnalysis'],
#            ['finalDataframeBundesliga.csv', 'BundesligaAnalysis'],
#            ['finalDataframeLigue1.csv', 'Ligue1Analysis']]
    
# for i in bigFive:
#     print(f"Uploading {i[0]} to database...")
#     df = pd.read_csv("data/" +  i[0]).T
#     df.columns = df.iloc[0]
#     df.drop(df.index[0], inplace = True)
#     uploadToDatabase(df, i[1])
#     exit()


#daily()

# daily(False, False)
# get finalDataframePremierLeague.csv final as a dataframe
# df = pd.read_csv("data/finalDataframePremierLeague.csv").T
# predictingResults("Everton", "Manchester City", pd.read_csv("data/finalDataframePremierLeague.csv")) 
# backtest(pd.read_csv("data/finalDataframePremierLeague.csv"))
# backtestWithOdds(pd.read_csv("data/finalDataframePremierLeague.csv"))

initFirebaseDatabase()

picks = getPicks(["soccer_epl"], pd.read_csv("data/finalDataframePremierLeague.csv"))

writePicksToDB(picks)

print("Code completed.")
