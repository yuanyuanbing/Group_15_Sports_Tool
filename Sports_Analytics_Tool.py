# -*- coding: utf-8 -*-
"""
Group 15

"""

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder  
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#### Descriptive @ author: Ma Mengliang; Wang xinye ####

datasetPath = 'dataset.csv'
defenderPath = 'Defender.csv'
forwardPath =  'Forward.csv'
goalkeeperPath = 'Goalkeeper.csv'
midfielderPath = 'Midfielder.csv'

head_row = pd.read_csv(datasetPath, nrows=0)
head_row_list = list(head_row)


datasetResult = pd.read_csv(datasetPath, usecols=head_row_list)

other_row = pd.read_csv(defenderPath, nrows=0)
other_row_list = list(other_row)

defenderResult = pd.read_csv(defenderPath, usecols=other_row_list)
forwardResult = pd.read_csv(forwardPath, usecols=other_row_list)
goalkeeperResult = pd.read_csv(goalkeeperPath, usecols=other_row_list)
midfielderResult = pd.read_csv(midfielderPath, usecols=other_row_list)

def getClubList(clubList):
    return list(set(clubList))

def initClubSelection():
    st.title('Historical Data Analysis')
    st.header('Player and Club profile')
    # get club information
    clubList = getClubList(datasetResult['Club'].values.tolist())
    option = st.selectbox('Choose Club', clubList)
    df = datasetResult[(datasetResult['Club'] == option)]
    firstTable(df)

def firstTable(dfTmp):  # player & club profile ①
    df = dfTmp[['Name', 'Club', 'Age', 'Goals per match', 'Shooting accuracy', 'Big chances missed per match', 'Assists per match',
                  'Passes per match', 'Big chances created per match', 'Accurate long balls per match']]


    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False, wrapText=False, autoHeight=True)
    grid_options = options_builder.build()
    AgGrid(df, grid_options, theme='streamlit')

def getClusteredColumnData():
    df = datasetResult[['Club', 'Wins', 'Losses']]
    arr = df.values.tolist()
    obj = {}
    for item in arr:
        if item[0] in obj.keys():
            obj[item[0]][0] += item[1]  
            obj[item[0]][1] += item[2]  
        else:
            obj[item[0]] = item[1:]  # initialize win and loss
            
    finalArr = []
    for key, value in obj.items():
        arrTmp1 = [key] + ['Wins'] + [value[0]]
        finalArr.append(arrTmp1)
        arrTmp2 = [key] + ['Losses'] + [value[1]]
        finalArr.append(arrTmp2)

    finalDf = pd.DataFrame(finalArr, columns=['Club', 'result', 'count'])
    return finalDf

def drawClusteredColumn():
    df = getClusteredColumnData()
    df.head()
    fig = px.bar(
        df,
        x="Club",
        y="count",
        color="result",
        title="Total win / loss chart",
        barmode='group',
    )
    st.plotly_chart(fig)

def initPositionSelection():
    option = st.selectbox('Choose Position', ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'])
    playerList = []
    df = pd.DataFrame()

    if option == 'Forward':
        playerList = forwardResult['Name'].values.tolist()
        df = forwardResult
    elif option == 'Midfielder':
        playerList = midfielderResult['Name'].values.tolist()
        df = midfielderResult
    elif option == 'Defender':
        playerList = defenderResult['Name'].values.tolist()
        df = defenderResult
    elif option == 'Goalkeeper':
        playerList = goalkeeperResult['Name'].values.tolist()
        df = goalkeeperResult

    initPlayerSelection(playerList, option, df)

def getConfigure(position):
    forwardConf = ['Goals per match', 'Shooting accuracy', 'Big chances missed per match', 'Assists per match', 'Big chances created per match']
    midfielderConf = ['Goals per match', 'Shooting accuracy', 'Big chances missed per match', 'Tackle success %', 'Successful 50/50s per match', 'Assists per match', 'Big chances created per match', 'Passes per match', 'Accurate long balls per match']
    defenderConf = ['Goals conceded per match', 'Tackle success %', 'Successful 50/50s per match', 'Big chances created per match', 'Accurate long balls per match']
    goalkeeperConf = ['Goals conceded  per match', 'Saves per match', 'Penalties saved per match']
    conf = []

    if position == 'Forward':
        conf = forwardConf
    elif position == 'Midfielder':
        conf = midfielderConf
    elif position == 'Defender':
        conf = defenderConf
    elif position == 'Goalkeeper':
        conf = goalkeeperConf

    return conf

def initPlayerSelection(playerList, position, df):
    option = st.selectbox('Choose Player', playerList)
    conf = getConfigure(position)
    conf.insert(0, 'Name')  
    tmpArr = df[conf].values.tolist()
    abilityList = []
    for item in tmpArr:
        if item[0] == option:
            abilityList = item
    drawRadarChart(abilityList[1:], conf[1:], position)
    getTop10(df[conf], position, conf[1:])

def getTop10(df, position, capabilities):
    st.header('Top10 ' + position + ' with higher + Kpi')
    option = st.selectbox('Please select a comparison capability', capabilities)
    df.sort_values(by=option, inplace=True, ascending=False)
    df = df.head(10)

    mid = df[option]  
    df.pop(option)  
    df.insert(1, option, mid) 

    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.configure_default_column(editable=False, wrapText=False)
    grid_options = options_builder.build()
    AgGrid(df, grid_options, theme='streamlit')


def drawRadarChart(abilityList, conf, position):
    if position != 'Goalkeeper':
        abilityList[1] = float(abilityList[1].strip('%')) / 100
    fig = go.Figure(data=go.Scatterpolar(
        r=abilityList,
        theta=conf,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )
    st.plotly_chart(fig)
    
    
#### Prdictive  @author: Yu zhen ####

def predictive():
    # ==============data processing===============
    player_datasets = pd.read_excel('player_datasets.xlsx')
    idx_list = []
    for idx, i in enumerate(player_datasets.isnull().sum(axis=0) / player_datasets.shape[0]):
        if i >= 0.3:
            idx_list.append(player_datasets.columns[idx])
    
    player_datasets.drop(labels=idx_list, axis=1, inplace=True)
    player_datasets.dropna(how='any', inplace=True)
    
    exclude_list = [
        'Name', 'Jersey Number', 'Goals', 'Nationality', 'Club', 'Shooting accuracy %',
        'Position', 'Headed goals', 'Goals with right foot', 'Goals with left foot',
        ]
    filt_cols = [col for col in player_datasets.columns if 'per' in col]
    
    var_list = [var for var in player_datasets.columns if var not in exclude_list and var not in filt_cols]
    X, y = player_datasets.loc[:, var_list], player_datasets['Goals']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
        )
    # ====================end===================
    
    # ================model building==============
    @st.cache(allow_output_mutation=True)
    def GBDT_Regression(params):
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
        mse = mean_squared_error(y_test, reg.predict(X_test))
        return reg, mse
    # ============================================
    
    st.title('Premier League Player Goals Prediction')
    
    option = st.selectbox('Please choose a player', player_datasets['Name'].unique())
    info = player_datasets.loc[player_datasets['Name']==option, :]
    # st.write(info)
    
    def update_play_metric(pre_goals):
        col1, col2, col3 = st.columns(3)
        col1.metric("Goals Prediction", pre_goals)
        # col2.metric("Wind", "9 mph", "-8%")
        # col3.metric("Humidity", "86%", "4%")
    
    st.header('GBDT Regression')
    
    st.sidebar.header('Model parameter setting')
    n_estimators = st.sidebar.slider(
        'Set the number of tree in the GBDT model(n_estimators)',
        min_value=5, max_value=500, value=300, step=5,
        # on_change=st.warning('Running model(...)'),
        )
    min_samples_split = st.sidebar.slider(
        'The minimum number of samples required to split an internal node:(min_samples_split)',
        min_value=2, max_value=10, value=5, step=1,
        # on_change=st.warning(''),
        )
    max_depth = st.sidebar.number_input(
        'The maximum depth of the tree:(max_depth)',
        min_value=1, max_value=5, value=4, step=1,
        # on_change=st.warning(''),
        )
    loss = st.sidebar.selectbox(
        'Set the loss function to be optimized',
        ['ls', 'lad', 'huber', 'quantile'],
        # on_change=st.warning(''),
        )
    criterion = st.sidebar.selectbox(
        'Set the loss function to be optimized',
        ['friedman_mse', 'mse', 'mae'],
        # on_change=st.warning(''),
        )
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "learning_rate": 0.01,
        "loss": loss,
        "criterion": criterion,
    }
    
    # update model result
    reg, mse = GBDT_Regression(params)
    # default sort on the columns name order
    feature_importance = reg.feature_importances_
    # argsort func return the element's index
    sorted_idx = list(np.argsort(feature_importance))
    key_feature = np.array(X_train.columns)[sorted_idx][::-1][:5]
    
    st.subheader("Player's key performance statistics")
    exp = st.expander(
        "Reset statistic to review the player's new Goals Prediction",
        expanded=True
        )
    for f in key_feature:
        if type(max(info[f])) == int:
            val = exp.number_input(f, min_value=0, max_value=max(X[f])+1, value=max(info[f]))
            info[f] = val
        elif type(max(info[f])) == float:
            val = exp.slider(f, min_value=0.0, max_value=float(max(X[f])+1), value=float(max(info[f])), format='%f')
            info[f] = val
    pre_goals = reg.predict(info[X.columns])
    update_play_metric(int(pre_goals))
    
    st.write("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    
    st.header('Model Performance')
    # =========================plot deviance=====================
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    # staged_predict: Predict regression target at each stage for X.
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)
    
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    st.pyplot(fig=fig)
    # ========================= end =====================
    
    # ======================plot feature importance==================
    # default sort on the columns name order
    feature_importance = reg.feature_importances_
    # get the decending order index
    
    # argsort func return the element's index
    sorted_idx = np.argsort(feature_importance)
    
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(1, 1, 1)
    
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    # reorder the data columns name by sorted_idx
    plt.yticks(pos, np.array(X_train.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")
    fig.tight_layout()
    st.pyplot(fig=fig)
    
    result = permutation_importance(
        reg, X_test, y_test, n_repeats=10, random_state=0, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(1, 1, 1)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(X_train.columns)[sorted_idx],
    )
    
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    st.pyplot(fig=fig)
    # ======================== end =======================
    
    # ========================plot y_predict==================
    y_pre = reg.predict(X)
    y_sort = np.sort(y)
    y_pre_sort = np.sort(y_pre)
    
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(111)
    plt.scatter(np.arange(len(y)), y_sort, label='Goals')
    plt.plot(np.arange(len(y)), y_pre_sort, label='Goals predict', c='r', ls='--')
    
    plt.xlabel('observatioins')
    plt.ylabel('Goals')
    plt.title('The predict vs real Goals value')
    plt.legend(loc='best')
    plt.tight_layout()
    st.pyplot(fig=fig)
    # =========================end=========================
    

#### Prescriptive @author: Chen Bing ####

def find_alternative():
    st.title('Find Alternative Players') 
    
    position_option = st.selectbox('Choose Position', ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'])
    
    st.header('Choose one from all players to find his alternative')
    data(position_option)

def data(p_option):
    if p_option == 'Forward':
        df = pd.read_csv('Forward.csv')
        df.fillna(0, inplace = True)
        df.rename(columns={'Goals per match':'Goals_per_match',
                           'Shooting accuracy':'Shooting_Accuracy',
                           'Big chances missed per match':'Big_chances_missed_per_match',
                           'Assists per match':'Assists_per_match',
                           'Big chances created per match':'Big_chances_created_per_match'} ,inplace = True)
        for i in df.index:
            df['Shooting_Accuracy'][i]= float(df['Shooting_Accuracy'][i].strip('%')) / 100
        df_used = df[['Name', 'Club', 'Age', 'Goals_per_match', 'Shooting_Accuracy', 'Big_chances_missed_per_match', 'Assists_per_match',
            'Big_chances_created_per_match']]
        df_used
        
        player_option = st.selectbox('Choose Player', df_used.Name, key = df_used.index)
        
        record = df_used.loc[df_used['Name'] == player_option]
     
        kpi_selection_forward(df_used, record)
        
    elif p_option == 'Midfielder':
        df = pd.read_csv('Midfielder.csv')
        df.fillna(0, inplace = True)
        df.rename(columns={'Goals per match':'Goals_per_match',
                           'Shooting accuracy':'Shooting_Accuracy',
                           'Big chances missed per match':'Big_chances_missed_per_match',
                           'Assists per match':'Assists_per_match',
                           'Passes per match':'Passes_per_match',
                           'Big chances created per match':'Big_chances_created_per_match',
                           'Accurate long balls per match':'Accurate_long_balls_per_match'} ,inplace = True)
        for i in df.index:
            df['Shooting_Accuracy'][i]= float(df['Shooting_Accuracy'][i].strip('%')) / 100
        df_used = df[['Name', 'Club', 'Age', 'Goals_per_match', 'Shooting_Accuracy', 'Big_chances_missed_per_match', 'Assists_per_match',
            'Big_chances_created_per_match', 'Passes_per_match', 'Accurate_long_balls_per_match']]
        df_used
        
        player_option = st.selectbox('Choose Player', df_used.Name, key = df_used.index)
        
        record = df_used.loc[df_used['Name'] == player_option]
     
        kpi_selection_midfielder(df_used, record)
        
    elif p_option == 'Defender':
        df = pd.read_csv('Defender.csv')
        df.fillna(0, inplace = True)
        df.rename(columns={'Goals conceded per match':'Goals_conceded_per_match',
                           'Tackle success %':'Tackle_success',
                           'Successful 50/50s per match':'Successful_vs_per_match',
                           'Accurate long balls per match':'Accurate_long_balls_per_match'} ,inplace = True)
        for i in df.index:
            df['Tackle_success'][i]= float(df['Tackle_success'][i].strip('%')) / 100
        df_used = df[['Name', 'Club', 'Age', 'Goals_conceded_per_match', 'Tackle_success', 'Successful_vs_per_match',
            'Accurate_long_balls_per_match']]
        df_used
        
        player_option = st.selectbox('Choose Player', df_used.Name, key = df_used.index)
        
        record = df_used.loc[df_used['Name'] == player_option]
     
        kpi_selection_defender(df_used, record)
        
    elif p_option == 'Goalkeeper':
        df = pd.read_csv('Goalkeeper.csv')
        df.fillna(0, inplace = True)
        df.rename(columns={'Goals conceded per match':'Goals_conceded_per_match',
                           'Saves per match':'Saves_per_match',
                           'Penalties saved per match':'Penalties_saved_per_match'} ,inplace = True)
        df_used = df[['Name', 'Club', 'Age', 'Goals_conceded_per_match', 'Saves_per_match', 'Penalties_saved_per_match']]
        df_used
        
        player_option = st.selectbox('Choose Player', df_used.Name, key = df_used.index)
        
        record = df_used.loc[df_used['Name'] == player_option]
     
        kpi_selection_goalkeeper(df_used, record)
    
def kpi_selection_forward(df_used, record):
    
        st.subheader('The player performance information') 
        record
        
        temp_name = record.Name.values.tolist()[0]
        temp_club = record.Club.values.tolist()[0]
        temp_gpm = record.Goals_per_match.values.tolist()[0]
        temp_sa = record.Shooting_Accuracy.values.tolist()[0]
        temp_bcmpm = record.Big_chances_missed_per_match.values.tolist()[0]
        temp_apm = record.Assists_per_match.tolist()[0]
        temp_bccpm = record.Big_chances_created_per_match.tolist()[0]
        
        st.subheader('Set Criteria')
        
        age_slider = st.slider('Select the desired player age', 18.0, 35.0, 20.0, 1.0)
        
        kpi_select = st.multiselect('Decide KPI',['goals per match (+/- 0.05)', 
                                                                  'shooting accuracy (+/- 0.05)', 
                                                                  'big chance missed per match (+/- 0.05)', 
                                                                  'assists per match (+/- 0.05)',
                                                                  'big chance created per match (+/- 0.05)'])

        st.subheader('Potential alternative players performance information')
        # 1
        if kpi_select == ['goals per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 2
        elif kpi_select == ['shooting accuracy (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 3
        elif kpi_select == ['big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 4           
        elif kpi_select == ['assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 5
        elif kpi_select == ['big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 12 
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 13
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 14
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 15
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 23
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 24
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 25
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 34
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 35
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 45
        elif kpi_select == ['assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 123
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 124
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 125
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 134
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1  & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 135
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 145
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1  & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 234
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif  condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 235
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 245
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 345
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 1234
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 1235
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 &condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 1245
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 1345
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 2345
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
        # 12345
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Big_chances_created_per_match']
                   
def kpi_selection_midfielder(df_used, record):
    
        st.subheader('The player performance information') 
        record
        
        temp_name = record.Name.values.tolist()[0]
        temp_club = record.Club.values.tolist()[0]
        temp_gpm = record.Goals_per_match.values.tolist()[0]
        temp_sa = record.Shooting_Accuracy.values.tolist()[0]
        temp_bcmpm = record.Big_chances_missed_per_match.values.tolist()[0]
        temp_apm = record.Assists_per_match.tolist()[0]
        temp_bccpm = record.Big_chances_created_per_match.tolist()[0]
        temp_ppm = record.Passes_per_match.tolist()[0]
        temp_albpm = record.Accurate_long_balls_per_match.tolist()[0]
        
        st.subheader('Set Criteria')
        
        age_slider = st.slider('Select the desired player age', 18.0, 35.0, 20.0, 1.0)
        
        kpi_select = st.multiselect('Decide KPI',['goals per match (+/- 0.05)', 
                                                                  'shooting accuracy (+/- 0.05)', 
                                                                  'big chance missed per match (+/- 0.05)', 
                                                                  'assists per match (+/- 0.05)',
                                                                  'big chance created per match (+/- 0.05)',
                                                                  'passes per match (+/- 10)',
                                                                  'accurate long balls per match (+/- 0.05)'])

        st.subheader('Potential alternative players performance information')
        # 1
        if kpi_select == ['goals per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2
        elif kpi_select == ['shooting accuracy (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 3
        elif kpi_select == ['big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 4           
        elif kpi_select == ['assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 5
        elif kpi_select == ['big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
        
        # 6
        elif kpi_select == ['passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
        # 7
        elif kpi_select == ['accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12 
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 14
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 15
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 16
        elif kpi_select == ['goals per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 17
        elif kpi_select == ['goals per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))               
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 24
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 25
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 26
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 27
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))               
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 34
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 35
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 36
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                
        # 37
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))               
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 45
        elif kpi_select == ['assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 46
        elif kpi_select == ['assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 47
        elif kpi_select == ['assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 56
        elif kpi_select == ['big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))                
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 57
        elif kpi_select == ['big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))                 
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 67
        elif kpi_select == ['passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_6 = ((temp_ppm - 0.05)<= df_used['Passes_per_match'][i] <= (temp_ppm + 0.05))   
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 123
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 124
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 125
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 126
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 127
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 134
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1  & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 135
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 136
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 137
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 145
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1  & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 146
        elif kpi_select == ['goals per match (+/- 0.05)','assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 147
        elif kpi_select == ['goals per match (+/- 0.05)','assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 156
        elif kpi_select == ['goals per match (+/- 0.05)','big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 157
        elif kpi_select == ['goals per match (+/- 0.05)','big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 167
        elif kpi_select == ['goals per match (+/- 0.05)','passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 234
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif  condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 235
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 236
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 237
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 245
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 246
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 247
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 256
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 257
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 267
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 345
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 346
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 347
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 356
        elif kpi_select == ['big chance missed per match (+/- 0.05)','big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 357
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 367
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 456
        elif kpi_select == ['assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 457
        elif kpi_select == ['assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 467
        elif kpi_select == ['assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 567
        elif kpi_select == ['big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1234
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1235
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 &condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1236
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1237
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
        # 1245
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
        
        # 1246
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1247
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1256
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1257
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1267
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1345
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match'] 
        
        # 1346
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1347
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 0.05)<= df_used['Passes_per_match'][i] <= (temp_ppm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1356
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1357
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1456
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1457
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1567
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2345
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2346
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2347
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2356
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2357
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2367
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2456
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2457
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif  condition_2 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2467
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2567
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 3456
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 3457
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 3567
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 4567
        elif kpi_select == ['assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12345
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12346
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12347
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12356
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12357
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12367
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12456
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12457
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12467
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 12567
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13456
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13457
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13467
        elif kpi_select == ['goals per match (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13567
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 14567
        elif kpi_select == ['goals per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23456
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23457
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23467
        elif kpi_select == ['shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23567
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 24567
        elif kpi_select == ['shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 34567
        elif kpi_select == ['big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 123456
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & condition_6 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 123457
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 123467
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 123567
        elif kpi_select == ['goals per match (+/- 0.05)', 'shooting accuracy (+/- 0.05)', 'big chance missed per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 124567
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 134567
        elif kpi_select == ['goals per match (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 1234567
        elif kpi_select == ['goals per match (+/- 0.05)','shooting accuracy (+/- 0.05)','big chance missed per match (+/- 0.05)', 'assists per match (+/- 0.05)', 'big chance created per match (+/- 0.05)', 'passes per match (+/- 0.05)', 'accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gpm - 0.05)<= df_used['Goals_per_match'][i] <= (temp_gpm + 0.05))
                condition_2 = ((temp_sa - 0.05)<= df_used['Shooting_Accuracy'][i] <= (temp_sa + 0.05))
                condition_3 = ((temp_bcmpm - 0.05)<= df_used['Big_chances_missed_per_match'][i] <= (temp_bcmpm + 0.05))
                condition_4 = ((temp_apm - 0.05)<= df_used['Assists_per_match'][i] <= (temp_apm + 0.05))
                condition_5 = ((temp_bccpm - 0.05)<= df_used['Big_chances_created_per_match'][i] <= (temp_bccpm + 0.05))
                condition_6 = ((temp_ppm - 10)<= df_used['Passes_per_match'][i] <= (temp_ppm + 10))
                condition_7 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & condition_6 & condition_7 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
def kpi_selection_defender(df_used, record):
        st.subheader('The player performance information') 
        record
        
        temp_name = record.Name.values.tolist()[0]
        temp_club = record.Club.values.tolist()[0]
        temp_gcpm = record.Goals_conceded_per_match.values.tolist()[0]
        temp_ts = record.Tackle_success.values.tolist()[0]
        temp_vspm = record.Successful_vs_per_match.values.tolist()[0]
        temp_albpm = record.Accurate_long_balls_per_match.tolist()[0]
        
        st.subheader('Set Criteria')
        
        age_slider = st.slider('Select the desired player age', 18.0, 35.0, 20.0, 1.0)
        
        kpi_select = st.multiselect('Decide KPI',['Goals conceded per match (+/- 0.05)', 
                                                                  'Tackle success (+/- 0.05)', 
                                                                  'Successful vs per match (+/- 0.05)', 
                                                                  'Accurate long balls per match (+/- 0.05)'])

        st.subheader('Potential alternative players performance information')
        # 1
        if kpi_select == ['Goals conceded per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 2
        elif kpi_select == ['Tackle success (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 3
        elif kpi_select == ['Successful vs per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 4           
        elif kpi_select == ['assists per match (+/- 0.05)']:
            for i in df_used.index:
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
                   
        # 12 
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Tackle success (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 13
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Successful vs per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 14
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 23
        elif kpi_select == ['Tackle success (+/- 0.05)', 'Successful vs per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 24
        elif kpi_select == ['Tackle success (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 34
        elif kpi_select == ['Successful vs per match (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
                   
        # 123
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Tackle success (+/- 0.05)', 'Successful vs per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 124
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Tackle success (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 134
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Successful vs per match (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
        # 234
        elif kpi_select == ['Tackle success (+/- 0.05)', 'Successful vs per match (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                       
        # 1234
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Tackle success (+/- 0.05)', 'Successful vs per match (+/- 0.05)', 'Accurate long balls per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 0.05)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 0.05))
                condition_2 = ((temp_ts - 0.05)<= df_used['Tackle_success'][i] <= (temp_ts + 0.05))
                condition_3 = ((temp_vspm - 0.05)<= df_used['Successful_vs_per_match'][i] <= (temp_vspm + 0.05))
                condition_4 = ((temp_albpm - 0.05)<= df_used['Accurate_long_balls_per_match'][i] <= (temp_albpm + 0.05))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & condition_4 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Accurate_long_balls_per_match']
                   
def kpi_selection_goalkeeper(df_used, record): 
     
        st.subheader('The player performance information') 
        record
        
        temp_name = record.Name.values.tolist()[0]
        temp_club = record.Club.values.tolist()[0]
        temp_gcpm = record.Goals_conceded_per_match.values.tolist()[0]
        temp_spm = record.Saves_per_match.values.tolist()[0]
        temp_pspm = record.Penalties_saved_per_match.tolist()[0]
        
        st.subheader('Set Criteria')
        
        age_slider = st.slider('Select the desired player age', 18.0, 35.0, 20.0, 1.0)
        
        kpi_select = st.multiselect('Decide KPI',['Goals conceded per match (+/- 0.05)', 
                                                                  'Saves per match (+/- 0.05)', 
                                                                  'Penalties saved per match (+/- 0.05)'])

        st.subheader('Potential alternative players performance information')
        # 1
        if kpi_select == ['Goals conceded per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 1)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 1))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
                   
        # 2
        elif kpi_select == ['Saves per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_spm - 1)<= df_used['Saves_per_match'][i] <= (temp_spm + 1))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
                   
        # 3
        elif kpi_select == ['Penalties saved per match (+/- 0.05)']:
            for i in df_used.index:
                condition_3 = ((temp_pspm - 1)<= df_used['Penalties_saved_per_match'][i] <= (temp_pspm + 1))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']             
                   
        # 12 
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Saves per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 1)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 1))
                condition_2 = ((temp_spm - 1)<= df_used['Saves_per_match'][i] <= (temp_spm + 1))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
                   
        # 13
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Penalties saved per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 1)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 1))
                condition_3 = ((temp_pspm - 0.01)<= df_used['Penalties_saved_per_match'][i] <= (temp_pspm + 0.01))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
                   
        # 23
        elif kpi_select == ['Saves per match (+/- 0.05)', 'Penalties saved per match (+/- 0.05)']:
            for i in df_used.index:
                condition_2 = ((temp_spm - 1)<= df_used['Saves_per_match'][i] <= (temp_spm + 1))
                condition_3 = ((temp_pspm - 0.01)<= df_used['Penalties_saved_per_match'][i] <= (temp_pspm + 0.01))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
                   
                   
        # 123
        elif kpi_select == ['Goals conceded per match (+/- 0.05)', 'Saves per match (+/- 0.05)', 'Penalties saved per match (+/- 0.05)']:
            for i in df_used.index:
                condition_1 = ((temp_gcpm - 1)<= df_used['Goals_conceded_per_match'][i] <= (temp_gcpm + 1))
                condition_2 = ((temp_spm - 1)<= df_used['Saves_per_match'][i] <= (temp_spm + 1))
                condition_3 = ((temp_pspm - 0.01)<= df_used['Penalties_saved_per_match'][i] <= (temp_pspm + 0.01))
                if (df_used['Name'][i] == temp_name) | (df_used['Club'][i] == temp_club):
                    continue
                elif condition_1 & condition_2 & condition_3 & (df_used['Age'][i] == age_slider):
                   df_used.loc[[i], 'Name':'Penalties_saved_per_match']
    


if __name__=="__main__":
    
    st.sidebar.title('Premier League Players data analysis (last 10 seasons)')
    side_opt = st.sidebar.selectbox('Choose Analytics type', ['Descriptive', 'Predictive', 'Prescriptive'])
    
    if side_opt == 'Descriptive':
       initClubSelection()
       drawClusteredColumn()
       initPositionSelection()
    if side_opt == 'Predictive':
       predictive()
    if side_opt == 'Prescriptive':
       find_alternative()


    
