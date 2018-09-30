#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #2 Main File
"""

import random
import numpy as np
import pandas as pd
import summary_stats
import matplotlib.pyplot as plt
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 1000)
plt.rcParams['figure.figsize'] = (15, 5)


def main():
    testing_df = pd.read_csv('input/test.csv')
    training_df = pd.read_csv('input/train.csv')
    combined_df = pd.concat([training_df, testing_df], sort=False)

    # task_3_5(training_df, 'Training data-set')
    # task_3_5(combined_df, 'Combined data-set')
    # task_3_7(training_df)
    # task_3_8(training_df)
    # task_3_9(training_df)
    # task_3_10(training_df)
    # task_3_11(training_df)
    # task_3_12(training_df)
    # task_3_13(training_df)
    # task_3_14(training_df)
    # task_3_15(combined_df)
    # task_3_16(training_df)
    # task_3_17(training_df)
    # task_3_18(training_df)
    # task_3_19(testing_df)
    # task_3_20(training_df)

    # print(testing_df)
    # print(training_df)
    # print(combined_df)


def task_3_5(frame, frame_name='given data-frame'):
    print('The ' + frame_name + ' has null values in the following features:')
    result = pd.DataFrame(frame.isnull().any(), columns=['nulls'])
    print(result[result['nulls']].index.values)
    print("")


def task_3_7(frame):
    numerical_sets = [frame['Age'].dropna(), frame['SibSp'], frame['Parch'], frame['Fare']]
    for data_frame in numerical_sets:
        print("Showing Statistics for numerical feature:", data_frame.name)
        summary_stats.df_print_summary_stats(data_frame, show_values=False, count_values=True)
        print("")


def task_3_8(frame):
    categorical_sets = [frame['PassengerId'],
                        frame['Survived'],
                        frame['Pclass'],
                        frame['Sex'],
                        frame['Ticket'],
                        frame['Cabin'].dropna(),
                        frame['Embarked'].dropna()]
    for data_frame in categorical_sets:
        print("Showing Statistics for categorical feature:", data_frame.name)
        print("Count:", data_frame.count())
        print("Unique:", data_frame.nunique())
        print("Top:", *data_frame.value_counts()[:1].index)
        print("Freq:", data_frame.value_counts().max())
        print("")


def task_3_9(frame):
    sub_frame = frame[['Pclass', 'Survived']]
    is_pclass_1 = []
    for i in range(len(sub_frame)):
        if sub_frame['Pclass'][i] == 1:
            is_pclass_1.append(1)
        else:
            is_pclass_1.append(0)
    sub_frame = sub_frame.assign(Is_Pclass_1=is_pclass_1)
    print("Correlation between Pclass==1 and Survived")
    print(sub_frame['Is_Pclass_1'].corr(sub_frame['Survived']))
    print()


def task_3_10(frame):
    men = frame[frame['Sex'] == 'male']
    women = frame[frame['Sex'] == 'female']
    m_s_r = len(men[men['Survived'] == 1]) / len(men)
    w_s_r = len(women[women['Survived'] == 1]) / len(women)
    print("Men's Survival Rate:", m_s_r)
    print("Women's Survival Rate:", w_s_r)


def task_3_11(frame):
    sub_frame = frame[frame['Age'].notnull()]

    plt.title("Age, Survived = 0")
    sub_frame[sub_frame['Survived'] == 0]['Age'].hist(bins=40)
    plt.figure()
    plt.title("Age, Survived = 1")
    sub_frame[sub_frame['Survived'] == 1]['Age'].hist(bins=40)

    plt.show()


def task_3_12(frame):
    fig, subplots = plt.subplots(3, 2, sharex='all', sharey='all')
    dead_frame = frame[frame['Survived'] == 0]
    survived_frame = frame[frame['Survived'] == 1]

    subplots[0, 0].set_title('Pclass = 1 | Survived = 0')
    dead_frame[dead_frame['Pclass'] == 1]['Age'].hist(ax=subplots[0, 0], bins=20)

    subplots[0, 1].set_title('Pclass = 1 | Survived = 1')
    survived_frame[survived_frame['Pclass'] == 1]['Age'].hist(ax=subplots[0, 1], bins=20)

    subplots[1, 0].set_title('Pclass = 2 | Survived = 0')
    dead_frame[dead_frame['Pclass'] == 2]['Age'].hist(ax=subplots[1, 0], bins=20)

    subplots[1, 1].set_title('Pclass = 2 | Survived = 1')
    survived_frame[survived_frame['Pclass'] == 2]['Age'].hist(ax=subplots[1, 1], bins=20)

    subplots[2, 0].set_title('Pclass = 3 | Survived = 0')
    dead_frame[dead_frame['Pclass'] == 3]['Age'].hist(ax=subplots[2, 0], bins=20)

    subplots[2, 1].set_title('Pclass = 3 | Survived = 1')
    survived_frame[survived_frame['Pclass'] == 3]['Age'].hist(ax=subplots[2, 1], bins=20)

    plt.setp([a.get_xticklabels() for a in subplots[:2, 0]], visible=False)
    plt.setp([a.get_xticklabels() for a in subplots[:2, 1]], visible=False)
    plt.show()


def task_3_13(frame):
    fig, subplots = plt.subplots(3, 2, sharex='all', sharey='all')
    dead_frame = frame[frame['Survived'] == 0]
    survived_frame = frame[frame['Survived'] == 1]

    subplots[0, 0].set_title('Embarked = S | Survived = 0')
    dead_frame[dead_frame['Embarked'] == 'S'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[0, 0], kind='bar')

    subplots[0, 1].set_title('Embarked = S | Survived = 1')
    survived_frame[survived_frame['Embarked'] == 'S'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[0, 1], kind='bar')

    subplots[1, 0].set_title('Embarked = C | Survived = 0')
    dead_frame[dead_frame['Embarked'] == 'C'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[1, 0], kind='bar')

    subplots[1, 1].set_title('Embarked = C | Survived = 1')
    survived_frame[survived_frame['Embarked'] == 'C'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[1, 1], kind='bar')

    subplots[2, 0].set_title('Embarked = Q | Survived = 0')
    dead_frame[dead_frame['Embarked'] == 'Q'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[2, 0], kind='bar')

    subplots[2, 1].set_title('Embarked = Q | Survived = 1')
    survived_frame[survived_frame['Embarked'] == 'Q'].groupby(['Sex'])['Fare'].mean().plot(ax=subplots[2, 1], kind='bar')

    plt.setp([a.get_xticklabels() for a in subplots[:2, 0]], visible=False)
    plt.setp([a.get_xticklabels() for a in subplots[:2, 1]], visible=False)
    plt.show()


def task_3_14(frame):
    sub_frame = frame['Ticket']
    total = sub_frame.count()
    unique = sub_frame.nunique()
    print('Ticket duplicate rate: %1.3f' % ((total - unique) / float(total)) + '%')


def task_3_15(frame):
    print("There are", len(frame[frame['Cabin'].isnull()]), "null cabin entries")


def task_3_16(frame):
    frame.loc[frame.Sex == 'female', 'Sex'] = 1
    frame.loc[frame.Sex == 'male', 'Sex'] = 0
    frame = frame.rename({'Sex': 'Gender'}, axis='columns')


def task_3_17(frame):
    frame_stats = summary_stats.df_build_summary_stats(frame['Age'].dropna())
    age_mean = frame_stats[2]
    age_sdev = frame_stats[4]
    age_lb = int(age_mean - age_sdev)
    age_ub = int(age_mean + age_sdev)
    frame['Age'] = frame['Age'].apply([lambda x: random.randrange(age_lb, age_ub) if np.isnan(x) else x])


def task_3_18(frame):
    mode = frame['Embarked'].dropna().value_counts()[:1].index[0]
    frame['Embarked'].fillna(mode, inplace=True)


def task_3_19(frame):
    mode = frame['Fare'].value_counts().index[0]
    frame['Fare'].fillna(mode, inplace=True)


def task_3_20(frame):
    frame['Fare'] = frame['Fare'].apply(lambda x: 3 if x > 31.0 else 2 if x > 14.454 else 1 if x > 7.91 else 0)


if __name__ == '__main__':
    main()
