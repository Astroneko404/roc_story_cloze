import pandas as pd
import numpy as np


def convert_train(df):
    df1 = df[['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3',
           'InputSentence4', 'RandomFifthSentenceQuiz1','AnswerRightEnding']]
    labels1 = df1['AnswerRightEnding'].values
    for i in range(len(labels1)):
        if labels1[i] == 1:
            labels1[i] = 1
        else:
            labels1[i] = 0
    df1['label'] = labels1

    df2 = df[['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3',
           'InputSentence4', 'RandomFifthSentenceQuiz2','AnswerRightEnding']]
    labels2 = df2['AnswerRightEnding'].values
    for i in range(len(labels2)):
        if labels2[i] == 2:
            labels2[i] = 1
        else:
            labels2[i] = 0
    df2['label'] = labels2

    df1 = df1.drop(columns=['AnswerRightEnding'])
    df1['Quiz'] = df1['RandomFifthSentenceQuiz1']
    df1 = df1.drop(columns=['RandomFifthSentenceQuiz1'])
    df2 = df2.drop(columns=['AnswerRightEnding'])
    df2['Quiz'] = df2['RandomFifthSentenceQuiz2']
    df2 = df2.drop(columns=['RandomFifthSentenceQuiz2'])
    df3 = pd.concat([df1,df2])
    df3.reset_index(drop=True,inplace=True)
    df3 = df3.sample(frac=1)
    del df1
    del df2
    df3["Story"] = df3['InputSentence1'].str.cat(df3['InputSentence2'])
    df3["Story"] = df3['Story'].str.cat(df3['InputSentence3'])
    df3["Story"] = df3['Story'].str.cat(df3['InputSentence4'])
    df3 = df3.drop(columns=['InputSentence1','InputSentence2','InputSentence3','InputSentence4'])
    df3 = df3[['InputStoryid', 'Story', 'Quiz', 'label']]
    return df3


def convert_val(df):
    df1 = df.drop('RandomFifthSentenceQuiz2', axis=1)
    df2 = df.drop('RandomFifthSentenceQuiz1', axis=1)
    df1.rename(columns={'RandomFifthSentenceQuiz1': 'Quiz'}, inplace=True)
    df2.rename(columns={'RandomFifthSentenceQuiz2': 'Quiz'}, inplace=True)

    df1["Story"] = df1['InputSentence1'].str.cat(df1['InputSentence2'])
    df1["Story"] = df1['Story'].str.cat(df1['InputSentence3'])
    df1["Story"] = df1['Story'].str.cat(df1['InputSentence4'])
    df1 = df1.drop(columns=['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4'])
    df1 = df1[['InputStoryid', 'Story', 'Quiz', 'AnswerRightEnding']]

    df2["Story"] = df2['InputSentence1'].str.cat(df2['InputSentence2'])
    df2["Story"] = df2['Story'].str.cat(df2['InputSentence3'])
    df2["Story"] = df2['Story'].str.cat(df2['InputSentence4'])
    df2 = df2.drop(columns=['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4'])
    df2 = df2[['InputStoryid', 'Story', 'Quiz', 'AnswerRightEnding']]

    return df1, df2


def convert_test(df):
    df1 = df.drop('RandomFifthSentenceQuiz2', axis=1)
    df2 = df.drop('RandomFifthSentenceQuiz1', axis=1)
    df1.rename(columns={'RandomFifthSentenceQuiz1': 'Quiz'}, inplace=True)
    df2.rename(columns={'RandomFifthSentenceQuiz2': 'Quiz'}, inplace=True)

    df1["Story"] = df1['InputSentence1'].str.cat(df1['InputSentence2'])
    df1["Story"] = df1['Story'].str.cat(df1['InputSentence3'])
    df1["Story"] = df1['Story'].str.cat(df1['InputSentence4'])
    df1 = df1.drop(columns=['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4'])
    df1 = df1[['InputStoryid', 'Story', 'Quiz']]

    df2["Story"] = df2['InputSentence1'].str.cat(df2['InputSentence2'])
    df2["Story"] = df2['Story'].str.cat(df2['InputSentence3'])
    df2["Story"] = df2['Story'].str.cat(df2['InputSentence4'])
    df2 = df2.drop(columns=['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4'])
    df2 = df2[['InputStoryid', 'Story', 'Quiz']]

    return df1, df2