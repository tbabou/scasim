import numpy as np
import pandas as pd
import collections

fixation_data = open('sample fixations.txt', 'r')
data_matrix = [['','ID number', 'Duration','X_coordinate','Word', 'Sentence']]
def get_data(file): # function that takes the raw data and turns into a nested list
    line_index = 0
    for line in file:
        line_index += 1
        fixation = line.split()
        relevant_data = []
        if fixation[23] != 'en-universal-test.conllu':
            relevant_data.extend(['fixation {}'.format(line_index),fixation[0],fixation[2],fixation[5], fixation[8], fixation[23]])
        else:
            relevant_data.extend(
                ['fixation {}'.format(line_index), fixation[0], fixation[2], fixation[5], fixation[8], fixation[24]])
        # the data that we are saving for later use is the ID, Duration of fixation, X_coordinate of fixation, Word that is being fixated on, and the sentence
        data_matrix.append(relevant_data)
    return data_matrix
#data munging
get_data(fixation_data)
np_data = np.array(data_matrix) # now we have a 2D numpy array that we will then turn into a pandas dataframe for ease of manipulation
fixation_dataframe = pd.DataFrame(data = np_data[1:,1:],index = np_data[1:,0], columns=np_data[0,1:])
fixations_by_user = {k:v for k,v in fixation_dataframe.groupby('ID number')}
scanpaths_l173 = {k:v for k,v in fixations_by_user['l173'].groupby('Sentence')}
scanpaths_cl277 = {i:j for i,j in fixations_by_user['cl277'].groupby('Sentence')}
l173 = collections.OrderedDict(scanpaths_l173)
cl277 = collections.OrderedDict(scanpaths_cl277)
#data munging

''' the above lines are used to take our larger pandas dataframe and group them into scanpaths of each of the participants
each participant is saved a dictionary whose keys are the sentence to which the scanpath belongs
and the value is a dataframe which contains the data for each fixation within that scanpath. Not
entirely sure if this is the most convenient format for the data to be in'''

# We then can implement Needleman-Wunsch to align different scanpaths
# gaps are introduced as null fixations with 0ms duration
def substitution_penalty(fixation1,fixation2):
    modulator = 0.83
    dur1, dur2 = float(fixation1['Duration']), float(fixation2['Duration'])
    distance = abs(float(fixation1['X_coordinate']) - float(fixation2['X_coordinate']))
    sub_score = abs((dur1 - dur2)/1000)*modulator**distance + ((dur1 + dur2)/1000)*(1 - modulator**distance)
    return float(sub_score)
def gap_penalty(fixation):
    modulator = 0.83
    dur = float(fixation['Duration'])
    distance = float(fixation['X_coordinate'])
    gap_score = ((dur / 1000) * modulator ** distance + (dur / 1000) * (1 - modulator ** distance))
    return float(gap_score)


def needle(path1,path2):
    m, n = path1.shape[0], path2.shape[0]
    score = np.zeros((m, n))
    for i in range(0, m):
        score[i][0] = -1*gap_penalty(path1.ix[i])+score[i-1][0]
    for j in range(0, n):
        score[0][j] = -1*gap_penalty(path2.ix[j])+score[0][j-1]
    # now to calculate the traceback matrix
    for i in range(1,m):
        for j in range(1, n):
            match = score[i-1][j-1] - substitution_penalty(path1.ix[i-1],path2.ix[j-1])
            delete = score[i-1][j] - gap_penalty(path1.ix[i-1])
            insert = score[i][j-1] - gap_penalty(path2.ix[j-1])
            score[i][j] = max(match, delete, insert)
    # now we want to traceback and compute the alignment
    align1, align2 = pd.DataFrame(columns = ['ID number', 'Duration','X_coordinate','Word', 'Sentence']), pd.DataFrame(columns = ['ID number', 'Duration','X_coordinate','Word', 'Sentence'])
    i,j = m -1,n -1
    index = 0
    while i > 0 and j>0:
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]

        if score_current == score_diagonal - substitution_penalty(path1.ix[i-1],path2.ix[j-1]):
            align1.loc[index] = path1.ix[i-1]
            align2.loc[index] = path2.ix[j-1]
            i -= 1
            j -= 1
            index += 1
        elif score_current == score_left - gap_penalty(path1.ix[i-1]):
            align1.loc[index] = path1.ix[i-1]
            align2.loc[index] = np.array([0,0,0,'',0])
            i -= 1
            index += 1
        elif score_current == score_up +-gap_penalty(path2.ix[j-1]):
            align1.loc[index] = np.array([0,0,0,'',0])
            align2.loc[index] =path2.ix[j-1]
            j -= 1
            index += 1
    while i > 0:
        align1.loc[index] =path1.ix[i-1]
        align2.loc[index] =np.array([0,0,0,'',0])
        i -= 1
        index += 1
    while j > 0:
        align1.loc[index] = np.array([0,0,0,'',0])
        align2.loc[index] = path2.ix[j-1]
        j -= 1
        index += 1
    return(align1,align2)

print(needle(l173['2301'],l173['604']))






















