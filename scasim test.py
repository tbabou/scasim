import pandas as pd
import numpy as np

fixations = pd.read_csv('eyemovements.csv')
scanpaths = {k:v for k,v in fixations.groupby('trial')}


def substitution_penalty(fixation1,fixation2, normalize):
    modulator = 0.83
    dur1, dur2 = float(fixation1['duration']), float(fixation2['duration'])
    total_dur = dur1+dur2
    fixation1_vector = np.array([60.0,(float(fixation1['x'])-512)/30,(float(fixation1['y'])-384.0)/30.0])
    fixation2_vector = np.array([60.0,(float(fixation2['x'])-512)/30,(float(fixation2['y'])-384.0)/30.0])
    dot = fixation1_vector.dot(fixation2_vector)
    distance = np.arccos(dot/float(((np.sqrt(fixation1_vector.dot(fixation1_vector)))*(np.sqrt(fixation2_vector.dot(fixation2_vector))))))*(180/np.pi)
    if np.isnan(distance):
        distance = 0.0
    sub_score = (abs((dur1 - dur2))*modulator**distance) + ((dur1 + dur2))*(1 - modulator**distance)
    if normalize:
        return float(sub_score)/(1000*total_dur)
    else:
        return float(sub_score)
def alignment_penalty(fixation1,fixation2):
    modulator = 0.83
    dur1, dur2 = float(fixation1['duration']), float(fixation2['duration'])
    distance = abs(float(fixation1['x']) - float(fixation2['x']))
    sub_score = abs((dur1 - dur2)) * modulator ** distance + ((dur1 + dur2)) * (1 - modulator ** distance)
    return float(sub_score)


def gap_penalty(fixation):
    dur = float(fixation['duration'])
    return float(dur)


def needle(path1,path2):
    path1 =path1.reset_index()
    path2 =path2.reset_index()
    m, n = path1.shape[0], path2.shape[0]
    score = np.zeros((m, n))
    for i in range(0, m):
        score[i][0] = -1*gap_penalty(path1.ix[i])+score[i-1][0]
    for j in range(0, n):
        score[0][j] = -1*gap_penalty(path2.ix[j])+score[0][j-1]
    # now to calculate the traceback matrix
    for i in range(1,m):
        for j in range(1, n):
            match = score[i-1][j-1] - substitution_penalty(path1.ix[i-1],path2.ix[j-1],False)
            delete = score[i-1][j] - gap_penalty(path1.ix[i-1])
            insert = score[i][j-1] - gap_penalty(path2.ix[j-1])
            score[i][j] = max(match, delete, insert)
    # now we want to traceback and compute the alignment
    align1, align2 = pd.DataFrame(columns = ['subject', 'trial', 'word','x','y', 'duration']), pd.DataFrame(columns = ['subject', 'trial', 'word','x','y', 'duration'])
    i,j = m -1,n -1
    index = 0
    while i > 0 and j>0:
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]

        if score_current == score_diagonal - substitution_penalty(path1.ix[i-1],path2.ix[j-1],False):
            align1.loc[index] = path1.ix[i-1]
            align2.loc[index] = path2.ix[j-1]
            i -= 1
            j -= 1
            index += 1
        elif score_current == score_left - gap_penalty(path1.ix[i-1]):
            align1.loc[index] = path1.ix[i-1]
            align2.loc[index] = np.array(['',0,0,0,0,0])
            i -= 1
            index += 1
        elif score_current == score_up +-gap_penalty(path2.ix[j-1]):
            align1.loc[index] = np.array(['',0,0,0,0,0])
            align2.loc[index] =path2.ix[j-1]
            j -= 1
            index += 1
    while i > 0:
        align1.loc[index] =path1.ix[i-1]
        align2.loc[index] =np.array(['',0,0,0,0,0])
        i -= 1
        index += 1
    while j > 0:
        align1.loc[index] = np.array(['',0,0,0,0,0])
        align2.loc[index] = path2.ix[j-1]
        j -= 1
        index += 1
    return(align1,align2)

def scasim1(dictionary): #this is for comparing all of the paths of a single participant
    print(len(dictionary.keys()))
    dissimilarity_matrix = np.zeros((len(dictionary.keys()),len(dictionary.keys())))
    index1 = 0
    for path in dictionary.values():
        index2 = 0
        for other_path in dictionary.values():
            if index1 != index2 and dissimilarity_matrix[index1,index2] == 0:
                score = 0
                aligned1, aligned2 = needle(path,other_path)
                for i in range(aligned1.shape[0]):
                   # print(substitution_penalty(aligned1.loc[i], aligned2.loc[i], True))
                    score += substitution_penalty(aligned1.loc[i], aligned2.loc[i], True)
                dissimilarity_matrix[index1,index2] = score
                dissimilarity_matrix[index2,index1] = score
            index2 +=1
            print(index1,index2)
        index1 += 1
        # #so I know the progress of the program
    scasim = pd.DataFrame(dissimilarity_matrix)
    #scasim.to_csv('scasim cl277.csv')
    return(scasim)

print(scasim1(scanpaths))
#print(needle(scanpaths[1],scanpaths[2]))

