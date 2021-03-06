import pandas as pd
import numpy as np

fixations = pd.read_csv('eyemovements.csv')
scanpaths = {k:v for k,v in fixations.groupby('trial')}


def inverse_gnomic(fixation, x_center, y_center, unit_size, distance):
    x = (float(fixation['x']) - x_center) * unit_size/distance
    y = (float(fixation['y']) - y_center) * unit_size/distance
    rho = np.sqrt(x**2 + y**2)
    c = np.arctan(rho)
    lat = np.arcsin(y * np.sin(c)/rho)
    lon = np.arctan2(x * np.sin(c), rho * np.cos(c))
    return lat, lon


def substitution_penalty(fixation1,fixation2):
    modulator = 0.83
    dur1, dur2 = float(fixation1['duration']), float(fixation2['duration'])
    total_dur = dur1+dur2
    x1, y1 = inverse_gnomic(fixation1,512,384,1/30,60)
    x2, y2 = inverse_gnomic(fixation2,512,384,1/30,60)
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    sub_score = (abs((dur1 - dur2))*modulator**distance) + ((dur1 + dur2))*(1 - modulator**distance)
    return sub_score,total_dur

def gap_penalty(fixation):
    dur = float(fixation['duration'])
    return float(dur)


def needle(path1,path2):
    path1 = path1.reset_index()
    path2 =path2.reset_index()
    m, n = path1.shape[0], path2.shape[0]
    score = np.zeros((m+1, n+1))
    score[0][0] = -1*substitution_penalty(path1.ix[0],path2.ix[0])[0]

    for i in range(1, m+1):
        score[i][0] = -1*gap_penalty(path1.ix[i-1])+score[i-1][0]
    for j in range(1, n+1):
        score[0][j] = -1*gap_penalty(path2.ix[j-1])+score[0][j-1]
    # now to calculate the traceback matrix
    for i in range(1,m+1):
        for j in range(1, n+1):
            match = score[i-1][j-1] - substitution_penalty(path1.ix[i-1],path2.ix[j-1])[0]
            delete = score[i-1][j] - gap_penalty(path1.ix[i-1])
            insert = score[i][j-1] - gap_penalty(path2.ix[j-1])
            score[i][j] = max(match, delete, insert)
    # now we want to traceback and compute the alignment
    align1, align2 = pd.DataFrame(columns = ['subject', 'trial', 'word','x','y', 'duration']), pd.DataFrame(columns = ['subject', 'trial', 'word','x','y', 'duration'])
    i,j = m ,n
    index = 0
    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]

        if score_current == score_diagonal - substitution_penalty(path1.ix[i-1],path2.ix[j-1])[0]:
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
        elif score_current == score_up -gap_penalty(path2.ix[j-1]):
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

def scasim1(dictionary, normalize): #this is for comparing all of the paths of a single participant
    print(len(dictionary.keys()))
    dissimilarity_matrix = np.zeros((len(dictionary.keys()),len(dictionary.keys())))
    index1 = 0
    for path in dictionary.values():
        index2 = 0
        for other_path in dictionary.values():
            if index1 != index2 and dissimilarity_matrix[index1,index2] == 0:
                score = 0
                total_duration = 0
                aligned1, aligned2 = needle(path,other_path)
                for i in range(aligned1.shape[0]):
                    score += substitution_penalty(aligned1.loc[i], aligned2.loc[i])[0]
                    total_duration += substitution_penalty(aligned1.loc[i], aligned2.loc[i])[1]
                if normalize:
                    dissimilarity_matrix[index1,index2] = score/total_duration
                    dissimilarity_matrix[index2,index1] = score/total_duration
                else:
                    dissimilarity_matrix[index1, index2] = score
                    dissimilarity_matrix[index2, index1] = score
            index2 +=1
            print(index1,index2)
        index1 += 1
        # #so I know the progress of the program
    scasim = pd.DataFrame(dissimilarity_matrix)
    #scasim.to_csv('test_scasim.csv')
    return(scasim)
