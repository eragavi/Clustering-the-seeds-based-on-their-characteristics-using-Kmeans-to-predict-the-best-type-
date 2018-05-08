#CLUSTERING
#IRIS DATASET
import csv
import math
import numpy as np
import itertools
import sys


samplefilepath="C:/Ragavi/Clustering Project/seeds.csv" 
samplefilepath1="C:/Ragavi/Clustering Project/seeds_orginal.csv" 

#samplefilepath="C:/Ragavi/Clustering Project/LifespansOfCricketers.csv" #uncomment this to check for the other dataset
#samplefilepath1="C:/Ragavi/Clustering Project/LifespansOfCricketers_orginal.csv"#uncomment this to check for the other dataset


k_Clusters=3

#function for flattening of the list
def flatten(S):
    if S == [] or not isinstance(S, list):
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

#function to calculate the distance between the data points
def calculate_distance(line1,line2):

    distance=0
    for k in range(0,len(line1)):
       distance=(distance+(float(line1[k])-float(line2[k]))**2)
    distance=math.sqrt(distance)

    return distance

row_names = []
#Function to get the matrix of the distances
def get_cluster_matrix(distance_matrix,type):
    index_min = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
    firstindex = list(index_min)

    if len(firstindex) == 1:
        return distance_matrix, firstindex
    combined_index1 = firstindex[0]
    combined_index2 = firstindex[1]

    firstindex[0] = row_names[firstindex[0]]
    firstindex[1] = row_names[firstindex[1]]

    row_names[combined_index1] = firstindex

    for i in range(0, len(distance_matrix)):

        row_names[combined_index1] = firstindex
        if type=="single":
            distance_matrix[combined_index1][i] = min(distance_matrix[combined_index1][i],distance_matrix[combined_index2][i])


        elif type=="complete":
            distance_matrix[combined_index1][i] = max(distance_matrix[combined_index1][i],distance_matrix[combined_index2][i])

        elif type=="average":
            flattened_combinedindex1=set(flatten([row_names[combined_index1]]))
            flattened_combinedindex2 = set(flatten([row_names[combined_index2]]))
            length=len(flattened_combinedindex1)+len(flattened_combinedindex2)
            distance_matrix[combined_index1][i] = ((len(flattened_combinedindex1) * distance_matrix[combined_index1][i])+ (len(flattened_combinedindex2) * distance_matrix[combined_index2][i]))/length


        distance_matrix[i][combined_index1] = distance_matrix[combined_index1][i]
    distance_matrix[combined_index1][combined_index1] = sys.maxsize

    row_names.pop(combined_index2)
    new_matrix = np.delete(distance_matrix, combined_index2, 0)
    new_matrix = np.delete(new_matrix, combined_index2, 1)


    return new_matrix


with open(samplefilepath1,'r')as f:
    reader1=csv.reader(f)
    data1=list(reader1)

#open the data set file for clustering, program runs from here
with open(samplefilepath, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    title = data.pop(0)
    title_len=len(title)
    targets = [row.pop(0) for row in data]
    unique_targets = set(targets)
    r=len(data)

    #dis matrix is initialized
    dis_matrix = np.zeros((r, r))
    for i in range(0, r):
      dis_matrix[i][i] = sys.maxsize
      row_names.append(i)
      for j in range(i + 1, r):
          distance = calculate_distance(data[i], data[j]) #calculate distance function is called to find distacne between the data points
          dis_matrix[i][j] = distance
          dis_matrix[j][i] = distance

    print("Enter single, complete or average")
    testVar = input("Enter the linkeage:::")
    if testVar == "single":
        print("############ SINGLE LINKEAGE ############")

    elif testVar == "complete":
        print("############ COMPLETE LINKEAGE ############")

    elif testVar == "average":
        print("############ AVERAGE LINKEAGE ############")

    i = 1
    new_matrix = dis_matrix.copy()
    if k_Clusters >= r:
        print ('error')
        exit(1)

    while k_Clusters  < len(row_names):
        i +=1
        new_matrix = get_cluster_matrix(new_matrix.copy(),testVar)
    calculated_matrix = np.zeros((r, r))
    finalmerged = {}
    for i in range(0, k_Clusters):
        cluster = [row_names[i]]
        merged=flatten(cluster)
        print ("Cluster-" + str(i) + ":" + str(merged))



        finalmerged[i]=merged

    #Hamming Distance Calculation is done
    calculated_matrix=np.zeros((r,r))
    for i in range(0,r):
        for j in range(0,r):
            calculated_matrix[i][j]=1


    orginal_matrix=np.zeros((r,r))
    for i in range(0,r):
        for j in range(0,r):
            if i==j:
                orginal_matrix[i][j]==0
            elif (data1[i])==(data1[j]):
                orginal_matrix[i][j]=0
                orginal_matrix[j][i]=0
            else:
                orginal_matrix[i][j]=1
                orginal_matrix[j][i]=1

    for i in range(0,k_Clusters):
        value=finalmerged[i]
        #print(value)
        for att in range(0,r):
              for j in range(0,r):
                if att in value and j in value:
                    calculated_matrix[att][j]=0
                    calculated_matrix[j][att]=0

    #print("ca",calculated_matrix)
    val1=np.array(orginal_matrix)
    val2=np.array(calculated_matrix)
    xp=np.sum(val1==val2)
    hamming_dist=(r*r-xp)/(0.5*r*(r-1))
    print("Hamminig distance=",hamming_dist)



