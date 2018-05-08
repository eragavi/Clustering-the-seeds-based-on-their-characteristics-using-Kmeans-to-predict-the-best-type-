#kmeans clustering
import csv
import numpy as np
import random
k=3

samplefilepath="C:/Ragavi/Clustering Project/seeds.csv"
samplefilepath1="C:/Ragavi/Clustering Project/seeds_orginal.csv"

#samplefilepath="C:/Ragavi/Clustering Project/LifespansOfCricketers.csv"#uncomment this to check for the other dataset
#samplefilepath1="C:/Ragavi/Clustering Project/LifespansOfCricketers_orginal.csv"#uncomment this to check for the other dataset

dist=[]
def get_cost(distance, finalclus,data):
    dist.clear()
    for i in range(0,len(distance)):
        x1=distance[i]
        final=finalclus[i]
        di=0

        for m in range(0,len(final)):
            for z in range(0,3):
                #print("finL M ODF Z",final[m][z])
                di=di+(float(x1[z])-float(data[final[m]][z]))**2

        dist.append(di)
    for i in range(0,len(distance)):
       sum= np.sum(dist)

    return sum


def calculate_distance(line1,line2):

   # print ("line1",line1)
   # print ("line2",line2)
    distance=0
    for k in range(0,len(line1)):
       distance=(distance+(float(line1[k])-float(line2[k]))**2)
    distance=distance**(1/2.0)
    #print(distance)
    return round(distance,1)

#function to calculate from centers
def calculate_from_centers(data,centroids,dis_matrix):


    for i in range(0, len(data)):
        for j in range(0,k):
         dist_from_centroid1 = calculate_distance(data[i], centroids[j])
         dis_matrix[i][j] = dist_from_centroid1

    min_eachrow=dis_matrix.argmin(1)


    length_of_mineachrow=len(min_eachrow)
    means = np.zeros((k, len(data[0])))
    for o in range(0, k):
        value=(j for j in range(0,length_of_mineachrow) if min_eachrow[j] ==o)
        count = sum(1 for i in range(0,length_of_mineachrow) if min_eachrow[i]==o)
        for x in value:
            for j in range(0,k):

                means[o][j]=means[o][j]+(float(data[x][j]))/count


    return means,min_eachrow

with open(samplefilepath1,'r')as f:
    reader1=csv.reader(f)
    data1=list(reader1)

sumFinalCluster={}

#Function to return the final cluster
def kMultipleValues(iterate,data,k):
    centroids = np.zeros((k, len(data[0])))
    for i in range(0, k):
        x = random.randint(0, len(data)-1)

        centroids[i] = data[x]

    l = len(data)
    dis_matrix = np.zeros((l, k))
    i = 0

    #finalcuster is initialized
    finalCluster = {}
    check=True
    while check:
        finalCluster.clear()
        dist1, final_min = calculate_from_centers(data, centroids,dis_matrix)
        olddist1 = centroids
        centroids = dist1
        cluster0 = [index for index, value in enumerate(final_min) if value == 0]
        cluster1 = [index for index, value in enumerate(final_min) if value == 1]
        cluster2 = [index for index, value in enumerate(final_min) if value == 2]


        finalCluster[0] = cluster0
        finalCluster[1] = cluster1
        finalCluster[2] = cluster2

        #Converging point is checked
        if (olddist1 == centroids).all():
            check=False
        else:
            check=True

    sums = get_cost(dist1, finalCluster, data)
    sumFinalCluster[iterate] = round(sums,3)

    return finalCluster

#Data set file for clustering is opened
with open(samplefilepath, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    title = data.pop(0)
    title_len=len(title)
    targets = [row.pop(0) for row in data]
    unique_targets = set(targets)
    flist={}
    sumFinalCluster.clear()
    for i in range(0,100):
        flist[i]=kMultipleValues(i,data,k)


    minimum= min(sumFinalCluster.values())
    index_of_min = min(sumFinalCluster, key=sumFinalCluster.get)
    print("minimum", minimum)
    #cluster with the minimum cost is printed
    print("cluster with minimum cost", flist[index_of_min])


    #Caclulation of Hamming Distancce
    calculated_matrix = np.zeros((len(data), len(data)))
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            calculated_matrix[i][j] = 1

    orginal_matrix = np.zeros((len(data), len(data)))
    for i in range(0, len(data)):
        for j in range(i, len(data)):
            if i == j:
                orginal_matrix[i][j] == 0
            elif (data1[i]) == (data1[j]):
                orginal_matrix[i][j] = 0
                orginal_matrix[j][i] = 0
            else:
                orginal_matrix[i][j] = 1
                orginal_matrix[j][i] = 1

    for i in range(0, k):
        value = flist[index_of_min][i]


        for att in range(0, len(data)):
            for j in range(0, len(data)):
                if att in value and j in value:
                    calculated_matrix[att][j] = 0
                    calculated_matrix[j][att] = 0

    val1 = np.array(orginal_matrix)
    val2 = np.array(calculated_matrix)
    xp = np.sum(val1 == val2)
    hamming_dist = (len(data) * len(data) - xp) / (0.5 * len(data) * (len(data) - 1))
    print("hamminig distance=", hamming_dist)





