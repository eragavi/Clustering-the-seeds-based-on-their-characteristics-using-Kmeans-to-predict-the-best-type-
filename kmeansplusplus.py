#kmeans clustering
import sys
import csv
import random
k=3
import numpy as np

samplefilepath="C:/Ragavi/Clustering Project/seeds.csv"
samplefilepath1="C:/Ragavi/Clustering Project/seeds_orginal.csv"


#samplefilepath="C:/Ragavi/Clustering Project/LifespansOfCricketers.csv"#uncomment this to check for the other dataset
#samplefilepath1="C:/Ragavi/Clustering Project/LifespansOfCricketers_orginal.csv"#uncomment this to check for the other dataset


distance1=[]
cum=[]
#function used to calculate the distance from the data towards the centroid points (kmeans++)
def get_kmeans_clust(data,initcentroid,counter):
    dis_min = np.zeros((len(data), k - 1))
    for i in range(0, len(data)):
        for j in range(0, k - 1):
            dis_min[i][j] = sys.maxsize
    dd=0
    for attr in range(0,len(data)):
        dd=(dd+(float(data[attr])-float(initcentroid[attr]))**2)
        dis_min[attr][counter]=dd
        min_eachrow_val = np.min(dis_min[attr])
        distance1.append(min_eachrow_val)

    return distance1

#calulcation of distance is done
def calculate_distance(line1,line2):

    distance=0
    for k in range(0,len(line1)):
       distance=(distance+(float(line1[k])-float(line2[k]))**2)
    distance=distance**(1/2.0)
    return round(distance,1)#returns the distance


#distance from the data points towards the centroid obtained
def calculate_from_centers(data,centroids):
    for i in range(0, len(data)):
        for j in range(0,k):
         dist_from_centroid1 = calculate_distance(data[i], centroids[j])#calculate distance is called (euclidean distance)
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

    #returns the centroid and the clustered indices
    return means,min_eachrow

#orginaldatafile for purpose of calculating the hamming distance
with open(samplefilepath1,'r')as f:
    reader1=csv.reader(f)
    data1=list(reader1)

#file for purpose of doing the clustering
with open(samplefilepath, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    title = data.pop(0)

    #centroids is initialized
    centroids=np.zeros((k, len(data[0])))
    counter=0
    newcenters=[]

    x=random.randint(0,len(data))
    newcenters.append(x)
    while len(newcenters)<k:
        for i in range(0,len(data)):
            #get_kmeans_clust is called where the data,particular new center and the counter is sent
            distance_initial=get_kmeans_clust(data[i],data[newcenters[counter]],counter)

        cummulative=[]
        cummulative.append(distance_initial[0])
        for max in range(1,len(data)):
            cc=0
            cc=cummulative[max-1]+distance_initial[max]
            cummulative.append(cc)
        cummulative[0]=distance_initial[0]
        a=cummulative.index(random.choice(cummulative))
        while a in newcenters:
            a=cummulative.index(random.choice(cummulative))
        newcenters.append(a)
        counter=counter+1

    l=len(data)

    #dis matrix is initialized
    dis_matrix = np.zeros((l, k))
    i=0
    while True:
        i+=1
        listx=[]
        for i in range(0,k):
            listx.append(data[newcenters[i]])
        #calculate_from_centers is called and the centroids and the clustered indices are obtained
        dist1,final_min=calculate_from_centers(data,listx)

       #When the centroids converge it stops
        if (dist1 == centroids).all():
            break
        centroids=dist1

    #final cluster is initialized
    finalCluster = {}
    for i in range(0,k):
        print("CLUSTER ",str(i))
        print ([index for index, value in enumerate(final_min) if value == i])
        #clusted get printed
        finalCluster[i]=[index for index, value in enumerate(final_min) if value == i]



    calculated_matrix = np.zeros((len(data), len(data))) #matrix for cluster data obtained through the implemented algorithm
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            calculated_matrix[i][j] = 1


    orginal_matrix = np.zeros((len(data), len(data))) #matrix for original cluster data
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
        value = finalCluster[i]
        for att in range(0, len(data)):
            for j in range(0, len(data)):
                if att in value and j in value:
                    calculated_matrix[att][j] = 0
                    calculated_matrix[j][att] = 0

    val1 = np.array(orginal_matrix)
    val2 = np.array(calculated_matrix)
    xp = np.sum(val1 == val2)

    #hamming distance for the data set is obtained
    hamming_dist = (len(data) * len(data) - xp) / (0.5 * len(data) * (len(data) - 1))
    print("\nHamming Distance=", hamming_dist)




