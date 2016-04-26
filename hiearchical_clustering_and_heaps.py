#!/usr/bin/python

from __future__ import division
import numpy as np
import heapq

"""
This script produces a program to conduct a Hierarchical clustering
procedure on set of vectors in n-dimensional Euclidean space.

The basic algorithm for the agglomerative clustering procedure is the
acc_() function. This function requires the maintenance of a clusters
list which tracks the clusters formed/remaining throughout the procedure.
This function iterates until we are left with one cluster, printing the 
details of each merge step.

The maintenance vector stores cluster x as follows: [(centroid of x),
[[all points in x]]].

*** This script also produces an improved version of the basic algorithm
by exploiting the heap data structure. In addition to the clusters maintenance
list, we require a heap (denoted by h), a task/cluster dictionary in the heap
(to keep track of removed clusters). 

For this part, the main function is agg_heap(), which requires the following
functions:

	1. remove_clusters(): 
		performs lazy deletion of merged clusters

	2. pop_cluster(): 
		heappops the merged cluster with smallest distance
		only if the that cluster was not marked as 'REMOVED'

	3. add_cluster(): 
		takes as input, the output from pop_cluster and 
		calculates the distances between this merged cluster and all of the 
		clusters in the maintenance list clusters_remaining. These distances,
		along with the candidate merged clusters are added to heap using 
		heappush. Note that we must also add the cluster that was actually
		merged to the clusters_remaining maintenance vector.

Things to do:
- Create a class for agg_heap.
"""

def Euclidean(x,y):
    """
    This function returns the Euclidean distance between two vectors
    of a Euclidean space.
    """
    xc = np.array(x)
    yc = np.array(y)
    return np.sqrt(np.dot(xc-yc,xc-yc))

def mean(x):
    """
    This function takes as input a lists of the points and outputs
    the overall average of these points. This output is stored as
    a tuple so that it can be used to access the cluster index. In other 
    words, the centroid of cluster x.
    """
    N = len(x)
    n = len(x[0])
    sum_vec = np.zeros(n)
    for point in x:
        sum_vec += np.array(point)
    mean_vec = sum_vec / N
    return tuple(mean_vec)

def mins(x,y):
    """
    This function takes as input two clusters of points (i.e. vectors) each of which 
    are represented by of their own individual lists. The output of this function
    is the minimum distance between any two points one from each cluster
    """
    nx = len(x)
    ny = len(y)
    running_min = 2**32 - 1
    for pt_x in x:
        for pt_y in y:
            if Euclidean(pt_x,pt_y) < running_min:
                running_min = Euclidean(pt_x,pt_y)
    return running_min

def avg(x,y):
    """
    This function takes as input two clusters of points (i.e. vectors) each of which 
    are represented by of their own individual lists. The output of this function
    is the average distance between any two points one from each of the two clusters.
    """
    nx = len(x)
    ny = len(y)
    running_sum = 0
    for pt_x in x:
        for pt_y in y:
            running_sum += Euclidean(pt_x,pt_y)
    return running_sum/(nx*ny) # total number of pairs is nx*ny (i.e., by multiplication rule)

def radius(x,y=[]):
    """
    This function takes as input two clusters of points (i.e. vectors) each of which 
    are represented by of their own individual lists. The output of this function
    is the radius of the custer which results from the merge of x and y.
    
    If the input is simply one cluster, then the output is the radius of that
    cluster.
    """
    nx = len(x)
    ny = len(y)
    # merge two clusters x and y
    merged_clus = x + y 
    # the centroid of the new merged cluster
    # in non-Euclidean setting, we should change centroid to a clustroid
    merged_cent = mean(merged_clus)
    # determine the radius of this merged cluster
    # radius is the maximum distance between all the points and the centroid
    radius = 0
    for pt in merged_clus:
        if Euclidean(pt,merged_cent) > radius:
            radius = Euclidean(pt,merged_cent)
    return radius

def diameter(x,y=[]):
    """
    This function takes as input two clusters of points (i.e. vectors) each of which 
    are represented by of their own individual lists. The output of this function
    is the diameter of the merged custer of x and y.
    
    If the input is simply one cluster, then the output is the diameter of that
    cluster.
    """
    # merge two clusters x and y
    merged_clus = x + y 
    n = len(merged_clus)
    # determine the diameter of this merged cluster
    # diameter is the maximum distance between any two points of the cluster
    diameter = 0
    for i in range(n-1):
        for j in range(i+1,n):
            distance_ij = Euclidean(merged_clus[i],merged_clus[j])
            if distance_ij > diameter:
                diameter = distance_ij
    return diameter

def agg_(clusters, print_summary = True, dist = 'Euclidean'):
    """
    This function takes as input a dictionary of clusters in 
    Euclidean space and returns the Agglomerative clustering. 
    The key of the dictionary is the centroid of the corresponding
    cluster.
    
    Note that the clustering agglomerative clustering is done in
    place with respect to the clusters list input.
    """
    
    # specifying the distance function used
    # r_ = 0 implies we consider centroids of the two clusters in merge step
    # r_ = 1 means that we consider the points of the two clusters themselves in merge step
    if dist == 'Euclidean':
        f_dist = Euclidean
        r_ = 0 
    if dist == 'mins':
        f_dist = mins
        r_ = 1 
    if dist == 'avg':
        f_dist = avg
        r_ = 1
    if dist == 'radius':
        f_dist = radius
        r_ = 1
    if dist == 'diameter':
        f_dist = diameter
        r_ = 1
    
    # start main code to conduct clustering
    step = 1    
    while len(clusters) > 1:
#     while step < 3:
        # clusters hash table (use centroids as hash keys)
        clusters_ix = {el[0]:i for i,el in enumerate(clusters)}
        # double loop to consider the minimal distance between all pairs of clusters
        n = len(clusters)
        min_dist = 2**32-1
        c1 = None
        c2 = None
        for i in range(n-1):
            for j in range(i+1,n):
                # the distance between centroids of cluster i and cluster j
                distance_ij = f_dist(clusters[i][r_], clusters[j][r_])
                if distance_ij < min_dist:
                    min_dist = distance_ij
                    c1 = clusters[i]
                    c2 = clusters[j]
        # merge the two clusters that result in minimum Euclidean distance
        new_cluster = c1[1] + c2[1]
        new_centroid = mean(new_cluster)
        clusters.append([new_centroid, new_cluster])
        # remove the merged clusters from the list 
        del clusters[max(clusters_ix[c1[0]],clusters_ix[c2[0]])]
        del clusters[min(clusters_ix[c1[0]],clusters_ix[c2[0]])]
        if print_summary:
            print 'Step %d:' % step
            print 'Merged clusters: %s and %s' %(str(c1[1]),str(c2[1]))
            print 'Minimum distance: %f' % min_dist
            print 'New clusters list:'
            print [el[1] for el in clusters] 
            print 'New centroids:'
            print [el[0] for el in clusters]
            print ''
            print '--------------------------------------------------------'
            print ''
        step += 1
    
# Alternatively, can use np.mean to create the new centroid
# new_centroid = tuple(np.mean(np.array(new_cluster),axis=0))

""" **************************************************************************

This part of the script defines agg_heap() and all of its necessary components

Sample input:
clusters = [[(4,10),[[4,10]]], [(7,10),[[7,10]]], [(4,8),[[4,8]]],
           [(6,8),[[6,8]]],[(3,4),[[3,4]]],[(2,2),[[2,2]]],[(5,2),[[5,2]]],
            [(12,6),[[12,6]]],[(10,5),[[10,5]]],[(11,4),[[11,4]]],[(9,3),[[9,3]]],
           [(12,3),[[12,3]]]]

# creating a dictionary tracking the remaining clusters
clusters_remaining = {tuple(tuple(el) for el in clusters[i][1]):clusters[i][1] 
                      for i in range(len(clusters))}
clusters_remaining

# creating a heap h with item keys (dist, pair)
h = []
n = len(clusters)
f_dist = Euclidean
r_ = 0
clusters_handle = {} # keys are centroids of the pairs of clusters
for i in range(n-1):
    for j in range(i+1,n):
        distance_ij = f_dist(clusters[i][r_],clusters[j][r_])
        ati = tuple(tuple(el) for el in clusters[i][1])
        tun = tuple(tuple(el) for el in clusters[j][1])
        foo = [distance_ij, (tuple(ati),tuple(tun)), 
               clusters[i][1]+clusters[j][1]]
        clusters_handle[(tuple(ati),tuple(tun))] = foo
        heapq.heappush(h,foo)

************************************************************************** """

REMOVED = '<removed-cluster>'
def remove_clusters(i):
    """
    This function lazily deletes any clusters that have been 
    merged from the dictionary clusters_handle.
    """
    for key in clusters_handle.keys():
        if i in key:
            # mark task as removed
            clusters_handle[key][1] = REMOVED

def pop_cluster():
    """
    To maintain the heap property, we lazily deleted merged clusters.
    In this function, we only pop (extract) minimum distance 
    clusters if the merged cluster has not been removed.
    """
    # this pops until it returns something
    while h: # while there are entries in the heap
        distance, tup, merged_clus = heapq.heappop(h)
        if tup != REMOVED:
            del clusters_handle[tup]
            # remove newly merged clusters from heap task dict and clusters_remaining dictionary
            for cluster in tup:
                remove_clusters(cluster)
                del clusters_remaining[cluster]
            return distance, tup, merged_clus
    raise KeyError('pop from an empty heap')


def add_cluster(entry):
    """
    This function takes the pop'ed entry and calculates the
    distances of the newly-merged cluster to all of the clusters
    in the clusters_remaining dictionary.
    """
    distance, tup, merged_clus = entry
    tup = tup[0] + tup[1]
    centroid = mean(merged_clus)
    # for every entry in the clusters_remaining compute new distances
    for tup_cmp, clus_cmp in clusters_remaining.items():
        centroid_cmp = mean(clus_cmp)
        new_distance = Euclidean(centroid, centroid_cmp)
        # generate new entry for the heap
        new_tup = (tup, tup_cmp)
        new_merged_clus = merged_clus + clus_cmp
        new_entry = [new_distance, new_tup, new_merged_clus]
        # add new entry to clusters_handle 
        clusters_handle[new_tup] = new_entry
        # add new entry to the heap
        heapq.heappush(h,new_entry)
    # add the recently merged cluster to clusters_remaining dict
    clusters_remaining[tup] = merged_clus

def agg_heap(clusters, print_summary = True):
    """
    This function takes as input a dictionary of clusters in 
    Euclidean space and returns the Agglomerative clustering. This is
    an improvement over agg_ since it exploits the heap data structure.
    
    Note that Python heapq and queue module do not support element deletion.
    In this code, we simply use lazy deletion.
    """
    # add methods here to create heap from clusters and 
    # other necessary components
    clusters_remaining = {tuple(tuple(el) for el in clusters[i][1]):clusters[i][1] 
                      for i in range(len(clusters))}
    
    # start main code to conduct clustering
    step = 1    
    while len(clusters_remaining) > 1:
        entry = pop_cluster()
        add_cluster(entry)
        distance, tup, merged_clus = entry
        if print_summary:
            print 'Step %d:' % step
            print 'Merged clusters: %s and %s' %(str(tup[0]),str(tup[1]))
            print 'Minimum distance: %f' % distance
            print 'New clusters list:'
            print [key for key in clusters_remaining.keys()] 
            print 'New centroids:'
            print [mean(el) for el in clusters_remaining.values()]
            print ''
            print '--------------------------------------------------------'
            print ''
        step += 1

