import pandas as pd
import matplotlib
import math
from numpy import *


def simplify_list(list):
    simple_list = []
    for item in list:
        if type(item) == type([]):
            simple_list.extend(simplify_list(item))
        else:
            simple_list.append(item)
    return simple_list


def avg_vectors(vec_one, vec_two):
    temp_vector = vec_two
    counter = 0
    for a, b in vec_one.iteritems():
        if b > 0:
            temp_vector[counter] = (temp_vector[counter] + b) / 2
        counter = counter + 1
    return temp_vector


def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    # read the cvs file into dataframe with latin encoding
    dt = pd.read_csv(file_name, delimiter=',', encoding="latin1")
    # removing the unnamed columns from the dataframe
    dt = dt.loc[:, ~dt.columns.str.contains('^Unnamed')]

    # remove attributes that seem unnecessary for our calculations
    dt = dt.drop(
        ['Region', 'Artist   ', 'Points', 'Year', 'Song language', 'Artist gender', 'Song   ', 'English translation ',
         'Artist gender', 'Group/Solo', 'Place', 'Host Country', 'Host region', 'Home/Away Country', 'Home/Away Region',
         'Approximate Betting Prices'], axis=1)

    dt.columns = dt.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')',
                                                                                                           '').str.replace(
        '&', 'and')

    # compare total points with sum of voting country attributes for every row
    # dt.iloc[2,3] == dt_countries.iloc[2].sum()
    # value of total point attribute is equal to the sum of all not nan voting attributes
    # which means we can replace nan values with 0, that way it wont influence the total points
    dt = dt.fillna(0)

    filtered_dic = {}

    # iterate trough all data frame rows
    for index, row in dt.iterrows():
        # separate key and value (key being country name, value being the rest of data)
        key = row[0].strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')
        value = row[1:len(row)]
        # if key already exists, means we have another row for this particular country
        if key in filtered_dic:
            temp_vector_dic = filtered_dic[key]
            # get the existing value for that key, and average it with the new value
            value = avg_vectors(temp_vector_dic, value)
        # update new value back in dictionary
        filtered_dic.update({key: value})

    return filtered_dic


class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        self.distance_dict = data
        self.all_clusters_distance = []
        self.clusters = [[name] for name in self.data.keys()]

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        r1 = r1.strip()
        r2 = r2.strip()
        vec_one = self.data.get(r1)
        vec_two = self.data.get(r2)
        distance = 0

        if vec_one is not None and vec_two is not None:
            length = len(vec_two)
            for x in range(length):
                distance += pow((vec_one[x] - vec_two[x]), 2)
            distance = math.sqrt(distance)
        else:
            distance = -1
        return distance

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        list_one = simplify_list(c1)
        list_two = simplify_list(c2)

        all_length = len(list_one) * len(list_two)
        distance = 0
        for item_one in list_one:
            for item_two in list_two:
                distance = distance + self.distance_dict[item_one][item_two]
        distance = distance / all_length
        return distance

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        distance = 10e+4931
        list_clusters = []
        for item_one in self.clusters:
            for item_two in self.clusters:
                if item_one != item_two:
                    temp_distance = self.cluster_distance(item_one, item_two)
                    if temp_distance < distance:
                        distance = temp_distance
                        list_clusters = [item_one, item_two]

        return [list_clusters, distance]

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """

        for key, value in self.distance_dict.items():
            for inner_key, inner_value in value.iteritems():
                distance = self.row_distance(key, inner_key)
                self.distance_dict[key][inner_key] = distance

        while len(self.clusters) != 2:
            list_item = self.closest_clusters()
            self.all_clusters_distance.append(list_item)
            pair = list_item[0]
            item_one = pair[0]
            item_two = pair[1]
            self.clusters.remove(item_one)
            self.clusters.remove(item_two)
            self.clusters.append(pair)

        pass

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        pass


if __name__ == "__main__":
    DATA_FILE = "eurovision-final.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
