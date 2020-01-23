import os
import json
import numpy as np
import random
import networkx as nx

class Data_loading(object):
    def __init__(self, data_set):
        """
        pick data set
        """
        self.data_set = data_set
        if data_set == 1:
            self.init_aminer()
        if data_set == 2 or data_set == 3 or data_set == 4:
            self.init_citceer()

    def init_aminer(self):
        file = open("/home/tingyi/data/full_team_members_year_c_0rmd.txt")
        file2 = open("/home/tingyi/data/merged_DiversityData2_0rmd.txt")
        self.G = nx.DiGraph()
        self.mean_count =None
        self.mean_top = None
        self.mean_prod = None
        self.mean_impact = None
        self.mean_sci = None

        """
        compute std for attribute
        """
        # std_year = np.std([G.nodes[k]['p_year'] for k in G.nodes()])
        # std_cit = np.std([G.nodes[k]['citation'] for k in G.nodes()])
        # std_avg = np.std([G.nodes[k]['avg_citation'] for k in G.nodes()])
        self.std_count = None
        self.std_top = None
        self.std_prod = None
        self.std_impact = None
        self.std_sci = None

        """
        Create dictionary of author to team
        """
        self.dic_author = {}
        for line in file:
            length = np.array(line.split('\t')).shape[0]
            a = np.int(np.array(line.split('\t'))[0])
            author_id = np.int(np.array(line.split('\t'))[4])
            self.dic_author.setdefault(author_id, []).append(a)
            # dic_author[author_id]=a
            if length == 5:
                author_id2 = 0
                author_id3 = 0
                author_id4 = 0
                author_id5 = 0
            if length == 6:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                self.dic_author.setdefault(author_id2, []).append(a)
                author_id3 = 0
                author_id4 = 0
                author_id5 = 0
            if length == 7:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = 0
                author_id5 = 0
                self.dic_author.setdefault(author_id2, []).append(a)
                self.dic_author.setdefault(author_id3, []).append(a)
            if length == 8:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                author_id5 = 0
                self.dic_author.setdefault(author_id2, []).append(a)
                self.dic_author.setdefault(author_id3, []).append(a)
                self.dic_author.setdefault(author_id4, []).append(a)
            if length == 9:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                author_id5 = np.int(np.array(line.split('\t'))[8])
                self.dic_author.setdefault(author_id2, []).append(a)
                self.dic_author.setdefault(author_id3, []).append(a)
                self.dic_author.setdefault(author_id4, []).append(a)
                self.dic_author.setdefault(author_id5, []).append(a)
            if length > 9:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                author_id5 = np.int(np.array(line.split('\t'))[8])
                self.dic_author.setdefault(author_id2, []).append(a)
                self.dic_author.setdefault(author_id3, []).append(a)
                self.dic_author.setdefault(author_id4, []).append(a)
                self.dic_author.setdefault(author_id5, []).append(a)
        file = open("/home/tingyi/data/full_team_members_year_c_0rmd.txt")
        file2 = open("/home/tingyi/data/merged_DiversityData2_0rmd.txt")

        """
        Create dictionary of team to author
        """
        self.dic_team = {}
        for line in file:
            length = np.array(line.split('\t')).shape[0]
            a = np.int(np.array(line.split('\t'))[0])
            author_id = np.int(np.array(line.split('\t'))[4])
            # dic_team[a]=[author_id]
            if length == 5:
                author_id2 = 0
                author_id3 = 0
                self.dic_team[a] = [author_id]
            if length == 6:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                self.dic_team[a] = [author_id, author_id2]
                author_id3 = 0
            if length == 7:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                self.dic_team[a] = [author_id, author_id2, author_id3]
            if length == 8:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                self.dic_team[a] = [author_id, author_id2, author_id3, author_id4]
            if length == 9:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                author_id5 = np.int(np.array(line.split('\t'))[8])
                self.dic_team[a] = [author_id, author_id2, author_id3, author_id4, author_id5]
            if length > 9:
                author_id2 = np.int(np.array(line.split('\t'))[5])
                author_id3 = np.int(np.array(line.split('\t'))[6])
                author_id4 = np.int(np.array(line.split('\t'))[7])
                author_id5 = np.int(np.array(line.split('\t'))[8])
                self.dic_team[a] = [author_id, author_id2, author_id3, author_id4, author_id5]

        """
        Create dictionary of connected team
        """
        self.dic_connect_team = {}
        for kk in self.dic_team.keys():
            author_set = self.dic_team[kk]
            for author_ind in author_set:
                for team_ind in self.dic_author[author_ind]:
                    self.dic_connect_team.setdefault(kk, []).append(team_ind)

        """
        Create dictionary for team attribute
        """
        self.dic_attribute = {}
        for line2 in file2:
            tid = np.int(np.array(line2.split('\t'))[0])
            p_year = np.float(np.array(line2.split('\t'))[1])
            citation = np.float(np.array(line2.split('\t'))[2])
            avg_citation = np.float(np.array(line2.split('\t'))[3])
            country_diversity = np.float(np.array(line2.split('\t'))[4])
            topic_diversity = np.float(np.array(line2.split('\t'))[5])
            productivity_diversity = np.float(np.array(line2.split('\t'))[6])
            impact_diversity = np.float(np.array(line2.split('\t'))[7])
            scientific_age_diversity = np.float(np.array(line2.split('\t'))[8])
            self.dic_attribute[tid] = [country_diversity, topic_diversity, productivity_diversity, impact_diversity,
                                  scientific_age_diversity, tid, p_year, citation]

        self.create_graph()

        """
        Create graph
        """
    def single_node_connection(self,node):
      neighbors = self.dic_connect_team[node]
      hop_nodes_one = []
      for tid in neighbors:
        if not self.G.has_node(tid):
          hop_nodes_one.append(tid)
          country_diversity = self.dic_attribute[tid][0]
          topic_diversity = self.dic_attribute[tid][1]
          productivity_diversity = self.dic_attribute[tid][2]
          impact_diversity = self.dic_attribute[tid][3]
          scientific_age_diversity = self.dic_attribute[tid][4]
          tid_ = self.dic_attribute[tid][5]
          p_year = self.dic_attribute[tid][6]
          citation = self.dic_attribute[tid][7]
          if len(self.dic_team[tid])==1:
            author_id1 = self.dic_team[tid][0]
            author_id2 = 0
            author_id3 = 0
            author_id4 = 0
            author_id5 = 0
          if len(self.dic_team[tid])==2:
            author_id1 = self.dic_team[tid][0]
            author_id2 = self.dic_team[tid][1]
            author_id3 = 0
            author_id4 = 0
            author_id5 = 0
          if len(self.dic_team[tid])==3:
            author_id1 = self.dic_team[tid][0]
            author_id2 = self.dic_team[tid][1]
            author_id3 = self.dic_team[tid][2]
            author_id4 = 0
            author_id5 = 0
          if len(self.dic_team[tid])==4:
            author_id1 = self.dic_team[tid][0]
            author_id2 = self.dic_team[tid][1]
            author_id3 = self.dic_team[tid][2]
            author_id4 = self.dic_team[tid][3]
            author_id5 = 0
          if len(self.dic_team[tid])==5:
            author_id1 = self.dic_team[tid][0]
            author_id2 = self.dic_team[tid][1]
            author_id3 = self.dic_team[tid][2]
            author_id4 = self.dic_team[tid][3]
            author_id5 = self.dic_team[tid][4]
          self.G.add_node(tid,country_diversity=country_diversity)
          self.G.add_node(tid,topic_diversity=topic_diversity)
          self.G.add_node(tid,productivity_diversity=productivity_diversity)
          self.G.add_node(tid,impact_diversity=impact_diversity)
          self.G.add_node(tid,scientific_age_diversity=scientific_age_diversity)
          #G.add_node(tid,node_index=node_index)
          self.G.add_node(tid,tid=tid_)
          self.G.add_node(tid,p_year=p_year)
          self.G.add_node(tid,citation=citation)
          self.G.add_node(tid,author_id1=author_id1)
          self.G.add_node(tid,author_id2=author_id2)
          self.G.add_node(tid,author_id3=author_id3)
          self.G.add_node(tid,author_id4=author_id4)
          self.G.add_node(tid,author_id5=author_id5)
        self.G.add_edge(tid,node)
        self.G.add_edge(node,tid)
      return hop_nodes_one

    def create_nodes_one_hop(self,hop_nodes):
      for nodes in hop_nodes:
        single_node_connection(self.G,nodes)

        #G = nx.DiGraph()
    def create_graph(self):
        tid = list(self.dic_attribute.keys())[0]
        country_diversity = self.dic_attribute[tid][0]
        topic_diversity = self.dic_attribute[tid][1]
        productivity_diversity = self.dic_attribute[tid][2]
        impact_diversity = self.dic_attribute[tid][3]
        scienctific_age_diversity = self.dic_attribute[tid][4]
        tid_ = self.dic_attribute[tid][5]
        p_year = self.dic_attribute[tid][6]
        citation = self.dic_attribute[tid][7]
        if len(self.dic_team[tid])==1:
          author_id1 = self.dic_team[tid][0]
          author_id2 = 0
          author_id3 = 0
          author_id4 = 0
          author_id5 = 0
        if len(self.dic_team[tid])==2:
          author_id1 = self.dic_team[tid][0]
          author_id2 = self.dic_team[tid][1]
          author_id3 = 0
          author_id4 = 0
          author_id5 = 0
        if len(self.dic_team[tid])==3:
          author_id1 = self.dic_team[tid][0]
          author_id2 = self.dic_team[tid][1]
          author_id3 = self.dic_team[tid][2]
          author_id4 = 0
          author_id5 = 0
        if len(self.dic_team[tid])==4:
          author_id1 = self.dic_team[tid][0]
          author_id2 = self.dic_team[tid][1]
          author_id3 = self.dic_team[tid][2]
          author_id4 = self.dic_team[tid][3]
          author_id5 = 0
        if len(self.dic_team[tid])==5:
          author_id1 = self.dic_team[tid][0]
          author_id2 = self.dic_team[tid][1]
          author_id3 = self.dic_team[tid][2]
          author_id4 = self.dic_team[tid][3]
          author_id5 = self.dic_team[tid][4]
        self.G.add_node(tid,country_diversity=country_diversity)
        self.G.add_node(tid,topic_diversity=topic_diversity)
        self.G.add_node(tid,productivity_diversity=productivity_diversity)
        self.G.add_node(tid,impact_diversity=impact_diversity)
        self.G.add_node(tid,scientific_age_diversity=scienctific_age_diversity)
        #G.add_node(tid,node_index=node_index)
        self.G.add_node(tid,tid=tid_)
        self.G.add_node(tid,p_year=p_year)
        self.G.add_node(tid,citation=citation)
        self.G.add_node(tid,author_id1=author_id1)
        self.G.add_node(tid,author_id2=author_id2)
        self.G.add_node(tid,author_id3=author_id3)
        self.G.add_node(tid,author_id4=author_id4)
        self.G.add_node(tid,author_id5=author_id5)

        current_hops = [tid]
        hop_next = []
        hop_index = 0
        hop_size = 10
        while(hop_index<3):
          for node_cur in current_hops:
            hop_one = self.single_node_connection(node_cur)
            hop_next = hop_next + hop_one
          current_hops = hop_next
          print(current_hops)
          hop_next = []
          hop_index += 1

        index = 0
        for node in self.G.nodes():
          self.G.add_node(node,node_index=index)
          index += 1

        """
        Compute mean for attributes, for later normalization
        """
        #mean_year = np.mean([G.nodes[k]['p_year'] for k in G.nodes()])
        #mean_cit = np.mean([G.nodes[k]['citation'] for k in G.nodes()])
        #mean_avg = np.mean([G.nodes[k]['avg_citation'] for k in G.nodes()])
        self.mean_count = np.mean([self.G.node[k]['country_diversity'] for k in self.G.nodes()])
        self.mean_top = np.mean([self.G.node[k]['topic_diversity'] for k in self.G.nodes()])
        self.mean_prod = np.mean([self.G.node[k]['productivity_diversity'] for k in self.G.nodes()])
        self.mean_impact = np.mean([self.G.node[k]['impact_diversity'] for k in self.G.nodes()])
        self.mean_sci = np.mean([self.G.node[k]['scientific_age_diversity'] for k in self.G.nodes()])

        """
        compute std for attribute
        """
        #std_year = np.std([G.nodes[k]['p_year'] for k in G.nodes()])
        #std_cit = np.std([G.nodes[k]['citation'] for k in G.nodes()])
        #std_avg = np.std([G.nodes[k]['avg_citation'] for k in G.nodes()])
        self.std_count = np.std([self.G.node[k]['country_diversity'] for k in self.G.nodes()])
        self.std_top = np.std([self.G.node[k]['topic_diversity'] for k in self.G.nodes()])
        self.std_prod = np.std([self.G.node[k]['productivity_diversity'] for k in self.G.nodes()])
        self.std_impact = np.std([self.G.node[k]['impact_diversity'] for k in self.G.nodes()])
        self.std_sci = np.std([self.G.node[k]['scientific_age_diversity'] for k in self.G.nodes()])

    def init_citceer(self):
        if self.data_set == 2:
            file = open("/home/tingyi/database/citeseer/edges.txt")
            file2 = open("/home/tingyi/database/citeseer/features.txt")
            file3 = open("/home/tingyi/database/citeseer/group.txt")
        if self.data_set == 3:
            file = open("/home/tingyi/database/cora/edges.txt")
            file2 = open("/home/tingyi/database/cora/features.txt")
            file3 = open("/home/tingyi/database/cora/group.txt")
        if self.data_set == 4:
            file = open("/home/tingyi/database/wiki/edges.txt")
            file2 = open("/home/tingyi/database/wiki/features.txt")
            file3 = open("/home/tingyi/database/wiki/group.txt")


        self.G = nx.DiGraph()
        #index = 0
        for line in file:
            line = line.rstrip('\n')
            a = np.int(np.array(line.split(' '))[0])
            b = np.int(np.array(line.split(' '))[1])
            """
            if not self.G.has_node(a):
                self.G.add_node(a, node_index=index)
                #index += 1
            if not self.G.has_node(b):
                self.G.add_node(b, node_index=index)
                #index += 1
            """
            self.G.add_edge(a, b)
            self.G.add_edge(b, a)

        for line in file2:
            line = line.rstrip('\n')
            feat = np.array(line.split(' '))
            feat = [float(i) for i in feat]
            index = np.int(feat[0])
            feat = feat[1:]
            self.G.add_node(index,node_index=index)
            self.G.add_node(index,feature=feat)

        for line in file3:
            line = line.rstrip('\n')
            node = np.int(np.array(line.split(' '))[0])
            label = np.int(np.array(line.split(' '))[1])
            self.G.add_node(node, label=label)





