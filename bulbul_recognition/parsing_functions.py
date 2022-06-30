# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:35:06 2022

@author: royf2

"""




def read_label_file(file_path):
    '''
    from label file (.txt) - reads starting and ending points (sec) of bulbul-labeled parts in coresponding wav file
    '''
    with open(file_path,'r') as label_file:
        line_count = 0
        label_tuples = []
        for line in label_file:
            line_count += 1
            # read 2 first 'words' of evry odd line
            if line_count % 2 == 1:
                label_tuples.append(line.split()[:2])
                       
    return label_tuples    
            
            
            
# file_path = 'train/20200816_053317.txt'         
# label_tuples = read_label_file(file_path)
# print(label_tuples)