# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:11:19 2022

@author: royf2
"""

def get_bin_label(frame_i, frame_length, hop_length, labels):
    '''
    scans labels to find overlappings with the frame,
    sums them up and returns 1 if at least 30% of the frame is labeled. 0 otherwise.
    '''
    frame_start = frame_i * hop_length
    frame_end = frame_start + frame_length
    sum_bulbul_labeled_samples = 0
    
    for label in labels:
        label_start = label[0]
        label_end = label[1]
        # ****   [     ]
        if frame_start > label_end:
            continue
        # [     ]   ****
        if  frame_end < label_start:
            break
        # ..[...**]***..
        if frame_start < label_start and label_start < frame_end < label_end:
            sum_bulbul_labeled_samples += (frame_end - label_start + 1)
        # ..**[***..]..
        elif label_start < frame_start < label_end and frame_end > label_end:
            sum_bulbul_labeled_samples += (label_end - frame_start + 1)
        # ..[.**..]..
        elif frame_start < label_start and frame_end > label_end:
            sum_bulbul_labeled_samples += (label_end - label_start + 1)
        # ..**[*****]*..
        else:
            return 1

    if sum_bulbul_labeled_samples/frame_length > 0.3:
        return 1
    
    return 0

            

    
    
               
               
               