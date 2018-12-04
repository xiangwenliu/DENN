"""
Select the test accuracy from the grid search results
"""
import os
import numpy as np


def file_name(file_dir):
    for lists in os.listdir(file_dir):
        print(lists)

def read_data(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            str = line.strip()
            if len(str) > 1:
                arrylist = str.split(' ')
                result.append(arrylist)
    arr = np.array(result)
    return arr


def compare_dendritevalue(path_dir):
    result = []
    for lists in os.listdir(path_dir):
        filename = str(lists)+'.txt'
        # filename = str(lists)
        arr = []
        result_path = os.path.join(path_dir, lists)
        file = os.path.join(result_path, filename)
        branch, val = compare_singleresultvalue(file)
        arr.append(lists)#dataset name
        arr.append(branch)#branch number
        arr.append(val)
        result.append(arr)
    # print(len(result))
    return result

"""select the best validation accuracy"""
def compare_singleresultvalue(filename):
    result = []
    with open(filename, 'r') as f:
        count = 0
        maxval = 0.0
        branch = 0
        val_maxval = 0.0
        for line in f.readlines():
            str = line.strip()
            str = str.strip('\n')
            if len(str) > 6:
                # print(str)
                arrylist = str.split(',')
                value = float(arrylist[7])
                dendrite = int(arrylist[1])
                val_value = float(arrylist[6])
                # if value > maxval:
                if val_value > val_maxval:
                    val_maxval = val_value
                    maxval = value
                    branch = dendrite
                if val_value == val_maxval:
                    if dendrite > branch:# use the biggest branch
                        maxval = value
                        branch = dendrite
    return branch, round(float(maxval),4)

if __name__ == '__main__':


    result_dir = './result'
    result = compare_dendritevalue(result_dir)
    for i in range(len(result)):
        print(result[i][2])

