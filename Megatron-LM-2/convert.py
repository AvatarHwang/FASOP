#!/usr/bin/env python3
def conv_array(array):
    num_list = array
    # num_list = list(map(int, array.strip().split(",")))

    balance= ''
    for i in range(len(num_list)-1):
        if num_list[i] == 0:
            balance += str(num_list[i+1] - num_list[i] - 1 )
        elif i == len(num_list) - 2 :
            balance += str(num_list[i+1] - num_list[i] - 1 )
            break
        else:
            balance += str(num_list[i+1] - num_list[i])
        balance += '-'
    print(balance)
    return balance

def conv_array(array):
    num_list = array
    # num_list = list(map(lambda x: int(x.strip()), array.split(",")))
    num_list[0] = num_list[0] - 1
    num_list[-1] = num_list[-1] - 1
    num_list = map(str, num_list)
    print("-".join(num_list))
    return "-".join(num_list)

conv_array([18, 11, 10, 11])
conv_array([9, 7, 6, 6, 5, 6, 5, 6])
conv_array([5, 4, 4, 4, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3])
conv_array([18, 11, 10, 11])
conv_array([9, 7, 6, 6, 5, 6, 5, 6])
conv_array([5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3])
conv_array([11, 6, 6, 6, 5, 5, 5, 6])
conv_array([5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
conv_array([2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 3])
conv_array([8, 6, 6, 6, 6, 6, 6, 6])
conv_array([4, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2])
conv_array([6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3])
conv_array([2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3])