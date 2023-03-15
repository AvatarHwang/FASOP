def conv_array(array):
    num_list = list(map(int, array.strip().split(",")))
    print(num_list)
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
    return balance

print(conv_array("0, 7, 13, 19, 25, 31, 37, 43, 50"))