# Ebony Clarke
# Assignment 1

#1. Flatten Function
def flatten(test_list):
    if isinstance(test_list, list):
        if len(any_list) == 0:
            return []
        first, rest = any_list[0], any_list[1:]
        return flatten(first) + flatten(rest)
    else:
        return [any_list]

#2. Powerset Function
def powerset(list):
	output = [[]]
	for i in list:
		output.extend([x + [i] for x in output])
	return output 


#3. Permuations Function
def all_perms(list):
    if len(list) == 0:
        return []
    if len(list) == 1:
        return [list]
    new_list = []
    for i in range(len(list)):
       rest_list = list[:i] + list[i+1:]
       for p in all_perms(rest_list):
           new_list.append([list[i]] + p)
    return new_list
