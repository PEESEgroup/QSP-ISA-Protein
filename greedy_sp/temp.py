from vqc_extension import flatten_list, unflatten_list

a = []
a.append([1, 2, 3])
a.append([1])
a.append([3, 4, 5, 6])
print(a)
b = flatten_list(a)
print(b)
c = unflatten_list(b, a)
print(c)
