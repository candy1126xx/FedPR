
import json, random, numpy as np

# def split_label(num_users, num_classes, n):
#     row_sols = []

#     def combine1(length, temp, list_num):
#         if length == n:
#             row_sols.append(temp[:])
#             return
#         for i, num in enumerate(list_num):
#             temp.append(num)
#             combine1(length+1, temp, list_num[i+1:])
#             temp.pop()

#     combine1(0, [], range(num_classes))
#     solutions = []

#     def check(sol):
#         label_counts = [0 for _ in range(num_classes)]
#         for s in sol:
#             for ss in s:
#                 label_counts[ss] += 1
#         for label_count in label_counts:
#             if label_count != num_users*n/num_classes:
#                 return False
#         return True

#     def combine2(length, temp, list_sol):
#         if length == num_users:
#             if check(temp):
#                 solutions.append(temp[:])
#                 print(len(solutions))
#             return
#         for i, sol in enumerate(list_sol):
#             temp.append(sol)
#             combine2(length+1, temp, list_sol)
#             temp.pop()
    
#     combine2(0, [], row_sols)
#     result_json = json.dumps(solutions)
#     with open('sss.json','w+') as file:
#         file.write(result_json)


def split_label(num_users, num_classes, n):
    shard_per_class = int(n * num_users / num_classes)
    classes_list = list(range(num_classes)) * shard_per_class
    classes_list = np.array(classes_list).reshape((num_users, -1)).tolist()

    i = 0
    while i < 10:# 0, 10000, 
        u1 = random.randint(0, num_users-1)
        u2 = random.randint(0, num_users-1)
        if u1 == u2:
            continue
        u1_labels = classes_list[u1]
        u2_labels = classes_list[u2]
        ava_u1_l = []
        for label in u1_labels:
            if label not in u2_labels:
                ava_u1_l.append(label)
        if len(ava_u1_l)==0:
            continue
        ava_u2_l = []
        for label in u2_labels:
            if label not in u1_labels:
                ava_u2_l.append(label)
        if len(ava_u2_l)==0:
            continue
        u1_change = random.choice(ava_u1_l)
        u2_change = random.choice(ava_u2_l)
        u1_labels.remove(u1_change)
        u1_labels.append(u2_change)
        u2_labels.remove(u2_change)
        u2_labels.append(u1_change)
        i += 1

    result_json = json.dumps(classes_list)
    with open('sss.json','w+') as file:
        file.write(result_json)

split_label(100, 10, 2)