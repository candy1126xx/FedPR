import numpy as np, random, torch, os, json


def split_label(num_users, num_classes, n):
    shard_per_class = int(n * num_users / num_classes)
    classes_list = list(range(num_classes)) * shard_per_class
    classes_list = np.array(classes_list).reshape((num_users, -1)).tolist()

    i = 0
    while i < 10000:# 0, 10000, 
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

    return classes_list

def pathological(train_dataset, seed, num_classes, num_users, n):
    train_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    # {label: [id]}
    idxs_dict = {}
    for i in range(len(train_dataset)):
        label = torch.tensor(train_dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    # {label: [[id],[id]...]}
    shard_per_class = int(n * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape(shard_per_class, -1)
        x = list(x)
        idxs_dict[label] = x

    classes_list = split_label(num_users, num_classes, n)
    # classes_list = [[0,1,2,3,4,5,6,7,8,9]]
    label_counts = torch.zeros(num_users, num_classes)
    #
    for i in range(num_users):
        classes = classes_list[i]
        train_data = []
        for label in classes:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            idx_pop = idxs_dict[label].pop(idx)
            train_data.append(idx_pop)
            label_counts[i, label] += len(idx_pop)
        train_groups[i] = np.concatenate(train_data)
        random.shuffle(train_groups[i])
    return train_groups, classes_list, label_counts


def pathological_lt(test_dataset, num_classes, num_users, way, classes_list):
    test_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    # {label: [id]}
    idxs_dict = {}
    for i in range(len(test_dataset)):
        label = torch.tensor(test_dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    # {label: [[id],[id]...]}
    shard_per_class = int(way * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape(shard_per_class, -1)
        x = list(x)
        idxs_dict[label] = x
    for i in range(num_users):
        classes = classes_list[i]
        test_data = []
        for label in classes:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            test_data.append(idxs_dict[label].pop(idx))
        test_groups[i] = np.concatenate(test_data)
        random.shuffle(test_groups[i])
    return test_groups


def practical(train_dataset, num_users, num_classes):
    train_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = np.array(train_dataset.targets)
    # {label: [id]}
    idxs_dict = {}
    for i in range(len(train_dataset)):
        label = torch.tensor(train_dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    label_counts = torch.zeros(num_users, num_classes, dtype=float)
    for i in range(num_users):
        user_data = np.array([], dtype=np.int64)
        if i==0:
            main_class = [0, 1, 2, 3, 4]
            amount = 1000
        elif i== 1:
            main_class = [5, 6, 7, 8, 9]
            amount = 1000
        main_class_idx = np.array([], dtype=np.int64)
        other_class_idx = np.array([], dtype=np.int64)
        for j in range(num_classes):
            if j in main_class:
                main_class_idx = np.concatenate((main_class_idx, idxs_dict[j]),axis=0)
            else:
                other_class_idx = np.concatenate((other_class_idx, idxs_dict[j]),axis=0)
        user_data = np.concatenate((user_data, np.random.choice(main_class_idx, int(amount*0.5), replace=False)),axis=0)
        user_data = np.concatenate((user_data, np.random.choice(other_class_idx, int(amount*0.5), replace=False)),axis=0)
        train_groups[i] = user_data
        random.shuffle(train_groups[i])
        for idx in user_data:
            label_counts[i, labels[idx]] += 1
    return train_groups, label_counts


def practical_lt(test_dataset, num_users, num_classes, label_counts):
    test_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = np.array(test_dataset.targets)
    idxs_dict = {}
    for i in range(len(test_dataset)):
        label = torch.tensor(test_dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    total_label_counts = torch.sum(label_counts, dim=1)
    for i in range(num_users):
        user_data = np.array([], dtype=np.int64)
        for j in range(num_classes):
            amount = int(100 * label_counts[i, j].item() / total_label_counts[i].item())
            if amount > 0:
                user_data = np.concatenate((user_data, random.sample(idxs_dict[j], amount)),axis=0)
        test_groups[i] = user_data
        random.shuffle(test_groups[i])
    return test_groups


def from_json(dataset, num_users, num_classes):
    datalist = []
    groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    label_counts = torch.zeros(num_users, num_classes)
    for i in range(num_users):
        b = len(datalist)
        x_list, y_list = dataset[str(i)]['x'], dataset[str(i)]['y']
        for row in range(len(x_list)):
            label = int(y_list[row])
            datalist.append((torch.tensor(x_list[row]), torch.tensor(label)))
            label_counts[i][label] += 1
        groups[i] = [j for j in range(b, len(datalist))]
    return datalist, groups, label_counts
    