import numpy as np
import operator
from scipy.spatial import distance

def cal_recall_rate(in_list, ground_truth):
    if len(in_list) != len(ground_truth):
        raise Exception("Total items in input and in ground-truth list should be equal")

    num_recall = 0
    num_total = 0
    recall_list = []

    for i in range(len(in_list)):
        in_set = set(in_list[i])
        gt_set = set(ground_truth[i])
        overlap = in_set & gt_set

        recall_list.append(overlap)
        num_recall += len(overlap)
        num_total += len(gt_set)

    recall_rate = num_recall/num_total   
    return recall_rate, recall_list


def cal_class_recall(test_labels, recall_list, test_tag_list):
    # Calculae recall rate of each class
    cls_recalls = dict()
    cls_tags = dict()
    for i in range(len(recall_list)):
        cls_name = test_labels[i]
        if cls_name in cls_recalls:
            cls_recalls[cls_name] += len(recall_list[i])
        else:
            cls_recalls[cls_name] = len(recall_list[i])
                                                
        if cls_name in cls_tags:
            cls_tags[cls_name] += len(test_tag_list[i])
        else:
            cls_tags[cls_name] = len(test_tag_list[i])

    recall_rates = dict()
    for key, val in cls_recalls.items():
        recall_rates[key] = val / cls_tags[key]
                                                                                            
    recall_sorted = sorted(recall_rates.items(), key=operator.itemgetter(1), reverse=True)
    return recall_rates, recall_sorted

# Get KNN of a vector
# Using cosine similarity
def get_knn_cos(in_vector, word_embeddings, KNN=10):
    num_feats = word_embeddings.shape[0]
    nn_dist = np.zeros(num_feats)
    for i in range(num_feats):
        vec1 = np.array(in_vector)
        vec2 = np.array(word_embeddings[i])
        # Cosine similarity is better but very slow
        nn_dist[i] = distance.cosine(vec1, vec2)
    knn_word_ind = nn_dist.argsort()[0:KNN+1]
    return knn_word_ind

def get_knn(in_vector, word_embeddings, KNN=10):
    scores = in_vector.dot(word_embeddings.transpose()) / np.linalg.norm(word_embeddings, axis=1)
    knn_word_ind = (-scores).argsort()[0:KNN+1]
    return knn_word_ind

# If the number of words is not too large (<1000), 
# we can use pre-computed kernel

def get_knn_in_kernel(wordID_list, kernel, KNN=10):
    # TODO: check kernel size
    
    for i in range(len(wordID_list)):
        for j in range(len(wordID_list[i])):
            dists = kernel[wordID_list[i][j], :]
            knn_indice = (-dists).argsort()[0:KNN] # ID 0 is target word itself
            out_list.append(knn_indice)

    return out_list


### HARRISON dataset preprocessing ###

# Convert TAG string to word2vec API

def tag_string_to_w2v_id(tag_strings, w2v_dictionary, separator=' '):
    tag_list = []
    miss_tags = dict()
    for tag_str in tag_strings:
        tags = tag_str.split(separator)
        tag_ind = []
        for tag in tags:
            try:
                tag_ind.append(w2v_dictionary[tag])
            except:
                # The tag is not in dictionary
                # print("Skip", tag)
                if tag not in miss_tags:
                    miss_tags[tag] = 0
                else:
                    miss_tags[tag] += 1

        tag_list.append(tag_ind)

    return tag_list, miss_tags


# Get HARRISON labels from image paths
# HARRISON dataset classifies images into 50 categories. Images are saved in corresponding folders

def img_paths_to_labels(image_list):
    harrison_labels = []  # Retrieve image folder as its label"
    for image_path in image_list:
        label = image_path.split('/')[-2]
        # HARRISON dataset have some label mismatches that need to be fixed
        if label == "girls":
            harrison_labels.append("girl")
        elif label == "friends":
            harrison_labels.append("friend")
        elif label == "shoes":
            harrison_labels.append("shoe")
        elif label == "eyes":
            harrison_labels.append("eye")
        else:
            harrison_labels.append(label)

    return harrison_labels


# Convert HARRISON text labels into sequential no. and dicionary

def labels_to_seq_no(label_list):
    lbl_id = -1
    prev_tag = ''
    label_id_list = []
    label_dict = dict()
    for label in label_list:
        if prev_tag != label:
            prev_tag = label
            lbl_id += 1
            label_dict[lbl_id] = label

        label_id_list.append(lbl_id) 

    return label_id_list, label_dict


# Load class name of ImageNet (1000 classes)
# The class names of ImageNet in Keras are in 'imagenet_class_index.json'

def load_imnet_class_name(filename):
    # Load ImageNet class names
    with open(filename) as f:
        imnet_class = json.load(f)

    im_cls_tag = []
    for i in range(0, len(imnet_class)):
        if imnet_class[str(i)][1] in tag_dictionary:
                im_cls_tag.append(imnet_class[str(i)][1])

    return im_cls_tag

# Convert HARRISON tag ground-truth string into wordID
def get_tag_id_list(tag_strings, tag_dictionary, select_ind):
    tag_list, miss_tags = tag_string_to_w2v_id(tag_strings, tag_dictionary)
    test_tag_list = []
    for tid in select_ind:
        test_tag_list.append(tag_list[tid])
    
    return test_tag_list
    
### For Kaggle hamming loss ###

# Id, Sample, Label, Predicted
def create_kaggle_hamming_loss_csv(filename, tag_list):
    with open(filename, 'w') as fp:
        fp.write("Id,Sample,Label,Predicted\n")
        i = 0
        for sid, tags in enumerate(tag_list):
            for tag in tags:
                fp.write('{},{},{},{}\n'.format(i, sid, tag, 'True'))
                i = i + 1
                
# Id, Sample, Label, Predicted
def save_kaggle_hamming_loss_csv(filename, in_list):
    with open(filename, 'w') as fp:
        fp.write("Id,Sample,Label,Predicted\n")
        for ret in in_list:
            fp.write('{},{},{},{}\n'.format(ret[0], ret[1], ret[2], ret[3]))

def cal_hamming_loss(in_list, gt_list):
    assert len(in_list) == len(gt_list)
    result_list = []
    loss = 0
    i = 0
    for sid, gt_tags in enumerate(gt_list):
        for gt_tag in gt_tags:
            predict = False
            if gt_tag in in_list[sid]:
                predict = True
            else:
                loss = loss + 1
            #fp.write('{},{},{},{}\n'.format(i, sid, tag, 'True'))
            result = []
            result.append(i)
            result.append(sid)
            result.append(gt_tag)
            result.append(predict)
            result_list.append(result)
            i = i + 1
    
    loss = loss / len(result_list)
    return loss, result_list
