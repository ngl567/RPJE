# RPJE
AAAI 2020: Rule-Guided Compositional Representation Learning on Knowledge Graphs

This is our c++ source code and data for the paper:
>Guanglin Niu, Yongfei Zhang, Bo Li, Peng Cui, Si Liu, Jingyang Li, Xiaowei Zhang. Rule-Guided Compositional Representation Learning on Knowledge Graphs. In AAAI, 2020. [Paper in arXiv](https://arxiv.org/abs/1911.08935).

Author: Dr. Guanglin Niu (beihangngl at buaa.edu.cn)

## Introduction
Rule and Path-based Joint Embedding (RPJE) takes full advantage of the explainability and accuracy of logic rules, the generalization of knowledge graph (KG) embedding as well as the supplementary semantic structure of paths. RPJE achieves better performance with higher accuracy and explainability on KG completion task.

## Dataset
We provide four datasets: FB15K, FB15K237, WN18 and NELL-995. You can find all the datasets as well as the encoded rules mined from each dataset in the folders ./data_FB15K, ./data_FB15K237, ./data_WN18, ./data_NELL-995, which containing the following files:
* entity2id.txt: Entity file containing all the entities in the dataset. Each line is an entity and its id: (entity name, entity id).
* relation2id.txt: Relation file containing all the relations in the dataset. Each line is an relation and its id: (relation name, relation id).
* train.txt: Training data file containing all the triples in train set. Each line is a triple in the format (head entity name, tail entity name, relation name).
* valid.txt: Validation data file containing all the triples in valid set. Each line is a triple in the format (head entity name, tail entity name, relation name).
* test.txt: Testing data file containing all the triples in test set. Each line is a triple in the format (head entity name, tail entity name, relation name).
* train_pra.txt: Training data file containing all the triples with the paths linking the entity pairs. Each train instance is composed of two lines. The former line is a triple in train.txt but in the format (head entity name, tail entity name, relation id), and the latter line is the paths information linking the entity pair of this triple in the format (the number of paths, relation id list in path 1, reliability of path 1, relation id list in path 2, reliability of path 2,...).
* test_pra.txt: Testing data file containing all the triples with the paths linking the entity pairs. Each test instance is composed of two lines. The former line is a triple in test.txt but in the format (head entity name, tail entity name, relation id), and the latter line is the paths information linking the entity pair of this triple in the format (the number of paths, length of path 1, relation id list in path 1, reliability of path 1, length of path 2, relation id list in path 2, reliability of path 2,...).
* confidence.txt: Confidence file containing all the paths with their corresponding direct relations in the dataset. The former line is a path in the format (length of the path, relation id list in the path), and the latter line is all the relations related to this path in the format (number of the relations, relation 1 id, reliability of the path representing relation 1, relation 2 id, reliability of the path representing relation 2,...).

In each folder of dataset, the folder ./rule containing all the encoded rules with various confidence threshold:
* rule_path[n].txt: Rules file containing all the encoded rules of length 2 mined from the dataset with the confidence threshold n. Each line is an encoded rule in the format (id of the first relation in rule body, id of the second relation in rule body, id of the relation in rule head).
* rule_relation[n].txt: Rules file containing all the encoded rules of length 1 mined from the dataset with the confidence threshold n. Each line is an encoded rule in the format (id of the relation in rule body, id of the relation in rule head).

Please note that all the above data contain the positive instances for training. The negative instances are generated in the process of training.

## Example to Run the codes
Firstly, select the dataset and the rules confidence threshold in the training file Train_RPJE.cpp. And then implement the settings:
* dimension:    dimension of entity and relation embeddings
* nbatches:     number of batches for each epoch
* nepoches:     number of epoches
* alpha:        learning rate
* maring:       margin in max-margin loss for training
* lambda:       weight of paths and length 2 rules in loss function
* lambda_rule:  weight of length 1 rules in loss function

### Compile
    g++ Train_RPJE.cpp -o Train_RPJE -O2
    g++ Test_RPJE.cpp -o Test_RPJE -O2
### Train
    ./Train_RPJE
### Test
    ./Test_RPJE
    
## Acknowledge

    @inproceedings{RPJE19,
      author    = {Guanglin Niu and
                   Yongfei Zhang and
                   Bo Li and
                   Peng Cui and
                   Si Liu and
                   Jingyang Li and
                   Xiaowei Zhang},
      title     = {Rule-Guided Compositional Representation Learning on Knowledge Graphs},
      booktitle = {arXiv preprint},
      year      = {2019}
    }
