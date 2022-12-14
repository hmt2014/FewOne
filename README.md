# FewOne

This directory is supplemented with the paper entitled
"Classical Sequence Match is Competitive Few-Shot One-Class Learner"

## Environment

    Python: 3.7.13
    Pytorch: 1.11.0
    transformers: 4.14.1
    sklearn: 1.0.2
    numpy: 1.21.2




## Datasets
The datasets directory contains the two datasets, ACD and Huffpost, in the experiments.
ACD is too large. Here we only upload the Huffpost.
Read dataset with the following code:


    text_word_id, text_id_to_word = pickle.load(open(path + "/text_id_info.p", "rb"))
    word_embedding = pickle.load(open(path + "/embedding_info.p", "rb")) # glove word embeddings
    train_data, dev_data, test_data = pickle.load(open(path + "/data_info.p", "rb"))  # glove
    train_data, dev_data, test_data = pickle.load(open(path + "/data_info_bert.p", "rb")) # bert
    train_data, dev_data, test_data = pickle.load(open(path + "/data_info_distilbert.p", "rb")) # distilbert

## Models
The code of all models.
It is worth noting that 4, 5, 6, 7. The models are directly fine-tuned after naive training,
with the support set of testing classes. 
Each single method can be runed directly

## compute_cov_score
The code about how to compute cov score for features.

## Citation
    @inproceedings{hu-etal-2022-classical,
        title = "Classical Sequence Match Is a Competitive Few-Shot One-Class Learner",
        author = "Hu, Mengting  and
          Gao, Hang  and
          Bai, Yinhao  and
          Liu, Mingming",
        booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
        month = oct,
        year = "2022",
        address = "Gyeongju, Republic of Korea",
        publisher = "International Committee on Computational Linguistics",
        url = "https://aclanthology.org/2022.coling-1.419",
        pages = "4728--4740"
    }

