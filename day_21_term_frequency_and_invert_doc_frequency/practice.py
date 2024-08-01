import numpy as np
import math

def compute_word_count(doc_np):
    idx, counter = np.unique(doc_np, return_index=True, return_counts=True)[1:3]
    unique_words = doc_np[idx]
    dtype = [('idx', 'int'), ('word', 'U10'), ('count', int)]
    original_words = np.array([(i, j, k) for i, j, k in zip(idx, unique_words, counter)], dtype=dtype)
    original_words = np.sort(original_words, order='idx')[['word', 'count']]

    return original_words

def compute_tf(docs):
    splitted_docs = [np.array(doc.split()) for doc in docs]
    docs_dict = dict()

    for i in range(len(splitted_docs)):
        original_words_count = compute_word_count(splitted_docs[i])
        dtype = [('word', 'U10'), ('tf', float)]
        sub_dict = np.array([(word, round(np.divide(n, len(splitted_docs[i])), 4)) for word, n in original_words_count], dtype=dtype)
        docs_dict[i] = sub_dict

    return docs_dict

def compute_idf(docs):
    splitted_docs = [np.array(doc.split()) for doc in docs]
    docs_dict = dict()

    for i in range(len(splitted_docs)):
        idx = sorted(np.unique(splitted_docs[i], return_index=True)[1])
        unique_words = splitted_docs[i][idx]
        word_in_docs_count_np = np.array([len(np.where(np.char.find(docs, word) >= 0)[0]) for word in unique_words])
        idf_np = np.round(np.log(len(docs) * 1 / (1 + word_in_docs_count_np)), decimals=4)
        dtype = [('word', 'U10'), ('idf', float)]
        sub_dict = np.array([(word, n) for word, n in zip(unique_words, idf_np)], dtype=dtype)
        docs_dict[i] = sub_dict
    
    return docs_dict

def compute_tf_idf(tf_np_dict, idf_np_dict):
    dtype = [('word', 'U10'), ('tf_idf', 'float64')]
    results = dict()
    for i in range(len(tf_np_dict.keys())):
        tf_idf_np = np.round(tf_np_dict[i]['tf'] * idf_np_dict[i]['idf'], decimals=4)
        new_np = np.empty(tf_idf_np.shape, dtype=dtype)
        new_np['word'] = tf_np_dict[i]['word']
        new_np['tf_idf'] = tf_idf_np
        results[i] = new_np
    
    return results

def main():
    documents = [
        "Tôi thích học AI",
        "AI là trí tuệ nhân tạo",
        "AGI là siêu trí tuệ nhân tạo"
    ]

    doc_tf_dict = compute_tf(documents)
    print(f"TF results:\n {doc_tf_dict}")

    doc_idf_dict = compute_idf(documents)
    print(f"IDF results:\n{doc_idf_dict}")

    print(f"TF-IDF results:\n{compute_tf_idf(doc_tf_dict, doc_idf_dict)}")


if __name__ == "__main__":
    main()
