import json
import faiss
import argparse
import numpy as np
import torch
import pickle
import os
import multiprocessing
import signal
from sentence_transformers import SentenceTransformer

def work(args, i, cluster_i_embeddings_ori, cluster_center_i_embeddings, sentences_flip):
    
    output, log_output = [], []
    print('cluster: ', i)
    print('size of cluster: ', len(cluster_i_embeddings_ori))
    # 余弦相似度降序
    _, index = torch.sort(torch.cosine_similarity(cluster_center_i_embeddings, cluster_i_embeddings_ori), descending=True)
    cluster_i_embeddings = cluster_i_embeddings_ori[index.numpy(), :] # 按降序翻转
    sentences_flip = sentences_flip[index.numpy(), :]
    # 互相算余弦相似度
    # cluster_i_embeddings = cluster_i_embeddings / torch.norm(cluster_i_embeddings, dim=-1, keepdim=True)
    # pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T
    pairwise_sim_matrix = torch.cosine_similarity(cluster_i_embeddings.unsqueeze(1), cluster_i_embeddings.unsqueeze(0), dim=-1) # 需要广播，时耗
    triu_sim_matrix = torch.triu(torch.tensor(pairwise_sim_matrix), diagonal=1) 
    keep_indices = []
    remove_indices, remove_likely_indices, remove_likely_sim = [], [], [] # 移除的元素，移除原因（和哪个保留最像，相似度）
    for j in range(len(cluster_i_embeddings)): # 遍历簇内元素
        if j == 0:
            keep_indices.append(j)
            continue
        # 从sim矩阵中取出keep_indices行
        max_prob, max_ids = torch.max(triu_sim_matrix[keep_indices, j], 0)
        if max_prob < 1 - args.epsilon: # 保留
            keep_indices.append(j)
        else:
            remove_indices.append(j)
            remove_likely_indices.append(max_ids)
            remove_likely_sim.append(str(max_prob.item()))
            
    keep_sentence = sentences_flip[keep_indices].squeeze(1)
    print('size of keep: ', len(keep_sentence))
    output.extend(list(zip(keep_sentence.tolist(), [i] * len(keep_sentence)))) # 存储(sentences1, cluster_id) 

    remove_sentence = sentences_flip[remove_indices].squeeze(1)
    remove_likely_sentence = sentences_flip[remove_likely_indices].squeeze(1)
    log_output.extend(list(zip(remove_sentence.tolist(), remove_likely_sim, remove_likely_sentence.tolist()))) # 存储(sentences1, cluster_id) 

    with open(os.path.join(args.log_path, 'log_out_'+args.embedding_path.split('/')[-1].split('-')[0]+'_'+str(i)+'.json'), 'w', encoding='utf-8') as fw:
        json.dump(log_output, fw, ensure_ascii=False, indent=4)

    with open(os.path.join(args.output_path, 'data_out_'+args.embedding_path.split('/')[-1].split('-')[0]+'_'+str(i)+'.json'), 'w', encoding='utf-8') as fw:
        json.dump(output, fw, ensure_ascii=False)


def throw_error(e): 
    print(e.__cause__) # 错误回调函数收到的已经是异常了，不用raise
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL) # 主进程和子进程在linux同一个进程组中，进程组id相同


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/data/px/chaodata/dedup_instruction_2.79M.json')
    parser.add_argument('--output_path', default='semdedupe_strip_test')
    parser.add_argument('--log_path', default='semdedupe_strip_log_test')
    parser.add_argument('--embedding_path', default='/data/px/jingyazang/text2vec-base-chinese')
    parser.add_argument('--saved_embedding_path', default='text2vec_sentence_embeddings_strip.pkl')
    parser.add_argument('--saved_cluster_path', default='text2vec_faiss_data_iter_100_centroid_20000_strip.pkl')
    parser.add_argument('--ncentroids', type=int, default=20000)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.175)
    
    parser.add_argument('--first_train', action='store_true')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    log_path = args.log_path
    embedding_path = args.embedding_path
    saved_embedding_path = args.saved_embedding_path
    first_train = args.first_train
    ncentroids = args.ncentroids # 聚类中心数
    niter = args.niter # 聚类迭代数
    saved_cluster_path = args.saved_cluster_path
    epsilon = args.epsilon # 1-eps为语义相似度threshold

    if not os.path.exists(output_path): os.mkdir(output_path)
    if not os.path.exists(log_path): os.mkdir(log_path)

    pool = multiprocessing.Pool(processes=1) # 传入1或者None默认最大cpu数，必须放在开头？
    # part 1. 读取数据和embedding
    with open(input_path, 'r', encoding='utf-8') as fr: # sentences: [句1, 句2, ...]
        sentences = json.load(fr) 
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()

    if first_train: # 第一次train要encode
        model = SentenceTransformer(embedding_path)
        sentence_embeddings = model.encode(sentences) # 载入embedding, 时耗

        with open(saved_embedding_path, 'wb') as fw:
            pickle.dump(sentence_embeddings, fw)
    else:
        fr = open(saved_embedding_path, 'rb')
        sentence_embeddings = pickle.load(fr)

    # part 2. 聚类 
    x = np.array(sentence_embeddings)
    n, m = x.shape # n: 数量，m: 维度
    
    if first_train:
        kmeans = faiss.Kmeans(m, ncentroids, niter=niter, verbose=True, gpu=True)
        kmeans.train(x)
        cluster_embeddings = kmeans.centroids # 每个簇中心的embedding
        D, I = kmeans.index.search(x, 1) # 每个点到簇中心的距离，每个点属于哪个簇

        with open(saved_cluster_path, 'wb') as fw:
            pickle.dump({'cluster_embeddings': cluster_embeddings, 'I': I}, fw)
    else:   
        fr = open(saved_cluster_path, 'rb')
        data = pickle.load(fr)

        cluster_embeddings = data['cluster_embeddings']
        I = data['I']

    # part 3. 语义去重
    cluster_center_embeddings = torch.tensor(data['cluster_embeddings']).float() # 11000 * 768
    x = torch.tensor(sentence_embeddings).float() # 2M * 768
    n, m = x.shape # n: 数量，m: 维度
    I = torch.tensor(data['I']).long() # 2M * 1    
    
    
    for i in range(ncentroids): # 每一个簇中
        choose = torch.where(I==i)[0]
        cluster_i_embeddings_ori = x[choose].reshape(-1, m) # 簇内数据的embedding
        cluster_center_i_embeddings = cluster_center_embeddings[i]
        
        sentences_flip = []
        for j in choose:
            sentences_flip.append(sentences[j.item()])
        sentences_flip = np.expand_dims(np.array(sentences_flip), 1)

        pool.apply_async(work, (args, i, cluster_i_embeddings_ori, cluster_center_i_embeddings, sentences_flip), error_callback=throw_error)
    
    pool.close()
    pool.join()
    print("Complete!!")
