'''
  统计分析每个类别数据的最大长度、最小长度、平均长度（llama分词）、数据比例
  并用jieba分词绘制词云图
'''
import argparse
import json
import os
from tokenizer import Tokenizer
import jieba
import collections
import re
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/data/px/jingyazang/cls_dedup_instruction_2.79M_semdedupe_strip')
    parser.add_argument('--output_path', default='/data/px/jingyazang/data_analysis/')
    parser.add_argument('--tokenizer_path', default='/data/px/jingyazang/llama-7b-hf/tokenizer.model')
    parser.add_argument('--stopwords_path', default='/home/jingyazang/projects/EDA_NLP_for_Chinese-master/stopwords/hit_stopwords.txt')
    args = parser.parse_args()

    output_path = args.output_path
    input_path = args.input_path
    tokenizer_path = args.tokenizer_path
    stopwords_path = args.stopwords_path
    
    lst = [name for name in os.listdir(input_path) if name.endswith('.json')]
    tokenizer = Tokenizer(tokenizer_path)

    f1 = open(os.path.join(output_path, 'statistics.txt'), 'w', encoding='utf-8')
    f1.write('type\taverage_len\tmin_len\tmax_len\tproportion\n')
    all_count = 0

    for name in lst: # 每个type下
        with open(os.path.join(input_path, name), 'r', encoding='utf-8') as fr:      
            all_count += len(fr.readlines())

    for name in lst: # 每个type下
        
        min_len = float('inf')
        max_len = -float('inf')
        avg_len, count = 0, 0
        data = []
        
        type_name = name.split('_')[0]
        with open(os.path.join(input_path, name), 'r', encoding='utf-8') as fr:
            
            for line in fr:
                json_line = json.loads(line)
                data.append(json_line['instruction'])

                encoded_text = tokenizer.encode(json_line['instruction'], bos=True, eos=True)
                text_len = len(encoded_text)

                min_len = min(text_len, min_len)
                max_len = max(text_len, max_len)
                avg_len += text_len

                count += 1
                
        f1.write('{}\t{:.2f}\t{}\t{}\t{:.2f}%\n'.format(type_name, avg_len / count, min_len, max_len, count / all_count * 100))
        print('{}\t{:.2f}\t{}\t{}\t{:.2f}%'.format(type_name, avg_len / count, min_len, max_len, count / all_count *100))
        
        data = ''.join(data)
        new_data = re.findall('[\u4e00-\u9fa5a-zA-Z]+', data, re.S) # 加入re.S将换行符当做字符串，不加只在行内匹配
        new_data = ' '.join(new_data) # 空格保证英文被分出
        seg_list_exact = jieba.cut(new_data, cut_all=False) # 默认精确模式false

        result_list = []
        with open(stopwords_path, encoding='utf-8') as f:
            con = f.readlines()
            stop_words = set()
            for i in con:
                stop_words.add(i.strip())

        for word in seg_list_exact:
            if word not in stop_words and len(word) > 1:
                result_list.append(word)

        word_counts = collections.Counter(result_list)
        word_counts_top100 = word_counts.most_common(100)

        with open(os.path.join(output_path, type_name + '.txt'), 'w', encoding='utf-8') as fw:
            for line in word_counts_top100:
                line = (line[0], str(line[1]))
                fw.write('\t'.join(line) + '\n')
                
        # 绘制词云
        my_cloud = WordCloud(
            background_color='white',  # 设置背景颜色  默认是black
            width=900, height=600,
            max_words=100,            # 词云显示的最大词语数量
            font_path='MSYH.TTC',   # 设置字体  显示中文
            max_font_size=99,         # 设置字体最大值
            min_font_size=16,         # 设置子图最小值
            random_state=50           # 设置随机生成状态，即多少种配色方案
        ).generate_from_frequencies(word_counts)

        # plt.text(0.55, 0.55, type_name, fontsize=20, color='black')
        
        # 显示生成的词云图片
        plt.title(type_name, fontsize=50)
        plt.imshow(my_cloud, interpolation='bilinear')
        plt.title(type_name, fontsize=50)
        # plt.colorbar(label=type_name)
        # 显示设置词云图中无坐标轴
        plt.axis('off')
        plt.show()

        my_cloud.to_file(os.path.join(output_path, type_name + '.jpg'))

    f1.close()
