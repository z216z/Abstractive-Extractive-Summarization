""" pretrain a word2vec on the corpus"""
import argparse
import gensim
from gensim.models import Word2Vec



def main(args):
    corpus=[]
    with open(corpus_processed_path,"r", encoding="utf8") as f:
        
        row= f.readline().strip()
        while(row):
            corpus.append(row.split(" "))
            row= f.readline().strip() #take next row
    w2v= Word2Vec(corpus,sg=1,min_count=3,window=2,size=300,sample=6e-5,alpha=0.05,negative=20,iter=15)
    w2v.save(save_model_path)
    
    print("Model has been saved in:",save_nodel_path)
            
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--corpus_processed_path', required=True, help='root of the corpus',default=os.path.join(DATASET_PATH, 'preprocess', 'corpus_filtered.txt'))
    parser.add_argument('--save_model_path', type=str,default=os.path.join(DATASET_PATH, 'preprocess', 'w2v_model'))
    args = parser.parse_args()

    main(args)
