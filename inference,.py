from predict_text import Predictor
import argparse 
import pandas as pd
import os

def main(args):
    p = Predictor()   
    df = pd.read_excel(args.dataset_path)   
    len_data = len(df)
    chunk_size = len_data//args.total_chunk
    start = args.chunk_id*chunk_size
    stop =  (args.chunk_id+1)*chunk_size
    df = df.iloc[start:stop]
    df['text'] = df['text'].apply(lambda x: ' . '.join(x.replace('\n',' ').split('.')[:args.max_sentences]))
    texts = df['text'].values
    out = p.predict_texts(texts)
    df_inf = pd.DataFrame(out)
    df = pd.concat([df, df_inf], axis=1)
    os.makedirs(args.output_path, exist_ok=True)
    df.to_excel(os.path.join(args.output_path , f'prediction_{args.chunk_id}.xlsx'), index=False)

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--chunk_id',  type=int)
    parse.add_argument('--total_chunk',  type=int)
    parse.add_argument('--dataset_path',  type=str)
    parse.add_argument('--output_path',  type=str)
    parse.add_argument('--max_sentences',  type=str)
    args = parse.parse_args()

    main(args)