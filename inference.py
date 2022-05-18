import argparse 
import pandas as pd
import os
import pickle
from model import Model
from tqdm import tqdm

class Predictor():
    def __init__(self):
        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()

    def load_models(self):
        M = Model()
        for trait in self.traits:
            with open('static/' + trait + '_model.pkl', 'rb') as f:
                self.models[trait] = pickle.load(f)

    def predict(self, X, traits='All', predictions='All'):
        predictions = {}
        if traits == 'All':
            for trait in self.traits:
                pkl_model = self.models[trait]

                
                trait_scores = pkl_model.predict(X, regression=True).reshape(1, -1)
                # scaler = MinMaxScaler(feature_range=(0, 50))
                # print(scaler.fit_transform(trait_scores))
                # scaled_trait_scores = scaler.fit_transform(trait_scores)
                predictions['pred_s'+trait] = trait_scores.flatten()[0]
                # predictions['pred_s'+trait] = scaled_trait_scores.flatten()

                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c'+trait] = str(trait_categories[0])
                # predictions['pred_c'+trait] = trait_categories

                trait_categories_probs = pkl_model.predict_proba(X)
                predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1][0]
                # predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1]

        return predictions

    def predict_texts(self, texts):
        outputs = [self.predict([text]) for text in tqdm(texts)]
        return outputs


if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--chunk_id',  type=int)
    parse.add_argument('--total_chunk',  type=int)
    parse.add_argument('--dataset_path',  type=str)
    parse.add_argument('--output_path',  type=str)
    parse.add_argument('--max_sentences',  type=int)
    args = parse.parse_args()

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
    df = pd.concat([df, df_inf], axis=0)
    os.makedirs(args.output_path, exist_ok=True)
    df.to_excel(os.path.join(args.output_path , f'prediction_{args.chunk_id}.xlsx'), index=False)