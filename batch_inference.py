import argparse
import pandas as pd
import pickle
from data_prep import DataPrep
from model import Model
from open_psychometrics import Big5
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import json

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
        outputs = [self.predict([text]) for text in texts]
        return outputs


def process(x):
    p = Predictor()
    out = p.predict_texts(x)
    return out

#df = pd.read_csv('data\myPersonality\mypersonality_final.csv', encoding='latin-1')

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--chunk_id',  type=int)
    parse.add_argument('--total_chunks',  type=int)
    parse.add_argument('--dataset_path',  type=str)
    
    args = parse.parse_args()
    
    dataset_path = args.dataset_path#r'C:\Users\Cristian\Documents\HolisticAI\repos\neural_nets_personality\outputs\organized_text\trait_activating_questions_clean.csv'

    df = pd.read_csv((dataset_path))
    text = df['text'].apply(lambda x: x.replace('\n',' '))
    
    chunk_size = (len(text)+args.total_chunks-1)//args.total_chunks

    batch_size = 10
    iteration = (chunk_size + batch_size-1)//batch_size
    args = []
    for i in tqdm(range(iteration)):
        start = i*batch_size
        stop = (i+1)*batch_size
        gtext = text.iloc[start:stop]
        args.append(gtext)

    outs = [process(a)  for a in tqdm(args)]

    with open(f'results_{args.chunk_id}.json') as file:
        json.dump(outs,file)