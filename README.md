# DrugMetric: Quantitative Drug-likeness Scoring Based on Chemical Space Distance


# Requirements
* RDKit (version >= 2019)
* Python (version >= 3.8)
* PyTorch (version >= 1.8)
* autogluon (version >= 0.6.2)
* streamlit (version >= 1.14.0)
  
To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start

## vae training
This project comprises the following three directories:
```
python preprocess.py  --train_path data/train/train_data.txt \
                      --output_path data/train_data_processed 
                      

```

```
python vae_train.py  --train data/train_data_processed \
                      --vocab data/vocab/all_data_vocab.txt
                      

```

## DrugMetric score
```
python DrugMetric_score.py --input molecules_files \
                           --vocab data/vocab/all_data_vocab.txt \
                           --model_path save/vae_model.pkl \
               
```
## webserver

DrugMetric webserver can run locally using Streamlit. To deploy DrugMetric locally, you need to install Streamlit.
```
pip install streamlit
               
```
Once Streamlit is installed, you can start the application by navigating to the DrugMetric directory in your terminal and running the following command:
```
streamlit run web_server/dashboard.py
               
```
