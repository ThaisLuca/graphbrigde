#!/bin/sh
python3 side_tune.py --source="imdb" --dataset="cora" --input_model_file="models_graphcl/graphcl_imdb_final.pth"
python3 side_tune.py --source="imdb" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_imdb_final.pth"
python3 side_tune.py --source="imdb" --dataset="yeast" --input_model_file="models_graphcl/graphcl_imdb_final.pth"

python3 side_tune.py --source="uwcse" --dataset="cora" --input_model_file="models_graphcl/graphcl_uwcse_final.pth"
python3 side_tune.py --source="uwcse" --dataset="imdb" --input_model_file="models_graphcl/graphcl_uwcse_final.pth"
python3 side_tune.py --source="uwcse" --dataset="yeast" --input_model_file="models_graphcl/graphcl_uwcse_final.pth"

python3 side_tune.py --source="cora" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_cora_final.pth"
python3 side_tune.py --source="cora" --dataset="imdb" --input_model_file="models_graphcl/graphcl_cora_final.pth"
python3 side_tune.py --source="cora" --dataset="yeast" --input_model_file="models_graphcl/graphcl_cora_final.pth"

python3 side_tune.py --source="yeast" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_yeast_final.pth"
python3 side_tune.py --source="yeast" --dataset="imdb" --input_model_file="models_graphcl/graphcl_yeast_final.pth"
python3 side_tune.py --source="yeast" --dataset="cora" --input_model_file="models_graphcl/graphcl_yeast_final.pth"