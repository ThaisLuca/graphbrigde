#!/bin/sh

python3 side_tune.py --source="zinc_standard_agent" --dataset="bace" --input_model_file="models_graphcl/graphcl_zinc_standard_agent_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="zinc_standard_agent" --dataset="bbbp" --input_model_file="models_graphcl/graphcl_zinc_standard_agent_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="zinc_standard_agent" --dataset="clintoxapproved" --input_model_file="models_graphcl/graphcl_zinc_standard_agent_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="zinc_standard_agent" --dataset="clintoxtoxicity" --input_model_file="models_graphcl/graphcl_zinc_standard_agent_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="zinc_standard_agent" --dataset="hiv" --input_model_file="models_graphcl/graphcl_zinc_standard_agent_final.pth" --epochs=1 --device=0

python3 side_tune.py --source="bace" --dataset="zinc_standard_agent" --input_model_file="models_graphcl/graphcl_bace_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bace" --dataset="bbbp" --input_model_file="models_graphcl/graphcl_bace_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bace" --dataset="clintoxapproved" --input_model_file="models_graphcl/graphcl_bace_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bace" --dataset="clintoxtoxicity" --input_model_file="models_graphcl/graphcl_bace_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bace" --dataset="hiv" --input_model_file="models_graphcl/graphcl_bace_final.pth" --epochs=1 --device=0

python3 side_tune.py --source="bbbp" --dataset="zinc_standard_agent" --input_model_file="models_graphcl/graphcl_bbbp_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bbbp" --dataset="bace" --input_model_file="models_graphcl/graphcl_bbbp_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bbbp" --dataset="clintoxapproved" --input_model_file="models_graphcl/graphcl_bbbp_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bbbp" --dataset="clintoxtoxicity" --input_model_file="models_graphcl/graphcl_bbbp_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="bbbp" --dataset="hiv" --input_model_file="models_graphcl/graphcl_bbbp_final.pth" --epochs=1 --device=0

python3 side_tune.py --source="hiv" --dataset="zinc_standard_agent" --input_model_file="models_graphcl/graphcl_hiv_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="hiv" --dataset="bace" --input_model_file="models_graphcl/graphcl_hiv_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="hiv" --dataset="bbbp" --input_model_file="models_graphcl/graphcl_hiv_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="hiv" --dataset="clintoxapproved" --input_model_file="models_graphcl/graphcl_hiv_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="hiv" --dataset="clintoxtoxicity" --input_model_file="models_graphcl/graphcl_hiv_final.pth" --epochs=1 --device=0

python3 side_tune.py --source="clintoxapproved" --dataset="zinc_standard_agent" --input_model_file="models_graphcl/graphcl_clintoxapproved_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxapproved" --dataset="bace" --input_model_file="models_graphcl/graphcl_clintoxapproved_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxapproved" --dataset="bbbp" --input_model_file="models_graphcl/graphcl_clintoxapproved_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxapproved" --dataset="clintoxtoxicity" --input_model_file="models_graphcl/graphcl_clintoxapproved_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxapproved" --dataset="hiv" --input_model_file="models_graphcl/graphcl_clintoxapproved_final.pth" --epochs=1 --device=0

python3 side_tune.py --source="clintoxtoxicity" --dataset="zinc_standard_agent" --input_model_file="models_graphcl/graphcl_clintoxtoxicity_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxtoxicity" --dataset="bace" --input_model_file="models_graphcl/graphcl_clintoxtoxicity_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxtoxicity" --dataset="bbbp" --input_model_file="models_graphcl/graphcl_clintoxtoxicity_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxtoxicity" --dataset="clintoxapproved" --input_model_file="models_graphcl/graphcl_clintoxtoxicity_final.pth" --epochs=1 --device=0
python3 side_tune.py --source="clintoxtoxicity" --dataset="hiv" --input_model_file="models_graphcl/graphcl_clintoxtoxicity_final.pth" --epochs=1 --device=0

#python3 side_tune.py --source="imdb" --dataset="cora" --input_model_file="models_graphcl/graphcl_imdb_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="imdb" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_imdb_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="imdb" --dataset="yeast" --input_model_file="models_graphcl/graphcl_imdb_final.pth" --epochs=1 --device=1

#python3 side_tune.py --source="uwcse" --dataset="cora" --input_model_file="models_graphcl/graphcl_uwcse_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="uwcse" --dataset="imdb" --input_model_file="models_graphcl/graphcl_uwcse_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="uwcse" --dataset="yeast" --input_model_file="models_graphcl/graphcl_uwcse_final.pth" --epochs=1 --device=1

#python3 side_tune.py --source="cora" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_cora_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="cora" --dataset="imdb" --input_model_file="models_graphcl/graphcl_cora_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="cora" --dataset="yeast" --input_model_file="models_graphcl/graphcl_cora_final.pth" --epochs=1 --device=1

#python3 side_tune.py --source="yeast" --dataset="uwcse" --input_model_file="models_graphcl/graphcl_yeast_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="yeast" --dataset="imdb" --input_model_file="models_graphcl/graphcl_yeast_final.pth" --epochs=1 --device=1
#python3 side_tune.py --source="yeast" --dataset="cora" --input_model_file="models_graphcl/graphcl_yeast_final.pth" --epochs=1 --device=1
