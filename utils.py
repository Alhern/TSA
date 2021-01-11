import json
from tensorflow.python.keras.models import model_from_json


###########################################
##########    SAVE/LOAD MODEL    ##########
###########################################

# FUNCTIONS:
# 1- save_modeljson(model)
# 2- load_modeljson(filename, weights)


# Save model configuration and weights

def save_modeljson(model):
    model_json = model.to_json()
    with open("model_config.json", "w") as f:
        f.write(model_json)
    model.save_weights("model_weights.h5")
    print("Saved model config and weights to disk")


# Loading the model configuration and its weights

def load_modeljson(filename, weights):
    model = model_from_json(open(filename).read())
    model.load_weights(weights)
    return model



###########################################
######### READ/VALIDATE JSON FILE #########
###########################################

# FUNCTIONS:
# 1- valid_json(file, new_file)
# 2- read_json(file)

# ------------------------------------------

# Le dataset récupéré avec miner.py n'a pas un format json valide, le fichier est une suite de listes contenant des chaînes.
# valid_json va transformer ce fichier en fichier json valide.

def valid_json(file, new_file):
    with open(file) as f, open(new_file, 'w', encoding='utf-8') as valid_file:
        data = json.loads("[" + f.read().replace("}{", "},\n{") + "]")  # magie
        json.dump(data, valid_file, ensure_ascii=False, indent=4)
    print("%s created." % new_file)


# Chargement du fichier json à partir de l'ordi

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
        return data
