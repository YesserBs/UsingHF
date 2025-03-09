from transformers import AutoModel, AutoTokenizer
model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#enregistrement en local je suppose
model.save_pretrained("./mon_modele_local")
tokenizer.save_pretrained("./mon_modele_local")
