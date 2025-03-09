# from transformers import AutoModel, AutoTokenizer
# model_name = "YesserYes/NoobProject"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # Test avec une phrase
# inputs = tokenizer("Ceci est un test", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)


from transformers import AutoModel, AutoTokenizer

# Nom de ton modèle sur Hugging Face
model_name = "YesserYes/NoobProject"

# Charger le modèle et le tokenizer depuis le Hub
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sauvegarder dans un dossier local de ton choix
dossier_local = "./mon_modele_recupere"  # Change ce chemin si tu veux
model.save_pretrained(dossier_local)
tokenizer.save_pretrained(dossier_local)

# (Optionnel) Vérifier que ça fonctionne en rechargeant depuis le dossier local
model_local = AutoModel.from_pretrained(dossier_local)
tokenizer_local = AutoTokenizer.from_pretrained(dossier_local)

# Test avec une phrase
inputs = tokenizer_local("Ceci est un test", return_tensors="pt")
outputs = model_local(**inputs)
print(outputs)