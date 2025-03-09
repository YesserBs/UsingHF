


# # Import necessary modules
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# import torch
# from datasets import Dataset
#
# from dataFile import data
# data = data
#
# # Replace with your Hugging Face token
# hf_token = "token_here"
#
# # Load tokenizer and model without quantization (pour CPU)
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)  # Pas de quantization_config ni device_map
#
# # Ajouter un pad_token si inexistant (requis pour certains modèles)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = model.config.eos_token_id
#
# # Configurer LoRA avec PEFT
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
#
# # Appliquer LoRA au modèle
# model = get_peft_model(model, lora_config)
#
#
# # Préparer les données pour l'entraînement
# def preprocess_function(examples):
#     inputs = [f"<s>[INST] {q} [/INST] {a} </s>" for q, a in zip(examples["question"], examples["answer"])]
#     model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
#     model_inputs["labels"] = model_inputs["input_ids"].copy()
#     return model_inputs
#
# # Créer le dataset et tokeniser
# dataset = Dataset.from_list(data)
# tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])
#
# # Configurer les paramètres d'entraînement (pas de fp16 car CPU ne le supporte pas bien en général)
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     logging_steps=1,
#     report_to="none"
# )
#
# # Initialiser le Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )
#
# # Lancer le fine-tuning
# trainer.train()
#
# # Sauvegarder le modèle fine-tuné
# model.save_pretrained("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_model")
#
# # Fonction pour tester (sans .to("cuda") car on reste sur CPU)
# def ask_question(question):
#     inputs = tokenizer(f"<s>[INST] {question} [/INST]", return_tensors="pt")  # Pas de .to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response
#
# # Tester
# question = "C'est quoi SLE dans l'université Côte d'Azur ?"
# answer = ask_question(question)
# print("Réponse après fine-tuning :", answer)