import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, pipeline
from openai import OpenAI
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# OpenAI API client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# OpenAI GPT modelleri ile yanıt

def openai_response(prompt, model_name):
    start = time.time()
    messages = [
        {"role": "system", "content": "Sen bir tıbbi danışman chatbotusun. Tahlil sonucuna göre ilgili uzmanlık alanına yönlendir."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model=model_name, messages=messages)
    answer = response.choices[0].message.content
    duration = round(time.time() - start, 3)
    tokens = count_tokens(prompt, model=model_name)
    return {
        "answer": answer,
        "inference_time": duration,
        "tokens": tokens,
        "model": model_name
    }

# BERT modeli ile cevap al

def bert_single_response(model_path, context, question):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        handle_impossible_answer=True
    )
    start = time.time()
    result = qa_pipeline(question=question, context=context)
    duration = round(time.time() - start, 3)
    return {
        "answer": result.get("answer", ""),
        "inference_time": duration,
        "score": result.get("score", 0.0)
    }

# BioMedLM için cevap üret

def generate_medical_response(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    start = time.time()
    response = generator(prompt, max_new_tokens=150)[0]["generated_text"]
    duration = round(time.time() - start, 3)

    return {
        "model": model_name,
        "answer": response,
        "inference_time": duration
    }

# ML Modelleri

def ml_models_comparison():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "MLPClassifier": MLPClassifier(max_iter=1000)
    }

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        duration = round(time.time() - start, 3)
        results.append({"model": name, "accuracy": accuracy, "inference_time": duration})
    return results

# Girdi
input_text = """
Hastanın kan tahlili sonuçları: 
- Glukoz: 155 mg/dL (yüksek)
- TSH: 7.1 uIU/mL (yüksek)
- ALT: 45 U/L (hafif yüksek)
Bu değerler, diyabet, tiroid disfonksiyonu veya karaciğer sorunlarına işaret edebilir.
"""

question = "Bu tahlil sonucuna göre hasta hangi branşa yönlendirilmelidir?"

bert_models = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BlueBERT": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
}

results = []

# GPT-3.5 ve GPT-4
for gpt_model in ["gpt-3.5-turbo", "gpt-4"]:
    try:
        result = openai_response(input_text, gpt_model)
        results.append(result)
        print(f"✅ {gpt_model} tamamlandı.")
    except Exception as e:
        print(f"❌ {gpt_model} başarısız oldu: {e}")

# BERT tabanlı modeller
for model_name, model_path in bert_models.items():
    try:
        result = bert_single_response(model_path, input_text, question)
        result["model"] = model_name
        results.append(result)
        print(f"✅ {model_name} tamamlandı.")
    except Exception as e:
        print(f"❌ {model_name} başarısız oldu: {e}")

# BioMedLM
try:
    result = generate_medical_response("stanford-crfm/BioMedLM", prompt=input_text + "\n" + question)
    results.append(result)
    print("✅ BioMedLM tamamlandı.")
except Exception as e:
    print(f"❌ BioMedLM başarısız oldu: {e}")

# ML modelleri
try:
    ml_results = ml_models_comparison()
    results.extend(ml_results)
    print("✅ ML modelleri tamamlandı.")
except Exception as e:
    print(f"❌ ML modelleri başarısız oldu: {e}")

# CSV’ye yaz
df = pd.DataFrame(results)
df.to_csv("model_karsilastirma_tum.csv", index=False)
print("📄 Karşılaştırma tamamlandı. Sonuçlar CSV olarak kaydedildi.")
