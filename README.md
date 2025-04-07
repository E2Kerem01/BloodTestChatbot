# Kan Tahlili Chatbotu - ChatBlood Projesi

Bu proje, kan tahlili verilerinin doğal dil işleme (NLP) ve yapay zeka (AI) modelleri kullanılarak yorumlanmasını sağlayan bir chatbot sistemidir. Sistem, kullanıcıdan alınan tahlil verilerini analiz ederek uygun tıbbi branşa yönlendirme önerilerinde bulunur.

## 🚀 Proje Özellikleri

- Kan tahlili değerlerine göre uzmanlık alanı önerisi
- GPT-3.5, GPT-4, BioBERT, ClinicalBERT gibi çoklu NLP modelleriyle test
- Türkçe dil desteği
- PDF yorumlama ve konuşma hafızası (isteğe bağlı entegrasyon)
- Karşılaştırmalı model değerlendirme: Yanıt süresi, doğruluk, skor

## 🔧 Kullanılan Teknolojiler

- Python
- Hugging Face Transformers
- OpenAI GPT API
- Scikit-learn (ML modelleri)
- Pandas, Matplotlib (veri analizi ve sonuçlar)

## 🧪 Test Edilen Modeller

| Model           | Yanıt Süresi (s) | Doğruluk Skoru | Açıklama                      |
|----------------|------------------|----------------|-------------------------------|
| GPT-3.5 Turbo  | 2.21             | -              | Diyabet, tiroid, karaciğer    |
| GPT-4          | 7.54             | -              | Endokrinoloji önerisi         |
| BioBERT        | 0.118            | 0.00026        | "yüksek"                      |
| ClinicalBERT   | 0.066            | 0.00022        | "Glukoz: 155 mg..."           |
| BioMedLM       | 331.375          | -              | Detaylı yorumlama             |
| RandomForest   | 0.18             | 89%            | ML sınıflandırıcı             |
| LogisticReg.   | 0.005            | 85.5%          | ML sınıflandırıcı             |

## 📂 Proje Yapısı


## 📥 Kurulum

1. Ortamı oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


pip install -r requirements.txt


api_key = "sk-..."


Hazırlayan: Kerem Metin
Sakarya Uygulamalı Bilimler Üniversitesi
İletişim: 24502405042@subu.edu.tr