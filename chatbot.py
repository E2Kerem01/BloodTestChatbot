# -*- coding: utf-8 -*-
# Kan Tahlili Chatbotu (Kerem Metin - UTD-88)

import os
import gradio as gr
import fitz  # PyMuPDF
from openai import OpenAI
import tiktoken

# API Key doğrudan kod içinde — teslimde .env'e alınacak!
client = OpenAI(
    api_key=""
)

def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file.name) as doc:
        for page in doc:
            text += page.get_text()
    return text

def analyze_input(user_input, uploaded_pdf, pdf_text_cache, chat_history):
    if uploaded_pdf and not pdf_text_cache:
        user_text = extract_text_from_pdf(uploaded_pdf)
    elif pdf_text_cache and not user_input.strip():
        user_text = pdf_text_cache
    elif user_input:
        user_text = user_input
        pdf_text_cache = ""  # Manuel giriş varsa PDF metnini temizle
    else:
        return "", None, chat_history, pdf_text_cache, chat_history  # 🔧 5 değer dönmeli

    # Selam gibi kısa mesajlar
    if user_text.strip().lower() in ["selam", "merhaba", "hey", "hi"]:
        assistant_reply = "Merhaba! 🩺 Kan tahlili sonucu yazabilir veya PDF yükleyebilirsin."
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        return "", None, chat_history, pdf_text_cache, chat_history  # 🔧 Burada da 5 değer dönmeli

    # Sistem mesajı + geçmiş konuşma
    messages = [{"role": "system", "content": """
Sen bir tıbbi danışman chatbotusun. Kan tahlili sonuçlarını yorumlarsın ve gerekirse kullanıcıyı ilgili doktor branşına yönlendirirsin.

📌 Aşağıdaki değerlere göre yönlendirme yap:
- Glukoz yüksekse → Dahiliye
- HbA1c yüksekse → Dahiliye
- Kreatinin, Üre yüksekse → Nefroloji
- ALT, AST, GGT, ALP yüksekse → Gastroenteroloji
- TSH, FT3, FT4 anormalse → Endokrinoloji
- WBC, CRP yüksekse → Enfeksiyon / Dahiliye
- Hemoglobin, Hematokrit düşükse → Hematoloji
- LDL, CK-MB, Troponin yüksekse → Kardiyoloji
- D-dimer yüksekse → Acil / Göğüs Hastalıkları
- Eozinofil, IgE yüksekse → Alerji / İmmünoloji

Cevabının sonunda **şöyle bir satır** eklemeyi unutma:
**Yönlendirme: [Uzmanlık Alanı]**

Tanı koyma. Sadece bilgilendir ve yönlendir.
"""}] + chat_history

    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    token_sayisi = count_tokens(user_text)
    assistant_reply += f"\n\n🔢 **Yaklaşık Token Sayısı:** {token_sayisi}"

    # PDF dosyasını sil
    if uploaded_pdf:
        try:
            os.remove(uploaded_pdf.name)
        except Exception as e:
            print(f"PDF silinirken hata oluştu: {e}")
        pdf_text_cache = ""

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": assistant_reply})

    return "", None, chat_history, pdf_text_cache, chat_history  # ✅ Tüm çıktı sayısı 5






# 🌐 Arayüz
with gr.Blocks(theme="soft") as app:
    gr.Markdown("<h1 style='text-align:center;'>🧬 Kan Tahlili Chatbotu</h1>")
    gr.Markdown("Tıbbi danışman. Tahlil verisi girin veya PDF yükleyin.")

    chatbot = gr.Chatbot(label="💬 Tahlil Yorumları", type="messages")

    user_input = gr.Textbox(
        label="📝 Tahlil Sonucu",
        placeholder="Örnek: Glukoz 145, ALT 60",
        lines=4
    )

    with gr.Row():
        analyze_btn = gr.Button("💬 YORUMLA", elem_id="analyze_button")

    pdf_file = gr.File(
        label="📎 PDF Dosyası Yükle (İsteğe Bağlı)",
        file_types=[".pdf"],
        elem_id="pdf_upload"
    )

    pdf_text_state = gr.State("")
    chat_history_state = gr.State([])

    analyze_btn.click(
    fn=analyze_input,
    inputs=[user_input, pdf_file, pdf_text_state, chat_history_state],
    outputs=[user_input, pdf_file, chatbot, pdf_text_state, chat_history_state]
    )

    gr.HTML("""
    <style>
        #pdf_upload {
            width: 1840px !important;
            height: 120px !important;
            background-color: #6A5ACD;
            border-radius: 8px;
            border: 1px solid #ccc;
            text-align: center;
        }

        #analyze_button {
            width: 300px;
            height: 60px;
            font-size: 18px;
            font-weight: bold;
            background-color: #4682B4;
            color: white;
            border-radius: 10px;
            border: none;
        }

        #analyze_button:hover {
            background-color: #3A6C99;
            cursor: pointer;
        }
    </style>
    """)

app.launch()
