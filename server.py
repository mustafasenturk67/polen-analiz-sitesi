# --- Gerekli Kütüphaneler ---
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import io
import base64
import os
import random
import json
from dotenv import load_dotenv  # .env dosyasını okumak için
from google import genai       # Gemini API ile iletişim için
from google.genai.errors import APIError # API hatalarını yakalamak için
import requests                # (Opsiyonel) Hava durumu vb. için eklenebilir

# --- Temel Flask Uygulama Yapılandırması ---\
load_dotenv() # Ortam değişkenlerini .env dosyasından yükle
# Flask uygulamasını başlat. template_folder='.' index.html'in ana dizinde olduğunu belirtir.
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app) # Tüm kaynaklardan gelen isteklere izin ver (Frontend'in bağlanabilmesi için)

# --- API Anahtarı Yönetimi ---\
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # API anahtarını ortam değişkeninden güvenli oku

# --- Gemini API İstemcisini Başlatma ---\
client = None
if GEMINI_API_KEY:
    try:
        # API Anahtarını kullanırken hata oluşursa (geçersiz anahtar vb.), istemci yine de None kalır.
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✓ Gemini API istemcisi başarıyla başlatıldı.")
    except Exception as e:
        print(f"✗ HATA: Gemini API istemcisi başlatılamadı: {e}")
else:
    print("UYARI: 'GEMINI_API_KEY' ortam değişkeni bulunamadı. API çağrıları başarısız olacaktır.")

# --- YARDIMCI FONKSİYONLAR (Gemini) ---

def analyze_with_gemini(image_data):
    # API İstemcisi kontrolü
    if not client:
        return {"error": "API istemcisi başlatılamadı. Lütfen 'GEMINI_API_KEY' ortam değişkenini kontrol edin."}
    
    try:
        # Görüntü verilerini modelin anlayacağı formata dönüştür
        image_part = {
            "mime_type": "image/jpeg",  # Görüntü tipini varsayıyoruz
            "data": image_data
        }

        # Analiz için istem (prompt)
        prompt = (
            "Bu, bir mikroskop altındaki polen görüntüsüdür. Görüntüyü dikkatlice incele. "
            "1. Polen var mı? Yanıtın: 'Evet' veya 'Hayır'. "
            "2. Polen varsa, en yüksek olasılıklı polen tipini (örneğin: Çam, Huş, Ot, Meşe, Ambrosia vb.) Türkçe olarak tek kelimeyle belirt. "
            "3. Güven seviyeni (0.0 ile 1.0 arasında) belirt. "
            "JSON formatında yanıtla: {\"is_pollen\": \"Evet/Hayır\", \"pollen_type\": \"Tip Adı\", \"confidence\": 0.95}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, {"text": prompt}],
            config={"response_mime_type": "application/json"}
        )
        
        # Yanıtı JSON olarak ayrıştır
        result_json = json.loads(response.text)
        
        # JSON alanlarını kontrol et ve dönüştür
        is_pollen = result_json.get("is_pollen", "Hayır").lower() == "evet"
        pollen_type = result_json.get("pollen_type", "Yok")
        confidence = float(result_json.get("confidence", 0.0))

        return {
            "is_pollen": is_pollen,
            "pollen_type": pollen_type,
            "confidence": confidence
        }

    except APIError as e:
        print(f"Gemini API Hatası: {e}")
        return {"error": f"Gemini API'den yanıt alınamadı: {str(e)}"}
    except json.JSONDecodeError:
        print(f"Gemini Yanıtı JSON Hatası: {response.text}")
        return {"error": "Gemini'den gelen yanıt JSON formatında değildi."}
    except Exception as e:
        print(f"Beklenmedik Analiz Hatası: {e}")
        return {"error": f"Beklenmedik bir hata oluştu: {str(e)}"}

def generate_text_gemini(user_prompt, system_instruction):
    # API İstemcisi kontrolü
    if not client:
        return {"error": "API istemcisi başlatılamadı. Lütfen 'GEMINI_API_KEY' ortam değişkenini kontrol edin."}

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                # Grounding (Google Search) sadece bilgi alma endpoint'i için kullanılacak
            }
        )
        return {"text": response.text}
    except APIError as e:
        print(f"Gemini API Hatası (Metin): {e}")
        return {"error": f"Metin oluşturma hatası: {str(e)}"}
    except Exception as e:
        print(f"Beklenmedik Metin Hatası: {e}")
        return {"error": f"Beklenmedik bir hata oluştu: {str(e)}"}


# --- ROUTE TANIMLAMALARI ---

@app.route('/')
def index():
    # index.html dosyasını ana sayfa olarak sun
    return render_template('index.html')

# --- API ENDPOINTS ---

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    # API İstemcisi kontrolü
    if not client:
        return jsonify({"error": "Sunucu hatası: API anahtarı eksik."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Geçersiz dosya adı"}), 400

    try:
        # Görüntüyü Base64'e dönüştür
        image_bytes = file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Gemini Analizi
        analysis_result = analyze_with_gemini(image_base64)

        if "error" in analysis_result:
             # Eğer hata varsa 500 kodu ile frontend'e gönder
            return jsonify({"error": analysis_result['error']}), 500

        # Başarılı yanıt
        return jsonify(analysis_result)

    except Exception as e:
        print(f"Endpoint'te Hata: {e}")
        return jsonify({"error": f"Sunucu tarafında işleme hatası: {str(e)}"}), 500


@app.route('/get_pollen_info', methods=['POST'])
def get_pollen_info_endpoint():
    # API İstemcisi kontrolü
    if not client:
        return jsonify({"error": "Sunucu hatası: API anahtarı eksik."}), 500

    data = request.json
    pollen_type = data.get('pollen_type')
    city = data.get('city') or 'Türkiye' # Şehir boş gelirse tüm Türkiye için bilgi ara

    if not pollen_type:
        return jsonify({"error": "Polen tipi belirtilmedi"}), 400
    
    # Grounding (Google Search) ile bilgi toplama
    prompt = (
        f"'{pollen_type}' poleninin özelliklerini, alerji mevsimini ve kaynaklarını özetle. "
        f"Ayrıca, şu anda {city} bölgesindeki '{pollen_type}' poleninin **güncel yoğunluk seviyesi** (düşük/orta/yüksek) nedir? "
        "Yanıtında, toplanan güncel yoğunluk verisini mutlaka belirt."
    )
    
    # Grounding'i etkinleştirerek Google'dan güncel bilgi al
    try:
        gemini_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"system_instruction": "Alerjenler ve güncel çevre verileri konusunda uzman bir biyolog gibi davran."},
            tools=[{"google_search": {}}]
        )
        
        info_text = gemini_response.text
        return jsonify({"info": info_text, "pollen_type": pollen_type, "city": city})

    except APIError as e:
        print(f"Gemini API Hatası (Bilgi): {e}")
        return jsonify({"error": f"Bilgi alma hatası: {str(e)}"}), 500
    except Exception as e:
        print(f"Beklenmedik Bilgi Hatası: {e}")
        return jsonify({"error": f"Beklenmedik bir hata oluştu: {str(e)}"}), 500


@app.route('/get_action_plan', methods=['POST'])
def get_action_plan_endpoint():
    # API İstemcisi kontrolü
    if not client:
        return jsonify({"error": "Sunucu hatası: API anahtarı eksik."}), 500
        
    data = request.json
    pollen_type = data.get('pollen_type')
    
    if not pollen_type:
        return jsonify({"error": "Polen tipi belirtilmedi"}), 400
        
    prompt = f"{pollen_type} polenine alerjisi olan biri için 5 adımlı detaylı bir alerji önleme planı oluştur. Her adımı kalın (bold) olarak başlat."

    # Grounding kullanılmayacak, sadece yaratıcı metin oluşturma
    gemini_response = generate_text_gemini(prompt, system_instruction="Bir halk sağlığı uzmanı ve alerji hekimi gibi davran.")
    
    if "error" in gemini_response:
        return jsonify({"error": gemini_response['error']}), 500
        
    return jsonify({"plan": gemini_response.get("text", "Plan oluşturulamadı."), "pollen_type": pollen_type})


# --- Sunucuyu Çalıştırma ---
if __name__ == '__main__':
    # Bu blok, kodu yerel makinenizde 'python server.py' ile test etmek içindir.
    port = int(os.environ.get('PORT', 5000))

    if not GEMINI_API_KEY:
        print("\n" + "="*50)
        print("!!! ⚠ UYARI: 'GEMINI_API_KEY' bulunamadı veya boş. !!!")
        print("Lütfen proje klasörünüzde '.env' dosyasını oluşturup")
        print("GEMINI_API_KEY=AIzaSy... şeklinde anahtarınızı eklediğinizden emin olun.")
        print("API çağrıları şu anda çalışmayacaktır.")
        print("="*50 + "\n")
    else:
        print("\n✓ API Anahtarı .env dosyasından başarıyla yüklendi.")

    print(f"🚀 Flask sunucusu başlatılıyor...")
    print(f"   -> Yerel Erişim: http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
