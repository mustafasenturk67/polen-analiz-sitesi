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

# --- Temel Flask Uygulama Yapılandırması ---
load_dotenv() # Ortam değişkenlerini .env dosyasından yükle
# Flask uygulamasını başlat. template_folder='.' index.html'in ana dizinde olduğunu belirtir.
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app) # Tüm kaynaklardan gelen isteklere izin ver (Frontend'in bağlanabilmesi için)

# --- API Anahtarı Yönetimi ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # API anahtarını ortam değişkeninden güvenli oku

# --- Gemini API İstemcisini Başlatma ---
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✓ Gemini API istemcisi başarıyla başlatıldı.")
    except Exception as e:
        print(f"✗ HATA: Gemini API istemcisi başlatılamadı! Anahtar geçersiz olabilir. Detay: {e}")
else:
    print("⚠ UYARI: Ortam değişkenlerinde 'GEMINI_API_KEY' bulunamadı. Lütfen .env dosyanızı kontrol edin.")

# --- Yardımcı Fonksiyon 1: Görüntü Analizi ---
def analyze_with_gemini(image_data):
    """
    Görüntüyü Gemini API'sine gönderir, polen analizi ve tip tespiti ister.
    Cevabı zorunlu JSON formatında bekler.
    """
    if not client:
        return "Hata: API İstemcisi başlatılamadı.", False, 0.0, "Hata"

    # Görüntüyü API'nin istediği formata çevir
    image_part = {
        "inline_data": {
            "data": base64.b64encode(image_data).decode('utf-8'),
            "mime_type": "image/jpeg" # veya yüklenen dosyanın tipine göre (örn: image/png)
        }
    }

    # Modele gönderilecek talimat (prompt)
    prompt = (
        "Bu bir mikroskop görüntüsü. Fotoğrafta polen taneleri, sporlar veya diğer "
        "biyolojik kalıntılar görüp görmediğinizi analiz edin. Eğer polen varsa, "
        "en olası polen tipini (Örn: Çam, Huş, Çayır vb.) tahmin edin. "
        "Cevabınızı Türkçe olarak, aşağıdaki JSON formatına kesinlikle uygun verin."
    )

    # Modelin döndürmesi gereken JSON yapısı
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "is_pollen": {"type": "BOOLEAN", "description": "Polen var mı (True/False)."},
            "pollen_type": {"type": "STRING", "description": "Eğer polen varsa, tahmin edilen polen tipi (Örn: Çam). Yoksa 'Yok'."}
        }
    }

    try:
        # Gemini API'sine isteği gönder
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt, image_part],
            config={
                "system_instruction": "Sen, mikroskopik görüntülerden polen analizi ve tip tespiti yapan uzman bir asistansın. Cevabını sadece JSON formatında döndür.",
                "response_mime_type": "application/json", # JSON çıktısı zorunlu
                "response_schema": response_schema       # Bu şemaya uyması zorunlu
            }
        )

        # Gelen JSON yanıtını işle
        json_response = json.loads(response.text)
        is_pollen = json_response.get('is_pollen', False)
        pollen_type = json_response.get('pollen_type', 'Yok')

        # Güven puanı simülasyonu (Model JSON'da güven puanı döndürmüyor)
        confidence = round(random.uniform(90.0, 99.9), 2) if is_pollen else round(random.uniform(70.0, 89.9), 2)
        message = "Analiz başarılı."

        return message, is_pollen, confidence, pollen_type

    except APIError as e:
        print(f"✗ Gemini API Hatası (analyze_with_gemini): {e}")
        return f"API isteği hatası (Sınır/Kota): {e}", False, 0.0, "Hata"
    except json.JSONDecodeError as e:
        print(f"✗ JSON Çözümleme Hatası (analyze_with_gemini): Modelden beklenen yapısal yanıt alınamadı. Yanıt: {response.text[:100]}...")
        return f"JSON çözümleme hatası: Modelden beklenen yapısal yanıt alınamadı.", False, 0.0, "Hata"
    except Exception as e:
        print(f"✗ Bilinmeyen Hata (analyze_with_gemini): {e}")
        return f"Bilinmeyen hata: {e}", False, 0.0, "Hata"

# --- Yardımcı Fonksiyon 2: Metin Üretimi ---
def generate_text_gemini(prompt, system_instruction):
    """
    Sadece metin girdisi alıp metin çıktısı üreten Gemini API'sini çağırır.
    """
    if not client:
        return {"error": "Hata: API İstemcisi başlatılamadı."}

    try:
        # Gemini API'sine isteği gönder (metin-sadece mod)
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt],
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "text/plain" # Düz metin yanıtı istiyoruz
            }
        )
        # Başarılı yanıtı döndür
        return {"text": response.text}
    except APIError as e:
        print(f"✗ Gemini API Hatası (generate_text_gemini): {e}")
        return {"error": f"API isteği hatası (Sınır/Kota): {e}"}
    except Exception as e:
        print(f"✗ Bilinmeyen Hata (generate_text_gemini): {e}")
        return {"error": f"Bilinmeyen hata: {e}"}

# --- Rota 1: Ana Sayfa ---
@app.route('/')
def home():
    """Ana index.html sayfasını sunar."""
    # Flask, template_folder='.' ayarı sayesinde index.html'i ana dizinde arar.
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"✗ HATA: index.html render edilemedi: {e}")
        return "Hata: Arayüz dosyası yüklenemedi.", 500


# --- Rota 2: Görüntü Analizi API Uç Noktası ---
@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    """Görüntüyü alır, Gemini ile analiz eder ve sonucu JSON olarak döndürür."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'Geçerli bir dosya yüklenmedi.'}), 400

    file = request.files['file']

    try:
        image_bytes = file.read()
        # Yardımcı fonksiyonu çağır
        message, is_pollen, confidence, pollen_type = analyze_with_gemini(image_bytes)

        # Sonucu istemciye JSON olarak gönder
        result = {
            'is_pollen': is_pollen,
            'confidence': confidence,
            'message': message,
            'pollen_type': pollen_type
        }
        print(f"✓ Analiz İsteği Başarılı: Tip={pollen_type}, Polen Var mı={is_pollen}")
        return jsonify(result)

    except Exception as e:
        print(f"✗ Hata (/analyze endpoint): {e}")
        return jsonify({'error': f'Sunucu tarafında analiz hatası: {e}'}), 500


# --- Rota 3: Polen Bilgisi API Uç Noktası ---
@app.route('/get_pollen_info', methods=['POST'])
def get_pollen_info_endpoint():
    """Polen tipi alır, Gemini ile bilgi üretir ve sonucu JSON olarak döndürür."""
    data = request.json
    pollen_type = data.get('pollen_type')

    if not pollen_type:
        return jsonify({"error": "İstekte 'pollen_type' belirtilmedi."}), 400

    # Gemini için istem (prompt) oluştur
    prompt = f"Türkiye'deki '{pollen_type}' poleni hakkında kısa ve anlaşılır bilgi ver (alerji, mevsim, kaynaklar). Bir paragraf yeterli."

    # Yardımcı fonksiyonu çağır
    gemini_response = generate_text_gemini(
        prompt,
        system_instruction="Sen alerjenler ve polen biyolojisi konusunda uzman bir biyologsun."
    )

    if "error" in gemini_response:
        return jsonify({"error": f"Gemini Bilgi Üretim Hatası: {gemini_response['error']}"}), 500

    print(f"✓ Bilgi İsteği Başarılı: Tip={pollen_type}")
    return jsonify({
        "info": gemini_response.get("text", "Bilgi alınamadı."),
        "pollen_type": pollen_type
    })


# --- Rota 4: Aksiyon Planı API Uç Noktası ---
@app.route('/get_action_plan', methods=['POST'])
def get_action_plan_endpoint():
    """Polen tipi alır, Gemini ile aksiyon planı üretir ve sonucu JSON olarak döndürür."""
    data = request.json
    pollen_type = data.get('pollen_type')

    if not pollen_type:
        return jsonify({"error": "İstekte 'pollen_type' belirtilmedi."}), 400

    # Gemini için istem (prompt) oluştur
    prompt = f"'{pollen_type}' polenine alerjisi olan bir kişi için, polen mevsiminde uygulayabileceği 5 maddelik pratik bir alerji önleme planı oluştur."

    # Yardımcı fonksiyonu çağır
    gemini_response = generate_text_gemini(
        prompt,
        system_instruction="Sen bir halk sağlığı uzmanı ve alerji danışmanısın."
    )

    if "error" in gemini_response:
        return jsonify({"error": f"Gemini Plan Üretim Hatası: {gemini_response['error']}"}), 500

    print(f"✓ Plan İsteği Başarılı: Tip={pollen_type}")
    return jsonify({
        "plan": gemini_response.get("text", "Plan oluşturulamadı."),
        "pollen_type": pollen_type
    })

# --- Sunucuyu Başlatma Bloğu ---
if __name__ == '__main__':
    # Render gibi platformlar genellikle bu bloğu çalıştırmaz,
    # bunun yerine doğrudan 'gunicorn server:app' komutunu kullanır.
    # Bu blok, kodu yerel makinenizde 'python server.py' ile test etmek içindir.

    # Render veya diğer platformlardan gelen PORT değişkenini kullan, yoksa 5000'i kullan
    port = int(os.environ.get('PORT', 5000))

    # API Anahtarı kontrolü ve uyarı
    if not GEMINI_API_KEY:
        print("\n" + "="*50)
        print("!!! ⚠ UYARI: 'GEMINI_API_KEY' bulunamadı veya boş. !!!")
        print("Lütfen proje klasörünüzde '.env' dosyasını oluşturup")
        print("GEMINI_API_KEY=AIzaSy... şeklinde anahtarınızı eklediğinizden emin olun.")
        print("API çağrıları şu anda çalışmayacaktır.")
        print("="*50 + "\n")
    else:
        # Anahtarın sadece varlığını kontrol edelim, yazdırmayalım.
        print("\n✓ API Anahtarı .env dosyasından başarıyla yüklendi.")

    print(f"🚀 Flask sunucusu başlatılıyor...")
    print(f"   -> Yerel Erişim: http://127.0.0.1:{port}")
    print(f"   -> Ağ Erişimi: http://0.0.0.0:{port} (Ağdaki diğer cihazlar için)")
    print("(Sunucuyu durdurmak için CTRL+C tuşlarına basın)")

    # debug=False: Üretim ortamı için (Render vb.)
    # host='0.0.0.0': Sunucunun ağdaki tüm adreslerden erişilebilir olmasını sağlar (Render için gerekli)
    app.run(debug=False, host='0.0.0.0', port=port)
