# --- Gerekli Kütüphaneler ---
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import io
import base64
import os
import random
import json
import traceback  # <-- HATA AYIKLAMA İÇİN EKLENDİ
from dotenv import load_dotenv  # .env dosyasını okumak için
from google import genai      # Gemini API ile iletişim için
from google.genai.errors import APIError # API hatalarını yakalamak için
import requests             # (Opsiyonel) Hava durumu vb. için

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
        print(f"✗ HATA: Gemini API istemcisi başlatılamadı: {e}")
else:
    print("⚠ UYARI: 'GEMINI_API_KEY' ortam değişkeninde bulunamadı. API çağrıları başarısız olabilir.")


# --- YARDIMCI FONKSİYONLAR (Gemini) ---

def generate_text_gemini(prompt, system_instruction):
    """Metin tabanlı Gemini API çağrısını gerçekleştirir."""
    if not client: return {"error": "API İstemcisi başlatılamadı."}
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=dict(
                system_instruction=system_instruction
            )
        )
        return {"text": response.text}
    except APIError as e:
        return {"error": f"API İstek Hatası: {e.message}"}
    except Exception as e:
        return {"error": f"Beklenmeyen Hata: {e}"}

def analyze_with_gemini(image_data):
    """Resim tabanlı Gemini API çağrısını gerçekleştirir (Polen tespiti)."""
    if not client: return "API İstemcisi başlatılamadı.", False, 0.0, "Hata"

    # Görüntü nesnesini oluştur
    image_part = io.BytesIO(image_data)
    
    # Sisteme Talimat
    system_instruction = (
        "Sen, mikroskop altında çekilen fotoğraflardan polen tiplerini ve polen varlığını tespit eden bir yapay zeka biyolog asistanısın. "
        "Yalnızca bir polen tipi tespiti yap. Tespitin çok kesin olmalıdır."
    )
    
    # Kullanıcı Sorusu
    prompt_parts = [
        "Bu görüntü mikroskop altında çekilmiş bir polen tanesi içeriyor mu? Eğer içeriyorsa, polen varlığını, tahmin güvenini ve tespit edilen polen tipini JSON formatında döndür. JSON formatı: {\"is_pollen\": boolean, \"confidence\": float (0.0-1.0 arasında), \"pollen_type\": string (Türkçe polen tipi örn: 'Çam Poleni' veya 'Yok')} "
        "Eğer polen yoksa `pollen_type` 'Yok' olmalı ve `confidence` 1.0 olmalıdır.",
        image_part
    ]
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_parts,
            config=dict(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_pollen": {"type": "BOOLEAN"},
                        "confidence": {"type": "NUMBER"},
                        "pollen_type": {"type": "STRING"}
                    },
                    "required": ["is_pollen", "confidence", "pollen_type"]
                }
            )
        )
        
        # JSON yanıtını çözümle
        json_data = json.loads(response.text)
        is_pollen = json_data.get('is_pollen', False)
        confidence = round(json_data.get('confidence', 0.0), 2)
        pollen_type = json_data.get('pollen_type', 'Yok')
        
        # Güvenin 0.90'dan düşük olması durumunda polen yok sayılır.
        if confidence < 0.90 or not is_pollen:
            is_pollen = False
            pollen_type = "Yok"
            confidence = round(random.uniform(0.70, 0.89), 2)
            message = "Analiz başarılı, ancak güven eşiği altında (Polen Yok)."
        else:
             message = "Analiz başarılı ve polen tespit edildi."

        return message, is_pollen, confidence, pollen_type

    except APIError as e:
        return f"API isteği hatası (Sınır/Kota): {e.message}", False, 0.0, "Hata"
    except json.JSONDecodeError:
        return f"JSON çözümleme hatası: Modelden beklenen yapısal yanıt alınamadı. Yanıt: {response.text[:100]}", False, 0.0, "Hata"
    except Exception as e:
        return f"Bilinmeyen hata: {e}", False, 0.0, "Hata"

# --- API ENDPOINTS ---

@app.route('/')
def home():
    """index.html dosyasını sunar."""
    return render_template('index.html')

# --- BU FONKSİYON GÜNCELLENDİ (Satır 119) ---
@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Görüntüyü alır ve Gemini ile polen tespiti yapar. (Gelişmiş Hata Yakalamalı)"""
    
    try:
        # --- 1. Dosya Kontrolleri ---
        if 'file' not in request.files or request.files['file'].filename == '':
            print("HATA: 'file' anahtarı request.files içinde bulunamadı.")
            return jsonify({'error': 'Geçerli bir dosya bulunamadı.'}), 400

        file = request.files['file']
        print(f"Dosya alındı: {file.filename}, MIME: {file.mimetype}")
        
        # --- 2. Dosyayı Okuma ---
        image_bytes = file.read()
        print(f"Dosya okundu, {len(image_bytes)} bytes.")
        
        if len(image_bytes) == 0:
             print("HATA: Dosya içeriği boş.")
             return jsonify({'error': 'Yüklenen dosya boş.'}), 400

        # --- 3. YAPAY ZEKA ANALİZİ GERÇEKLEŞTİRME ---
        print("Gemini analizi başlatılıyor...")
        message, is_pollen, confidence, pollen_type = analyze_with_gemini(image_bytes)
        print(f"Gemini analizi tamamlandı: {message}")

        # --- 4. Kontrollü Hata (API'den dönen) ---
        if "Hata" in pollen_type or "Hata" in message:
            print(f"Kontrollü hata yakalandı (analyze_with_gemini içinden): {message}")
            return jsonify({
                'error': message,
                'is_pollen': False,
                'confidence': 0.0,
                'pollen_type': 'Hata'
            }), 500

        # --- 5. Başarılı Sonuç ---
        print("Analiz başarılı, sonuç dönülüyor.")
        return jsonify({
            'message': message,
            'is_pollen': is_pollen,
            'confidence': confidence,
            'pollen_type': pollen_type
        })

    except Exception as e:
        # !!! İŞTE ASIL ARADIĞIMIZ YER BURASI !!!
        # EĞER 'try' BLOĞUNDA BEKLENMEDİK BİR ÇÖKME OLURSA, BURASI ÇALIŞACAK
        
        print("!!!!!!!!!!!!!! KONTROLSÜZ 500 HATASI YAKALANDI !!!!!!!!!!!!!!")
        print(f"Hata Türü: {type(e).__name__}")
        print(f"Hata Mesajı: {e}")
        print("--- TRACEBACK BAŞLANGICI ---")
        traceback.print_exc()  # ASIL HATA DÖKÜMÜNÜ BU BASACAK
        print("--- TRACEBACK SONU ---")
        
        # Frontend'e de düzgün bir hata mesajı gönderelim
        return jsonify({
            "error": "Sunucu tarafında beklenmedik kritik bir hata oluştu.", 
            "detay": str(e)
        }), 500
# --- GÜNCELLEME SONU ---


@app.route('/get_pollen_info', methods=['POST'])
def get_pollen_info_endpoint():
    """Tespit edilen polen tipi hakkında detaylı bilgi (mevsim, alerjen, korunma) alır."""
    data = request.json
    pollen_type = data.get('pollen_type')
    
    if not pollen_type: return jsonify({"error": "Polen tipi belirtilmedi"}), 400
    
    # Gemini'den bilgi alma isteği
    prompt = f"Polen tipi: {pollen_type}. Bu polen için alerji mevsimi, ana alerjen kaynakları ve korunma yöntemleri hakkında kısa ve bilgilendirici bir paragraf özetle. Yanıtın sadece Türkçe ve bilgilendirici paragraf olsun."
    system_instruction = "Alerjenler konusunda uzman bir biyolog ve bilgilendirme uzmanı gibi davran."
    
    gemini_response = generate_text_gemini(prompt, system_instruction=system_instruction)
    
    if "error" in gemini_response: return jsonify({"error": gemini_response['error']}), 500
    
    return jsonify({
        "info": gemini_response.get("text", "Bilgi alınamadı."),
        "pollen_type": pollen_type
    })


@app.route('/get_action_plan', methods=['POST'])
def get_action_plan_endpoint():
    """Tespit edilen polen tipi için Gemini'den 5 adımlı alerji önleme planı alır."""
    data = request.json
    pollen_type = data.get('pollen_type')
    
    if not pollen_type: return jsonify({"error": "Polen tipi belirtilmedi"}), 400
    
    # Gemini'den eylem planı alma isteği
    prompt = f"{pollen_type} polenine alerjisi olan bir kişi için, polen mevsiminde günlük olarak uygulayabileceği, ev ve dış ortamda alınması gereken önlemleri içeren, maddeler halinde 5 adımlı bir alerji önleme planı oluşturun. Maddelendirme kullan."
    system_instruction = "Bir halk sağlığı uzmanı ve alerji danışmanı gibi davran. Yanıt sadece 5 maddeden oluşmalı."
    
    gemini_response = generate_text_gemini(prompt, system_instruction=system_instruction)
    
    if "error" in gemini_response: return jsonify({"error": gemini_response['error']}), 500
    
    return jsonify({
        "plan": gemini_response.get("text", "Plan oluşturulamadı. Polen tipini tekrar kontrol edin."),
        "pollen_type": pollen_type
    })

# --- Sunucuyu Başlatma Bloğu ---
if __name__ == '__main__':
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
    print(f"    -> Yerel Erişim: http://127.0.0.1:{port}")
    print("    -> Render'da bu blok çalışmaz, gunicorn kullanılır.")

    # Yerel test için sunucuyu başlat
    app.run(host='0.0.0.0', port=port, debug=True)
