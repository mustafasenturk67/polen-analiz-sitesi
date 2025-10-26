# --- Gerekli Kütüphaneler ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
import base64
import os
import random
import json
import traceback
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# --- Temel Flask Uygulama Yapılandırması ---
load_dotenv()
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# --- API Anahtarı Yönetimi ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

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


# --- Yardımcı Fonksiyonlar (Gemini) ---
def generate_text_gemini(prompt, system_instruction):
    """Metin tabanlı Gemini API çağrısını gerçekleştirir."""
    if not client:
        return {"error": "API İstemcisi başlatılamadı."}
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=dict(system_instruction=system_instruction)
        )
        return {"text": response.text}
    except APIError as e:
        return {"error": f"API İstek Hatası: {e.message}"}
    except Exception as e:
        return {"error": f"Beklenmeyen Hata: {e}"}


# --- Yeni Özellik: Analiz Sonuçlarını Kaydetme ---
def save_analysis_log(filename, is_pollen, confidence, pollen_type):
    """Analiz sonucunu logs/results.json dosyasına kaydeder."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "results.json")

    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "is_pollen": is_pollen,
        "confidence": confidence,
        "pollen_type": pollen_type
    }

    try:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        print(f"✓ Log kaydedildi: {entry}")

    except Exception as e:
        print(f"⚠ Log kaydedilemedi: {e}")


# --- Görsel Analiz Fonksiyonu ---
def analyze_with_gemini(image_data, mime_type):
    """Resim tabanlı Gemini API çağrısını gerçekleştirir (Polen tespiti)."""
    if not client:
        return "API İstemcisi başlatılamadı.", False, 0.0, "Hata"

    # Base64 formatına çevir
    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_data).decode('utf-8')
    }

    system_instruction = (
        "Sen, mikroskop altında çekilen fotoğraflardan polen tiplerini ve polen varlığını tespit eden bir yapay zeka biyolog asistanısın. "
        "Yalnızca bir polen tipi tespiti yap. Tespitin çok kesin olmalıdır."
    )

    prompt_parts = [
        "Bu görüntü mikroskop altında çekilmiş bir polen tanesi içeriyor mu? Eğer içeriyorsa, polen varlığını, tahmin güvenini ve tespit edilen polen tipini JSON formatında döndür. JSON formatı: "
        "{\"is_pollen\": boolean, \"confidence\": float (0.0-1.0 arasında), \"pollen_type\": string (örnek: 'Çam Poleni' veya 'Yok')}. "
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

        json_data = json.loads(response.text)
        is_pollen = json_data.get('is_pollen', False)
        confidence = round(json_data.get('confidence', 0.0), 2)
        pollen_type = json_data.get('pollen_type', 'Yok')

        if confidence < 0.90 or not is_pollen:
            is_pollen = False
            pollen_type = "Yok"
            confidence = round(random.uniform(0.70, 0.89), 2)
            message = "Analiz başarılı, ancak güven eşiği altında (Polen Yok)."
        else:
            message = "Analiz başarılı ve polen tespit edildi."

        return message, is_pollen, confidence, pollen_type

    except APIError as e:
        return f"API isteği hatası: {e.message}", False, 0.0, "Hata"
    except json.JSONDecodeError:
        return f"JSON çözümleme hatası: Modelden beklenen yapısal yanıt alınamadı.", False, 0.0, "Hata"
    except Exception as e:
        print("TRACEBACK:", traceback.format_exc())
        return f"Gemini API çağrısında bilinmeyen hata: {e}", False, 0.0, "Hata"


# --- API ENDPOINTS ---
@app.route('/')
def home():
    """index.html dosyasını sunar."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Görüntüyü alır ve Gemini ile polen tespiti yapar. (Gelişmiş Hata Yakalamalı)"""
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'Geçerli bir dosya bulunamadı.'}), 400

        file = request.files['file']
        mime_type = file.mimetype
        image_bytes = file.read()

        if len(image_bytes) == 0:
            return jsonify({'error': 'Yüklenen dosya boş.'}), 400

        print("Gemini analizi başlatılıyor...")
        message, is_pollen, confidence, pollen_type = analyze_with_gemini(image_bytes, mime_type)
        print(f"Gemini analizi tamamlandı: {message}")

        if "Hata" in pollen_type or "Hata" in message:
            return jsonify({
                'error': message,
                'is_pollen': False,
                'confidence': 0.0,
                'pollen_type': 'Hata'
            }), 500

        # ✅ Yeni: Log Kaydı
        save_analysis_log(file.filename, is_pollen, confidence, pollen_type)

        print("Analiz başarılı, sonuç dönülüyor.")
        return jsonify({
            'message': message,
            'is_pollen': is_pollen,
            'confidence': confidence,
            'pollen_type': pollen_type
        })

    except Exception as e:
        print("!!!!!!!!!!!!!! KONTROLSÜZ 500 HATASI YAKALANDI !!!!!!!!!!!!!!")
        traceback.print_exc()
        return jsonify({
            "error": "Sunucu tarafında beklenmedik kritik bir hata oluştu.",
            "detay": str(e)
        }), 500


# --- Bilgi ve Plan Endpointleri ---
@app.route('/get_pollen_info', methods=['POST'])
def get_pollen_info_endpoint():
    data = request.json
    pollen_type = data.get('pollen_type')
    if not pollen_type:
        return jsonify({"error": "Polen tipi belirtilmedi"}), 400

    prompt = f"Polen tipi: {pollen_type}. Bu polen için alerji mevsimi, ana alerjen kaynakları ve korunma yöntemleri hakkında kısa bir bilgilendirme yap."
    system_instruction = "Alerjenler konusunda uzman bir biyolog gibi davran."
    gemini_response = generate_text_gemini(prompt, system_instruction)
    if "error" in gemini_response:
        return jsonify({"error": gemini_response['error']}), 500

    return jsonify({
        "info": gemini_response.get("text", "Bilgi alınamadı."),
        "pollen_type": pollen_type
    })


@app.route('/get_action_plan', methods=['POST'])
def get_action_plan_endpoint():
    data = request.json
    pollen_type = data.get('pollen_type')
    if not pollen_type:
        return jsonify({"error": "Polen tipi belirtilmedi"}), 400

    prompt = f"{pollen_type} polenine alerjisi olan biri için 5 adımlı korunma planı oluştur."
    system_instruction = "Bir halk sağlığı uzmanı gibi davran."
    gemini_response = generate_text_gemini(prompt, system_instruction)
    if "error" in gemini_response:
        return jsonify({"error": gemini_response['error']}), 500

    return jsonify({
        "plan": gemini_response.get("text", "Plan oluşturulamadı."),
        "pollen_type": pollen_type
    })


# --- Logları Görüntüleme (Opsiyonel) ---
@app.route('/logs', methods=['GET'])
def get_logs():
    log_path = os.path.join("logs", "results.json")
    if not os.path.exists(log_path):
        return jsonify([])

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


# --- Sunucu Başlatma ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
