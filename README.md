# Shinol AI Services

AI Services API cho Shinol Studio - bao gồm TTS, Image Generation, Transcription.

## Features

- **TTS (Text-to-Speech)**
  - Edge TTS - Microsoft Edge AI voices
  - Kokoro TTS - High-quality local TTS

- **Image Generation**
  - Stable Diffusion XL (via ComfyUI)

- **Audio Transcription**
  - Whisper - Audio to text

- **LLM**
  - Ollama integration

## Yêu cầu

- Python 3.10+
- Windows 10/11 hoặc WSL
- GPU khuyến nghị cho Image Generation

## Cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd shinol-ai-services
```

### 2. Tạo virtual environment

```bash
# Tạo venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/WSL)
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Các AI models (Kokoro, SDXL, Whisper) sẽ được tự động download khi sử dụng lần đầu. Không cần tải thủ công.

### 4. Chạy API

```bash
python main.py
```

API sẽ chạy tại: http://localhost:8000

API Docs: http://localhost:8000/docs

## Cấu trúc thư mục

```
shinol-ai-services/
├── controllers/          # API endpoints
│   ├── tts_controller.py
│   ├── kokoro_controller.py
│   ├── image_controller.py
│   ├── whisper_controller.py
│   └── ...
├── models/              # AI models (auto-download)
├── output/              # Generated files
│   ├── proj_12/
│   │   └── epi_1/
│   │       └── audio/
│   └── tts/
├── main.py              # Entry point
└── requirements.txt    # Python dependencies
```

## API Endpoints

### TTS (Edge)

```bash
POST /tts/generate_sync
{
  "text": "Hello world",
  "voice": "en-US-AndrewNeural",
  "speed": 1.0,
  "project_code": "proj_12",
  "episode_code": "epi_1"
}
```

### TTS (Kokoro)

```bash
POST /tts/kokoro/generate_sync
{
  "text": "Hello world",
  "voice": "af_sarah",
  "speed": 1.0,
  "project_code": "proj_12",
  "episode_code": "epi_1"
}
```

### Image Generation

```bash
POST /image/generate_sync
{
  "prompt": "A beautiful sunset",
  "width": 1024,
  "height": 576,
  "project_code": "proj_12",
  "episode_code": "epi_1"
}
```

### Audio Transcription

```bash
POST /audio/transcribe
{
  "audio_url": "http://example.com/audio.mp3"
}
```

## Output Files

Files được lưu tại thư mục `output/`:

- **TTS Audio:** `output/{project_code}/{episode_code}/audio/{filename}.mp3`
- **Kokoro Audio:** `output/{project_code}/{episode_code}/audio/{filename}.wav`
- **Images:** `output/{project_code}/{episode_code}/image/{filename}.png`

Truy cập qua: `http://localhost:8000/{project_code}/{episode_code}/audio/{filename}`

## Troubleshooting

### Lỗi module không tìm thấy

```bash
pip install -r requirements.txt
```

### Lỗi CUDA/GPU

Đảm bảo đã cài đặt PyTorch với CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Port bị chiếm

Đổi port trong `main.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## License

MIT
