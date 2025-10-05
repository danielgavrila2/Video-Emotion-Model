import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import sys
import json
import boto3
import tempfile

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def install_ffmpeg():
    print("Starting FFMPEG installation")

    # 1. Install (Upgrade) pip
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])

    # 2. Install setuptools
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "setuptools"])

    # 3. Install ffmpeg
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed FFMPEG successfully!")

    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFMPEG binary successfully!")

    except Exception as e:
        print(f"Failed to install static FFMPEG: {e}")

    try:
        result = subprocess.run(["ffmpeg", "-version"],
                                capture_output=True, text=True, check=True)
        print("FFMPEG version:")
        print(result.stdout)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFMPEG installation verification failed!")
        return False


class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # We'll try to read the first frame to validate the video
            ret, frame = cap.read()

            if (not ret) or (frame is None):
                raise ValueError(f"Video file is corrupt: {video_path}")

            # Reset the index in order to start from the begging
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Normalise the RGB channels
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames if it is necessary
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permutation -> [frames, height, width, channels]
        # After permutation  -> [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    def process_audio(self, video_path, max_length=300):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectogram(waveform)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, output_path="/tmp"):
        os.makedirs(output_path, exist_ok=True)
        segment_path = os.path.join(
            output_path, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed " + segment_path)

        return segment_path


def download_from_s3(s3_uri):
    s3_client = boto3.client('s3')
    bucket = s3_uri.split('/')[2]
    key = "/".join(s3_uri.split("/")[3:])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        s3_client.download_fileobj(bucket, key, tmp_file.name)
        return tmp_file.name


def input_fn(req_body, req_content_type):
    if req_content_type == "application/json":
        input_data = json.loads(req_body)
        s3_uri = input_data['video_path']
        local_path = download_from_s3(s3_uri)
        return {"video_path": local_path}
    else:
        raise ValueError(f"Unsupported content type: {req_content_type}")


def output_fn(prediction, res_content_type):
    if res_content_type == "application/json":
        return json.dumps(prediction), "application/json"
    else:
        raise ValueError(f"Unsupported content type: {res_content_type}")


def model_fn(model_dir):
    # load the model
    if not install_ffmpeg():
        raise EnvironmentError("FFMPEG installation failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_path}")

    print("Loading model from:", model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained("bert-base-uncased"),
        'transcriber': whisper.load_model("base", device=device),
        'device': device
    }


def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data["video_path"]

    if not video_path or not os.path.exists(video_path):
        raise ValueError("Invalid or missing video path")

    result = model_dict['transcriber'].transcribe(
        video_path, word_timestamps=True)

    video_utterance_processor = VideoUtteranceProcessor()
    predictions = []

    # Transcribe the video to get utterance

    for segment in result['segments']:
        try:
            segment_path = video_utterance_processor.extract_segment(
                video_path,
                segment['start'],
                segment['end']
            )

            video_frames = video_utterance_processor.video_processor.process_video(
                segment_path)
            audio_features = video_utterance_processor.audio_processor.process_audio(
                segment_path)
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # move to the device

            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # get the prediction
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(
                    outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(
                    sentiment_probs, 3)

                predictions.append({
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "emotions": [
                        {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(emotion_indices, emotion_values)
                    ],
                    "sentiments": [
                        {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(sentiment_indices, sentiment_values)
                    ]
                })

        except Exception as e:
            print("Segment processing failed:", str(e))

        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)

    return {"utterances": predictions}


# def process_local_video(video_path, model_dir="./model"):
#     model_dict = model_fn(model_dir)
#     input_data = {"video_path": video_path}

#     predictions = predict_fn(input_data, model_dict)

#     for utterance in predictions["utterances"]:
#         print(f"\nUtterance: {utterance['text']}")
#         print(
#             f"Time: {utterance['start_time']}s - {utterance['end_time']}s")
#         print("Top Emotions:")
#         for emotion in utterance["emotions"]:
#             print(f"  - {emotion['label']}: {emotion['confidence']:.4f}")
#         print("Top Sentiments:")
#         for sentiment in utterance["sentiments"]:
#             print(f"  - {sentiment['label']}: {sentiment['confidence']:.4f}")
#         print("-" * 50)


# if __name__ == "__main__":
#     process_local_video("./deployment/joy.mp4",
#                         model_dir="./deployment/model_normalized")
