import cv2
from fer import FER
from llama_cpp import Llama
import time

# -------------------------------
# MODEL PATH
# -------------------------------
model_path = r"C:\models\gemma-4-E2B-it-UD-Q4_K_XL.gguf"

# -------------------------------
# LOAD MODEL
# -------------------------------
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=20   # reduce to 10 if GPU error
)

# -------------------------------
# EMOTION DETECTOR
# -------------------------------
detector = FER(mtcnn=False)
emotion_buffer = []

def stable_emotion(new_emotion):
    emotion_buffer.append(new_emotion)
    if len(emotion_buffer) > 5:
        emotion_buffer.pop(0)
    return max(set(emotion_buffer), key=emotion_buffer.count)

def get_emotion(frame):
    try:
        result = detector.detect_emotions(frame)
        if not result:
            return None
        emotions = result[0]["emotions"]
        return max(emotions, key=emotions.get)
    except:
        return None

# -------------------------------
# AI RESPONSE (FIXED PROMPT)
# -------------------------------
def get_response(emotion):

    tone_map = {
        "happy": "cheerful",
        "sad": "comforting",
        "angry": "calm",
        "neutral": "friendly",
        "surprise": "excited"
    }

    tone = tone_map.get(emotion, "friendly")

    prompt = f"""<|user|>
The person looks {emotion}. Respond in a {tone} way.
Give a natural human reply (max 2 sentences).

<|assistant|>
"""

    output = llm(
        prompt,
        max_tokens=80,
        temperature=0.7,
        stop=["<|user|>", "<|assistant|>"]
    )

    return output["choices"][0]["text"].strip()

# -------------------------------
# CAMERA LOOP
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press S = get response")
print("Press Q = quit")

last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    raw_emotion = get_emotion(frame)
    if raw_emotion is None:
        continue

    emotion = stable_emotion(raw_emotion)

    cv2.putText(frame, f"Emotion: {emotion}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("MindMate", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if time.time() - last_time > 2:
            print("\nDetected Emotion:", emotion)
            reply = get_response(emotion)
            print("AI:", reply)
            last_time = time.time()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()