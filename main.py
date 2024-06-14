import cv2
import dlib
import threading
from ultralytics import YOLO
from drowsines_dwtwction_for_jetson_nano import realtime_inferences
from face_id_for_jetson_nano import face_reco_from_camera_ot
from object_detection_jetson_nano import yolo_inference
from emotion_detection_for_jetson import test


def process_yolo(frame, result_holder, yolo_inferences1):
    result_holder['yolo'] = yolo_inferences1.process_frame(frame)


def process_face(frame, result_holder, face_detector):
    annotated_frame = face_detector.FaceProcess(frame)
    result_holder['face'] = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)


def process_emotion(frame, result_holder, emotion_music_player):
    result_holder['emotion'] = emotion_music_player.process_image(frame)


def detect_drowsiness(frame, result_holder, drowsy_inference):
    result_holder['drowsiness'] = drowsy_inference.detect_drowsiness(frame)

def main():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./face_id_for_jetson_nano/data/data_dlib/shape_predictor_68_face_landmarks.dat')
    face_reco_model = dlib.face_recognition_model_v1(
        "./face_id_for_jetson_nano/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
    face_detector = face_reco_from_camera_ot.FaceRecognizer(detector, predictor, face_reco_model)
    mod = YOLO('./drowsines_dwtwction_for_jetson_nano/prod_deployment_path/best.pt')
    emotion_music_player = test.EmotionMusicPlayer()
    yolo_inferences1 = yolo_inference.YOLOInference()
    drowsy_inference = realtime_inferences.RealTimeDrowsyInference(mod)

    # Use the process_image method to detect emotion and play music
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        result_holder = {'yolo': None, 'face': None, 'emotion': None, 'drowsiness': None}

        threads = [
            threading.Thread(target=process_yolo, args=(frame, result_holder, yolo_inferences1)),
            threading.Thread(target=process_face, args=(frame, result_holder, face_detector)),
            threading.Thread(target=process_emotion, args=(frame, result_holder, emotion_music_player)),
            threading.Thread(target=detect_drowsiness, args=(frame, result_holder, drowsy_inference))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        xx = result_holder['yolo']
        annotated_frame = result_holder['face']
        emotion = result_holder['emotion']
        drowsy = result_holder['drowsiness']

        # Check if the annotated_frame is None to avoid errors
        if annotated_frame is not None:
            cv2.imshow('Annotated Frame', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
