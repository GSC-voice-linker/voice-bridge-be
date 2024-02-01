from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import speech
import io, os
import tempfile
import subprocess

def convert_aac_to_flac(audio_file):
    try:
        # 임시 파일로 오디오 파일 저장
        with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as tmp_aac_file:
            for chunk in audio_file.chunks():
                tmp_aac_file.write(chunk)
            aac_file_path = tmp_aac_file.name

        # FLAC 파일 경로 생성 (임시 파일 생성하지 않고 경로만 사용)
        flac_file_path = tempfile.mktemp(suffix='.flac')

        # ffmpeg를 사용하여 AAC를 FLAC로 변환
        command = ['ffmpeg', '-i', aac_file_path, '-vn', '-acodec', 'flac', flac_file_path]
        subprocess.run(command, check=True)

    except Exception as e:
        return None

    finally:
        os.remove(aac_file_path)  # 변환 후 임시 AAC 파일 삭제
        
    return flac_file_path

@csrf_exempt
def speech_to_text(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio']  # HTML form을 통해 업로드된 오디오 파일
        
        flac_file_path = convert_aac_to_flac(audio_file) # AAC 파일을 FLAC 파일로 변환

        if flac_file_path is None:
            return JsonResponse({'error': 'Invalid audio file'}, status=400)

        # Google Cloud Speech-to-Text 클라이언트 초기화
        client = speech.SpeechClient()

        # FLAC 파일을 읽습니다.
        with io.open(flac_file_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        # 인식 설정을 정의합니다.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            language_code='ko-KR',  # 원하는 언어 코드 설정
            audio_channel_count=2,  # Set this to 2 for stereo files
            enable_separate_recognition_per_channel=True  # Set to True if you want separate recognition for each channel
        )

        # Speech-to-Text API 호출
        response = client.recognize(config=config, audio=audio)

        # 결과 처리
        for result in response.results:
            print('text: {}'.format(result.alternatives[0].transcript))
            return JsonResponse({'text': result.alternatives[0].transcript})

    return JsonResponse({'error': 'Invalid request'}, status=400)