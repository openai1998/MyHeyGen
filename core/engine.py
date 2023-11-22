import re
import time
from core.voice_cloner import VoiceCloner
from core.dereverb import MDXNetDereverb
from core.audio_pre import AudioPre
from core.scene_preprocessor import ScenePreprocessor
from core.face.lipsync import LipSync
from core.helpers import (
    to_segments, 
    to_extended_frames, 
    to_avi, 
    merge, 
    merge_voices, 
    find_speaker, 
    get_voice_segments
)
from core.translator import TextHelper
from core.audio import speedup_audio, combine_audio,speed_change
from core.temp_manager import TempFileManager
from pydub import AudioSegment
from core.whisperx.asr import load_model, load_audio
from core.whisperx.alignment import load_align_model, align
from core.whisperx.diarize import DiarizationPipeline, assign_word_speakers
import torch
from itertools import groupby
import torch
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm
# from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
import soundfile as sf

class Engine:
    def __init__(self, config, output_language):
        #if not config['HF_TOKEN']:
           # raise Exception('No HuggingFace token providen!')
        self.output_language = output_language
        print("output_language:{}".format(output_language))
        self.config = config
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)
        self.whisper_batch_size = 16
        self.whisper = load_model('large-v2', device=device_type, compute_type='int8')
        self.diarize_model = DiarizationPipeline(use_auth_token=config['HF_TOKEN'],device=device_type)
        self.text_helper = TextHelper(config)
        self.temp_manager = TempFileManager()
        
        if config["AUDIO_H5"] == 0:
            self.audio_pre = MDXNetDereverb(15)
        else:
            print("enable H5 for splitting vocal and bgm")
            self.audio_pre = AudioPre(10)
    
    def __call__(self, input_file_path, output_file_path):
        # [Step 1] Reading the video, getting audio (voice + noise), as well as the text of the voice -------
        print("[Step 1] Reading the video, getting audio (voice + noise), as well as the text of the voice")
        original_audio_file = self.temp_manager.create_temp_file(suffix='.wav').name
        if input_file_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
            orig_clip = VideoFileClip(input_file_path, verbose=True)
            orig_clip.audio.write_audiofile(original_audio_file, codec='pcm_s16le', verbose=True, logger=None)
        else:
            orig_clip = None
            print("VIOCE_ONLY enable,just voice cloning!you can set VIOCE_ONLY as 0 in config.json to disable")
            assert input_file_path.endswith(('mp3', 'wav', 'WAV', 'MP3'))
            original_audio_file = input_file_path
            
        original_audio = AudioSegment.from_file(original_audio_file)
        audio_res = self.audio_pre.split(original_audio_file)
        voice_audio = AudioSegment.from_file(audio_res['voice_file'], format='wav')
        noise_audio = AudioSegment.from_file(audio_res['noise_file'], format='wav')

        speakers, lang = self.transcribe_audio_extended(audio_res['voice_file'])
        # ---------------------------------------------------------------------------------------------------

        # [Step 2] Merging voices, translating speech, cloning voices ---------------------------------------
        print("[Step 2] Merging voices, translating speech, cloning voices")
        merged_voices = merge_voices(speakers, voice_audio)

        updates = []
        zimu_path = Path(output_file_path).parent.joinpath('zimu.txt')
        self.empty_cache()
        cloner = VoiceCloner(self.config,self.output_language)
        
        total_change_duration = 0
        
        speech_video = []
        speech_audio = AudioSegment.silent(duration=0)
        new_noise_audio = AudioSegment.silent(duration=0) 
        prev_end = 0
        for i, speaker in enumerate(speakers):
            start = speaker['start'] * 1000
            end = speaker['end'] * 1000
            org_duration = end - start
            if 'id' in speaker:
                voice = merged_voices[speaker['id']]
            else:
                voice = voice_audio[start: end]
            
            voice_wav = self.temp_manager.create_temp_file(suffix='.wav').name
            voice.export(voice_wav, format='wav')
            
            voice_audio_wav = self.temp_manager.create_temp_file(suffix='.wav').name
            voice_audio.export(voice_audio_wav, format='wav')

            dst_text = self.text_helper.translate(speaker['text'], src_lang=lang, dst_lang=self.output_language)
            
            with open(zimu_path, 'a', encoding="utf-8") as f:
                f.write("\n" + speaker['text'])
                f.write("\n" + dst_text)
                f.write("\n")
                
            cloned_wav = cloner.process(
                speaker_wav_filename=[voice_wav,voice_audio_wav],
                text=dst_text
            )
            
            if start > prev_end:
                ## add empty video and audio
                speech_audio += AudioSegment.silent(duration=start - prev_end)
                new_noise_audio += noise_audio[prev_end:start]
            
        
            tmp_audio = AudioSegment.from_file(cloned_wav)
            tmp_noise_audio = noise_audio[start:end]
            
            tmp_noise_audio_path = self.temp_manager.create_temp_file(suffix='.wav').name
            tmp_noise_audio.export(tmp_noise_audio_path, format='wav')
            
            speech_audio += tmp_audio
            
            
            tmp_noise_audio_path_ = self.temp_manager.create_temp_file(suffix='.wav').name
            audio_noise_ratio = tmp_audio.duration_seconds / tmp_noise_audio.duration_seconds
            y,sr = sf.read(tmp_noise_audio_path)
            sf.write(tmp_noise_audio_path_,y,int(sr*audio_noise_ratio))
            new_noise_audio += AudioSegment.from_file(tmp_noise_audio_path_)
        
            if i + 1 == len(speakers):
                speech_audio += voice_audio[end:]
                new_noise_audio += noise_audio[end:]
            prev_end = end
        # ---------------------------------------------------------------------------------------------------
        cloner = None
        torch.cuda.empty_cache()
        # [Step 3] Creating final speech audio --------------------------------------------------------------
        print("[Step 3] Creating final speech audio and video")
        
        speech_audio_wav = self.temp_manager.create_temp_file(suffix='.wav').name
        speech_audio.export(speech_audio_wav, format='wav')
        
        noise_audio_wav = self.temp_manager.create_temp_file(suffix='.wav').name
        new_noise_audio.export(noise_audio_wav, format='wav')
        combined_audio = combine_audio(speech_audio_wav, noise_audio_wav)
        
        if orig_clip is not None:
            new_video_mp4 = os.path.join(Path(output_file_path).parent, "video_extend.mp4")
            new_video_clip = self.video_map_audio(speech_audio, orig_clip)
            new_video_clip.write_videofile(new_video_mp4,audio=False,fps=30)
            # some bug made write twice!!
            new_video_clip.write_videofile(new_video_mp4,audio=False)
            input_file_path = new_video_mp4
            
           
        
        subprocess.call("cp {} {}/voice_cloned.wav".format(
            combined_audio,Path(output_file_path).parent
        ), shell=True)
        
        bgm_path = Path(output_file_path).parent.joinpath('bgm.wav')
        new_noise_audio.export(bgm_path,format='wav')
        
        
        if self.config['VOICE_ONLY'] == 1:
            print("VOICE_ONLY all done")
            return
        # ---------------------------------------------------------------------------------------------------
        
        # [Step 4] Using video-retalking merge speech voice and video, creating output ------------------------------------
        print("Video-retalking merge speech voice and video, creating output!!!")
        command = 'pip install librosa==0.9.2 &&  cd ./video-retalking && rm -rf ./temp/* && python inference.py \
        --face {} --audio {} --outfile {} --LNet_batch_size {}'.format(
            input_file_path, combined_audio, output_file_path, 8
        )
        subprocess.call(command, shell=True)
        
        # [Step 5] Using CodeFormer upscale video, creating output ------------------------------------
        print("Using CodeFormer upscale video, creating output!!!")
        up_output_file_path = Path(output_file_path).parent.joinpath('out_upscale')
        command = 'cd ./CodeFormer && python inference_codeformer.py -i {} -o {} \
        --bg_upsampler realesrgan --face_upsample -w 0.9 -s 1'.format(
            output_file_path, up_output_file_path
        )
        subprocess.call(command, shell=True)
    
    def empty_cache(self):
        self.whisper = None
        self.diarize_model = None
        self.audio_pre = None
        torch.cuda.empty_cache()
        print("cuda memeroy:{}".format(torch.cuda.memory_reserved()))
        
        
    def transcribe_audio_extended(self, audio_file):
        audio = load_audio(audio_file)
        result = self.whisper.transcribe(audio, batch_size=self.whisper_batch_size,chunk_size=15)
        language = result['language']
        model_a, metadata = load_align_model(language_code=language, device=self.device)
        result = align(result['segments'], model_a, metadata, audio, self.device, return_char_alignments=False)
        print("diarizing ... wait moment")
        diarize_segments = self.diarize_model(audio)
        result = assign_word_speakers(diarize_segments, result)
        return result['segments'], language
    
    def video_map_audio(self,audio, video):
        audio_duration = audio.duration_seconds
        
        video_duration = video.duration
        
        ratio = video_duration / audio_duration 
        print("video_duration / audio_duration  =ratio:{}".format(ratio))
        
        new_video = video.fl_time(lambda t:  ratio*t,apply_to=['mask', 'audio'])
        new_video1 = new_video.set_duration(audio_duration)
        new_video2 = new_video1.set_fps(new_video1.fps / video_duration * audio_duration)
        return new_video2
