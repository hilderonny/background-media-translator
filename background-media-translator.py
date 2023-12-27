PROGRAM_VERSION = '2.0.0'

import time
import os
import json
import datetime
import shutil
import stat
from typing import Sequence

# Set Path for cublas DLLs, neede by ctranslate2 and torch
script_path = os.path.realpath(__file__)
cublas_path = os.path.join(script_path, "python", "Lib", "cublas")
os.environ["PATH"] = os.environ["PATH"] + ";" + cublas_path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', type=str, action='store', required=True, help='Directory where the media files to process are obtained from. Must be writable.')
parser.add_argument('--processingpath', type=str, action='store', required=True, help='Directory where the currently processed media file gets stored. Must be writable.')
parser.add_argument('--outputpath', type=str, action='store', required=True, help='Directory where the output JSON files will be stored. Must be writable.')
parser.add_argument('--fasterwhisperpath', type=str, action='store', required=True, help='Directory where the Faster-Whisper models are stored.')
parser.add_argument('--huggingfacepath', type=str, action='store', required=True, help='Directory where the Hugging Face models are stored.')
parser.add_argument('--whispermodel', type=str, default='small', action='store', help='Whisper model size to use. Can be "tiny", "base", "small" (default), "medium" or "large-v2".')
parser.add_argument('--usegpu', action='store_true', help='Use GPU for neural network calculations. Needs to have cuBLAS and cuDNN installed from https://github.com/Purfview/whisper-standalone-win/releases/tag/libs')
parser.add_argument('--version', action='version', version=PROGRAM_VERSION)
args = parser.parse_args()

# Check write access to directories
import sys
import os
INPUTPATH = args.inputpath
PROCESSINGPATH = args.processingpath
OUTPUTPATH = args.outputpath
if not os.access(INPUTPATH, os.R_OK | os.W_OK):
    sys.exit(f'ERROR: Cannot read and write input directory {INPUTPATH}')
if not os.access(PROCESSINGPATH, os.R_OK | os.W_OK):
    sys.exit(f'ERROR: Cannot read and write processing directory {PROCESSINGPATH}')
if not os.access(OUTPUTPATH, os.R_OK | os.W_OK):
    sys.exit(f'ERROR: Cannot read and write output directory {OUTPUTPATH}')

# Check existence of Whisper files
WHISPERPATH = args.fasterwhisperpath
WHISPERTINYMODELPATH = os.path.join(WHISPERPATH, 'models--guillaumekln--faster-whisper-tiny')
WHISPERBASEMODELPATH = os.path.join(WHISPERPATH, 'models--guillaumekln--faster-whisper-base')
WHISPERSMALLMODELPATH = os.path.join(WHISPERPATH, 'models--guillaumekln--faster-whisper-small')
WHISPERMEDIUMMODELPATH = os.path.join(WHISPERPATH, 'models--guillaumekln--faster-whisper-medium')
WHISPERLARGEV2MODELPATH = os.path.join(WHISPERPATH, 'models--guillaumekln--faster-whisper-large-v2')
if not os.access(WHISPERPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper directory {WHISPERPATH}')
if not os.access(WHISPERTINYMODELPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper tiny model directory {WHISPERTINYMODELPATH}')
if not os.access(WHISPERBASEMODELPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper base model directory {WHISPERBASEMODELPATH}')
if not os.access(WHISPERSMALLMODELPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper small model directory {WHISPERSMALLMODELPATH}')
if not os.access(WHISPERMEDIUMMODELPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper medium model directory {WHISPERMEDIUMMODELPATH}')
if not os.access(WHISPERLARGEV2MODELPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Whisper large V2 model directory {WHISPERLARGEV2MODELPATH}')

# Check existence of Hugging Face model files
HUGGINGFACEPATH = args.huggingfacepath
HUGGINGFACEENDEPACKAGEPATH = os.path.join(HUGGINGFACEPATH, "hub", "models--Helsinki-NLP--opus-mt-en-de")
if not os.access(HUGGINGFACEPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Hugging Face directory {HUGGINGFACEPATH}')
if not os.access(HUGGINGFACEENDEPACKAGEPATH, os.R_OK):
    sys.exit(f'ERROR: Cannot read Hugging Face EN->DE model directory {HUGGINGFACEENDEPACKAGEPATH}')

USEGPU = args.usegpu

# Load Faster Whisper
print('Loading Faster Whisper')
from faster_whisper import WhisperModel    
compute_type = 'float16' if USEGPU else 'int8'
device = 'cuda' if USEGPU else 'cpu'
whisper_model = WhisperModel( model_size_or_path = args.whispermodel, device = device, local_files_only = False, compute_type = compute_type, download_root = WHISPERPATH )

class Translator:
    def __init__(self, source_lang: str, dest_lang: str, use_gpu: bool=False) -> None:
        self.use_gpu = use_gpu
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        if use_gpu:
            self.model = self.model.cuda()
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        if self.use_gpu:
            tokens = {k:v.cuda() for k, v in tokens.items()}
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]

# Load Marian MT
print('Loading Marian MT')
os.environ['HF_HOME'] = "./data/transformers_cache"
from transformers import MarianMTModel, MarianTokenizer
translator = Translator("en", "de", USEGPU)

def translate_into_german(segments_en):
    translation_segments_de = list(map(lambda segment: { 'start': segment['start'], 'end': segment['end'], 'text': translator.translate([segment['text']])[0] }, segments_en))
    return translation_segments_de

def process_file(file_path):
    start_time = datetime.datetime.now()
    result = {}
    try:
        print('Processing file ' + file_path)
        print('Transcribing')
        transcribe_segments_generator, transcribe_info = whisper_model.transcribe(file_path, task = 'transcribe')
        transcribe_segments = list(map(lambda segment: { 'start': segment.start, 'end': segment.end, 'text': segment.text }, transcribe_segments_generator))
        original_language = transcribe_info.language
        result['language'] = original_language
        result['original'] = { 'segments': transcribe_segments, 'fulltext':  ' '.join(map(lambda segment: segment['text'], transcribe_segments)) }
        if original_language == 'de': # Deutsch brauchen wir weder uebersetzen noch in Englisch
            result['en'] = None
            result['de'] = result['original']
        elif original_language == 'en': # Englisch muss nicht ins Englische uebersetzt werden
            result['en'] = result['original']
            print('Translating into german')
            segments_de = translate_into_german(result['en']['segments'])
            result['de'] = { 'segments': segments_de, 'fulltext':  ' '.join(map(lambda segment: segment['text'], segments_de)) }
        else:
            print('Translating into english')
            translation_segments_generator_en, _ = whisper_model.transcribe(file_path, task = 'translate')
            translation_segments_en = list(map(lambda segment: { 'start': segment.start, 'end': segment.end, 'text': segment.text }, translation_segments_generator_en))
            result['en'] = { 'segments':  translation_segments_en, 'fulltext':  ''.join(map(lambda segment: segment['text'], translation_segments_en)) }
            print('Translating into german')
            segments_de = translate_into_german(result['en']['segments'])
            result['de'] = { 'segments': segments_de, 'fulltext':  ' '.join(map(lambda segment: segment['text'], segments_de)) }
    except Exception as ex:
        print(ex)
        result['exception'] = str(ex)
    finally:
        print('Deleting file ' + file_path)
        os.remove(file_path)
        pass
    end_time = datetime.datetime.now()
    result['duration'] = (end_time - start_time).total_seconds()
    return result

def check_and_process_files():
    file_was_processed = False
    for file_name in os.listdir(INPUTPATH):
        input_file_path = os.path.join(INPUTPATH, file_name)
        if os.path.isfile(input_file_path):
            try:
                # Erst mal Datei aus INPUT Verzeichnis bewegen, damit andere Prozesse diese nicht ebenfalls verarbeiten
                processing_file_path = os.path.join(PROCESSINGPATH, file_name)
                shutil.move(input_file_path, processing_file_path)
                os.chmod(processing_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO ) # Let the background process delete the file afterwards
                # Datei verarbeiten
                result = process_file(processing_file_path)
                json_result = json.dumps(result, indent=2)
                output_file_path = os.path.join(OUTPUTPATH, file_name + '.json')
                print('Writing output file ' + output_file_path)
                output_file = os.open(output_file_path, os.O_RDWR|os.O_CREAT)
                os.write(output_file, str.encode(json_result))
                os.close(output_file)
                print(json_result)
                file_was_processed = True
                return file_was_processed # Let the program wait a moment and recheck the uplopad directory
            except Exception as ex:
                print(ex)
            finally: # Hat nicht geklappt. Eventuell hat ein anderer Prozess die Datei bereits weg geschnappt. Egal.
                return
    return file_was_processed

try:
    print('Ready and waiting for action')
    while True:
        file_was_processed = check_and_process_files()
        if file_was_processed == False:
            time.sleep(3)
finally:
    pass
