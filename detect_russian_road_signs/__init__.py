import os

from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_DISABLED'] = 'true'

load_dotenv()
API_KEY = os.getenv('ROBOFLOW_API_KEY')
# https://universe.roboflow.com/ksenia-komlach/roud-signs-rus/dataset/5/download/yolov8
# Register in get a code with API-KEY
