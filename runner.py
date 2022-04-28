import os
from os.path import join, dirname
import main
from dotenv import load_dotenv 
from main import *
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get('API_KEY')
STOCK_FILE = os.environ.get('STOCK_FILE')

engine = QMS(STOCK_FILE, API_KEY)
engine.QMS_process()