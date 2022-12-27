#!/bin/bash/

gdown https://drive.google.com/file/d/1aoKXDRVDVGdHHdQv4DrIxIWAusptyRfW/view?usp=sharing

uvicorn app:app --host 0.0.0.0 --port 5001 