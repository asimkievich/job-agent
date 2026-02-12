@echo off
cd /d C:\Users\aleja\job-agent
call .venv\Scripts\activate
python batch_run.py >> runs\scheduler.log 2>&1
