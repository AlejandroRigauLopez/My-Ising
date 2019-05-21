@echo off
mode con: cols=130 lines=20
python ktrain.py --l 20 --epochs 100 --batchSize 1024 --lamb 1e-3 --lRate 1e-3 --neurons 2
pause