@echo off
title Retinex Vision Research Studio
echo ===================================================
echo  Starting Retinex Vision Research Studio Server...
echo ===================================================
echo.
timeout /t 2 /nobreak >nul
start http://localhost:5000
py dashboard\app.py
pause
