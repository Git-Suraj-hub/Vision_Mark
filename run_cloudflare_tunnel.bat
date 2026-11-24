@echo off
title Vision Mark - Cloudflare Tunnel
echo ====================================================
echo Starting Vision Mark Cloudflare Tunnel...
echo ====================================================
echo.
"%~dp0tools\cloudflared.exe" tunnel --url http://127.0.0.1:8000
pause
