@echo off
REM Watchdog level-proses: restart web_ui.py otomatis kalau crash/exit.
REM Auto-start saat boot: Task Scheduler -> Trigger "At startup" -> Action jalanin run.bat.
cd /d "%~dp0"
:loop
echo [%date% %time%] starting web_ui...
python web_ui.py %*
echo [%date% %time%] web_ui exited, restart 5 detik...
timeout /t 5 /nobreak >nul
goto loop
