@echo off
set "source=.\dependencies\SDL2-2.30.4\lib\x64\SDL2.dll"
set "destination=.\bin\"
set "dllpath=.\bin\SDL2.dll"

echo Verifying the existance of SDL2.dll file in bin folder
if exist "%dllpath%" (
    echo SDL2.dll file exists in bin folder
) else (
    echo Copying SDL2.dll to bin
    xcopy "%source%" "%destination%" /Y
    echo File copied
)

pause
