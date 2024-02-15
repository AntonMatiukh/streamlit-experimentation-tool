set "timeout_period=10"

if not "%1" == "" (
    set "timeout_period=%1"
)

docker build -f Dockerfile -t experimentation_tool:latest .

docker run -d -p 8501:8501 experimentation_tool:latest

timeout /t %timeout_period%

for /f %%i in ('docker ps -q -f "ancestor=experimentation_tool:latest"') do set container_id=%%i

echo %container_id%

docker stop %container_id%