docker build -f Dockerfile -t experimentation_tool:latest .
docker run -p 8501:8501 experimentation_tool:latest