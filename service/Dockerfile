# ReadForMe VLM service. GPU image: runs North-Micro-Vision through transformers.
# For vLLM serving instead, run `vllm serve CohereLabs/North-Micro-Vision-Instruct` and
# set READFORME_BACKEND=openai READFORME_VLM_URL=http://vllm:8000/v1 on this container.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml ./
COPY readforme ./readforme
COPY eval ./eval
RUN pip install --no-cache-dir -e ".[transformers]"
ENV READFORME_BACKEND=transformers
EXPOSE 8090
CMD ["uvicorn", "readforme.server:app", "--host", "0.0.0.0", "--port", "8090"]
