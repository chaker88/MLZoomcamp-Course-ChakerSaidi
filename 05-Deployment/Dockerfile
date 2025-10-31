# 1️⃣ Base image with the prebuilt model pipeline
FROM agrigorev/zoomcamp-model:2025

# 2️⃣ Set the working directory inside the container
WORKDIR /app

# 3️⃣ Make sure the container uses the uv virtual environment for Python
ENV PATH="/app/.venv/bin:$PATH"

# 4️⃣ Copy all project files into the container
COPY . .

# 5️⃣ Install uv and uvicorn inside the container
RUN pip install uv "uvicorn[standard]"

# 6️⃣ Install project dependencies using uv
RUN uv sync --locked

# 7️⃣ Expose the port your app will listen on
EXPOSE 9696

# 8️⃣ ENTRYPOINT to run the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
