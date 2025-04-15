from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()


@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}


mount_chainlit(app=app, target="chat_interface.py", path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
