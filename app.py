from src.components.predictor import predict
import gradio as gr

if __name__=="__main__":
    print("Launching the chat interface...")
    gr.ChatInterface(predict).launch()
