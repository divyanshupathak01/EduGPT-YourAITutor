import os
import time

# import your Google/OpenAI/Groq key (put in your .env file)
try:
    with open(".env", "r") as f:
        env_file = f.readlines()
    envs_dict = {
        key.strip("'"): value.strip("\n")
        for key, value in [i.split("=", 1) for i in env_file if "=" in i]
    }
    os.environ["GROQ_API_KEY"] = envs_dict.get(
        "GROQ_API_KEY", envs_dict.get("OPENAI_API_KEY", envs_dict.get("GOOGLE_API_KEY", ""))
    )
except FileNotFoundError:
    pass

import gradio as gr
from generating_syllabus import generate_syllabus
from teaching_agent import teaching_agent

with gr.Blocks() as demo:
    gr.Markdown("# Your AI Instructor")
    with gr.Tab("Input Your Information"):

        def perform_task(input_text):
            import traceback
            try:
                # Perform the desired task based on the user input
                task = (
                    "Generate a course syllabus to teach the topic: " + input_text
                )
                syllabus = generate_syllabus(input_text, task)
                teaching_agent.seed_agent(syllabus, task)
                return syllabus
            except Exception as e:
                error_msg = traceback.format_exc()
                print(error_msg)
                return f"ERROR OCCURRED:\n\n{error_msg}"

        text_input = gr.Textbox(
            label="State the name of topic you want to learn:"
        )
        text_output = gr.Textbox(label="Your syllabus will be showed here:")
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(perform_task, text_input, text_output)
    with gr.Tab("AI Instructor"):
        #       inputbox = gr.Textbox("Input your text to build a Q&A Bot here.....")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="What do you concern about?")
        clear = gr.Button("Clear")

        def user(user_message, history):
            teaching_agent.human_step(user_message)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": ""})
            return "", history

        def bot(history):
            bot_message = teaching_agent.instructor_step()
            history[-1]["content"] = bot_message
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
