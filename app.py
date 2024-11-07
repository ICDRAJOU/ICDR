import gradio as gr
from inference import infer

def greet(image, prompt):
    restore_img = infer(img=image, text_prompt=prompt)
    return restore_img

title = "🖼️ ICDR 🖼️"
description = ''' ## ICDR: Image Restoration Framework for Composite Degradation following Human Instructions
Our Github : https://github.com/

Siwon Kim, Donghyeon Yoon

Ajou Univ
'''


article = "<p style='text-align: center'><a href='https://github.com/' target='_blank'>ICDR</a></p>"

#### Image,Prompts examples
examples = [['input/00013.png', "Remove the rain as much as possible like the picture taken on a clear day."],
            ['input/00010.png', "I love this photo, could you remove the haze and more brighter?"],
            ['input/00058.png', "I have to post an emotional shot on Instagram, but it was shot too foggy and too dark. Change it like a sunny day and brighten it up!"],
            ['input/00075.png', "Remove the rain from the video, remove the brightness and fog"],
            ]

css = """
    .image-frame img, .image-container img {
        width: auto;
        height: auto;
        max-width: none;
    }
"""


demo = gr.Interface(
        fn=greet, 
        inputs=[gr.Image(type="pil", label="Input"),
                gr.Text(label="Prompt") ],
        outputs=[gr.Image(type="pil", label="Ouput")],
        title=title,
        description=description,
        article=article,
        examples=examples,
        css=css,
    )

if __name__ == "__main__":
    demo.launch()