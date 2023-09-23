


def get_midjourney_prompt(title,style):
    gpt_prompt = f"""As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image.

                            Please adhere to the structure and formatting below, and follow these guidelines:

                            Do not use the words "description" or ":" in any form.
                            Do not place a comma between [ar] and [v].
                            Write each prompt in one line without using return.
                            Structure:
                            [1] = ${title}
                            [2] = a detailed description of [1] with specific imagery details.
                            [3] = a detailed description of the scene's environment.
                            [4] = a detailed description of the compositions.
                            [5] = a detailed description of the scene's mood, feelings, and atmosphere.
                            [6] = A {style} style for [1].
                            [7] =  a detailed description of the scene's mood, feelings, and atmosphere.
                            [ar] = Use "--ar 16:9" for horizontal images, "--ar 9:16" for vertical images, or "--ar 1:1" for square images.
                            [v] = Use "--niji" for Japanese art style, or "--v 5" for other styles.

                            Formatting:
                            Follow this prompt structure: "/imagine prompt: [1], [2], [3], [4], [5], [6], [7], [ar] [v]".

                            Your task: Create 1 distinct prompts for each concept [1], varying in details description, environment,compositions,atmosphere, and realization.

                            Write your prompts in english.
                            Do not describe unreal concepts as "real" or "photographic".
                            Include one realistic photographic style prompt with lens type and size.
                            Separate different prompts with two new lines.
                            Example Prompts:

                            /imagine prompt: cute dog, fluffy fur, wagging tail, playful expression, sitting on a grassy field, under a clear blue sky, with a colorful collar, in a natural and vibrant setting, by a lake, captured with a Nikon D750 camera, 50mm lens, shallow depth of field, composition focused on the dog's face, capturing its joyful spirit, in a style reminiscent of William Wegman's iconic dog portraits. --ar 1:1 --v 5.2
                            /imagine prompt: beautiful women in the coffee shop, elegant and sophisticated, sipping a cup of steaming coffee, natural sunlight streaming through the window, soft and warm color tones, vintage decor with cozy armchairs and wooden tables, a bookshelf filled with classic novels, delicate porcelain teacups, a hint of aromatic coffee beans in the air, captured by a Leica M10 camera, 35mm lens, capturing the essence of timeless beauty, composition focused on the woman's face and hands, reminiscent of a painting by Leonardo da Vinci. --ar 1:1 --v 5.2
                            /imagine prompt:professional photograph of organic pea protein powder packaged in high end packaging - recyclable material, eye level, warm cinematic, Sony A7 105mm, close-up, centred shot, octane render --ar 2:1 --v 5"""
    reply = [{'role': 'user', 'content': gpt_prompt}]
    completion = openai.ChatCompletion.create(model="gpt-4", messages=reply)
    p = completion['choices'][0]['message']['content']
    prompt = re.findall('/imagine prompt:(.*--v 5)', p)[0]
    return prompt