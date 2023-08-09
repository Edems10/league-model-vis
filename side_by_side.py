from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

font_size = 30
save_image_dir = os.path.join(os.path.dirname(__file__), 'output', 'model_comparison', '3117488')
os.makedirs(save_image_dir, exist_ok=True)

for enum, current in tqdm(enumerate(range(7640, 9760, 10)), total=(9760-7640)//10, desc='Processing images'):
    current = f'{current}_movements.png'
    model_path_fow = os.path.join(os.path.dirname(__file__), 'output', 'macro_prediction_fow', 'position_predictions', '3117488', current)
    model_path_old = os.path.join(os.path.dirname(__file__), 'output', 'macro_prediction_muller', 'position_predictions', '3117488', current)
    
    save_image_path = os.path.join(save_image_dir, f'{enum}.png')

    image_fow = Image.open(model_path_fow)
    image_old = Image.open(model_path_old)

    image_fow = image_fow.resize((image_fow.width * image_old.height // image_fow.height, image_old.height))
    result_image = Image.new('RGB', (image_old.width + image_fow.width, image_old.height + font_size + 10))

    font = ImageFont.truetype("arial.ttf", font_size) 

    draw = ImageDraw.Draw(result_image)
    draw.text((10, 5), "Old Model", font=font, fill="white")
    draw.text((image_old.width + 10, 5), "Fow Model", font=font, fill="white")

    # Paste images below text
    result_image.paste(image_old, (0, font_size + 10))
    result_image.paste(image_fow, (image_old.width, font_size + 10))
    result_image.save(save_image_path)
