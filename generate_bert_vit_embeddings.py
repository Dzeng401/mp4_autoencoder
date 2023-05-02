from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "google/vit-base-patch16-224", "bert-base-uncased"
)



# contrastive training
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
)
outputs = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    pixel_values=inputs.pixel_values,
    return_loss=False,
)

print(outputs.image_embeds.shape, outputs.text_embeds.shape)


print('done')