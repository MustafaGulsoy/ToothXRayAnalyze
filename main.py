# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

processor = AutoImageProcessor.from_pretrained("Sankpan/vit-teeth-segment")
model = SegformerForSemanticSegmentation.from_pretrained("Sankpan/vit-teeth-segment")